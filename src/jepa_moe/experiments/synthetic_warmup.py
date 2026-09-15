"""Test whether delaying Jacobian specialization repairs full-IID MoE learning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch
from torch import Tensor

from ..evaluation import evaluate_iid_predictor, one_step_mse
from ..losses import control_jacobian_specialization_terms
from ..models import JacobianMoEPredictor, next_state_mse
from ..synthetic import TransitionBatch
from .synthetic_basis import (
    ExperimentConfig,
    ExperimentData,
    ShuffledBatchStream,
    _append_json_line,
    _cpu_state_dict,
    _moe_diagnostics,
    _scalar,
    _seed_everything,
    make_experiment_data,
)


CONDITIONS: dict[str, int | None] = {
    "vanilla": None,
    "immediate": 0,
    "warmup_250": 250,
    "warmup_500": 500,
    "warmup_1000": 1000,
}
RAMP_STEPS = 500
GRADIENT_INTERVAL = 250
GRADIENT_EPS = 1e-12


def specialization_weight(
    condition: str, step: int, *, lambda_jac: float = 0.1
) -> float:
    """Keep step <= warmup prediction-only; finish the ramp at W + 500."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if step < 0 or lambda_jac < 0:
        raise ValueError("step and lambda_jac must be non-negative")
    warmup = CONDITIONS[condition]
    if warmup is None:
        return 0.0
    if warmup == 0:
        return lambda_jac
    return lambda_jac * min(max(step - warmup, 0) / RAMP_STEPS, 1.0)


def _expert_grad_norm(grads: tuple[Tensor | None, ...]) -> float:
    return math.sqrt(
        sum(float(grad.detach().double().square().sum()) for grad in grads if grad is not None)
    )


def expert_gradient_probe(
    model: JacobianMoEPredictor,
    state: Tensor,
    action: Tensor,
    next_state: Tensor,
    *,
    condition: str,
    step: int,
    config: ExperimentConfig,
) -> dict[str, float | int]:
    """Measure both expert gradients on the exact same batch without writing .grad."""

    weight = specialization_weight(condition, step, lambda_jac=config.lambda_jac)
    prediction = next_state_mse(model(state, action).pred, next_state)
    terms = control_jacobian_specialization_terms(
        model,
        state,
        action,
        create_graph=True,
        margin=config.margin,
        min_jacobian_norm=config.min_jacobian_norm,
        beta_activity=config.beta_activity,
    )
    parameters = tuple(model.experts.parameters())
    prediction_grads = torch.autograd.grad(
        prediction, parameters, retain_graph=True, allow_unused=True
    )
    if terms.total.requires_grad:
        specialization_grads = torch.autograd.grad(
            terms.total, parameters, allow_unused=True
        )
        specialization_norm = _expert_grad_norm(specialization_grads)
    else:
        specialization_norm = 0.0
    prediction_norm = _expert_grad_norm(prediction_grads)
    return {
        "step": step,
        "lambda_jac": weight,
        "prediction_loss": _scalar(prediction),
        "jacobian_diversity_loss": _scalar(terms.diversity),
        "activity_loss": _scalar(terms.activity),
        "specialization_loss": _scalar(terms.total),
        "expert_prediction_grad_norm": prediction_norm,
        "expert_specialization_grad_norm_unweighted": specialization_norm,
        "expert_specialization_grad_norm_weighted": weight * specialization_norm,
        "r_grad": weight * specialization_norm / (prediction_norm + GRADIENT_EPS),
    }


def _training_loss(
    model: JacobianMoEPredictor,
    state: Tensor,
    action: Tensor,
    next_state: Tensor,
    *,
    weight: float,
    config: ExperimentConfig,
) -> tuple[Tensor, Tensor]:
    if weight == 0:
        prediction = next_state_mse(model(state, action).pred, next_state)
        return prediction, prediction
    loss = model.training_loss(
        state,
        action,
        next_state,
        lambda_jac=weight,
        margin=config.margin,
        min_jacobian_norm=config.min_jacobian_norm,
        beta_activity=config.beta_activity,
    )
    return loss.total, loss.prediction


def train_condition(
    condition: str,
    seed: int,
    config: ExperimentConfig,
    data: ExperimentData,
    *,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the same IID predictor and batch stream with one schedule change."""

    if condition == "vanilla":
        raise ValueError("vanilla is imported from the identical full-IID run")
    _seed_everything(seed)
    model = JacobianMoEPredictor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    stream = ShuffledBatchStream(
        data.train, config.batch_size, seed=config.data_seed + 10_000 + seed
    )
    diagnostic_batch = TransitionBatch(
        data.train.state[: config.jacobian_batch_size],
        data.train.action[: config.jacobian_batch_size],
        data.train.next_state[: config.jacobian_batch_size],
    )
    log_path = output_dir / "logs" / f"{condition}_seed_{seed}.jsonl"
    grad_path = output_dir / "gradients" / f"{condition}_seed_{seed}.jsonl"
    checkpoint_path = output_dir / "checkpoints" / f"{condition}_seed_{seed}.pt"
    start = time.perf_counter()
    best_validation_mse = math.inf
    best_step = -1
    best_state: dict[str, Tensor] | None = None

    def validate(step: int, train_mse: float | None) -> None:
        nonlocal best_validation_mse, best_step, best_state
        validation_mse = one_step_mse(
            model, data.validation, device=device, batch_size=config.evaluation_batch_size
        )
        record: dict[str, Any] = {
            "step": step,
            "lambda_jac": specialization_weight(
                condition, step, lambda_jac=config.lambda_jac
            ),
            "train_prediction_loss": train_mse,
            "validation_prediction_loss": validation_mse,
            "elapsed_seconds": time.perf_counter() - start,
        }
        record.update(_moe_diagnostics(model, diagnostic_batch, config, device))
        _append_json_line(log_path, record)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_step = step
            best_state = _cpu_state_dict(model)
        model.train()

    validate(0, None)
    initial = diagnostic_batch
    _append_json_line(
        grad_path,
        expert_gradient_probe(
            model,
            initial.state.to(device),
            initial.action.to(device),
            initial.next_state.to(device),
            condition=condition,
            step=0,
            config=config,
        ),
    )
    for step in range(1, config.training_steps + 1):
        batch = stream.next()
        state = batch.state.to(device)
        action = batch.action.to(device)
        target = batch.next_state.to(device)
        weight = specialization_weight(condition, step, lambda_jac=config.lambda_jac)
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % GRADIENT_INTERVAL == 0:
            _append_json_line(
                grad_path,
                expert_gradient_probe(
                    model,
                    state,
                    action,
                    target,
                    condition=condition,
                    step=step,
                    config=config,
                ),
            )
        total, prediction = _training_loss(
            model, state, action, target, weight=weight, config=config
        )
        total.backward()
        optimizer.step()
        if step % config.validation_interval == 0 or step == config.training_steps:
            validate(step, _scalar(prediction))
            print(
                f"[{condition} seed={seed}] {step}/{config.training_steps} "
                f"lambda={weight:.4g} best_val={best_validation_mse:.6g}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint = {
        "condition": condition,
        "seed": seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "state_dict": best_state,
        "config": asdict(config),
        "ramp_steps": RAMP_STEPS,
        "gradient_interval": GRADIENT_INTERVAL,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics = evaluate_iid_predictor(
        model,
        iid_test=data.iid_test,
        iid_rollout=data.iid_rollout,
        basis_probe=data.basis_probe,
        device=device,
        prediction_batch_size=config.evaluation_batch_size,
        jacobian_batch_size=config.jacobian_batch_size,
    )
    result = {
        "condition": condition,
        "seed": seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "checkpoint": str(checkpoint_path),
        "training_seconds": time.perf_counter() - start,
        **metrics,
    }
    result_path = output_dir / "results" / f"{condition}_seed_{seed}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def import_vanilla(
    seed: int, *, source_dir: Path, output_dir: Path
) -> dict[str, Any]:
    source = json.loads(
        (source_dir / "results" / f"vanilla_moe_seed_{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_path = source_dir / "checkpoints" / f"vanilla_moe_seed_{seed}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    result = {
        **source,
        "condition": "vanilla",
        "checkpoint": str(checkpoint_path),
        "reused_iid_checkpoint": True,
        "gradient_logging": "r_grad is identically zero because lambda_jac=0",
    }
    result_path = output_dir / "results" / f"vanilla_seed_{seed}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_results(
    results: list[dict[str, Any]], config: ExperimentConfig
) -> dict[str, Any]:
    metrics = (
        "iid_one_step_mse",
        "iid_rollout_mse",
        "expert_redundancy_mean_abs_cosine",
        "dynamics_subspace_principal_cosine_similarity",
        "dynamics_subspace_projection_reconstruction_error",
        "normalized_routing_usage_entropy",
        "minimum_mean_routing_weight",
        "basis_recovery_score",
    )
    aggregate: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [result for result in results if result["condition"] == condition]
        values: dict[str, Any] = {"seeds": sorted(row["seed"] for row in rows)}
        for metric in metrics:
            samples = [float(row[metric]) for row in rows if metric in row]
            if samples:
                values[metric] = _mean_std(samples)
        aggregate[condition] = values

    lower_is_better = {
        "iid_one_step_mse": True,
        "iid_rollout_mse": True,
        "expert_redundancy_mean_abs_cosine": True,
        "dynamics_subspace_principal_cosine_similarity": False,
        "dynamics_subspace_projection_reconstruction_error": True,
        "normalized_routing_usage_entropy": False,
    }
    indexed = {
        (row["condition"], int(row["seed"])): row
        for row in results
    }
    comparisons: dict[str, Any] = {}
    for condition in ("warmup_250", "warmup_500", "warmup_1000"):
        comparisons[condition] = {}
        for reference in ("immediate", "vanilla"):
            seeds = sorted(
                seed
                for seed in config.seeds
                if (condition, seed) in indexed and (reference, seed) in indexed
            )
            if not seeds:
                comparisons[condition][reference] = {}
                continue
            metric_comparisons = {}
            for metric, minimize in lower_is_better.items():
                candidate = [float(indexed[(condition, seed)][metric]) for seed in seeds]
                baseline = [float(indexed[(reference, seed)][metric]) for seed in seeds]
                candidate_mean = statistics.fmean(candidate)
                reference_mean = statistics.fmean(baseline)
                metric_comparisons[metric] = {
                    "candidate_mean": candidate_mean,
                    "reference_mean": reference_mean,
                    "ratio_of_means": candidate_mean / reference_mean,
                    "mean_difference": candidate_mean - reference_mean,
                    "paired_wins": sum(
                        left < right if minimize else left > right
                        for left, right in zip(candidate, baseline, strict=True)
                    ),
                    "paired_seeds": len(seeds),
                }
            comparisons[condition][reference] = metric_comparisons
    return {
        "config": asdict(config),
        "ramp_steps": RAMP_STEPS,
        "gradient_interval": GRADIENT_INTERVAL,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
    }


def summarize_gradients(output_dir: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Aggregate pre-update same-batch probes across completed runs."""

    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        if condition == "vanilla":
            output[condition] = {"gradient_ratio_is_zero_by_design": True}
            continue
        step_rows: dict[int, list[dict[str, Any]]] = {}
        for seed in seeds:
            result_path = output_dir / "results" / f"{condition}_seed_{seed}.json"
            path = output_dir / "gradients" / f"{condition}_seed_{seed}.jsonl"
            if not result_path.exists() or not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                step_rows.setdefault(row["step"], []).append(row)
        by_step = {}
        for step, rows in sorted(step_rows.items()):
            by_step[str(step)] = {
                "seeds": len(rows),
                **{
                    key: _mean_std([float(row[key]) for row in rows])
                    for key in (
                        "lambda_jac",
                        "prediction_loss",
                        "specialization_loss",
                        "expert_prediction_grad_norm",
                        "expert_specialization_grad_norm_unweighted",
                        "expert_specialization_grad_norm_weighted",
                        "r_grad",
                    )
                },
            }
        output[condition] = {"by_step": by_step}
    return output


def render_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Delayed Jacobian specialization on full IID",
        "",
        "Shared protocol: 50k/10k/10k full-IID splits, 10k rollout-25 trajectories, "
        f"{summary['config']['training_steps']} updates, batch "
        f"{summary['config']['batch_size']}, AdamW lr="
        f"{summary['config']['learning_rate']}, five matched seeds, "
        "validation prediction MSE-only checkpoint selection. Vanilla reuses the "
        "identical full-IID baseline; all other conditions are retrained.",
        "",
        "Warm-up conditions use prediction alone through W, ramp lambda linearly "
        "from zero after W to 0.1 at W+500, then keep 0.1. Immediate uses 0.1 "
        "at every optimizer update.",
        "",
        "## Aggregate results",
        "",
        "| Condition | IID one-step MSE | Rollout-25 MSE | Redundancy | Principal similarity | Projection error | Usage entropy | Minimum usage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = (
        "iid_one_step_mse",
        "iid_rollout_mse",
        "expert_redundancy_mean_abs_cosine",
        "dynamics_subspace_principal_cosine_similarity",
        "dynamics_subspace_projection_reconstruction_error",
        "normalized_routing_usage_entropy",
        "minimum_mean_routing_weight",
    )
    for condition in CONDITIONS:
        values = aggregate[condition]
        formatted = [
            (
                f"{values[key]['mean']:.6g} +/- {values[key]['std']:.3g}"
                if key in values else "n/a"
            )
            for key in keys
        ]
        lines.append(f"| {condition} | " + " | ".join(formatted) + " |")

    lines.extend(
        [
            "",
            "## Per-seed results",
            "",
            "| Condition | Seed | Best step | IID MSE | Rollout MSE | Redundancy | Principal similarity | Projection error | Usage entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(results, key=lambda result: (result["condition"], result["seed"])):
        lines.append(
            f"| {row['condition']} | {row['seed']} | {row['best_step']} | "
            f"{row['iid_one_step_mse']:.6g} | {row['iid_rollout_mse']:.6g} | "
            f"{row['expert_redundancy_mean_abs_cosine']:.4f} | "
            f"{row['dynamics_subspace_principal_cosine_similarity']:.4f} | "
            f"{row['dynamics_subspace_projection_reconstruction_error']:.4f} | "
            f"{row['normalized_routing_usage_entropy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired warm-up comparisons",
            "",
            "Ratios use aggregate means (below 1 is better for errors and "
            "redundancy). Deltas are candidate minus reference. Wins count "
            "matched seeds in the favorable direction.",
        ]
    )
    for reference in ("immediate", "vanilla"):
        lines.extend(
            [
                "",
                f"### Relative to {reference}",
                "",
                "| Condition | IID ratio (wins) | Rollout ratio (wins) | Redundancy ratio (wins) | Principal delta (wins) | Projection ratio (wins) | Entropy delta (wins) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in ("warmup_250", "warmup_500", "warmup_1000"):
            comparison = summary["paired_comparisons"][condition][reference]
            if not comparison:
                lines.append(f"| {condition} | " + " | ".join(["n/a"] * 6) + " |")
                continue

            def ratio(metric: str) -> str:
                row = comparison[metric]
                return (
                    f"{row['ratio_of_means']:.3f} "
                    f"({row['paired_wins']}/{row['paired_seeds']})"
                )

            def delta(metric: str) -> str:
                row = comparison[metric]
                return (
                    f"{row['mean_difference']:+.4f} "
                    f"({row['paired_wins']}/{row['paired_seeds']})"
                )

            lines.append(
                f"| {condition} | {ratio('iid_one_step_mse')} | "
                f"{ratio('iid_rollout_mse')} | "
                f"{ratio('expert_redundancy_mean_abs_cosine')} | "
                f"{delta('dynamics_subspace_principal_cosine_similarity')} | "
                f"{ratio('dynamics_subspace_projection_reconstruction_error')} | "
                f"{delta('normalized_routing_usage_entropy')} |"
            )
    lines.extend(
        [
            "",
            "## Expert gradient ratios during training",
            "",
            "`r_grad = ||lambda(step) * grad_expert(L_spec)|| / "
            "(||grad_expert(L_pred)|| + 1e-12)`; measured before the update on "
            "the same batch. Values are mean +/- sample std across completed seeds.",
            "",
            "| Condition | Step 0 | Step 1 | Step 250 | Step 500 | Step 750 | Step 1000 | Step 1250 | Step 1500 | Step 2000 | Step 5000 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    gradient_summary = summary.get("gradients", {})
    for condition in CONDITIONS:
        if condition == "vanilla":
            lines.append("| vanilla | " + " | ".join(["0 by design"] * 10) + " |")
            continue
        by_step = gradient_summary.get(condition, {}).get("by_step", {})
        values = []
        for step in (0, 1, 250, 500, 750, 1000, 1250, 1500, 2000, 5000):
            row = by_step.get(str(step))
            values.append(
                f"{row['r_grad']['mean']:.3g} +/- {row['r_grad']['std']:.2g}"
                if row is not None else "n/a"
            )
        lines.append(f"| {condition} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "`gradients/*.jsonl` contains the pre-update expert gradient ratio on "
            "the same shuffled training batch at step 1 and every 250 updates; "
            "step 0 is an unshuffled, shared fixed-batch initialization probe. "
            "Unweighted specialization gradient is also recorded while lambda is zero. "
            "The probe never writes optimizer gradients. Vanilla has algebraically "
            "zero weighted specialization gradient and reuses prior checkpoints.",
            "",
            "Hungarian matching remains an auxiliary metric in `results/*.json`, "
            "not a basis target or success criterion.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/synthetic_warmup_v1")
    )
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/synthetic_iid_v1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ExperimentConfig.seeds))
    parser.add_argument("--run-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--training-steps", type=int, default=ExperimentConfig.training_steps)
    parser.add_argument("--validation-interval", type=int, default=ExperimentConfig.validation_interval)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--train-samples", type=int, default=ExperimentConfig.train_samples)
    parser.add_argument("--validation-samples", type=int, default=ExperimentConfig.validation_samples)
    parser.add_argument("--iid-test-samples", type=int, default=ExperimentConfig.iid_test_samples)
    parser.add_argument("--rollout-trajectories", type=int, default=ExperimentConfig.rollout_trajectories)
    parser.add_argument("--basis-probe-samples", type=int, default=ExperimentConfig.basis_probe_samples)
    parser.add_argument("--lambda-jac", type=float, default=ExperimentConfig.lambda_jac)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = _parse_args()
    config = ExperimentConfig(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        iid_test_samples=args.iid_test_samples,
        rollout_trajectories=args.rollout_trajectories,
        basis_probe_samples=args.basis_probe_samples,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        validation_interval=args.validation_interval,
        lambda_jac=args.lambda_jac,
        seeds=tuple(args.seeds),
    )
    if min(
        config.train_samples, config.validation_samples, config.iid_test_samples,
        config.rollout_trajectories, config.basis_probe_samples, config.batch_size,
        config.training_steps, config.validation_interval,
    ) <= 0:
        raise ValueError("all sizes and training intervals must be positive")
    if config.batch_size > config.train_samples:
        raise ValueError("batch size cannot exceed training split size")
    run_seeds = config.seeds if args.run_seeds is None else tuple(args.run_seeds)
    if not run_seeds or not set(run_seeds).issubset(config.seeds):
        raise ValueError("--run-seeds must be a non-empty subset of --seeds")

    source_config_path = args.source_dir / "config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    normalized_config = json.loads(json.dumps(asdict(config)))
    if source_config != normalized_config:
        raise RuntimeError("full-IID source protocol differs; use matching settings")

    output_dir: Path = args.output_dir
    for child in ("checkpoints", "gradients", "logs", "results"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    serialized = json.dumps(
        {"config": normalized_config, "ramp_steps": RAMP_STEPS,
         "gradient_interval": GRADIENT_INTERVAL, "source_dir": str(args.source_dir)},
        indent=2, sort_keys=True,
    ) + "\n"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"{config_path} contains a different protocol")
    config_path.write_text(serialized, encoding="utf-8")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Generating shared full-IID data; device={device}", flush=True)
    data = make_experiment_data(config)
    for seed in run_seeds:
        for condition in args.conditions:
            path = output_dir / "results" / f"{condition}_seed_{seed}.json"
            if args.resume and path.exists():
                print(f"[resume] {path}", flush=True)
                continue
            if condition == "vanilla":
                import_vanilla(seed, source_dir=args.source_dir, output_dir=output_dir)
            else:
                train_condition(
                    condition, seed, config, data, device=device, output_dir=output_dir
                )

    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for seed in config.seeds for condition in CONDITIONS
        if (path := output_dir / "results" / f"{condition}_seed_{seed}.json").exists()
    ]
    summary = summarize_results(results, config)
    summary["gradients"] = summarize_gradients(output_dir, config.seeds)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_report(summary, results), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
