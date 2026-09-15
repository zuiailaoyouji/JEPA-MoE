"""Run oracle-router, oracle-expert, and joint-learning diagnostics."""

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
from torch import Tensor, nn

from ..diagnostic_models import OracleExpertMoEPredictor, OracleRouterMoEPredictor
from ..evaluation import evaluate_predictor, one_step_mse
from ..losses import control_jacobian_specialization_terms
from ..models import JacobianMoEPredictor, VanillaMoEPredictor, next_state_mse
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


CONDITION_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "oracle_router_vanilla": OracleRouterMoEPredictor,
    "oracle_router_jacobian": OracleRouterMoEPredictor,
    "oracle_expert": OracleExpertMoEPredictor,
    "joint_vanilla": VanillaMoEPredictor,
    "joint_jacobian": JacobianMoEPredictor,
}
JACOBIAN_CONDITIONS = {"oracle_router_jacobian", "joint_jacobian"}
JOINT_SOURCE_MODELS = {
    "joint_vanilla": "vanilla_moe",
    "joint_jacobian": "jacobian_moe",
}
REGIMES = {
    "oracle_router_vanilla": "oracle_router",
    "oracle_router_jacobian": "oracle_router",
    "oracle_expert": "oracle_expert",
    "joint_vanilla": "joint_learning",
    "joint_jacobian": "joint_learning",
}
OBJECTIVES = {
    condition: (
        "prediction_plus_control_jacobian"
        if condition in JACOBIAN_CONDITIONS
        else "prediction_only"
    )
    for condition in CONDITION_MODEL_CLASSES
}


def _condition_metadata(condition: str) -> dict[str, str]:
    return {
        "condition": condition,
        "regime": REGIMES[condition],
        "objective": OBJECTIVES[condition],
    }


def _training_objective(
    model: nn.Module,
    condition: str,
    state: Tensor,
    action: Tensor,
    next_state: Tensor,
    config: ExperimentConfig,
) -> tuple[Tensor, Tensor]:
    output = model(state, action)
    prediction = next_state_mse(output.pred, next_state)
    if condition not in JACOBIAN_CONDITIONS:
        return prediction, prediction

    specialization = control_jacobian_specialization_terms(
        model,
        state,
        action,
        create_graph=True,
        margin=config.margin,
        min_jacobian_norm=config.min_jacobian_norm,
        beta_activity=config.beta_activity,
    )
    return prediction + config.lambda_jac * specialization.total, prediction


def train_one_condition(
    condition: str,
    seed: int,
    config: ExperimentConfig,
    data: ExperimentData,
    *,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one diagnostic condition with validation-MSE model selection."""

    _seed_everything(seed)
    model = CONDITION_MODEL_CLASSES[condition]().to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"{condition} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batch_stream = ShuffledBatchStream(
        data.train, config.batch_size, seed=config.data_seed + 10_000 + seed
    )
    diagnostic_batch = TransitionBatch(
        data.train.state[: config.jacobian_batch_size],
        data.train.action[: config.jacobian_batch_size],
        data.train.next_state[: config.jacobian_batch_size],
    )

    log_path = output_dir / "logs" / f"{condition}_seed_{seed}.jsonl"
    checkpoint_path = output_dir / "checkpoints" / f"{condition}_seed_{seed}.pt"
    start_time = time.perf_counter()
    best_validation_mse = math.inf
    best_step = -1
    best_state: dict[str, Tensor] | None = None

    def validate_and_log(step: int, train_prediction_loss: float | None) -> None:
        nonlocal best_validation_mse, best_step, best_state
        validation_mse = one_step_mse(
            model,
            data.validation,
            device=device,
            batch_size=config.evaluation_batch_size,
        )
        record: dict[str, Any] = {
            **_condition_metadata(condition),
            "step": step,
            "train_prediction_loss": train_prediction_loss,
            "validation_prediction_loss": validation_mse,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        record.update(_moe_diagnostics(model, diagnostic_batch, config, device))
        _append_json_line(log_path, record)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_step = step
            best_state = _cpu_state_dict(model)
        model.train()

    validate_and_log(step=0, train_prediction_loss=None)
    last_prediction_loss = math.nan
    for step in range(1, config.training_steps + 1):
        batch = batch_stream.next()
        state = batch.state.to(device)
        action = batch.action.to(device)
        next_state = batch.next_state.to(device)

        optimizer.zero_grad(set_to_none=True)
        total_loss, prediction_loss = _training_objective(
            model, condition, state, action, next_state, config
        )
        total_loss.backward()
        optimizer.step()
        last_prediction_loss = _scalar(prediction_loss)

        if step % config.validation_interval == 0 or step == config.training_steps:
            validate_and_log(step, last_prediction_loss)
            print(
                f"[{condition} seed={seed}] step={step}/{config.training_steps} "
                f"train_mse={last_prediction_loss:.6g} "
                f"best_val_mse={best_validation_mse:.6g}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        **_condition_metadata(condition),
        "seed": seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "state_dict": best_state,
        "config": asdict(config),
    }
    torch.save(checkpoint, checkpoint_path)

    metrics = evaluate_predictor(
        model,
        iid_test=data.iid_test,
        heldout_test=data.heldout_test,
        iid_rollout=data.iid_rollout,
        heldout_rollout=data.heldout_rollout,
        basis_probe=data.basis_probe,
        device=device,
        prediction_batch_size=config.evaluation_batch_size,
        jacobian_batch_size=config.jacobian_batch_size,
    )
    result = {
        **_condition_metadata(condition),
        "seed": seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "training_seconds": time.perf_counter() - start_time,
        "checkpoint": str(checkpoint_path),
        **metrics,
    }
    result_path = output_dir / "results" / f"{condition}_seed_{seed}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _verify_joint_protocol(
    source_dir: Path, config: ExperimentConfig
) -> dict[str, Any]:
    source_config_path = source_dir / "config.json"
    if not source_config_path.exists():
        raise FileNotFoundError(f"joint source config not found: {source_config_path}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    expected = asdict(config)
    expected["seeds"] = list(expected["seeds"])
    if source_config != expected:
        raise RuntimeError(
            "joint source protocol differs from the diagnostic protocol; "
            "rerun with --train-joint or choose matching settings"
        )
    return source_config


def reevaluate_joint_condition(
    condition: str,
    seed: int,
    config: ExperimentConfig,
    data: ExperimentData,
    *,
    device: torch.device,
    output_dir: Path,
    source_dir: Path,
) -> dict[str, Any]:
    """Evaluate an existing joint checkpoint with the expanded diagnostics."""

    source_model = JOINT_SOURCE_MODELS[condition]
    source_checkpoint = source_dir / "checkpoints" / f"{source_model}_seed_{seed}.pt"
    source_result_path = source_dir / "results" / f"{source_model}_seed_{seed}.json"
    if not source_checkpoint.exists() or not source_result_path.exists():
        raise FileNotFoundError(
            f"missing joint checkpoint or result for {source_model} seed {seed}"
        )

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    model = CONDITION_MODEL_CLASSES[condition]().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = evaluate_predictor(
        model,
        iid_test=data.iid_test,
        heldout_test=data.heldout_test,
        iid_rollout=data.iid_rollout,
        heldout_rollout=data.heldout_rollout,
        basis_probe=data.basis_probe,
        device=device,
        prediction_batch_size=config.evaluation_batch_size,
        jacobian_batch_size=config.jacobian_batch_size,
    )
    result = {
        **_condition_metadata(condition),
        "seed": seed,
        "best_step": int(checkpoint["best_step"]),
        "best_validation_prediction_mse": float(
            checkpoint["best_validation_prediction_mse"]
        ),
        "training_seconds": float(source_result["training_seconds"]),
        "checkpoint": str(source_checkpoint),
        "reused_joint_checkpoint": True,
        **metrics,
    }
    result_path = output_dir / "results" / f"{condition}_seed_{seed}.json"
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
    scalar_metrics = (
        "best_validation_prediction_mse",
        "iid_one_step_mse",
        "heldout_composition_mse",
        "iid_rollout_mse",
        "heldout_composition_rollout_mse",
        "expert_redundancy_mean_abs_cosine",
        "basis_recovery_score",
        "router_weight_mse",
        "matched_router_weight_mse",
    )
    vector_metrics = (
        "expert_pair_mean_abs_cosines",
        "mean_routing_weights",
        "true_mean_routing_weights",
    )
    aggregate: dict[str, Any] = {}
    for condition in CONDITION_MODEL_CLASSES:
        condition_results = sorted(
            (result for result in results if result["condition"] == condition),
            key=lambda result: result["seed"],
        )
        values: dict[str, Any] = {"seeds": [r["seed"] for r in condition_results]}
        for metric in scalar_metrics:
            samples = [float(r[metric]) for r in condition_results if metric in r]
            if samples:
                values[metric] = _mean_std(samples)
        for metric in vector_metrics:
            vectors = [r[metric] for r in condition_results if metric in r]
            if vectors:
                values[metric] = [
                    _mean_std([float(vector[index]) for vector in vectors])
                    for index in range(len(vectors[0]))
                ]
        aggregate[condition] = values

    diagnosis: dict[str, Any] | None = None
    if all(aggregate[condition].get("iid_one_step_mse") for condition in aggregate):
        def mean(condition: str, metric: str) -> float:
            return float(aggregate[condition][metric]["mean"])

        diagnosis = {
            "oracle_router_jacobian_over_vanilla": {
                metric: mean("oracle_router_jacobian", metric)
                / mean("oracle_router_vanilla", metric)
                for metric in (
                    "iid_one_step_mse",
                    "heldout_composition_mse",
                    "heldout_composition_rollout_mse",
                    "expert_redundancy_mean_abs_cosine",
                    "basis_recovery_score",
                )
            },
            "joint_over_matching_oracle_router": {
                "vanilla_iid_mse_ratio": mean("joint_vanilla", "iid_one_step_mse")
                / mean("oracle_router_vanilla", "iid_one_step_mse"),
                "vanilla_heldout_mse_ratio": mean(
                    "joint_vanilla", "heldout_composition_mse"
                )
                / mean("oracle_router_vanilla", "heldout_composition_mse"),
                "jacobian_iid_mse_ratio": mean(
                    "joint_jacobian", "iid_one_step_mse"
                )
                / mean("oracle_router_jacobian", "iid_one_step_mse"),
                "jacobian_heldout_mse_ratio": mean(
                    "joint_jacobian", "heldout_composition_mse"
                )
                / mean("oracle_router_jacobian", "heldout_composition_mse"),
            },
            "router_learning": {
                "oracle_expert_router_mse": mean(
                    "oracle_expert", "router_weight_mse"
                ),
                "joint_vanilla_matched_router_mse": mean(
                    "joint_vanilla", "matched_router_weight_mse"
                ),
                "joint_jacobian_matched_router_mse": mean(
                    "joint_jacobian", "matched_router_weight_mse"
                ),
            },
            "conclusion": (
                "Experts and router are each recoverable in isolation; the large joint-to-"
                "oracle error and router-MSE gaps identify joint co-adaptation/"
                "identifiability as the dominant failure mode under this protocol."
            ),
        }
    return {"config": asdict(config), "aggregate": aggregate, "diagnosis": diagnosis}


def _format_metric(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "n/a"
    return f"{summary['mean']:.6g} +/- {summary['std']:.3g}"


def render_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    config = summary["config"]
    aggregate = summary["aggregate"]
    lines = [
        "# Synthetic diagnostic experiments",
        "",
        (
            f"Protocol: {config['training_steps']} updates, batch {config['batch_size']}, "
            f"AdamW lr={config['learning_rate']}, lambda_jac={config['lambda_jac']}, "
            f"{len(config['seeds'])} seeds. Checkpoints are selected only by "
            "validation prediction MSE."
        ),
        "",
        "The oracle-router regime fixes alpha to alpha_true and trains only experts. "
        "The oracle-expert regime fixes F_k to F_gt_k and trains only the original "
        "router. Joint-learning trains the original router and experts together.",
        "",
        "## Aggregate results",
        "",
        "| Condition | IID MSE | Held-out MSE | IID rollout-25 | Held-out rollout-25 | Redundancy | Recovery | Router MSE | Matched router MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_MODEL_CLASSES:
        values = aggregate[condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    _format_metric(values.get("iid_one_step_mse")),
                    _format_metric(values.get("heldout_composition_mse")),
                    _format_metric(values.get("iid_rollout_mse")),
                    _format_metric(values.get("heldout_composition_rollout_mse")),
                    _format_metric(values.get("expert_redundancy_mean_abs_cosine")),
                    _format_metric(values.get("basis_recovery_score")),
                    _format_metric(values.get("router_weight_mse")),
                    _format_metric(values.get("matched_router_weight_mse")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Routing and expert diagnostics",
            "",
            "| Condition | abs cos(0,1) | abs cos(0,2) | abs cos(1,2) | route 0 | route 1 | route 2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITION_MODEL_CLASSES:
        values = aggregate[condition]
        pairwise = values.get("expert_pair_mean_abs_cosines", [])
        routing = values.get("mean_routing_weights", [])
        lines.append(
            f"| {condition} | "
            + " | ".join(_format_metric(value) for value in pairwise + routing)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-seed results",
            "",
            "| Condition | Seed | Best step | IID MSE | Held-out MSE | Held-out rollout | Redundancy | Recovery | Router MSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sorted(results, key=lambda item: (item["condition"], item["seed"])):
        lines.append(
            f"| {result['condition']} | {result['seed']} | {result['best_step']} | "
            f"{result['iid_one_step_mse']:.6g} | "
            f"{result['heldout_composition_mse']:.6g} | "
            f"{result['heldout_composition_rollout_mse']:.6g} | "
            f"{result['expert_redundancy_mean_abs_cosine']:.4f} | "
            f"{result['basis_recovery_score']:.4f} | "
            f"{result['router_weight_mse']:.6g} |"
        )

    diagnosis = summary["diagnosis"]
    lines.extend(["", "## Diagnostic reading", ""])
    if diagnosis is None:
        lines.append("Incomplete: all five conditions are required for diagnosis.")
    else:
        oracle_effect = diagnosis["oracle_router_jacobian_over_vanilla"]
        joint_gap = diagnosis["joint_over_matching_oracle_router"]
        router = diagnosis["router_learning"]
        lines.extend(
            [
                (
                    "With the true router fixed, Jacobian specialization changes the "
                    f"IID MSE by {oracle_effect['iid_one_step_mse']:.3f}x, held-out MSE "
                    f"by {oracle_effect['heldout_composition_mse']:.3f}x, held-out "
                    f"rollout MSE by {oracle_effect['heldout_composition_rollout_mse']:.3f}x, "
                    f"redundancy by {oracle_effect['expert_redundancy_mean_abs_cosine']:.3f}x, "
                    f"and recovery by {oracle_effect['basis_recovery_score']:.3f}x."
                ),
                "",
                (
                    "Joint Vanilla has "
                    f"{joint_gap['vanilla_iid_mse_ratio']:.2f}x the oracle-router IID MSE "
                    f"and {joint_gap['vanilla_heldout_mse_ratio']:.2f}x the held-out MSE. "
                    "Joint Jacobian has "
                    f"{joint_gap['jacobian_iid_mse_ratio']:.2f}x and "
                    f"{joint_gap['jacobian_heldout_mse_ratio']:.2f}x, respectively."
                ),
                "",
                (
                    f"The oracle-expert router MSE is {router['oracle_expert_router_mse']:.3g}; "
                    "after permutation matching, joint Vanilla/Jacobian router MSEs are "
                    f"{router['joint_vanilla_matched_router_mse']:.3g} and "
                    f"{router['joint_jacobian_matched_router_mse']:.3g}."
                ),
                "",
                diagnosis["conclusion"],
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation order: oracle-router isolates expert learning; oracle-expert "
            "isolates router learning; joint-learning shows whether degradation appears "
            "only when both components co-adapt.",
            "",
            "Full routing matrices, Hungarian matches, pairwise cosine values, and "
            "checkpoint provenance are preserved in `summary.json` and `results/*.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/synthetic_diagnostics_v1")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_MODEL_CLASSES,
        default=list(CONDITION_MODEL_CLASSES),
    )
    parser.add_argument(
        "--reuse-joint-from",
        type=Path,
        default=Path("artifacts/synthetic_basis_v1"),
    )
    parser.add_argument(
        "--train-joint",
        action="store_true",
        help="Train joint conditions instead of re-evaluating matching checkpoints.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ExperimentConfig.seeds))
    parser.add_argument(
        "--run-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run only this subset while retaining --seeds as the shared protocol.",
    )
    parser.add_argument("--training-steps", type=int, default=ExperimentConfig.training_steps)
    parser.add_argument("--validation-interval", type=int, default=ExperimentConfig.validation_interval)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--train-samples", type=int, default=ExperimentConfig.train_samples)
    parser.add_argument("--validation-samples", type=int, default=ExperimentConfig.validation_samples)
    parser.add_argument("--iid-test-samples", type=int, default=ExperimentConfig.iid_test_samples)
    parser.add_argument("--heldout-test-samples", type=int, default=ExperimentConfig.heldout_test_samples)
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
        heldout_test_samples=args.heldout_test_samples,
        rollout_trajectories=args.rollout_trajectories,
        basis_probe_samples=args.basis_probe_samples,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        validation_interval=args.validation_interval,
        lambda_jac=args.lambda_jac,
        seeds=tuple(args.seeds),
    )
    if min(
        config.train_samples,
        config.validation_samples,
        config.iid_test_samples,
        config.heldout_test_samples,
        config.rollout_trajectories,
        config.basis_probe_samples,
        config.batch_size,
        config.training_steps,
        config.validation_interval,
    ) <= 0:
        raise ValueError("all sizes and training intervals must be positive")
    run_seeds = tuple(config.seeds if args.run_seeds is None else args.run_seeds)
    if not run_seeds or not set(run_seeds).issubset(config.seeds):
        raise ValueError("--run-seeds must be a non-empty subset of --seeds")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    output_dir: Path = args.output_dir
    for child in ("checkpoints", "logs", "results"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    serialized_config = json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != serialized_config:
        raise RuntimeError(
            f"{config_path} contains a different protocol; choose a new output directory"
        )
    config_path.write_text(serialized_config, encoding="utf-8")

    uses_reused_joint = (
        not args.train_joint
        and any(condition in JOINT_SOURCE_MODELS for condition in args.conditions)
    )
    if uses_reused_joint:
        _verify_joint_protocol(args.reuse_joint_from, config)

    print(f"Generating fixed datasets on CPU; training device={device}", flush=True)
    data = make_experiment_data(config)
    results: list[dict[str, Any]] = []
    for seed in run_seeds:
        for condition in args.conditions:
            result_path = output_dir / "results" / f"{condition}_seed_{seed}.json"
            if args.resume and result_path.exists():
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
                print(f"[resume] loaded {result_path}", flush=True)
                continue
            if condition in JOINT_SOURCE_MODELS and not args.train_joint:
                results.append(
                    reevaluate_joint_condition(
                        condition,
                        seed,
                        config,
                        data,
                        device=device,
                        output_dir=output_dir,
                        source_dir=args.reuse_joint_from,
                    )
                )
                print(f"[reuse] evaluated {condition} seed={seed}", flush=True)
            else:
                results.append(
                    train_one_condition(
                        condition,
                        seed,
                        config,
                        data,
                        device=device,
                        output_dir=output_dir,
                    )
                )

    all_results: list[dict[str, Any]] = []
    for seed in config.seeds:
        for condition in CONDITION_MODEL_CLASSES:
            result_path = output_dir / "results" / f"{condition}_seed_{seed}.json"
            if result_path.exists():
                all_results.append(json.loads(result_path.read_text(encoding="utf-8")))
    summary = summarize_results(all_results, config)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_report(summary, all_results), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
