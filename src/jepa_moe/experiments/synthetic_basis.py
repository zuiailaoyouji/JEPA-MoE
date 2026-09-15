"""Run synthetic basis-dynamics recovery and unseen-composition experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..evaluation import evaluate_predictor, one_step_mse
from ..losses import control_jacobian_specialization_terms
from ..models import (
    DensePredictor,
    JacobianMoEPredictor,
    VanillaMoEPredictor,
    next_state_mse,
)
from ..synthetic import (
    RolloutBatch,
    TransitionBatch,
    make_rollout_batch,
    make_transition_batch,
)


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "dense": DensePredictor,
    "vanilla_moe": VanillaMoEPredictor,
    "jacobian_moe": JacobianMoEPredictor,
}


@dataclass(frozen=True)
class ExperimentConfig:
    train_samples: int = 50_000
    validation_samples: int = 10_000
    iid_test_samples: int = 10_000
    heldout_test_samples: int = 10_000
    rollout_trajectories: int = 10_000
    rollout_horizon: int = 25
    basis_probe_samples: int = 10_000
    batch_size: int = 512
    evaluation_batch_size: int = 2_048
    jacobian_batch_size: int = 512
    training_steps: int = 5_000
    validation_interval: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_jac: float = 0.1
    margin: float = 0.3
    min_jacobian_norm: float = 0.05
    beta_activity: float = 0.1
    data_seed: int = 20_260_914
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class ExperimentData:
    train: TransitionBatch
    validation: TransitionBatch
    iid_test: TransitionBatch
    heldout_test: TransitionBatch
    iid_rollout: RolloutBatch
    heldout_rollout: RolloutBatch
    basis_probe: TransitionBatch


def _concatenate_batches(*batches: TransitionBatch) -> TransitionBatch:
    return TransitionBatch(
        state=torch.cat([batch.state for batch in batches], dim=0),
        action=torch.cat([batch.action for batch in batches], dim=0),
        next_state=torch.cat([batch.next_state for batch in batches], dim=0),
    )


def make_experiment_data(config: ExperimentConfig) -> ExperimentData:
    """Create every fixed split once so all model conditions see identical data."""

    base = config.data_seed
    train = make_transition_batch(config.train_samples, "iid", seed=base)
    validation = make_transition_batch(
        config.validation_samples, "iid", seed=base + 1
    )
    iid_test = make_transition_batch(
        config.iid_test_samples, "iid", seed=base + 2
    )
    heldout_test = make_transition_batch(
        config.heldout_test_samples, "heldout", seed=base + 3
    )
    iid_rollout = make_rollout_batch(
        config.rollout_trajectories,
        config.rollout_horizon,
        "iid",
        seed=base + 4,
    )
    heldout_rollout = make_rollout_batch(
        config.rollout_trajectories,
        config.rollout_horizon,
        "heldout",
        seed=base + 5,
    )

    iid_probe_count = config.basis_probe_samples // 2
    heldout_probe_count = config.basis_probe_samples - iid_probe_count
    basis_probe = _concatenate_batches(
        make_transition_batch(iid_probe_count, "iid", seed=base + 6),
        make_transition_batch(heldout_probe_count, "heldout", seed=base + 7),
    )
    return ExperimentData(
        train=train,
        validation=validation,
        iid_test=iid_test,
        heldout_test=heldout_test,
        iid_rollout=iid_rollout,
        heldout_rollout=heldout_rollout,
        basis_probe=basis_probe,
    )


class ShuffledBatchStream:
    """Deterministic, epoch-shuffled batches shared across model conditions."""

    def __init__(self, data: TransitionBatch, batch_size: int, seed: int) -> None:
        if batch_size > len(data):
            raise ValueError("batch_size cannot exceed the training split size")
        self.data = data
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.permutation = torch.empty(0, dtype=torch.long)
        self.cursor = len(data)

    def next(self) -> TransitionBatch:
        if self.cursor + self.batch_size > len(self.data):
            self.permutation = torch.randperm(
                len(self.data), generator=self.generator
            )
            self.cursor = 0
        indices = self.permutation[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return TransitionBatch(
            self.data.state[indices],
            self.data.action[indices],
            self.data.next_state[indices],
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _scalar(value: Tensor) -> float:
    return float(value.detach().cpu())


def _moe_diagnostics(
    model: nn.Module,
    batch: TransitionBatch,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    state = batch.state.to(device)
    action = batch.action.to(device)
    terms = control_jacobian_specialization_terms(
        model,
        state,
        action,
        create_graph=False,
        margin=config.margin,
        min_jacobian_norm=config.min_jacobian_norm,
        beta_activity=config.beta_activity,
    )
    return {
        "jacobian_diversity_loss": _scalar(terms.diversity),
        "activity_loss": _scalar(terms.activity),
        "control_jacobian_specialization_loss": _scalar(terms.total),
        "expert_pair_mean_cosines": [
            float(value) for value in terms.pairwise_cosines.detach().cpu()
        ],
        "expert_mean_jacobian_norms": [
            float(value) for value in terms.expert_jacobian_norms.detach().cpu()
        ],
        "mean_routing_weights": [
            float(value) for value in terms.routing_weights.detach().cpu()
        ],
    }


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def train_one_model(
    model_name: str,
    seed: int,
    config: ExperimentConfig,
    data: ExperimentData,
    *,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one condition and select only by validation prediction MSE."""

    _seed_everything(seed)
    model = MODEL_CLASSES[model_name]().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    log_path = output_dir / "logs" / f"{model_name}_seed_{seed}.jsonl"
    checkpoint_path = output_dir / "checkpoints" / f"{model_name}_seed_{seed}.pt"
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
            "step": step,
            "train_prediction_loss": train_prediction_loss,
            "validation_prediction_loss": validation_mse,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if model_name != "dense":
            record.update(
                _moe_diagnostics(model, diagnostic_batch, config, device)
            )
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
        if model_name == "jacobian_moe":
            loss = model.training_loss(
                state,
                action,
                next_state,
                lambda_jac=config.lambda_jac,
                margin=config.margin,
                min_jacobian_norm=config.min_jacobian_norm,
                beta_activity=config.beta_activity,
            )
            total_loss = loss.total
            prediction_loss = loss.prediction
        else:
            pred = model(state, action).pred
            prediction_loss = next_state_mse(pred, next_state)
            total_loss = prediction_loss
        total_loss.backward()
        optimizer.step()
        last_prediction_loss = _scalar(prediction_loss)

        if step % config.validation_interval == 0 or step == config.training_steps:
            validate_and_log(step, last_prediction_loss)
            print(
                f"[{model_name} seed={seed}] step={step}/{config.training_steps} "
                f"train_mse={last_prediction_loss:.6g} "
                f"best_val_mse={best_validation_mse:.6g}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_name": model_name,
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
        "model": model_name,
        "seed": seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "training_seconds": time.perf_counter() - start_time,
        **metrics,
    }
    result_path = output_dir / "results" / f"{model_name}_seed_{seed}.json"
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
    """Aggregate scalar/vector metrics and make the support criterion explicit."""

    by_model = {
        model_name: sorted(
            (result for result in results if result["model"] == model_name),
            key=lambda result: result["seed"],
        )
        for model_name in MODEL_CLASSES
    }
    scalar_metrics = (
        "best_validation_prediction_mse",
        "iid_one_step_mse",
        "heldout_composition_mse",
        "iid_rollout_mse",
        "heldout_composition_rollout_mse",
        "expert_redundancy_mean_abs_cosine",
        "basis_recovery_score",
    )
    aggregate: dict[str, Any] = {}
    for model_name, model_results in by_model.items():
        model_summary: dict[str, Any] = {}
        for metric in scalar_metrics:
            values = [float(result[metric]) for result in model_results if metric in result]
            if values:
                model_summary[metric] = _mean_std(values)
        for metric in ("expert_pair_mean_abs_cosines", "mean_routing_weights"):
            vectors = [result[metric] for result in model_results if metric in result]
            if vectors:
                model_summary[metric] = [
                    _mean_std([float(vector[index]) for vector in vectors])
                    for index in range(len(vectors[0]))
                ]
        aggregate[model_name] = model_summary

    vanilla_by_seed = {result["seed"]: result for result in by_model["vanilla_moe"]}
    jacobian_by_seed = {result["seed"]: result for result in by_model["jacobian_moe"]}
    common_seeds = sorted(vanilla_by_seed.keys() & jacobian_by_seed.keys())
    support: dict[str, Any] | None = None
    if common_seeds:
        vanilla_iid = aggregate["vanilla_moe"]["iid_one_step_mse"]["mean"]
        jacobian_iid = aggregate["jacobian_moe"]["iid_one_step_mse"]["mean"]
        heldout_one_step_wins = sum(
            jacobian_by_seed[seed]["heldout_composition_mse"]
            < vanilla_by_seed[seed]["heldout_composition_mse"]
            for seed in common_seeds
        )
        heldout_rollout_wins = sum(
            jacobian_by_seed[seed]["heldout_composition_rollout_mse"]
            < vanilla_by_seed[seed]["heldout_composition_rollout_mse"]
            for seed in common_seeds
        )
        conditions = {
            "iid_not_more_than_5_percent_worse": jacobian_iid <= 1.05 * vanilla_iid,
            "lower_mean_expert_redundancy": (
                aggregate["jacobian_moe"]["expert_redundancy_mean_abs_cosine"]["mean"]
                < aggregate["vanilla_moe"]["expert_redundancy_mean_abs_cosine"]["mean"]
            ),
            "higher_mean_basis_recovery": (
                aggregate["jacobian_moe"]["basis_recovery_score"]["mean"]
                > aggregate["vanilla_moe"]["basis_recovery_score"]["mean"]
            ),
            "lower_mean_heldout_one_step_mse": (
                aggregate["jacobian_moe"]["heldout_composition_mse"]["mean"]
                < aggregate["vanilla_moe"]["heldout_composition_mse"]["mean"]
            ),
            "lower_mean_heldout_rollout_mse": (
                aggregate["jacobian_moe"][
                    "heldout_composition_rollout_mse"
                ]["mean"]
                < aggregate["vanilla_moe"][
                    "heldout_composition_rollout_mse"
                ]["mean"]
            ),
        }
        support = {
            "definition": (
                "IID mean MSE no more than 5% worse, lower mean redundancy, higher "
                "mean basis recovery, and lower mean held-out one-step and rollout MSE."
            ),
            "paired_seeds": common_seeds,
            "heldout_one_step_wins": heldout_one_step_wins,
            "heldout_rollout_wins": heldout_rollout_wins,
            "conditions": conditions,
            "provides_preliminary_support": all(conditions.values()),
        }
    return {"config": asdict(config), "aggregate": aggregate, "support": support}


def _format_metric(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "n/a"
    return f"{summary['mean']:.6g} +/- {summary['std']:.3g}"


def render_report(
    summary: dict[str, Any], results: list[dict[str, Any]]
) -> str:
    config = summary["config"]
    lines = [
        "# Synthetic basis-dynamics recovery",
        "",
        (
            f"Protocol: {config['training_steps']} updates, batch {config['batch_size']}, "
            f"AdamW lr={config['learning_rate']}, lambda_jac={config['lambda_jac']}, "
            f"{len(config['seeds'])} seeds. Checkpoints selected only by validation MSE."
        ),
        "",
        "## Aggregate results",
        "",
        "| Model | IID one-step MSE | Held-out one-step MSE | IID rollout-25 MSE | Held-out rollout-25 MSE | Redundancy | Basis recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    aggregate = summary["aggregate"]
    for model_name in MODEL_CLASSES:
        values = aggregate[model_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    _format_metric(values.get("iid_one_step_mse")),
                    _format_metric(values.get("heldout_composition_mse")),
                    _format_metric(values.get("iid_rollout_mse")),
                    _format_metric(values.get("heldout_composition_rollout_mse")),
                    _format_metric(values.get("expert_redundancy_mean_abs_cosine")),
                    _format_metric(values.get("basis_recovery_score")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Aggregate MoE diagnostics",
            "",
            "| Model | abs cos(0,1) | abs cos(0,2) | abs cos(1,2) | route 0 | route 1 | route 2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in ("vanilla_moe", "jacobian_moe"):
        pairwise = aggregate[model_name].get("expert_pair_mean_abs_cosines", [])
        routing = aggregate[model_name].get("mean_routing_weights", [])
        lines.append(
            f"| {model_name} | "
            + " | ".join(_format_metric(value) for value in pairwise + routing)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-seed results",
            "",
            "| Model | Seed | Best step | IID MSE | Held-out MSE | IID rollout | Held-out rollout | Redundancy | Recovery | Routing weights |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for result in sorted(results, key=lambda item: (item["model"], item["seed"])):
        routing = result.get("mean_routing_weights")
        routing_text = (
            "[" + ", ".join(f"{value:.3f}" for value in routing) + "]"
            if routing is not None
            else "n/a"
        )
        lines.append(
            f"| {result['model']} | {result['seed']} | {result['best_step']} | "
            f"{result['iid_one_step_mse']:.6g} | "
            f"{result['heldout_composition_mse']:.6g} | "
            f"{result['iid_rollout_mse']:.6g} | "
            f"{result['heldout_composition_rollout_mse']:.6g} | "
            f"{result.get('expert_redundancy_mean_abs_cosine', float('nan')):.4f} | "
            f"{result.get('basis_recovery_score', float('nan')):.4f} | {routing_text} |"
        )

    support = summary["support"]
    lines.extend(["", "## Preliminary-support criterion", ""])
    if support is None:
        lines.append("Not evaluated: paired Vanilla/Jacobian seeds are incomplete.")
    else:
        lines.append(support["definition"])
        lines.append("")
        for name, passed in support["conditions"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
        lines.append("")
        lines.append(
            f"Paired-seed wins: held-out one-step {support['heldout_one_step_wins']}/"
            f"{len(support['paired_seeds'])}; held-out rollout "
            f"{support['heldout_rollout_wins']}/{len(support['paired_seeds'])}."
        )
        lines.append("")
        lines.append(
            "Overall: "
            + (
                "provides preliminary support."
                if support["provides_preliminary_support"]
                else "does not meet the preliminary-support criterion."
            )
        )
    lines.append("")
    lines.append(
        "Full pairwise cosines, routing weights, recovery matrices, and Hungarian matches are preserved in `summary.json` and `results/*.json`."
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/synthetic_basis"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", nargs="+", choices=MODEL_CLASSES, default=list(MODEL_CLASSES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ExperimentConfig.seeds))
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

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
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

    print(f"Generating fixed datasets on CPU; training device={device}", flush=True)
    data = make_experiment_data(config)
    results: list[dict[str, Any]] = []
    for seed in config.seeds:
        for model_name in args.models:
            result_path = output_dir / "results" / f"{model_name}_seed_{seed}.json"
            if args.resume and result_path.exists():
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
                print(f"[resume] loaded {result_path}", flush=True)
                continue
            results.append(
                train_one_model(
                    model_name,
                    seed,
                    config,
                    data,
                    device=device,
                    output_dir=output_dir,
                )
            )

    all_expected_results = []
    for seed in config.seeds:
        for model_name in MODEL_CLASSES:
            result_path = output_dir / "results" / f"{model_name}_seed_{seed}.json"
            if result_path.exists():
                all_expected_results.append(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
    summary = summarize_results(all_expected_results, config)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_report(summary, all_expected_results), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
