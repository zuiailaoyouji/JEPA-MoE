"""Run and summarize the frozen five-seed Experiment 1 protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

import torch

from ..evaluation import evaluate_experiment1_model
from ..models import Experiment1ModelName, build_experiment1_model
from ..synthetic import (
    make_basis_composition_rollout_batch,
    make_basis_composition_transition_batch,
)
from .experiment1_horizon import make_horizon_data, train_validation_horizon
from .experiment1_pilot import MODEL_NAMES, PilotConfig


FORMAL_SEEDS = (0, 1, 2, 3, 4)
SCALAR_METRICS = (
    "best_validation_prediction_mse",
    "iid_one_step_mse",
    "rollout_25_mse",
    "directional_response_mse",
    "jacobian_frobenius_mse",
    "dynamics_subspace_principal_cosine_similarity",
    "dynamics_subspace_projection_reconstruction_error",
    "routing_entropy",
    "normalized_routing_entropy",
    "routing_effective_experts",
    "load_balance_loss",
    "best_step",
)


def formal_protocol() -> dict[str, Any]:
    base = asdict(PilotConfig(training_steps=15_000, seed=0))
    base.pop("seed")
    return {
        **base,
        "seeds": list(FORMAL_SEEDS),
        "models": list(MODEL_NAMES),
        "checkpoint_selection_metric": "validation prediction MSE",
        "standard_deviation": "sample standard deviation (n - 1)",
    }


def _ensure_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "logs", "results", "workers"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    serialized = json.dumps(
        formal_protocol(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"{path} contains a different experiment protocol")
    if not path.exists():
        path.write_text(serialized, encoding="utf-8")


def _load_best_model(
    model_name: Experiment1ModelName,
    seed: int,
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_name"] != model_name or checkpoint["seed"] != seed:
        raise RuntimeError(f"checkpoint identity mismatch: {checkpoint_path}")
    model = build_experiment1_model(
        model_name, initialization_seed=seed
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def run_seed_subset(
    seeds: tuple[int, ...],
    *,
    device: torch.device,
    output_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    """Train and evaluate a disjoint subset of formal model seeds."""

    if not seeds or not set(seeds).issubset(FORMAL_SEEDS):
        raise ValueError("seeds must be a non-empty subset of (0, 1, 2, 3, 4)")
    _ensure_protocol(output_dir)
    base_config = PilotConfig(training_steps=15_000, seed=0)
    train_validation_data = make_horizon_data(base_config)
    test_data = make_basis_composition_transition_batch(
        base_config.test_samples, seed=base_config.data_seed + 2
    )
    rollout_25 = make_basis_composition_rollout_batch(
        base_config.rollout_trajectories,
        base_config.rollout_horizon,
        seed=base_config.data_seed + 3,
    )

    completed: list[dict[str, Any]] = []
    for seed in seeds:
        config = replace(base_config, seed=seed)
        for model_name in MODEL_NAMES:
            result_path = output_dir / "results" / f"{model_name}_seed_{seed}.json"
            if resume and result_path.exists():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if existing.get("formal_evaluation_complete") is True:
                    completed.append(existing)
                    print(f"[resume] loaded {result_path}", flush=True)
                    continue

            validation_result = train_validation_horizon(
                model_name,
                config,
                train_validation_data,
                device=device,
                output_dir=output_dir,
            )
            checkpoint_path = (
                output_dir / "checkpoints" / f"{model_name}_seed_{seed}.pt"
            )
            model = _load_best_model(
                model_name,
                seed,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            metrics = evaluate_experiment1_model(
                model,
                iid_test=test_data,
                rollout_25=rollout_25,
                device=device,
                prediction_batch_size=config.evaluation_batch_size,
                jacobian_batch_size=config.jacobian_batch_size,
                direction_seed=(
                    config.data_seed
                    + config.evaluation_direction_seed_offset
                    + seed
                ),
            )
            result = {
                **validation_result,
                **metrics,
                "checkpoint_path": str(checkpoint_path),
                "test_evaluation_performed": True,
                "formal_evaluation_complete": True,
            }
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            completed.append(result)
            print(
                f"[complete] {model_name} seed={seed} "
                f"best_step={result['best_step']} "
                f"test_mse={result['iid_one_step_mse']:.6g}",
                flush=True,
            )

    worker_path = output_dir / "workers" / (
        "seeds_" + "_".join(str(seed) for seed in seeds) + ".json"
    )
    worker_path.write_text(
        json.dumps(completed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return completed


def _mean_std(values: list[float]) -> dict[str, float]:
    if len(values) != len(FORMAL_SEEDS):
        raise ValueError("formal summary requires exactly five values")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
    }


def _aggregate_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in SCALAR_METRICS:
        values = [result.get(metric) for result in results]
        if all(value is None for value in values):
            aggregate[metric] = None
        elif any(value is None for value in values):
            raise ValueError(f"partially missing metric: {metric}")
        else:
            aggregate[metric] = _mean_std([float(value) for value in values])

    usage = [result["mean_expert_usage"] for result in results]
    if all(value is None for value in usage):
        aggregate["mean_expert_usage"] = None
    elif any(value is None for value in usage):
        raise ValueError("partially missing mean_expert_usage")
    else:
        aggregate["mean_expert_usage"] = [
            _mean_std([float(value[index]) for value in usage])
            for index in range(len(usage[0]))
        ]
    return aggregate


def _paired_comparison(
    ours: dict[str, Any], moe_s: dict[str, Any]
) -> dict[str, Any]:
    metric_names = (
        "iid_one_step_mse",
        "directional_response_mse",
        "jacobian_frobenius_mse",
        "dynamics_subspace_projection_reconstruction_error",
    )
    return {
        "seed": ours["seed"],
        **{
            metric: {
                "moe_s": moe_s[metric],
                "ours": ours[metric],
                "ours_minus_moe_s": ours[metric] - moe_s[metric],
                "ours_over_moe_s": ours[metric] / moe_s[metric],
            }
            for metric in metric_names
        },
    }


def collect_complete_results(output_dir: Path) -> list[dict[str, Any]]:
    results = []
    missing = []
    for seed in FORMAL_SEEDS:
        for model_name in MODEL_NAMES:
            path = output_dir / "results" / f"{model_name}_seed_{seed}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("formal_evaluation_complete") is not True:
                missing.append(f"{path} (incomplete)")
                continue
            results.append(result)
    if missing:
        raise RuntimeError("formal results are incomplete:\n" + "\n".join(missing))
    return results


def summarize(output_dir: Path) -> dict[str, Any]:
    """Aggregate all formal results and write JSON plus Markdown reports."""

    _ensure_protocol(output_dir)
    results = collect_complete_results(output_dir)
    results.sort(key=lambda value: (MODEL_NAMES.index(value["model"]), value["seed"]))
    (output_dir / "per_seed_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    by_model = {
        model_name: [
            result for result in results if result["model"] == model_name
        ]
        for model_name in MODEL_NAMES
    }
    aggregate = {
        model_name: _aggregate_model(model_results)
        for model_name, model_results in by_model.items()
    }
    moe_s_by_seed = {result["seed"]: result for result in by_model["moe_s"]}
    ours_by_seed = {result["seed"]: result for result in by_model["ours"]}
    paired = [
        _paired_comparison(ours_by_seed[seed], moe_s_by_seed[seed])
        for seed in FORMAL_SEEDS
    ]
    summary = {
        "config": formal_protocol(),
        "aggregate": aggregate,
        "ours_vs_moe_s_by_seed": paired,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_report(summary, results), encoding="utf-8"
    )
    return summary


def _format_summary(value: dict[str, float] | None) -> str:
    if value is None:
        return "N/A"
    return f"{value['mean']:.6g} +/- {value['std']:.3g}"


def _format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def render_report(
    summary: dict[str, Any], results: list[dict[str, Any]]
) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Experiment 1: frozen five-seed results",
        "",
        "All checkpoints were selected only by validation prediction MSE. "
        "Reported standard deviations use n - 1.",
        "",
        "## Main results",
        "",
        "| Model | Test prediction MSE | Rollout-25 MSE | Directional response MSE | Jacobian Frobenius error | Subspace similarity | Projection error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        values = aggregate[model_name]
        lines.append(
            "| "
            + " | ".join(
                (
                    model_name,
                    _format_summary(values["iid_one_step_mse"]),
                    _format_summary(values["rollout_25_mse"]),
                    _format_summary(values["directional_response_mse"]),
                    _format_summary(values["jacobian_frobenius_mse"]),
                    _format_summary(
                        values["dynamics_subspace_principal_cosine_similarity"]
                    ),
                    _format_summary(
                        values[
                            "dynamics_subspace_projection_reconstruction_error"
                        ]
                    ),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Routing diagnostics",
            "",
            "| Model | Routing entropy | Effective experts | Usage 0 | Usage 1 | Usage 2 | Best step |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODEL_NAMES:
        values = aggregate[model_name]
        usage = values["mean_expert_usage"]
        usage_text = (
            ["N/A", "N/A", "N/A"]
            if usage is None
            else [_format_summary(item) for item in usage]
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    model_name,
                    _format_summary(values["routing_entropy"]),
                    _format_summary(values["routing_effective_experts"]),
                    *usage_text,
                    _format_summary(values["best_step"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Ours vs MoE-S by seed",
            "",
            "| Seed | Prediction: MoE-S | Prediction: Ours | Directional: MoE-S | Directional: Ours | Jacobian: MoE-S | Jacobian: Ours | Projection: MoE-S | Projection: Ours |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in summary["ours_vs_moe_s_by_seed"]:
        pred = comparison["iid_one_step_mse"]
        directional = comparison["directional_response_mse"]
        jacobian = comparison["jacobian_frobenius_mse"]
        projection = comparison[
            "dynamics_subspace_projection_reconstruction_error"
        ]
        lines.append(
            f"| {comparison['seed']} | {pred['moe_s']:.6g} | {pred['ours']:.6g} | "
            f"{directional['moe_s']:.6g} | {directional['ours']:.6g} | "
            f"{jacobian['moe_s']:.6g} | {jacobian['ours']:.6g} | "
            f"{projection['moe_s']:.6g} | {projection['ours']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed checkpoint and routing diagnostics",
            "",
            "| Model | Seed | Best step | Best val MSE | Routing entropy | Effective experts | Mean usage |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        usage = result["mean_expert_usage"]
        usage_text = (
            "N/A" if usage is None else ", ".join(f"{item:.6g}" for item in usage)
        )
        lines.append(
            f"| {result['model']} | {result['seed']} | {result['best_step']} | "
            f"{result['best_validation_prediction_mse']:.6g} | "
            f"{_format_value(result['routing_entropy'])} | "
            f"{_format_value(result['routing_effective_experts'])} | "
            f"{usage_text} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiment1_5seeds")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=FORMAL_SEEDS, default=list(FORMAL_SEEDS)
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = _parse_args()
    if args.summarize_only:
        summarize(args.output_dir)
        print(f"Wrote {args.output_dir / 'report.md'}", flush=True)
        return
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    run_seed_subset(
        tuple(args.seeds),
        device=device,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
