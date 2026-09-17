"""Run the fixed single-seed pilot for the state-conditioned Experiment 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..evaluation import evaluate_experiment1_model
from ..models import (
    ACTION_DIM,
    Experiment1ModelName,
    TrainingLoss,
    build_experiment1_model,
)
from ..objectives import (
    Experiment1ObjectiveConfig,
    experiment1_training_loss,
    sample_unit_action_directions,
)
from ..synthetic import (
    RolloutBatch,
    TransitionBatch,
    make_basis_composition_rollout_batch,
    make_basis_composition_transition_batch,
)


MODEL_NAMES: tuple[Experiment1ModelName, ...] = (
    "dense",
    "moe_a",
    "moe_s",
    "ours",
)


@dataclass(frozen=True)
class PilotConfig:
    """Frozen protocol for the one-seed pilot; values are written to JSON."""

    train_samples: int = 50_000
    validation_samples: int = 10_000
    test_samples: int = 10_000
    rollout_trajectories: int = 10_000
    rollout_horizon: int = 25
    batch_size: int = 512
    evaluation_batch_size: int = 2_048
    jacobian_batch_size: int = 512
    training_steps: int = 5_000
    validation_interval: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_cr: float = 1.0
    lambda_bal: float = 0.01
    num_directions: int = 1
    seed: int = 0
    data_seed: int = 20_260_916
    batch_seed_offset: int = 10_000
    train_direction_seed_offset: int = 20_000
    validation_direction_seed_offset: int = 30_000
    evaluation_direction_seed_offset: int = 40_000

    def __post_init__(self) -> None:
        sizes = (
            self.train_samples,
            self.validation_samples,
            self.test_samples,
            self.rollout_trajectories,
            self.rollout_horizon,
            self.batch_size,
            self.evaluation_batch_size,
            self.jacobian_batch_size,
            self.training_steps,
            self.validation_interval,
        )
        if min(sizes) <= 0:
            raise ValueError("all dataset, batch, and schedule sizes must be positive")
        if self.batch_size > self.train_samples:
            raise ValueError("batch_size cannot exceed train_samples")
        if self.rollout_horizon != 25:
            raise ValueError("the Experiment 1 pilot requires rollout_horizon=25")
        if self.num_directions != 1:
            raise ValueError("the fixed pilot requires num_directions=1")
        if self.lambda_cr != 1.0 or self.lambda_bal != 0.01:
            raise ValueError("the fixed pilot requires lambda_cr=1.0, lambda_bal=0.01")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters must be valid")

    @property
    def objective(self) -> Experiment1ObjectiveConfig:
        return Experiment1ObjectiveConfig(
            lambda_cr=self.lambda_cr,
            lambda_bal=self.lambda_bal,
        )


@dataclass(frozen=True)
class PilotData:
    train: TransitionBatch
    validation: TransitionBatch
    test: TransitionBatch
    rollout_25: RolloutBatch


class ShuffledBatchStream:
    """Deterministic epoch-shuffled batches, reset identically per model."""

    def __init__(self, data: TransitionBatch, batch_size: int, *, seed: int) -> None:
        if batch_size > len(data):
            raise ValueError("batch_size cannot exceed the dataset size")
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
            state=self.data.state[indices],
            action=self.data.action[indices],
            next_state=self.data.next_state[indices],
        )


def make_pilot_data(config: PilotConfig) -> PilotData:
    """Build the shared full-IID train/validation/test and rollout splits."""

    base = config.data_seed
    return PilotData(
        train=make_basis_composition_transition_batch(
            config.train_samples, seed=base
        ),
        validation=make_basis_composition_transition_batch(
            config.validation_samples, seed=base + 1
        ),
        test=make_basis_composition_transition_batch(
            config.test_samples, seed=base + 2
        ),
        rollout_25=make_basis_composition_rollout_batch(
            config.rollout_trajectories,
            config.rollout_horizon,
            seed=base + 3,
        ),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _loss_scalars(loss: TrainingLoss) -> dict[str, float]:
    return {
        "prediction_loss": float(loss.prediction.detach().cpu()),
        "control_response_loss": float(loss.control_response.detach().cpu()),
        "load_balance_loss": float(loss.balance.detach().cpu()),
        "total_loss": float(loss.total.detach().cpu()),
        "prediction_mse": float(loss.prediction.detach().cpu()),
    }


def _zero_loss_accumulator() -> dict[str, float]:
    return {
        "prediction_loss": 0.0,
        "control_response_loss": 0.0,
        "load_balance_loss": 0.0,
        "total_loss": 0.0,
        "prediction_mse": 0.0,
    }


def _accumulate_losses(
    accumulator: dict[str, float], values: dict[str, float], weight: int
) -> None:
    for name in accumulator:
        accumulator[name] += values[name] * weight


def _divide_losses(
    accumulator: dict[str, float], denominator: int
) -> dict[str, float]:
    if denominator <= 0:
        raise ValueError("loss denominator must be positive")
    return {name: value / denominator for name, value in accumulator.items()}


def evaluate_objective_losses(
    model_name: Experiment1ModelName,
    model: nn.Module,
    data: TransitionBatch,
    *,
    objective_config: Experiment1ObjectiveConfig,
    device: torch.device,
    batch_size: int,
    directions: Tensor | None,
) -> dict[str, float]:
    """Measure raw validation objectives using a fixed direction set."""

    if model_name == "ours":
        if directions is None or directions.shape != data.action.shape:
            raise ValueError("ours requires one fixed direction per validation sample")
    elif directions is not None:
        raise ValueError("only ours accepts validation directions")

    model.eval()
    totals = _zero_loss_accumulator()
    sample_count = 0
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        state = data.state[start:stop].to(device)
        action = data.action[start:stop].to(device)
        next_state = data.next_state[start:stop].to(device)
        direction = (
            directions[start:stop].to(device) if directions is not None else None
        )
        loss = experiment1_training_loss(
            model_name,
            model,
            state,
            action,
            next_state,
            config=objective_config,
            direction=direction,
        )
        batch_count = stop - start
        _accumulate_losses(totals, _loss_scalars(loss), batch_count)
        sample_count += batch_count
    return _divide_losses(totals, sample_count)


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def train_one_model(
    model_name: Experiment1ModelName,
    config: PilotConfig,
    data: PilotData,
    *,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one condition and select its checkpoint only by validation MSE."""

    _seed_everything(config.seed)
    model = build_experiment1_model(
        model_name, initialization_seed=config.seed
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batch_stream = ShuffledBatchStream(
        data.train,
        config.batch_size,
        seed=config.data_seed + config.batch_seed_offset + config.seed,
    )
    train_direction_generator = torch.Generator(device="cpu").manual_seed(
        config.data_seed + config.train_direction_seed_offset + config.seed
    )
    validation_directions = None
    if model_name == "ours":
        validation_directions = sample_unit_action_directions(
            len(data.validation),
            ACTION_DIM,
            generator=torch.Generator(device="cpu").manual_seed(
                config.data_seed
                + config.validation_direction_seed_offset
                + config.seed
            ),
            dtype=data.validation.action.dtype,
        )

    log_path = output_dir / "logs" / f"{model_name}_seed_{config.seed}.jsonl"
    checkpoint_path = (
        output_dir / "checkpoints" / f"{model_name}_seed_{config.seed}.pt"
    )
    log_path.write_text("", encoding="utf-8")
    start_time = time.perf_counter()
    best_validation_mse = math.inf
    best_step = -1
    best_state: dict[str, Tensor] | None = None
    initial_train_losses: dict[str, float] | None = None
    final_train_losses: dict[str, float] | None = None
    initial_validation_losses: dict[str, float] | None = None
    final_validation_losses: dict[str, float] | None = None
    train_window = _zero_loss_accumulator()
    train_window_samples = 0

    def validate_and_log(
        step: int, train_losses: dict[str, float] | None
    ) -> None:
        nonlocal best_validation_mse, best_step, best_state
        nonlocal initial_validation_losses, final_validation_losses
        validation_losses = evaluate_objective_losses(
            model_name,
            model,
            data.validation,
            objective_config=config.objective,
            device=device,
            batch_size=config.evaluation_batch_size,
            directions=validation_directions,
        )
        if initial_validation_losses is None:
            initial_validation_losses = validation_losses
        final_validation_losses = validation_losses
        record = {
            "step": step,
            "train": train_losses,
            "validation": validation_losses,
            "checkpoint_selection_metric": "validation.prediction_mse",
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        _append_json_line(log_path, record)
        validation_mse = validation_losses["prediction_mse"]
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_step = step
            best_state = _cpu_state_dict(model)
        model.train()

    validate_and_log(0, None)
    for step in range(1, config.training_steps + 1):
        batch = batch_stream.next()
        state = batch.state.to(device)
        action = batch.action.to(device)
        next_state = batch.next_state.to(device)
        direction = None
        if model_name == "ours":
            direction = sample_unit_action_directions(
                state.shape[0],
                ACTION_DIM,
                generator=train_direction_generator,
                dtype=state.dtype,
                device=device,
            )

        optimizer.zero_grad(set_to_none=True)
        loss = experiment1_training_loss(
            model_name,
            model,
            state,
            action,
            next_state,
            config=config.objective,
            direction=direction,
        )
        loss_values = _loss_scalars(loss)
        if not all(math.isfinite(value) for value in loss_values.values()):
            raise FloatingPointError(
                f"non-finite training loss for {model_name} at step {step}: "
                f"{loss_values}"
            )
        loss.total.backward()
        optimizer.step()

        if initial_train_losses is None:
            initial_train_losses = loss_values
        final_train_losses = loss_values
        _accumulate_losses(train_window, loss_values, state.shape[0])
        train_window_samples += state.shape[0]

        if step % config.validation_interval == 0 or step == config.training_steps:
            averaged_train = _divide_losses(
                train_window, train_window_samples
            )
            validate_and_log(step, averaged_train)
            print(
                f"[{model_name}] step={step}/{config.training_steps} "
                f"train_pred={averaged_train['prediction_loss']:.6g} "
                f"train_cr={averaged_train['control_response_loss']:.6g} "
                f"val_pred={final_validation_losses['prediction_loss']:.6g} "
                f"best_val={best_validation_mse:.6g}",
                flush=True,
            )
            train_window = _zero_loss_accumulator()
            train_window_samples = 0

    if (
        best_state is None
        or initial_train_losses is None
        or final_train_losses is None
        or initial_validation_losses is None
        or final_validation_losses is None
    ):
        raise RuntimeError("training completed without complete pilot records")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_name": model_name,
            "seed": config.seed,
            "best_step": best_step,
            "best_validation_prediction_mse": best_validation_mse,
            "selection_metric": "validation prediction MSE",
            "state_dict": best_state,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    metrics = evaluate_experiment1_model(
        model,
        iid_test=data.test,
        rollout_25=data.rollout_25,
        device=device,
        prediction_batch_size=config.evaluation_batch_size,
        jacobian_batch_size=config.jacobian_batch_size,
        direction_seed=(
            config.data_seed
            + config.evaluation_direction_seed_offset
            + config.seed
        ),
    )
    result = {
        "model": model_name,
        "seed": config.seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "checkpoint_selection_metric": "validation prediction MSE",
        "initial_train_losses": initial_train_losses,
        "final_train_losses": final_train_losses,
        "initial_validation_losses": initial_validation_losses,
        "final_validation_losses": final_validation_losses,
        "training_and_evaluation_seconds": time.perf_counter() - start_time,
        **metrics,
    }
    result_path = output_dir / "results" / f"{model_name}_seed_{config.seed}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6g}"


def render_report(config: PilotConfig, results: list[dict[str, Any]]) -> str:
    """Render the requested single-seed table and Ours trajectory summary."""

    by_model = {result["model"]: result for result in results}
    lines = [
        "# Experiment 1 single-seed pilot",
        "",
        (
            f"Seed {config.seed}; full-IID train/validation/test = "
            f"{config.train_samples}/{config.validation_samples}/{config.test_samples}; "
            f"AdamW lr={config.learning_rate}, weight_decay={config.weight_decay}, "
            f"batch={config.batch_size}, updates={config.training_steps}; "
            f"lambda_cr={config.lambda_cr}, lambda_bal={config.lambda_bal}, "
            f"num_directions={config.num_directions}. Checkpoints are selected only "
            "by validation prediction MSE."
        ),
        "",
        "| Model | Best Val Prediction MSE | Test Prediction MSE | 25-step Rollout MSE | Directional Response MSE | Jacobian Frobenius Error | Subspace Similarity | Projection Error | Routing Entropy | Effective Expert Count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        result = by_model[model_name]
        lines.append(
            "| "
            + " | ".join(
                (
                    model_name,
                    _format(result["best_validation_prediction_mse"]),
                    _format(result["iid_one_step_mse"]),
                    _format(result["rollout_25_mse"]),
                    _format(result["directional_response_mse"]),
                    _format(result["jacobian_frobenius_mse"]),
                    _format(
                        result["dynamics_subspace_principal_cosine_similarity"]
                    ),
                    _format(
                        result[
                            "dynamics_subspace_projection_reconstruction_error"
                        ]
                    ),
                    _format(result["routing_entropy"]),
                    _format(result["routing_effective_experts"]),
                )
            )
            + " |"
        )

    lines.extend(["", "## Routing usage", ""])
    for model_name in MODEL_NAMES:
        result = by_model[model_name]
        usage = result["mean_expert_usage"]
        lines.append(
            f"- {model_name}: "
            + ("N/A" if usage is None else ", ".join(_format(x) for x in usage))
        )

    ours = by_model["ours"]
    initial = ours["initial_validation_losses"]
    final = ours["final_validation_losses"]
    lines.extend(
        [
            "",
            "## Ours validation loss trajectory",
            "",
            "| Point | L_pred | L_CR | L_balance | Total |",
            "|---|---:|---:|---:|---:|",
            (
                f"| initial | {_format(initial['prediction_loss'])} | "
                f"{_format(initial['control_response_loss'])} | "
                f"{_format(initial['load_balance_loss'])} | "
                f"{_format(initial['total_loss'])} |"
            ),
            (
                f"| final | {_format(final['prediction_loss'])} | "
                f"{_format(final['control_response_loss'])} | "
                f"{_format(final['load_balance_loss'])} | "
                f"{_format(final['total_loss'])} |"
            ),
            "",
            f"Best Ours checkpoint: step {ours['best_step']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary(
    config: PilotConfig,
    results: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    by_model = {result["model"]: result for result in results}
    ours = by_model["ours"]
    moe_s = by_model["moe_s"]
    minimum_usage = min(ours["mean_expert_usage"])
    return {
        "config": asdict(config),
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "results": results,
        "pilot_diagnostics": {
            "all_losses_finite": all(
                math.isfinite(float(result[key]))
                for result in results
                for key in (
                    "best_validation_prediction_mse",
                    "iid_one_step_mse",
                    "rollout_25_mse",
                    "directional_response_mse",
                    "jacobian_frobenius_mse",
                )
            ),
            "ours_validation_prediction_decreased": (
                ours["final_validation_losses"]["prediction_loss"]
                < ours["initial_validation_losses"]["prediction_loss"]
            ),
            "ours_validation_control_response_decreased": (
                ours["final_validation_losses"]["control_response_loss"]
                < ours["initial_validation_losses"]["control_response_loss"]
            ),
            "ours_minimum_mean_usage": minimum_usage,
            "ours_expert_collapse_threshold": 0.05,
            "ours_expert_collapse_observed": minimum_usage < 0.05,
            "ours_to_moe_s_test_prediction_mse_ratio": (
                ours["iid_one_step_mse"] / moe_s["iid_one_step_mse"]
            ),
            "ours_to_moe_s_directional_response_mse_ratio": (
                ours["directional_response_mse"]
                / moe_s["directional_response_mse"]
            ),
            "ours_to_moe_s_jacobian_error_ratio": (
                ours["jacobian_frobenius_mse"]
                / moe_s["jacobian_frobenius_mse"]
            ),
        },
    }


def run_pilot(
    config: PilotConfig,
    *,
    device: torch.device,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Run or resume all four fixed pilot conditions."""

    for child in ("checkpoints", "logs", "results"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    serialized_config = json.dumps(
        asdict(config), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != serialized_config:
            raise RuntimeError(
                f"{config_path} contains a different protocol; use a new directory"
            )
    else:
        config_path.write_text(serialized_config, encoding="utf-8")

    print(f"Generating shared full-IID pilot data; device={device}", flush=True)
    data = make_pilot_data(config)
    results: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        result_path = output_dir / "results" / f"{model_name}_seed_{config.seed}.json"
        if resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(result)
            print(f"[resume] loaded {result_path}", flush=True)
            continue
        results.append(
            train_one_model(
                model_name,
                config,
                data,
                device=device,
                output_dir=output_dir,
            )
        )

    summary = build_summary(config, results, device=device)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_report(config, results), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'report.md'}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment1_pilot_seed0"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    run_pilot(
        PilotConfig(),
        device=device,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
