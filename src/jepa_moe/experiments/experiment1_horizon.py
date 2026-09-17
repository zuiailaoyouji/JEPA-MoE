"""Run a validation-only training-horizon check for Experiment 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from ..models import (
    ACTION_DIM,
    Experiment1ModelName,
    build_experiment1_model,
)
from ..objectives import experiment1_training_loss, sample_unit_action_directions
from ..synthetic import (
    TransitionBatch,
    make_basis_composition_transition_batch,
)
from .experiment1_pilot import (
    MODEL_NAMES,
    PilotConfig,
    ShuffledBatchStream,
    _accumulate_losses,
    _cpu_state_dict,
    _divide_losses,
    _loss_scalars,
    _seed_everything,
    _zero_loss_accumulator,
    evaluate_objective_losses,
)


@dataclass(frozen=True)
class HorizonData:
    train: TransitionBatch
    validation: TransitionBatch


def make_horizon_data(config: PilotConfig) -> HorizonData:
    """Generate only the shared train and validation splits."""

    return HorizonData(
        train=make_basis_composition_transition_batch(
            config.train_samples, seed=config.data_seed
        ),
        validation=make_basis_composition_transition_batch(
            config.validation_samples, seed=config.data_seed + 1
        ),
    )


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def train_validation_horizon(
    model_name: Experiment1ModelName,
    config: PilotConfig,
    data: HorizonData,
    *,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one model without evaluating any test split."""

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
    best_validation_losses: dict[str, float] | None = None
    validation_by_step: dict[str, dict[str, float]] = {}
    train_window = _zero_loss_accumulator()
    train_window_samples = 0

    def validate_and_log(
        step: int, train_losses: dict[str, float] | None
    ) -> None:
        nonlocal best_validation_mse, best_step, best_state
        nonlocal best_validation_losses
        validation_losses = evaluate_objective_losses(
            model_name,
            model,
            data.validation,
            objective_config=config.objective,
            device=device,
            batch_size=config.evaluation_batch_size,
            directions=validation_directions,
        )
        validation_by_step[str(step)] = validation_losses
        _append_json_line(
            log_path,
            {
                "step": step,
                "train": train_losses,
                "validation": validation_losses,
                "checkpoint_selection_metric": "validation.prediction_mse",
                "elapsed_seconds": time.perf_counter() - start_time,
            },
        )
        validation_mse = validation_losses["prediction_mse"]
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_step = step
            best_state = _cpu_state_dict(model)
            best_validation_losses = validation_losses
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
        _accumulate_losses(train_window, loss_values, state.shape[0])
        train_window_samples += state.shape[0]

        if step % config.validation_interval == 0 or step == config.training_steps:
            averaged_train = _divide_losses(
                train_window, train_window_samples
            )
            validate_and_log(step, averaged_train)
            current = validation_by_step[str(step)]
            print(
                f"[{model_name}] step={step}/{config.training_steps} "
                f"val_pred={current['prediction_mse']:.6g} "
                f"best_val={best_validation_mse:.6g} "
                f"best_step={best_step}",
                flush=True,
            )
            train_window = _zero_loss_accumulator()
            train_window_samples = 0

    if best_state is None or best_validation_losses is None:
        raise RuntimeError("training completed without a validation checkpoint")
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
    result = {
        "model": model_name,
        "seed": config.seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "best_validation_losses": best_validation_losses,
        "validation_by_step": validation_by_step,
        "training_seconds": time.perf_counter() - start_time,
        "test_evaluation_performed": False,
    }
    (output_dir / "results" / f"{model_name}_seed_{config.seed}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def run_horizon_check(
    config: PilotConfig,
    *,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Run all four models and persist validation-only horizon results."""

    for child in ("checkpoints", "logs", "results"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    data = make_horizon_data(config)
    results = [
        train_validation_horizon(
            model_name,
            config,
            data,
            device=device,
            output_dir=output_dir,
        )
        for model_name in MODEL_NAMES
    ]
    summary = {
        "config": asdict(config),
        "device": str(device),
        "test_evaluation_performed": False,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment1_horizon_15000_seed0"),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = _parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    run_horizon_check(
        PilotConfig(training_steps=15_000),
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
