"""Extend the seed-0 history-conditioned sanity model from 15k to 25k steps."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .experiment2_history_sanity import (
    HistorySanityConfig,
    ShuffledHistoryBatchStream,
    _cpu_state_dict,
    _seed_everything,
    _to_device,
    build_sanity_model,
    evaluate_validation,
    history_sanity_loss,
    make_sanity_data,
)
from ..models import HistoryConditionedMoEPredictor


REFERENCE_STEP = 15_000
FINAL_STEP = 25_000
REPORT_STEPS = (15_000, 17_500, 20_000, 22_500, 25_000)


def _optimizer(model: HistoryConditionedMoEPredictor, config: HistorySanityConfig):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _train_step(
    model: HistoryConditionedMoEPredictor,
    optimizer: torch.optim.Optimizer,
    stream: ShuffledHistoryBatchStream,
    *,
    config: HistorySanityConfig,
    device: torch.device,
    step: int,
) -> None:
    batch = _to_device(stream.next(), device)
    optimizer.zero_grad(set_to_none=True)
    total, prediction, balance = history_sanity_loss(
        "history", model, batch, lambda_bal=config.lambda_bal
    )
    if not all(
        math.isfinite(float(value.detach().cpu()))
        for value in (total, prediction, balance)
    ):
        raise FloatingPointError(f"non-finite loss at step {step}")
    total.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"invalid gradient for {name} at step {step}")
    optimizer.step()


def _maximum_state_error(
    first: dict[str, Tensor], second: dict[str, Tensor]
) -> float:
    if first.keys() != second.keys():
        raise RuntimeError("model state keys differ")
    return max(
        float((first[name].cpu() - second[name].cpu()).abs().max())
        for name in first
    )


def _capture_resume_state(
    model: HistoryConditionedMoEPredictor,
    optimizer: torch.optim.Optimizer,
    stream: ShuffledHistoryBatchStream,
    *,
    current_step: int,
    config: HistorySanityConfig,
    best_validation_mse: float,
    best_step: int,
    direction_rng_state: Tensor | None = None,
) -> dict[str, Any]:
    return {
        "step": current_step,
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "batch_stream_state_dict": stream.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "direction_rng_state": direction_rng_state,
        "best_validation_prediction_mse": best_validation_mse,
        "best_step": best_step,
        "config": asdict(config),
    }


def _restore_resume_state(
    snapshot_path: Path,
    data,
    *,
    config: HistorySanityConfig,
    device: torch.device,
) -> tuple[
    HistoryConditionedMoEPredictor,
    torch.optim.Optimizer,
    ShuffledHistoryBatchStream,
    dict[str, Any],
]:
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    model = build_sanity_model("history", initialization_seed=config.seed)
    assert isinstance(model, HistoryConditionedMoEPredictor)
    model.load_state_dict(snapshot["model_state_dict"])
    model = model.to(device)
    optimizer = _optimizer(model, config)
    optimizer.load_state_dict(snapshot["optimizer_state_dict"])
    stream = ShuffledHistoryBatchStream(
        data.train,
        config.batch_size,
        seed=config.data_seed + config.batch_seed_offset + config.seed,
    )
    stream.load_state_dict(snapshot["batch_stream_state_dict"])
    random.setstate(snapshot["python_rng_state"])
    np.random.set_state(snapshot["numpy_rng_state"])
    torch.set_rng_state(snapshot["torch_cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["torch_cuda_rng_state_all"])
    return model, optimizer, stream, snapshot


def run_horizon_check(
    *,
    device: torch.device,
    reference_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = HistorySanityConfig()
    if config.training_steps != REFERENCE_STEP:
        raise RuntimeError("reference protocol must end at 15,000 steps")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_record = {**asdict(config), "extended_training_steps": FINAL_STEP}
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reference_path = reference_dir / "checkpoints" / "history_seed_0.pt"
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference["best_step"] != REFERENCE_STEP:
        raise RuntimeError("the reference best checkpoint is not step 15,000")

    data = make_sanity_data(config)
    _seed_everything(config.seed)
    replay_model = build_sanity_model("history", initialization_seed=config.seed)
    assert isinstance(replay_model, HistoryConditionedMoEPredictor)
    replay_model = replay_model.to(device)
    replay_optimizer = _optimizer(replay_model, config)
    replay_stream = ShuffledHistoryBatchStream(
        data.train,
        config.batch_size,
        seed=config.data_seed + config.batch_seed_offset + config.seed,
    )
    replay_model.train()
    for step in range(1, REFERENCE_STEP + 1):
        _train_step(
            replay_model,
            replay_optimizer,
            replay_stream,
            config=config,
            device=device,
            step=step,
        )
        if step % 2_500 == 0:
            print(f"[replay] step={step}/{REFERENCE_STEP}", flush=True)

    replay_state = _cpu_state_dict(replay_model)
    replay_error = _maximum_state_error(replay_state, reference["state_dict"])
    if replay_error != 0.0:
        raise RuntimeError(
            "deterministic replay did not exactly recover the step-15k checkpoint: "
            f"max parameter error={replay_error}"
        )
    validation_15k = evaluate_validation(
        "history",
        replay_model,
        data.validation,
        lambda_bal=config.lambda_bal,
        device=device,
        batch_size=config.evaluation_batch_size,
    )["prediction_loss"]
    snapshot_path = output_dir / "resume_step_15000.pt"
    torch.save(
        _capture_resume_state(
            replay_model,
            replay_optimizer,
            replay_stream,
            current_step=REFERENCE_STEP,
            config=config,
            best_validation_mse=float(reference["best_validation_prediction_mse"]),
            best_step=int(reference["best_step"]),
        ),
        snapshot_path,
    )

    model, optimizer, stream, snapshot = _restore_resume_state(
        snapshot_path, data, config=config, device=device
    )
    restored_error = _maximum_state_error(
        _cpu_state_dict(model), reference["state_dict"]
    )
    if restored_error != 0.0:
        raise RuntimeError("restored model differs from the reference checkpoint")

    best_validation_mse = float(snapshot["best_validation_prediction_mse"])
    best_step = int(snapshot["best_step"])
    best_state = _cpu_state_dict(model)
    requested_validation = {REFERENCE_STEP: validation_15k}
    validation_curve: list[dict[str, float | int]] = [
        {"step": REFERENCE_STEP, "validation_prediction_mse": validation_15k}
    ]
    model.train()
    for step in range(REFERENCE_STEP + 1, FINAL_STEP + 1):
        _train_step(
            model, optimizer, stream, config=config, device=device, step=step
        )
        if step % config.validation_interval == 0:
            validation_mse = evaluate_validation(
                "history",
                model,
                data.validation,
                lambda_bal=config.lambda_bal,
                device=device,
                batch_size=config.evaluation_batch_size,
            )["prediction_loss"]
            validation_curve.append(
                {"step": step, "validation_prediction_mse": validation_mse}
            )
            if step in REPORT_STEPS:
                requested_validation[step] = validation_mse
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_step = step
                best_state = _cpu_state_dict(model)
            model.train()
            if step % 2_500 == 0:
                print(
                    f"[resume] step={step}/{FINAL_STEP} val={validation_mse:.8g} "
                    f"best={best_validation_mse:.8g}@{best_step}",
                    flush=True,
                )

    if set(requested_validation) != set(REPORT_STEPS):
        raise RuntimeError("one or more requested validation points are missing")
    torch.save(
        {
            "model_name": "history",
            "seed": config.seed,
            "best_step": best_step,
            "best_validation_prediction_mse": best_validation_mse,
            "selection_metric": "validation prediction MSE",
            "state_dict": best_state,
            "config": config_record,
        },
        output_dir / "best_checkpoint.pt",
    )
    result = {
        "replay": {
            "reason": "the original checkpoint did not contain optimizer, batch-stream, or RNG state",
            "reference_checkpoint": str(reference_path),
            "maximum_parameter_error_at_step_15000": replay_error,
            "maximum_parameter_error_after_restore": restored_error,
            "complete_resume_state_saved": str(snapshot_path),
        },
        "checkpoint_selection_metric": "validation prediction MSE",
        "validation_prediction_mse": {
            str(step): requested_validation[step] for step in REPORT_STEPS
        },
        "best_validation_prediction_mse": best_validation_mse,
        "best_step": best_step,
        "validation_curve": validation_curve,
        "test_evaluation_performed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("artifacts/experiment2_history_sanity_seed0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment2_history_horizon_25000_seed0"),
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
    result = run_horizon_check(
        device=device,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
