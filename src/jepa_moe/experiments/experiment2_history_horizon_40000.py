"""Resume the history-conditioned seed-0 saturation check from 25k to 40k."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from .experiment2_history_horizon import (
    _capture_resume_state,
    _maximum_state_error,
    _restore_resume_state,
    _train_step,
)
from .experiment2_history_sanity import (
    HistorySanityConfig,
    _cpu_state_dict,
    evaluate_validation,
    make_sanity_data,
)
from ..models import HistoryConditionedMoEPredictor


BRIDGE_START_STEP = 15_000
RESUME_STEP = 25_000
FINAL_STEP = 40_000
REPORT_STEPS = (25_000, 27_500, 30_000, 32_500, 35_000, 37_500, 40_000)
VALIDATION_INTERVAL = 250


def _validation_mse(
    model: HistoryConditionedMoEPredictor,
    data,
    config: HistorySanityConfig,
    device: torch.device,
) -> float:
    return evaluate_validation(
        "history",
        model,
        data.validation,
        lambda_bal=config.lambda_bal,
        device=device,
        batch_size=config.evaluation_batch_size,
    )["prediction_loss"]


def _save_full_snapshot(
    path: Path,
    model: HistoryConditionedMoEPredictor,
    optimizer: torch.optim.Optimizer,
    stream,
    *,
    current_step: int,
    config: HistorySanityConfig,
    best_validation_mse: float,
    best_step: int,
) -> None:
    torch.save(
        _capture_resume_state(
            model,
            optimizer,
            stream,
            current_step=current_step,
            config=config,
            best_validation_mse=best_validation_mse,
            best_step=best_step,
            direction_rng_state=None,
        ),
        path,
    )


def run_horizon_check(
    *,
    device: torch.device,
    reference_dir: Path,
    output_dir: Path,
) -> dict:
    config = HistorySanityConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        **asdict(config),
        "resume_step": RESUME_STEP,
        "extended_training_steps": FINAL_STEP,
        "reported_validation_interval": 2_500,
        "checkpoint_validation_interval": VALIDATION_INTERVAL,
        "saturation_criterion": {
            "A": "best checkpoint is not in the final 2,500 steps",
            "B": "relative validation-MSE improvement from 35k to 40k is below 5%",
            "combination": "A or B",
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    data = make_sanity_data(config)
    source_snapshot = reference_dir / "resume_step_15000.pt"
    reference_25k_path = reference_dir / "best_checkpoint.pt"
    reference_25k = torch.load(
        reference_25k_path, map_location="cpu", weights_only=True
    )
    if reference_25k["best_step"] != RESUME_STEP:
        raise RuntimeError("the step-25k reference checkpoint is not at step 25,000")

    model, optimizer, stream, source = _restore_resume_state(
        source_snapshot, data, config=config, device=device
    )
    if source["step"] != BRIDGE_START_STEP:
        raise RuntimeError("the full source snapshot is not at step 15,000")
    model.train()
    for step in range(BRIDGE_START_STEP + 1, RESUME_STEP + 1):
        _train_step(model, optimizer, stream, config=config, device=device, step=step)
        if step % 2_500 == 0:
            print(f"[bridge] step={step}/{RESUME_STEP}", flush=True)

    bridge_error = _maximum_state_error(
        _cpu_state_dict(model), reference_25k["state_dict"]
    )
    if bridge_error != 0.0:
        raise RuntimeError(
            "15k full-state continuation did not recover the 25k reference: "
            f"max parameter error={bridge_error}"
        )
    validation_25k = _validation_mse(model, data, config, device)
    full_25k_path = output_dir / "resume_step_25000.pt"
    _save_full_snapshot(
        full_25k_path,
        model,
        optimizer,
        stream,
        current_step=RESUME_STEP,
        config=config,
        best_validation_mse=float(reference_25k["best_validation_prediction_mse"]),
        best_step=int(reference_25k["best_step"]),
    )

    model, optimizer, stream, restored = _restore_resume_state(
        full_25k_path, data, config=config, device=device
    )
    restore_error = _maximum_state_error(
        _cpu_state_dict(model), reference_25k["state_dict"]
    )
    if restored["step"] != RESUME_STEP or restore_error != 0.0:
        raise RuntimeError("full step-25k resume verification failed")

    best_validation_mse = float(restored["best_validation_prediction_mse"])
    best_step = int(restored["best_step"])
    requested_validation = {RESUME_STEP: validation_25k}
    validation_curve = [
        {"step": RESUME_STEP, "validation_prediction_mse": validation_25k}
    ]
    latest_path = output_dir / "latest_resume.pt"
    best_path = output_dir / "best_resume.pt"
    _save_full_snapshot(
        latest_path,
        model,
        optimizer,
        stream,
        current_step=RESUME_STEP,
        config=config,
        best_validation_mse=best_validation_mse,
        best_step=best_step,
    )
    _save_full_snapshot(
        best_path,
        model,
        optimizer,
        stream,
        current_step=RESUME_STEP,
        config=config,
        best_validation_mse=best_validation_mse,
        best_step=best_step,
    )

    model.train()
    for step in range(RESUME_STEP + 1, FINAL_STEP + 1):
        _train_step(model, optimizer, stream, config=config, device=device, step=step)
        if step % VALIDATION_INTERVAL != 0:
            continue
        validation_mse = _validation_mse(model, data, config, device)
        validation_curve.append(
            {"step": step, "validation_prediction_mse": validation_mse}
        )
        if step in REPORT_STEPS:
            requested_validation[step] = validation_mse
        improved = validation_mse < best_validation_mse
        if improved:
            best_validation_mse = validation_mse
            best_step = step
        model.train()
        _save_full_snapshot(
            latest_path,
            model,
            optimizer,
            stream,
            current_step=step,
            config=config,
            best_validation_mse=best_validation_mse,
            best_step=best_step,
        )
        if improved:
            _save_full_snapshot(
                best_path,
                model,
                optimizer,
                stream,
                current_step=step,
                config=config,
                best_validation_mse=best_validation_mse,
                best_step=best_step,
            )
        if step % 2_500 == 0:
            print(
                f"[resume] step={step}/{FINAL_STEP} val={validation_mse:.8g} "
                f"best={best_validation_mse:.8g}@{best_step}",
                flush=True,
            )

    if set(requested_validation) != set(REPORT_STEPS):
        raise RuntimeError("one or more requested validation points are missing")
    relative_improvement = (
        (requested_validation[35_000] - requested_validation[40_000])
        / requested_validation[35_000]
    )
    criterion_a = best_step <= FINAL_STEP - 2_500
    criterion_b = relative_improvement < 0.05
    result = {
        "validation_prediction_mse": {
            str(step): requested_validation[step] for step in REPORT_STEPS
        },
        "best_validation_prediction_mse": best_validation_mse,
        "best_step": best_step,
        "relative_improvement_35000_to_40000": relative_improvement,
        "saturation": {
            "criterion_a": criterion_a,
            "criterion_b": criterion_b,
            "satisfied": criterion_a or criterion_b,
        },
        "full_resume": {
            "source_step": BRIDGE_START_STEP,
            "bridge_parameter_error_at_25000": bridge_error,
            "restored_parameter_error_at_25000": restore_error,
            "snapshot_step": restored["step"],
            "verified": bridge_error == 0.0 and restore_error == 0.0,
            "latest_snapshot": str(latest_path),
            "best_snapshot": str(best_path),
            "direction_rng_state": None,
        },
        "checkpoint_selection_metric": "validation prediction MSE",
        "test_evaluation_performed": False,
        "validation_curve": validation_curve,
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
        default=Path("artifacts/experiment2_history_horizon_25000_seed0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment2_history_horizon_40000_seed0"),
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
