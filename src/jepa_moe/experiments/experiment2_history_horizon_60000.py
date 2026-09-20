"""Resume the history-conditioned seed-0 saturation check from 40k to 60k."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

import torch

from .experiment2_history_horizon import (
    _maximum_state_error,
    _restore_resume_state,
    _train_step,
)
from .experiment2_history_horizon_40000 import (
    _save_full_snapshot,
    _validation_mse,
)
from .experiment2_history_sanity import (
    HistorySanityConfig,
    _cpu_state_dict,
    make_sanity_data,
)
from ..models import HistoryConditionedMoEPredictor


RESUME_STEP = 40_000
FINAL_STEP = 60_000
REPORT_STEPS = (
    40_000,
    42_500,
    45_000,
    47_500,
    50_000,
    52_500,
    55_000,
    57_500,
    60_000,
)
VALIDATION_INTERVAL = 250


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
            "B": "relative validation-MSE improvement from 55k to 60k is below 5%",
            "combination": "A or B",
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    data = make_sanity_data(config)
    source_path = reference_dir / "latest_resume.pt"
    source_raw = torch.load(source_path, map_location="cpu", weights_only=False)
    model, optimizer, stream, restored = _restore_resume_state(
        source_path, data, config=config, device=device
    )
    if restored["step"] != RESUME_STEP:
        raise RuntimeError("the full source snapshot is not at step 40,000")
    restore_error = _maximum_state_error(
        _cpu_state_dict(model), source_raw["model_state_dict"]
    )
    if restore_error != 0.0:
        raise RuntimeError("full step-40k resume verification failed")

    best_validation_mse = float(restored["best_validation_prediction_mse"])
    best_step = int(restored["best_step"])
    validation_40k = _validation_mse(model, data, config, device)
    requested_validation = {RESUME_STEP: validation_40k}
    validation_curve = [
        {"step": RESUME_STEP, "validation_prediction_mse": validation_40k}
    ]

    resume_path = output_dir / "resume_step_40000.pt"
    latest_path = output_dir / "latest_resume.pt"
    best_path = output_dir / "best_resume.pt"
    _save_full_snapshot(
        resume_path,
        model,
        optimizer,
        stream,
        current_step=RESUME_STEP,
        config=config,
        best_validation_mse=best_validation_mse,
        best_step=best_step,
    )
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
    shutil.copy2(reference_dir / "best_resume.pt", best_path)

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
        (requested_validation[55_000] - requested_validation[60_000])
        / requested_validation[55_000]
    )
    criterion_a = best_step <= FINAL_STEP - 2_500
    criterion_b = relative_improvement < 0.05
    result = {
        "validation_prediction_mse": {
            str(step): requested_validation[step] for step in REPORT_STEPS
        },
        "best_validation_prediction_mse": best_validation_mse,
        "best_step": best_step,
        "relative_improvement_55000_to_60000": relative_improvement,
        "saturation": {
            "criterion_a": criterion_a,
            "criterion_b": criterion_b,
            "satisfied": criterion_a or criterion_b,
        },
        "full_resume": {
            "source_snapshot": str(source_path),
            "source_step": restored["step"],
            "restored_parameter_error_at_40000": restore_error,
            "verified": restored["step"] == RESUME_STEP and restore_error == 0.0,
            "resume_snapshot": str(resume_path),
            "latest_snapshot": str(latest_path),
            "best_snapshot": str(best_path),
            "direction_rng_state": restored["direction_rng_state"],
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
        default=Path("artifacts/experiment2_history_horizon_40000_seed0"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiment2_history_horizon_60000_seed0"),
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
