"""Run the single-seed Experiment 2 history-context training sanity check."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from ..losses import load_balance_loss
from ..models import (
    ACTION_DIM,
    STATE_DIM,
    HistoryConditionedMoEPredictor,
    MoEPredictor,
    PredictorOutput,
    build_experiment1_model,
    build_experiment2_model,
    next_state_mse,
)
from ..synthetic import (
    HISTORY_LENGTH,
    NUM_SYSTEMS,
    HistoryTransitionBatch,
    make_history_transition_batch,
    sample_actions,
    system_true_transition,
)


SanityModelName = Literal["state_only", "history"]
MODEL_NAMES: tuple[SanityModelName, ...] = ("state_only", "history")


@dataclass(frozen=True)
class HistorySanityConfig:
    train_samples: int = 50_000
    validation_samples: int = 10_000
    test_samples: int = 10_000
    batch_size: int = 512
    evaluation_batch_size: int = 2_048
    training_steps: int = 15_000
    validation_interval: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_bal: float = 0.01
    seed: int = 0
    data_seed: int = 20_260_920
    batch_seed_offset: int = 10_000
    diagnostic_samples: int = 512
    diagnostic_seed_offset: int = 20_000

    def __post_init__(self) -> None:
        positive = (
            self.train_samples,
            self.validation_samples,
            self.test_samples,
            self.batch_size,
            self.evaluation_batch_size,
            self.training_steps,
            self.validation_interval,
            self.diagnostic_samples,
        )
        if min(positive) <= 0:
            raise ValueError("all dataset, batch, and schedule sizes must be positive")
        if self.batch_size > self.train_samples:
            raise ValueError("batch_size cannot exceed train_samples")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters must be valid")
        if not math.isfinite(self.lambda_bal) or self.lambda_bal < 0:
            raise ValueError("lambda_bal must be finite and non-negative")


@dataclass(frozen=True)
class HistorySanityData:
    train: HistoryTransitionBatch
    validation: HistoryTransitionBatch
    test: HistoryTransitionBatch


class ShuffledHistoryBatchStream:
    """Deterministic shuffled batches reset identically for both models."""

    def __init__(
        self, data: HistoryTransitionBatch, batch_size: int, *, seed: int
    ) -> None:
        if batch_size > len(data):
            raise ValueError("batch_size cannot exceed dataset size")
        self.data = data
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.permutation = torch.empty(0, dtype=torch.long)
        self.cursor = len(data)

    def next(self) -> HistoryTransitionBatch:
        if self.cursor + self.batch_size > len(self.data):
            self.permutation = torch.randperm(len(self.data), generator=self.generator)
            self.cursor = 0
        indices = self.permutation[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return HistoryTransitionBatch(
            state_history=self.data.state_history[indices],
            past_actions=self.data.past_actions[indices],
            current_action=self.data.current_action[indices],
            next_state=self.data.next_state[indices],
            system_id=self.data.system_id[indices],
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state().clone(),
            "permutation": self.permutation.clone(),
            "cursor": self.cursor,
            "batch_size": self.batch_size,
            "dataset_size": len(self.data),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["batch_size"] != self.batch_size:
            raise ValueError("batch stream checkpoint has a different batch size")
        if state["dataset_size"] != len(self.data):
            raise ValueError("batch stream checkpoint has a different dataset size")
        self.generator.set_state(state["generator_state"])
        self.permutation = state["permutation"].clone()
        self.cursor = int(state["cursor"])


def make_sanity_data(config: HistorySanityConfig) -> HistorySanityData:
    """Generate shared pooled-IID splits with system IDs retained as metadata."""

    return HistorySanityData(
        train=make_history_transition_batch(
            config.train_samples, seed=config.data_seed
        ),
        validation=make_history_transition_batch(
            config.validation_samples, seed=config.data_seed + 1
        ),
        test=make_history_transition_batch(
            config.test_samples, seed=config.data_seed + 2
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


def build_sanity_model(
    model_name: SanityModelName, *, initialization_seed: int
) -> MoEPredictor | HistoryConditionedMoEPredictor:
    if model_name == "state_only":
        model = build_experiment1_model(
            "moe_s", initialization_seed=initialization_seed
        )
        assert isinstance(model, MoEPredictor)
        return model
    if model_name == "history":
        return build_experiment2_model(initialization_seed=initialization_seed)
    raise ValueError(f"unknown sanity model: {model_name}")


def _forward(
    model_name: SanityModelName,
    model: nn.Module,
    batch: HistoryTransitionBatch,
) -> PredictorOutput:
    if model_name == "state_only":
        if not isinstance(model, MoEPredictor):
            raise TypeError("state_only requires MoEPredictor")
        return model(batch.state_history[:, -1], batch.current_action)
    if model_name == "history":
        if not isinstance(model, HistoryConditionedMoEPredictor):
            raise TypeError("history requires HistoryConditionedMoEPredictor")
        return model(
            batch.state_history, batch.past_actions, batch.current_action
        )
    raise ValueError(f"unknown sanity model: {model_name}")


def history_sanity_loss(
    model_name: SanityModelName,
    model: nn.Module,
    batch: HistoryTransitionBatch,
    *,
    lambda_bal: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, prediction, and balance losses; no CR term is present."""

    output = _forward(model_name, model, batch)
    if output.alpha is None:
        raise TypeError("both sanity models must return routing weights")
    prediction = next_state_mse(output.pred, batch.next_state)
    balance = load_balance_loss(output.alpha)
    total = prediction + lambda_bal * balance
    return total, prediction, balance


def _to_device(
    batch: HistoryTransitionBatch, device: torch.device
) -> HistoryTransitionBatch:
    return HistoryTransitionBatch(
        state_history=batch.state_history.to(device),
        past_actions=batch.past_actions.to(device),
        current_action=batch.current_action.to(device),
        next_state=batch.next_state.to(device),
        system_id=batch.system_id.to(device),
    )


def _slice_batch(
    data: HistoryTransitionBatch, start: int, stop: int
) -> HistoryTransitionBatch:
    return HistoryTransitionBatch(
        state_history=data.state_history[start:stop],
        past_actions=data.past_actions[start:stop],
        current_action=data.current_action[start:stop],
        next_state=data.next_state[start:stop],
        system_id=data.system_id[start:stop],
    )


@torch.no_grad()
def evaluate_prediction(
    model_name: SanityModelName,
    model: nn.Module,
    data: HistoryTransitionBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate overall and per-system elementwise next-state MSE."""

    model.eval()
    total_squared_error = 0.0
    total_elements = 0
    system_squared_error = torch.zeros(NUM_SYSTEMS, dtype=torch.float64)
    system_elements = torch.zeros(NUM_SYSTEMS, dtype=torch.long)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        batch = _to_device(_slice_batch(data, start, stop), device)
        prediction = _forward(model_name, model, batch).pred
        squared_error = (prediction - batch.next_state).square()
        total_squared_error += float(squared_error.sum().cpu())
        total_elements += squared_error.numel()
        per_sample = squared_error.sum(dim=-1).cpu().to(torch.float64)
        system_id = batch.system_id.cpu()
        for selected_system in range(NUM_SYSTEMS):
            mask = system_id == selected_system
            system_squared_error[selected_system] += per_sample[mask].sum()
            system_elements[selected_system] += int(mask.sum()) * STATE_DIM

    return {
        "prediction_mse": total_squared_error / total_elements,
        "per_system_prediction_mse": [
            float(system_squared_error[index] / system_elements[index])
            for index in range(NUM_SYSTEMS)
        ],
    }


def evaluate_validation(
    model_name: SanityModelName,
    model: nn.Module,
    data: HistoryTransitionBatch,
    *,
    lambda_bal: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    sums = {"prediction_loss": 0.0, "load_balance_loss": 0.0, "total_loss": 0.0}
    samples = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            batch = _to_device(_slice_batch(data, start, stop), device)
            total, prediction, balance = history_sanity_loss(
                model_name, model, batch, lambda_bal=lambda_bal
            )
            count = stop - start
            sums["prediction_loss"] += float(prediction.cpu()) * count
            sums["load_balance_loss"] += float(balance.cpu()) * count
            sums["total_loss"] += float(total.cpu()) * count
            samples += count
    return {name: value / samples for name, value in sums.items()}


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def train_one_model(
    model_name: SanityModelName,
    config: HistorySanityConfig,
    data: HistorySanityData,
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one condition and select only by validation prediction MSE."""

    _seed_everything(config.seed)
    model = build_sanity_model(
        model_name, initialization_seed=config.seed
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    stream = ShuffledHistoryBatchStream(
        data.train,
        config.batch_size,
        seed=config.data_seed + config.batch_seed_offset + config.seed,
    )
    log_path = output_dir / "logs" / f"{model_name}_seed_{config.seed}.jsonl"
    checkpoint_path = (
        output_dir / "checkpoints" / f"{model_name}_seed_{config.seed}.pt"
    )
    log_path.write_text("", encoding="utf-8")
    started = time.perf_counter()
    best_validation_mse = math.inf
    best_step = -1
    best_state: dict[str, Tensor] | None = None
    window = {"prediction_loss": 0.0, "load_balance_loss": 0.0, "total_loss": 0.0}
    window_samples = 0

    def validate(step: int, train_metrics: dict[str, float] | None) -> None:
        nonlocal best_validation_mse, best_step, best_state
        validation = evaluate_validation(
            model_name,
            model,
            data.validation,
            lambda_bal=config.lambda_bal,
            device=device,
            batch_size=config.evaluation_batch_size,
        )
        if not all(math.isfinite(value) for value in validation.values()):
            raise FloatingPointError(f"non-finite validation metrics: {validation}")
        _append_json_line(
            log_path,
            {
                "step": step,
                "train": train_metrics,
                "validation": validation,
                "checkpoint_selection_metric": "validation.prediction_loss",
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        if validation["prediction_loss"] < best_validation_mse:
            best_validation_mse = validation["prediction_loss"]
            best_step = step
            best_state = _cpu_state_dict(model)
        model.train()

    validate(0, None)
    for step in range(1, config.training_steps + 1):
        batch = _to_device(stream.next(), device)
        optimizer.zero_grad(set_to_none=True)
        total, prediction, balance = history_sanity_loss(
            model_name, model, batch, lambda_bal=config.lambda_bal
        )
        values = {
            "prediction_loss": float(prediction.detach().cpu()),
            "load_balance_loss": float(balance.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(
                f"non-finite {model_name} loss at step {step}: {values}"
            )
        total.backward()
        for parameter_name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"invalid gradient for {model_name}.{parameter_name} at step {step}"
                )
        optimizer.step()

        for name, value in values.items():
            window[name] += value * len(batch)
        window_samples += len(batch)
        if step % config.validation_interval == 0 or step == config.training_steps:
            train_metrics = {
                name: value / window_samples for name, value in window.items()
            }
            validate(step, train_metrics)
            print(
                f"[{model_name}] step={step}/{config.training_steps} "
                f"train_pred={train_metrics['prediction_loss']:.6g} "
                f"val_pred={best_validation_mse:.6g} best_step={best_step}",
                flush=True,
            )
            window = {name: 0.0 for name in window}
            window_samples = 0

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
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
    test_metrics = evaluate_prediction(
        model_name,
        model,
        data.test,
        device=device,
        batch_size=config.evaluation_batch_size,
    )
    result = {
        "model": model_name,
        "seed": config.seed,
        "best_step": best_step,
        "best_validation_prediction_mse": best_validation_mse,
        "checkpoint_selection_metric": "validation prediction MSE",
        "test": test_metrics,
        "training_seconds": time.perf_counter() - started,
    }
    (output_dir / "results" / f"{model_name}_seed_{config.seed}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model, result


def _state_jacobian(state: Tensor, action: Tensor, system_id: int) -> Tensor:
    state = state.detach().requires_grad_(True)
    output = system_true_transition(state, action, system_id)
    return torch.stack(
        [
            torch.autograd.grad(
                output[:, index].sum(), state, retain_graph=True
            )[0]
            for index in range(STATE_DIM)
        ],
        dim=1,
    )


def _invert_transition(
    target_state: Tensor,
    action: Tensor,
    system_id: int,
    *,
    iterations: int = 15,
    tolerance: float = 1e-10,
) -> Tensor:
    """Solve F^(s)(previous_state, action)=target_state by batched Newton steps."""

    previous = target_state.clone()
    for _ in range(iterations):
        residual = system_true_transition(previous, action, system_id) - target_state
        if float(residual.abs().max()) <= tolerance:
            break
        jacobian = _state_jacobian(previous, action, system_id)
        update = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
        previous = (previous - update).detach()
    error = (
        system_true_transition(previous, action, system_id) - target_state
    ).abs().max()
    if float(error) > 1e-8:
        raise RuntimeError(
            f"history inversion failed for system {system_id}: max error={float(error)}"
        )
    return previous


def make_matched_terminal_histories(
    sample_count: int, *, seed: int
) -> tuple[list[Tensor], Tensor, Tensor, float]:
    """Build valid system histories sharing exact terminal states and actions."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    current_state = torch.rand(
        (sample_count, STATE_DIM), generator=generator, dtype=torch.float64
    ) - 0.5
    past_actions = sample_actions(
        sample_count * HISTORY_LENGTH,
        "full",
        generator=generator,
        dtype=torch.float64,
    ).reshape(sample_count, HISTORY_LENGTH, ACTION_DIM)
    current_action = sample_actions(
        sample_count, "full", generator=generator, dtype=torch.float64
    )
    histories: list[Tensor] = []
    max_error = 0.0
    for system_id in range(NUM_SYSTEMS):
        reverse_states = [current_state]
        target = current_state
        for action_index in reversed(range(HISTORY_LENGTH)):
            target = _invert_transition(
                target, past_actions[:, action_index], system_id
            )
            reverse_states.append(target)
        history = torch.stack(list(reversed(reverse_states)), dim=1)
        for action_index in range(HISTORY_LENGTH):
            expected = system_true_transition(
                history[:, action_index], past_actions[:, action_index], system_id
            )
            max_error = max(
                max_error,
                float((expected - history[:, action_index + 1]).abs().max()),
            )
        histories.append(history)
    return histories, past_actions, current_action, max_error


def history_routing_diagnostics(
    model: HistoryConditionedMoEPredictor,
    *,
    sample_count: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    histories, past_actions, current_action, validity_error = (
        make_matched_terminal_histories(sample_count, seed=seed)
    )
    del current_action
    model.eval()
    contexts = []
    alphas = []
    with torch.no_grad():
        for history in histories:
            context, alpha = model.routing_weights(
                history.to(device=device, dtype=torch.float32),
                past_actions.to(device=device, dtype=torch.float32),
            )
            contexts.append(context.cpu())
            alphas.append(alpha.cpu())
    context_tensor = torch.stack(contexts)
    alpha_tensor = torch.stack(alphas)
    distance_matrix = torch.zeros(NUM_SYSTEMS, NUM_SYSTEMS)
    for first in range(NUM_SYSTEMS):
        for second in range(first + 1, NUM_SYSTEMS):
            distance = torch.linalg.vector_norm(
                alpha_tensor[first] - alpha_tensor[second], dim=-1
            ).mean()
            distance_matrix[first, second] = distance_matrix[second, first] = distance

    flattened_contexts = context_tensor.flatten(end_dim=1)
    feature_std = flattened_contexts.std(dim=0, unbiased=False)
    mean_feature_std = float(feature_std.mean())
    return {
        "matched_current_state": True,
        "matched_past_actions": True,
        "matched_current_actions": True,
        "history_transition_max_abs_error": validity_error,
        "pairwise_mean_alpha_distance": distance_matrix.tolist(),
        "context_mean_norm": float(
            torch.linalg.vector_norm(flattened_contexts, dim=-1).mean()
        ),
        "context_global_std": float(flattened_contexts.std(unbiased=False)),
        "context_mean_feature_std": mean_feature_std,
        "context_min_feature_std": float(feature_std.min()),
        "context_collapse_threshold": 1e-3,
        "context_collapsed": mean_feature_std < 1e-3,
    }


def _assert_fair_expert_initialization(seed: int) -> None:
    state_only = build_sanity_model("state_only", initialization_seed=seed)
    history = build_sanity_model("history", initialization_seed=seed)
    first = dict(state_only.experts.named_parameters())
    second = dict(history.experts.named_parameters())
    if first.keys() != second.keys():
        raise RuntimeError("expert parameter names differ between sanity models")
    for name in first:
        if not torch.equal(first[name], second[name]):
            raise RuntimeError(f"expert initialization differs at {name}")


def run_history_sanity(
    config: HistorySanityConfig,
    *,
    device: torch.device,
    output_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    for child in ("checkpoints", "logs", "results"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    serialized_config = json.dumps(
        asdict(config), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    config_path = output_dir / "config.json"
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != serialized_config:
            raise RuntimeError(f"{config_path} contains a different protocol")
    else:
        config_path.write_text(serialized_config, encoding="utf-8")

    _assert_fair_expert_initialization(config.seed)
    print(f"Generating shared pooled-IID history data; device={device}", flush=True)
    data = make_sanity_data(config)
    results = []
    history_model: HistoryConditionedMoEPredictor | None = None
    for model_name in MODEL_NAMES:
        result_path = output_dir / "results" / f"{model_name}_seed_{config.seed}.json"
        checkpoint_path = (
            output_dir / "checkpoints" / f"{model_name}_seed_{config.seed}.pt"
        )
        if resume and result_path.exists() and checkpoint_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            model = build_sanity_model(model_name, initialization_seed=config.seed)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(device)
            print(f"[resume] loaded {result_path}", flush=True)
        else:
            model, result = train_one_model(
                model_name, config, data, device=device, output_dir=output_dir
            )
        results.append(result)
        if model_name == "history":
            assert isinstance(model, HistoryConditionedMoEPredictor)
            history_model = model

    if history_model is None:
        raise RuntimeError("history model result is missing")
    routing = history_routing_diagnostics(
        history_model,
        sample_count=config.diagnostic_samples,
        seed=config.data_seed + config.diagnostic_seed_offset + config.seed,
        device=device,
    )
    summary = {
        "config": asdict(config),
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "shared_dataset_object": True,
        "results": results,
        "history_routing_diagnostics": routing,
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
        default=Path("artifacts/experiment2_history_sanity_seed0"),
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
    run_history_sanity(
        HistorySanityConfig(),
        device=device,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
