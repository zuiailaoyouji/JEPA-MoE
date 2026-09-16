"""Unified first-version training objective for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .jacobian import (
    basis_composition_target_response,
    mixture_action_response,
)
from .losses import control_response_loss, load_balance_loss
from .models import (
    Experiment1ModelName,
    MoEPredictor,
    TrainingLoss,
    next_state_mse,
)


@dataclass(frozen=True)
class Experiment1ObjectiveConfig:
    lambda_cr: float = 1.0
    lambda_bal: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_cr) or self.lambda_cr < 0:
            raise ValueError("lambda_cr must be finite and non-negative")
        if not math.isfinite(self.lambda_bal) or self.lambda_bal < 0:
            raise ValueError("lambda_bal must be finite and non-negative")


def sample_unit_action_directions(
    batch_size: int,
    action_dim: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Sample one reproducible unit Gaussian action direction per sample."""

    if batch_size <= 0 or action_dim <= 0:
        raise ValueError("batch_size and action_dim must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    sample_device = generator.device
    direction = torch.randn(
        (batch_size, action_dim),
        generator=generator,
        dtype=dtype,
        device=sample_device,
    )
    direction = direction / (
        torch.linalg.vector_norm(direction, dim=-1, keepdim=True) + eps
    )
    if device is not None:
        direction = direction.to(device=device)
    return direction


def experiment1_training_loss(
    model_name: Experiment1ModelName,
    model: nn.Module,
    state: Tensor,
    action: Tensor,
    next_state: Tensor,
    *,
    config: Experiment1ObjectiveConfig,
    direction: Tensor | None = None,
) -> TrainingLoss:
    """Combine the raw losses for one of the four Experiment 1 conditions."""

    output = model(state, action)
    prediction = next_state_mse(output.pred, next_state)
    zero = prediction.new_zeros(())

    if model_name == "dense":
        if output.alpha is not None:
            raise TypeError("dense condition must not return routing weights")
        return TrainingLoss(prediction, prediction, zero, zero)

    if not isinstance(model, MoEPredictor) or output.alpha is None:
        raise TypeError("MoE conditions require an Experiment 1 MoEPredictor")
    expected_router_input = "state_action" if model_name == "moe_a" else "state"
    if model.router_input != expected_router_input:
        raise ValueError(
            f"{model_name} requires router_input={expected_router_input!r}"
        )

    balance = load_balance_loss(output.alpha)
    control_response = zero
    if model_name == "ours":
        if direction is None:
            raise ValueError("ours requires an externally sampled action direction")
        predicted_response = mixture_action_response(
            model, state, action, direction
        )
        target_response = basis_composition_target_response(
            state, action, direction
        )
        control_response = control_response_loss(
            predicted_response, target_response
        )
    elif model_name not in ("moe_a", "moe_s"):
        raise ValueError(f"unknown Experiment 1 model: {model_name}")

    total = (
        prediction
        + config.lambda_cr * control_response
        + config.lambda_bal * balance
    )
    return TrainingLoss(total, prediction, control_response, balance)
