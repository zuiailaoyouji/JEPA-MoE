"""Training objectives that distinguish the Jacobian MoE condition."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .jacobian import expert_action_jacobians
from .models import NUM_EXPERTS, _MoEPredictor


class ControlJacobianSpecialization(NamedTuple):
    """Scalar loss terms and batch-mean specialization diagnostics."""

    total: Tensor
    diversity: Tensor
    activity: Tensor
    pairwise_cosines: Tensor
    expert_jacobian_norms: Tensor
    routing_weights: Tensor


def control_response_loss(
    predicted_response: Tensor,
    target_response: Tensor,
) -> Tensor:
    """Return mean_batch ||target_response - predicted_response||_2^2."""

    if predicted_response.ndim != 2:
        raise ValueError(
            "predicted_response must have shape [batch_size, response_dim]"
        )
    if target_response.shape != predicted_response.shape:
        raise ValueError(
            f"target_response must have shape {tuple(predicted_response.shape)}, "
            f"got {tuple(target_response.shape)}"
        )
    return (predicted_response - target_response).square().sum(dim=-1).mean()


def load_balance_loss(routing_weights: Tensor) -> Tensor:
    """Penalize deviation of mean soft-routing usage from uniform usage."""

    if routing_weights.ndim != 2 or routing_weights.shape[-1] <= 0:
        raise ValueError(
            "routing_weights must have shape [batch_size, num_experts]"
        )
    if routing_weights.shape[0] <= 0:
        raise ValueError("routing_weights batch size must be positive")
    num_experts = routing_weights.shape[-1]
    mean_usage = routing_weights.mean(dim=0)
    uniform_usage = 1.0 / num_experts
    return num_experts * (mean_usage - uniform_usage).square().sum()


def control_jacobian_specialization_terms(
    model: _MoEPredictor,
    state: Tensor,
    action: Tensor,
    *,
    create_graph: bool = True,
    margin: float = 0.3,
    min_jacobian_norm: float = 0.05,
    beta_activity: float = 0.1,
    eps: float = 1e-8,
) -> ControlJacobianSpecialization:
    """Compute weighted margin diversity, activity, and logging statistics."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if not 0 <= margin <= 1:
        raise ValueError("margin must be between 0 and 1")
    if min_jacobian_norm < 0:
        raise ValueError("min_jacobian_norm must be non-negative")
    if beta_activity < 0:
        raise ValueError("beta_activity must be non-negative")

    jacobians = expert_action_jacobians(
        model, state, action, create_graph=create_graph
    )
    output = model(state, action)
    assert output.alpha is not None

    flattened = jacobians.flatten(start_dim=2)
    jacobian_norms = torch.linalg.vector_norm(flattened, ord=2, dim=-1)
    normalized = F.normalize(flattened, p=2, dim=-1, eps=eps)

    pair_losses = []
    pair_weights = []
    pairwise_cosines = []
    for first in range(NUM_EXPERTS):
        for second in range(first + 1, NUM_EXPERTS):
            cosine = torch.sum(
                normalized[:, first, :] * normalized[:, second, :], dim=-1
            )
            pairwise_cosines.append(cosine)
            pair_losses.append(F.relu(cosine.abs() - margin).square())
            pair_weights.append(
                (output.alpha[:, first] * output.alpha[:, second]).detach()
            )

    pair_loss_tensor = torch.stack(pair_losses, dim=-1)
    pair_weight_tensor = torch.stack(pair_weights, dim=-1)
    diversity = torch.sum(pair_weight_tensor * pair_loss_tensor) / (
        torch.sum(pair_weight_tensor) + eps
    )
    activity = F.relu(min_jacobian_norm - jacobian_norms).square().mean()
    total = diversity + beta_activity * activity

    return ControlJacobianSpecialization(
        total=total,
        diversity=diversity,
        activity=activity,
        pairwise_cosines=torch.stack(pairwise_cosines, dim=-1).mean(dim=0),
        expert_jacobian_norms=jacobian_norms.mean(dim=0),
        routing_weights=output.alpha.mean(dim=0),
    )


def control_jacobian_specialization_loss(
    model: _MoEPredictor,
    state: Tensor,
    action: Tensor,
    *,
    create_graph: bool = True,
    margin: float = 0.3,
    min_jacobian_norm: float = 0.05,
    beta_activity: float = 0.1,
    eps: float = 1e-8,
) -> Tensor:
    """Return the scalar control-Jacobian structured specialization loss."""

    return control_jacobian_specialization_terms(
        model,
        state,
        action,
        create_graph=create_graph,
        margin=margin,
        min_jacobian_norm=min_jacobian_norm,
        beta_activity=beta_activity,
        eps=eps,
    ).total
