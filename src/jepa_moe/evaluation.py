"""Prediction, routing, and dynamics-dictionary evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
import torch.nn.functional as F

from .jacobian import expert_action_jacobians
from .models import NUM_EXPERTS
from .synthetic import (
    RolloutBatch,
    TransitionBatch,
    ground_truth_expert_action_jacobians,
    true_routing_weights,
)


def _slices(length: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def one_step_mse(
    model: nn.Module,
    data: TransitionBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    """Evaluate elementwise one-step MSE without retaining predictions."""

    model.eval()
    squared_error = 0.0
    element_count = 0
    with torch.inference_mode():
        for batch_slice in _slices(len(data), batch_size):
            state = data.state[batch_slice].to(device)
            action = data.action[batch_slice].to(device)
            target = data.next_state[batch_slice].to(device)
            pred = model(state, action).pred
            squared_error += (pred - target).square().sum().item()
            element_count += target.numel()
    return squared_error / element_count


def rollout_mse(
    model: nn.Module,
    data: RolloutBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    """Evaluate autoregressive state MSE over every step of each rollout."""

    model.eval()
    squared_error = 0.0
    element_count = 0
    with torch.inference_mode():
        for batch_slice in _slices(len(data), batch_size):
            state = data.initial_state[batch_slice].to(device)
            actions = data.actions[batch_slice].to(device)
            targets = data.target_states[batch_slice].to(device)
            for time_index in range(actions.shape[1]):
                state = model(state, actions[:, time_index]).pred
                target = targets[:, time_index]
                squared_error += (state - target).square().sum().item()
                element_count += target.numel()
    return squared_error / element_count


def expert_jacobian_metrics(
    model: nn.Module,
    data: TransitionBatch,
    *,
    device: torch.device,
    batch_size: int,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Measure redundancy, routing, subspace coverage, and auxiliary matching."""

    model.eval()
    pair_indices = ((0, 1), (0, 2), (1, 2))
    pair_abs_cosine_sums = torch.zeros(len(pair_indices), dtype=torch.float64)
    routing_sums = torch.zeros(NUM_EXPERTS, dtype=torch.float64)
    true_routing_sums = torch.zeros(NUM_EXPERTS, dtype=torch.float64)
    raw_routing_squared_error = torch.zeros((), dtype=torch.float64)
    routing_pair_squared_error = torch.zeros(
        (NUM_EXPERTS, NUM_EXPERTS), dtype=torch.float64
    )
    sample_routing_entropy_sum = torch.zeros((), dtype=torch.float64)
    principal_angle_cosine_sums = torch.zeros(NUM_EXPERTS, dtype=torch.float64)
    projection_error_sum = torch.zeros((), dtype=torch.float64)
    recovery_abs_cosine_sum = torch.zeros(
        (NUM_EXPERTS, NUM_EXPERTS), dtype=torch.float64
    )
    sample_count = 0

    for batch_slice in _slices(len(data), batch_size):
        state = data.state[batch_slice].to(device)
        action = data.action[batch_slice].to(device)
        learned_flat = expert_action_jacobians(
            model, state, action, create_graph=False
        ).flatten(start_dim=2)
        learned = F.normalize(learned_flat, p=2, dim=-1, eps=eps)

        ground_truth_flat = ground_truth_expert_action_jacobians(state).flatten(
            start_dim=2
        )
        ground_truth = F.normalize(ground_truth_flat, p=2, dim=-1, eps=eps)
        recovery_abs_cosine_sum += torch.einsum(
            "blf,bgf->blg", learned, ground_truth
        ).abs().sum(dim=0).double().cpu()

        learned64 = learned.double()
        ground_truth64 = ground_truth.double()
        learned_u, learned_s, learned_vh = torch.linalg.svd(
            learned64, full_matrices=False
        )
        del learned_u
        ground_truth_vh = torch.linalg.svd(
            ground_truth64, full_matrices=False
        ).Vh
        rank_threshold = learned_s[:, :1] * 1e-5
        learned_rank_mask = learned_s > rank_threshold
        learned_basis = learned_vh * learned_rank_mask.unsqueeze(-1)
        principal_cosines = torch.linalg.svdvals(
            learned_basis @ ground_truth_vh.transpose(-2, -1)
        ).clamp(0.0, 1.0)
        principal_angle_cosine_sums += principal_cosines.sum(dim=0).cpu()

        learned_pinv = torch.linalg.pinv(learned64, rtol=1e-5)
        reconstructed_ground_truth = (
            ground_truth64 @ learned_pinv @ learned64
        )
        projection_error = (
            (ground_truth64 - reconstructed_ground_truth).square().sum(dim=(-2, -1))
            / (ground_truth64.square().sum(dim=(-2, -1)) + eps)
        )
        projection_error_sum += projection_error.sum().cpu()

        for pair_index, (first, second) in enumerate(pair_indices):
            cosine = torch.sum(
                learned[:, first] * learned[:, second], dim=-1
            ).abs()
            pair_abs_cosine_sums[pair_index] += cosine.sum().double().cpu()

        with torch.no_grad():
            alpha = model(state, action).alpha
        if alpha is None:
            raise TypeError("expert_jacobian_metrics requires an MoE predictor")
        alpha_true = true_routing_weights(action)
        routing_sums += alpha.sum(dim=0).double().cpu()
        true_routing_sums += alpha_true.sum(dim=0).double().cpu()
        raw_routing_squared_error += (
            alpha - alpha_true
        ).square().sum().double().cpu()
        routing_pair_squared_error += (
            alpha.unsqueeze(-1) - alpha_true.unsqueeze(1)
        ).square().sum(dim=0).double().cpu()
        sample_routing_entropy_sum += (
            -(alpha * alpha.clamp_min(eps).log()).sum(dim=-1).sum().double().cpu()
        )
        sample_count += state.shape[0]

    cosine_matrix = recovery_abs_cosine_sum / sample_count
    learned_indices, ground_truth_indices = linear_sum_assignment(
        -cosine_matrix.numpy()
    )
    matched_cosines = cosine_matrix[learned_indices, ground_truth_indices]
    pair_values = pair_abs_cosine_sums / sample_count
    routing_values = routing_sums / sample_count
    true_routing_values = true_routing_sums / sample_count
    routing_pair_mse = routing_pair_squared_error / sample_count
    matched_routing_mse = routing_pair_mse[
        learned_indices, ground_truth_indices
    ].mean()
    maximum_entropy = math.log(NUM_EXPERTS)
    mean_sample_routing_entropy = (sample_routing_entropy_sum / sample_count).item()
    routing_usage_entropy = (
        -(routing_values * routing_values.clamp_min(eps).log()).sum().item()
    )
    principal_angle_cosines = principal_angle_cosine_sums / sample_count

    return {
        "expert_pair_mean_abs_cosines": pair_values.tolist(),
        "expert_redundancy_mean_abs_cosine": pair_values.mean().item(),
        "mean_routing_weights": routing_values.tolist(),
        "minimum_mean_routing_weight": routing_values.min().item(),
        "maximum_mean_routing_weight": routing_values.max().item(),
        "mean_sample_routing_entropy": mean_sample_routing_entropy,
        "normalized_mean_sample_routing_entropy": (
            mean_sample_routing_entropy / maximum_entropy
        ),
        "routing_usage_entropy": routing_usage_entropy,
        "normalized_routing_usage_entropy": routing_usage_entropy / maximum_entropy,
        "routing_effective_experts": math.exp(routing_usage_entropy),
        "true_mean_routing_weights": true_routing_values.tolist(),
        "router_weight_mse": (
            raw_routing_squared_error / (sample_count * NUM_EXPERTS)
        ).item(),
        "matched_router_weight_mse": matched_routing_mse.item(),
        "routing_pair_mse_matrix": routing_pair_mse.tolist(),
        "basis_cosine_matrix": cosine_matrix.tolist(),
        "basis_matching_learned_to_ground_truth": [
            int(index) for index in ground_truth_indices.tolist()
        ],
        "basis_matched_cosines": matched_cosines.tolist(),
        "basis_recovery_score": matched_cosines.mean().item(),
        "dynamics_subspace_principal_angle_cosines": (
            principal_angle_cosines.tolist()
        ),
        "dynamics_subspace_principal_cosine_similarity": (
            principal_angle_cosines.mean().item()
        ),
        "dynamics_subspace_projection_reconstruction_error": (
            projection_error_sum / sample_count
        ).item(),
    }


def evaluate_iid_predictor(
    model: nn.Module,
    *,
    iid_test: TransitionBatch,
    iid_rollout: RolloutBatch,
    basis_probe: TransitionBatch,
    device: torch.device,
    prediction_batch_size: int,
    jacobian_batch_size: int,
) -> dict[str, Any]:
    """Run the full-IID metrics used by the revised first experiment."""

    metrics: dict[str, Any] = {
        "iid_one_step_mse": one_step_mse(
            model, iid_test, device=device, batch_size=prediction_batch_size
        ),
        "iid_rollout_mse": rollout_mse(
            model, iid_rollout, device=device, batch_size=prediction_batch_size
        ),
    }
    with torch.no_grad():
        has_router = (
            model(
                state=basis_probe.state[:1].to(device),
                action=basis_probe.action[:1].to(device),
            ).alpha
            is not None
        )
    if has_router:
        metrics.update(
            expert_jacobian_metrics(
                model,
                basis_probe,
                device=device,
                batch_size=jacobian_batch_size,
            )
        )
    return metrics


def evaluate_predictor(
    model: nn.Module,
    *,
    iid_test: TransitionBatch,
    heldout_test: TransitionBatch,
    iid_rollout: RolloutBatch,
    heldout_rollout: RolloutBatch,
    basis_probe: TransitionBatch,
    device: torch.device,
    prediction_batch_size: int,
    jacobian_batch_size: int,
) -> dict[str, Any]:
    """Run all required predictive and expert-level evaluations."""

    metrics: dict[str, Any] = {
        "iid_one_step_mse": one_step_mse(
            model, iid_test, device=device, batch_size=prediction_batch_size
        ),
        "heldout_composition_mse": one_step_mse(
            model, heldout_test, device=device, batch_size=prediction_batch_size
        ),
        "iid_rollout_mse": rollout_mse(
            model, iid_rollout, device=device, batch_size=prediction_batch_size
        ),
        "heldout_composition_rollout_mse": rollout_mse(
            model,
            heldout_rollout,
            device=device,
            batch_size=prediction_batch_size,
        ),
    }
    with torch.no_grad():
        has_router = (
            model(
                state=basis_probe.state[:1].to(device),
                action=basis_probe.action[:1].to(device),
            ).alpha
            is not None
        )
    if has_router:
        metrics.update(
            expert_jacobian_metrics(
                model,
                basis_probe,
                device=device,
                batch_size=jacobian_batch_size,
            )
        )
    return metrics
