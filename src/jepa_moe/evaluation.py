"""Evaluation metrics for synthetic basis recovery and unseen composition."""

from __future__ import annotations

from collections.abc import Iterator
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
    """Measure redundancy, routing, and permutation-invariant basis recovery."""

    model.eval()
    pair_indices = ((0, 1), (0, 2), (1, 2))
    pair_abs_cosine_sums = torch.zeros(len(pair_indices), dtype=torch.float64)
    routing_sums = torch.zeros(NUM_EXPERTS, dtype=torch.float64)
    recovery_abs_cosine_sum = torch.zeros(
        (NUM_EXPERTS, NUM_EXPERTS), dtype=torch.float64
    )
    sample_count = 0

    for batch_slice in _slices(len(data), batch_size):
        state = data.state[batch_slice].to(device)
        action = data.action[batch_slice].to(device)
        learned = expert_action_jacobians(
            model, state, action, create_graph=False
        ).flatten(start_dim=2)
        learned = F.normalize(learned, p=2, dim=-1, eps=eps)

        ground_truth = ground_truth_expert_action_jacobians(state).flatten(
            start_dim=2
        )
        ground_truth = F.normalize(ground_truth, p=2, dim=-1, eps=eps)
        recovery_abs_cosine_sum += torch.einsum(
            "blf,bgf->blg", learned, ground_truth
        ).abs().sum(dim=0).double().cpu()

        for pair_index, (first, second) in enumerate(pair_indices):
            cosine = torch.sum(
                learned[:, first] * learned[:, second], dim=-1
            ).abs()
            pair_abs_cosine_sums[pair_index] += cosine.sum().double().cpu()

        with torch.no_grad():
            alpha = model(state, action).alpha
        if alpha is None:
            raise TypeError("expert_jacobian_metrics requires an MoE predictor")
        routing_sums += alpha.sum(dim=0).double().cpu()
        sample_count += state.shape[0]

    cosine_matrix = recovery_abs_cosine_sum / sample_count
    learned_indices, ground_truth_indices = linear_sum_assignment(
        -cosine_matrix.numpy()
    )
    matched_cosines = cosine_matrix[learned_indices, ground_truth_indices]
    pair_values = pair_abs_cosine_sums / sample_count
    routing_values = routing_sums / sample_count

    return {
        "expert_pair_mean_abs_cosines": pair_values.tolist(),
        "expert_redundancy_mean_abs_cosine": pair_values.mean().item(),
        "mean_routing_weights": routing_values.tolist(),
        "basis_cosine_matrix": cosine_matrix.tolist(),
        "basis_matching_learned_to_ground_truth": [
            int(index) for index in ground_truth_indices.tolist()
        ],
        "basis_matched_cosines": matched_cosines.tolist(),
        "basis_recovery_score": matched_cosines.mean().item(),
    }


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
