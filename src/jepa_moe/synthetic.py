"""Deterministic synthetic next-state experts and controlled data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .models import ACTION_DIM, HISTORY_LENGTH, NUM_EXPERTS, STATE_DIM


HELDOUT_ACTION_BOUND = 0.35
NUM_SYSTEMS = 4
ActionRegion = Literal["full", "iid", "heldout"]


@dataclass(frozen=True)
class TransitionBatch:
    state: Tensor
    action: Tensor
    next_state: Tensor

    def __len__(self) -> int:
        return self.state.shape[0]


@dataclass(frozen=True)
class RolloutBatch:
    initial_state: Tensor
    actions: Tensor
    target_states: Tensor

    def __len__(self) -> int:
        return self.initial_state.shape[0]


@dataclass(frozen=True)
class HistoryTransitionBatch:
    state_history: Tensor
    past_actions: Tensor
    current_action: Tensor
    next_state: Tensor
    system_id: Tensor

    def __len__(self) -> int:
        return self.state_history.shape[0]


def _validate_state_action(state: Tensor, action: Tensor) -> None:
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have shape [batch, {STATE_DIM}]")
    if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must have shape [batch, {ACTION_DIM}]")
    if state.shape[0] != action.shape[0]:
        raise ValueError("state and action batch sizes must match")
    if state.device != action.device or state.dtype != action.dtype:
        raise ValueError("state and action must share device and dtype")


def true_routing_weights(action: Tensor) -> Tensor:
    """Return legacy action-conditioned ground-truth weights [batch, 3]."""

    if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must have shape [batch, {ACTION_DIM}]")
    a0, a1 = action.unbind(dim=-1)
    logits = torch.stack((2.0 * a0, 2.0 * a1, -2.0 * (a0 + a1)), dim=-1)
    return torch.softmax(logits, dim=-1)


def basis_composition_weights(state: Tensor) -> Tensor:
    """Return Experiment 1 state-conditioned mixture weights [batch, 3]."""

    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have shape [batch, {STATE_DIM}]")
    x0, x1 = state[:, 0], state[:, 1]
    logits = torch.stack((2.0 * x0, 2.0 * x1, -2.0 * (x0 + x1)), dim=-1)
    return torch.softmax(logits, dim=-1)


def system_composition_weights(state: Tensor, system_id: int) -> Tensor:
    """Return Experiment 2 state-conditioned weights for one synthetic system."""

    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have shape [batch, {STATE_DIM}]")
    if isinstance(system_id, bool) or not isinstance(system_id, int):
        raise TypeError("system_id must be an integer")
    if not 0 <= system_id < NUM_SYSTEMS:
        raise ValueError(f"system_id must be in [0, {NUM_SYSTEMS})")

    x0, x1, x2, x3 = state.unbind(dim=-1)
    if system_id == 0:
        logits = torch.stack((2.0 * x0, 2.0 * x1, -2.0 * (x0 + x1)), dim=-1)
    elif system_id == 1:
        logits = torch.stack((2.0 * x2, 2.0 * x3, -2.0 * (x2 + x3)), dim=-1)
    elif system_id == 2:
        logits = torch.stack((-2.0 * x0, 2.0 * x1, 2.0 * (x0 - x1)), dim=-1)
    else:
        logits = torch.stack((2.0 * x2, -2.0 * x3, -2.0 * (x2 - x3)), dim=-1)
    return torch.softmax(logits, dim=-1)


def ground_truth_expert_outputs(state: Tensor, action: Tensor) -> Tensor:
    """Return the three direct next-state experts with shape [batch, 3, 4]."""

    _validate_state_action(state, action)
    a0, a1 = action.unbind(dim=-1)
    x0, x1, x2, x3 = state.unbind(dim=-1)

    expert_1 = torch.stack(
        (0.9 * x0 + 0.20 * a0, 0.8 * x1 + 0.20 * a1, 0.7 * x2, 0.7 * x3),
        dim=-1,
    )
    expert_2 = torch.stack(
        (
            0.7 * x0,
            0.7 * x1,
            0.85 * x2 + 0.20 * (1.0 + 0.3 * torch.tanh(x0)) * a0,
            0.85 * x3 + 0.20 * (1.0 + 0.3 * torch.tanh(x1)) * a1,
        ),
        dim=-1,
    )
    expert_3 = torch.stack(
        (
            0.8 * x0 + 0.15 * a1,
            0.8 * x1 - 0.15 * a0,
            0.8 * x2 + 0.10 * (1.0 + 0.3 * torch.tanh(x2)) * a1,
            0.8 * x3 - 0.10 * (1.0 + 0.3 * torch.tanh(x3)) * a0,
        ),
        dim=-1,
    )
    return torch.stack((expert_1, expert_2, expert_3), dim=1)


def synthetic_transition(state: Tensor, action: Tensor) -> Tensor:
    """Apply the legacy deterministic action-conditioned mixture transition."""

    experts = ground_truth_expert_outputs(state, action)
    alpha = true_routing_weights(action)
    return torch.sum(alpha.unsqueeze(-1) * experts, dim=1)


def basis_composition_transition(state: Tensor, action: Tensor) -> Tensor:
    """Apply the Experiment 1 deterministic state-conditioned transition."""

    experts = ground_truth_expert_outputs(state, action)
    alpha = basis_composition_weights(state)
    return torch.sum(alpha.unsqueeze(-1) * experts, dim=1)


def system_true_transition(state: Tensor, action: Tensor, system_id: int) -> Tensor:
    """Apply one Experiment 2 system using the shared ground-truth bases."""

    experts = ground_truth_expert_outputs(state, action)
    alpha = system_composition_weights(state, system_id)
    return torch.sum(alpha.unsqueeze(-1) * experts, dim=1)


def _batched_system_true_transition(
    state: Tensor, action: Tensor, system_id: Tensor
) -> Tensor:
    """Apply per-sample synthetic systems without exposing IDs to a learned model."""

    _validate_state_action(state, action)
    if system_id.ndim != 1 or system_id.shape[0] != state.shape[0]:
        raise ValueError("system_id must have shape [batch]")
    if system_id.device != state.device:
        raise ValueError("system_id must be on the same device as state")
    if torch.any((system_id < 0) | (system_id >= NUM_SYSTEMS)):
        raise ValueError(f"system_id values must be in [0, {NUM_SYSTEMS})")

    next_state = torch.empty_like(state)
    for selected_system in range(NUM_SYSTEMS):
        mask = system_id == selected_system
        if torch.any(mask):
            next_state[mask] = system_true_transition(
                state[mask], action[mask], selected_system
            )
    return next_state


def ground_truth_expert_action_jacobians(state: Tensor) -> Tensor:
    """Analytic d F_gt_k / d a with shape [batch, 3, 4, 2]."""

    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have shape [batch, {STATE_DIM}]")

    batch_size = state.shape[0]
    result = state.new_zeros((batch_size, NUM_EXPERTS, STATE_DIM, ACTION_DIM))
    result[:, 0, 0, 0] = 0.20
    result[:, 0, 1, 1] = 0.20

    result[:, 1, 2, 0] = 0.20 * (1.0 + 0.3 * torch.tanh(state[:, 0]))
    result[:, 1, 3, 1] = 0.20 * (1.0 + 0.3 * torch.tanh(state[:, 1]))

    result[:, 2, 0, 1] = 0.15
    result[:, 2, 1, 0] = -0.15
    result[:, 2, 2, 1] = (
        0.10 * (1.0 + 0.3 * torch.tanh(state[:, 2]))
    )
    result[:, 2, 3, 0] = (
        -0.10 * (1.0 + 0.3 * torch.tanh(state[:, 3]))
    )
    return result


def basis_composition_action_jacobian(state: Tensor, action: Tensor) -> Tensor:
    """Return d F_true(x, a) / d a for the state-conditioned environment."""

    _validate_state_action(state, action)
    expert_jacobians = ground_truth_expert_action_jacobians(state)
    alpha = basis_composition_weights(state)
    return torch.einsum("bk,bksa->bsa", alpha, expert_jacobians)


def system_true_action_jacobian(
    state: Tensor, action: Tensor, system_id: int
) -> Tensor:
    """Return d F_true^(system_id)(x, a) / d a with shape [batch, 4, 2]."""

    _validate_state_action(state, action)
    expert_jacobians = ground_truth_expert_action_jacobians(state)
    alpha = system_composition_weights(state, system_id)
    return torch.einsum("bk,bksa->bsa", alpha, expert_jacobians)


def action_is_heldout(action: Tensor) -> Tensor:
    """Return the per-sample center-region membership mask."""

    return torch.all(action.abs() < HELDOUT_ACTION_BOUND, dim=-1)


def sample_actions(
    sample_count: int,
    region: ActionRegion,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample full IID actions or either legacy diagnostic action region."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if region == "full":
        return (
            torch.rand(
                (sample_count, ACTION_DIM), generator=generator, dtype=dtype
            )
            * 2.0
            - 1.0
        )
    if region == "heldout":
        inner_bound = HELDOUT_ACTION_BOUND - torch.finfo(dtype).eps
        return (
            torch.rand((sample_count, ACTION_DIM), generator=generator, dtype=dtype)
            * (2.0 * inner_bound)
            - inner_bound
        )
    if region != "iid":
        raise ValueError(f"unknown action region: {region}")

    accepted: list[Tensor] = []
    remaining = sample_count
    while remaining > 0:
        candidates = (
            torch.rand(
                (max(remaining * 2, 32), ACTION_DIM),
                generator=generator,
                dtype=dtype,
            )
            * 2.0
            - 1.0
        )
        outside = candidates[~action_is_heldout(candidates)]
        selected = outside[:remaining]
        accepted.append(selected)
        remaining -= selected.shape[0]
    return torch.cat(accepted, dim=0)


def make_history_transition_batch(
    sample_count: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> HistoryTransitionBatch:
    """Generate H=4 continuous histories from the Experiment 2 systems."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    system_id = torch.randint(
        NUM_SYSTEMS, (sample_count,), generator=generator, dtype=torch.long
    )
    initial_state = torch.rand(
        (sample_count, STATE_DIM), generator=generator, dtype=dtype
    ) - 0.5
    past_actions = sample_actions(
        sample_count * HISTORY_LENGTH,
        "full",
        generator=generator,
        dtype=dtype,
    ).reshape(sample_count, HISTORY_LENGTH, ACTION_DIM)
    current_action = sample_actions(
        sample_count,
        "full",
        generator=generator,
        dtype=dtype,
    )

    states = [initial_state]
    state = initial_state
    for history_index in range(HISTORY_LENGTH):
        state = _batched_system_true_transition(
            state, past_actions[:, history_index], system_id
        )
        states.append(state)

    next_state = _batched_system_true_transition(
        state, current_action, system_id
    )
    return HistoryTransitionBatch(
        state_history=torch.stack(states, dim=1),
        past_actions=past_actions,
        current_action=current_action,
        next_state=next_state,
        system_id=system_id,
    )


def make_transition_batch(
    sample_count: int,
    region: ActionRegion,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> TransitionBatch:
    """Generate a deterministic one-step split from a local random generator."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.rand(
        (sample_count, STATE_DIM), generator=generator, dtype=dtype
    ) - 0.5
    action = sample_actions(
        sample_count, region, generator=generator, dtype=dtype
    )
    return TransitionBatch(state, action, synthetic_transition(state, action))


def make_basis_composition_transition_batch(
    sample_count: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> TransitionBatch:
    """Generate a reproducible full-IID split for the new Experiment 1."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.rand(
        (sample_count, STATE_DIM), generator=generator, dtype=dtype
    ) - 0.5
    action = sample_actions(
        sample_count, "full", generator=generator, dtype=dtype
    )
    return TransitionBatch(
        state,
        action,
        basis_composition_transition(state, action),
    )


def make_rollout_batch(
    trajectory_count: int,
    horizon: int,
    region: ActionRegion,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> RolloutBatch:
    """Generate deterministic ground-truth trajectories under fixed action sequences."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_state = torch.rand(
        (trajectory_count, STATE_DIM), generator=generator, dtype=dtype
    ) - 0.5
    actions = sample_actions(
        trajectory_count * horizon,
        region,
        generator=generator,
        dtype=dtype,
    ).reshape(trajectory_count, horizon, ACTION_DIM)

    state = initial_state
    target_states = []
    for time_index in range(horizon):
        state = synthetic_transition(state, actions[:, time_index])
        target_states.append(state)
    return RolloutBatch(
        initial_state=initial_state,
        actions=actions,
        target_states=torch.stack(target_states, dim=1),
    )


def make_basis_composition_rollout_batch(
    trajectory_count: int,
    horizon: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> RolloutBatch:
    """Generate deterministic full-IID rollouts for the new Experiment 1."""

    if trajectory_count <= 0 or horizon <= 0:
        raise ValueError("trajectory_count and horizon must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_state = torch.rand(
        (trajectory_count, STATE_DIM), generator=generator, dtype=dtype
    ) - 0.5
    actions = sample_actions(
        trajectory_count * horizon,
        "full",
        generator=generator,
        dtype=dtype,
    ).reshape(trajectory_count, horizon, ACTION_DIM)

    state = initial_state
    target_states = []
    for time_index in range(horizon):
        state = basis_composition_transition(state, actions[:, time_index])
        target_states.append(state)
    return RolloutBatch(
        initial_state=initial_state,
        actions=actions,
        target_states=torch.stack(target_states, dim=1),
    )
