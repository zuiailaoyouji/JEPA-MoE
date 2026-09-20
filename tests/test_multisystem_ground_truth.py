import inspect

import pytest
import torch

from jepa_moe import (
    NUM_SYSTEMS,
    ground_truth_expert_action_jacobians,
    ground_truth_expert_outputs,
    system_composition_weights,
    system_true_action_jacobian,
    system_true_transition,
)


def _sample_state_action(
    batch_size: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state = torch.rand(batch_size, 4, generator=generator, dtype=torch.float64) - 0.5
    action = (
        torch.rand(batch_size, 2, generator=generator, dtype=torch.float64) * 2.0
        - 1.0
    )
    return state, action


def _autograd_action_jacobian(
    state: torch.Tensor, action: torch.Tensor, system_id: int
) -> torch.Tensor:
    differentiable_action = action.detach().requires_grad_(True)
    output = system_true_transition(state, differentiable_action, system_id)
    rows = [
        torch.autograd.grad(
            output[:, index].sum(), differentiable_action, retain_graph=True
        )[0]
        for index in range(output.shape[-1])
    ]
    return torch.stack(rows, dim=1)


@pytest.mark.parametrize("system_id", range(NUM_SYSTEMS))
def test_each_system_forwards_with_normalized_finite_state_only_weights(
    system_id: int,
) -> None:
    state, action = _sample_state_action(64, seed=200)
    alpha = system_composition_weights(state, system_id)
    next_state = system_true_transition(state, action, system_id)

    assert tuple(inspect.signature(system_composition_weights).parameters) == (
        "state",
        "system_id",
    )
    assert alpha.shape == (64, 3)
    assert next_state.shape == (64, 4)
    assert torch.all(alpha >= 0)
    torch.testing.assert_close(alpha.sum(dim=-1), torch.ones(64, dtype=alpha.dtype))
    assert torch.isfinite(alpha).all()
    assert torch.isfinite(next_state).all()


def test_system_weights_are_action_independent_and_different() -> None:
    state, _ = _sample_state_action(128, seed=201)
    assert "action" not in inspect.signature(system_composition_weights).parameters

    all_alpha = [
        system_composition_weights(state, system_id)
        for system_id in range(NUM_SYSTEMS)
    ]

    for system_id, alpha in enumerate(all_alpha):
        torch.testing.assert_close(
            alpha, system_composition_weights(state.clone(), system_id)
        )

    for first in range(NUM_SYSTEMS):
        for second in range(first + 1, NUM_SYSTEMS):
            assert not torch.allclose(all_alpha[first], all_alpha[second])


@pytest.mark.parametrize("system_id", range(NUM_SYSTEMS))
def test_all_systems_reuse_exactly_the_same_ground_truth_bases(
    system_id: int,
) -> None:
    state, action = _sample_state_action(31, seed=203)
    shared_bases = ground_truth_expert_outputs(state, action)
    alpha = system_composition_weights(state, system_id)
    expected = torch.sum(alpha.unsqueeze(-1) * shared_bases, dim=1)

    torch.testing.assert_close(
        system_true_transition(state, action, system_id), expected
    )


@pytest.mark.parametrize("system_id", range(NUM_SYSTEMS))
def test_each_system_action_jacobian_matches_shared_basis_composition(
    system_id: int,
) -> None:
    state, action = _sample_state_action(23, seed=204)
    expert_jacobians = ground_truth_expert_action_jacobians(state)
    expected = torch.einsum(
        "bk,bksa->bsa",
        system_composition_weights(state, system_id),
        expert_jacobians,
    )
    analytic = system_true_action_jacobian(state, action, system_id)
    direct = _autograd_action_jacobian(state, action, system_id)

    assert analytic.shape == (23, 4, 2)
    assert torch.isfinite(analytic).all()
    torch.testing.assert_close(analytic, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(direct, expected, rtol=1e-12, atol=1e-12)


def test_system_dynamics_are_reproducible_and_do_not_collapse_by_definition() -> None:
    first_state, first_action = _sample_state_action(4_096, seed=205)
    second_state, second_action = _sample_state_action(4_096, seed=205)

    torch.testing.assert_close(first_state, second_state)
    torch.testing.assert_close(first_action, second_action)
    for system_id in range(NUM_SYSTEMS):
        first_alpha = system_composition_weights(first_state, system_id)
        second_alpha = system_composition_weights(second_state, system_id)
        first_transition = system_true_transition(
            first_state, first_action, system_id
        )
        second_transition = system_true_transition(
            second_state, second_action, system_id
        )

        torch.testing.assert_close(first_alpha, second_alpha)
        torch.testing.assert_close(first_transition, second_transition)
        assert torch.all(first_alpha.mean(dim=0) > 0.2)
        assert torch.all(first_alpha.max(dim=-1).values < 0.95)


@pytest.mark.parametrize("system_id", [-1, NUM_SYSTEMS])
def test_invalid_system_id_is_rejected(system_id: int) -> None:
    state, action = _sample_state_action(2, seed=206)

    with pytest.raises(ValueError, match="system_id"):
        system_composition_weights(state, system_id)
    with pytest.raises(ValueError, match="system_id"):
        system_true_transition(state, action, system_id)
