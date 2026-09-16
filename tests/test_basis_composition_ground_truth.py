import inspect

import torch

from jepa_moe import (
    basis_composition_action_jacobian,
    basis_composition_transition,
    basis_composition_weights,
    ground_truth_expert_action_jacobians,
    ground_truth_expert_outputs,
    make_basis_composition_transition_batch,
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
    state: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    differentiable_action = action.detach().requires_grad_(True)
    output = basis_composition_transition(state, differentiable_action)
    rows = [
        torch.autograd.grad(
            output[:, index].sum(), differentiable_action, retain_graph=True
        )[0]
        for index in range(output.shape[-1])
    ]
    return torch.stack(rows, dim=1)


def test_basis_composition_weights_are_state_only_and_normalized() -> None:
    state, _ = _sample_state_action(32, seed=101)
    alpha = basis_composition_weights(state)

    assert tuple(inspect.signature(basis_composition_weights).parameters) == ("state",)
    assert alpha.shape == (32, 3)
    assert torch.all(alpha >= 0)
    torch.testing.assert_close(alpha.sum(dim=-1), torch.ones(32, dtype=alpha.dtype))
    torch.testing.assert_close(alpha, basis_composition_weights(state.clone()))


def test_state_conditioned_transition_reuses_ground_truth_experts() -> None:
    state, action = _sample_state_action(17, seed=102)
    alpha = basis_composition_weights(state)
    experts = ground_truth_expert_outputs(state, action)
    expected = torch.sum(alpha.unsqueeze(-1) * experts, dim=1)

    torch.testing.assert_close(
        basis_composition_transition(state, action), expected
    )


def test_aggregate_action_jacobian_matches_autograd_and_basis_composition() -> None:
    state, action = _sample_state_action(19, seed=103)
    analytic = basis_composition_action_jacobian(state, action)
    expert_jacobians = ground_truth_expert_action_jacobians(state)
    composed = torch.einsum(
        "bk,bksa->bsa", basis_composition_weights(state), expert_jacobians
    )
    direct = _autograd_action_jacobian(state, action)

    assert analytic.shape == (19, 4, 2)
    torch.testing.assert_close(analytic, composed, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(direct, composed, rtol=1e-12, atol=1e-12)


def test_directional_response_matches_central_finite_difference() -> None:
    state, action = _sample_state_action(23, seed=104)
    generator = torch.Generator().manual_seed(105)
    direction = torch.randn(23, 2, generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    )
    epsilon = 1e-5

    jacobian_response = torch.einsum(
        "bsa,ba->bs",
        basis_composition_action_jacobian(state, action),
        direction,
    )
    finite_difference = (
        basis_composition_transition(state, action + epsilon * direction)
        - basis_composition_transition(state, action - epsilon * direction)
    ) / (2.0 * epsilon)

    torch.testing.assert_close(
        jacobian_response, finite_difference, rtol=1e-9, atol=1e-9
    )


def test_ground_truth_expert_control_responses_are_distinct() -> None:
    state, _ = _sample_state_action(29, seed=106)
    jacobians = ground_truth_expert_action_jacobians(state)

    assert jacobians.shape == (29, 3, 4, 2)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert not torch.equal(jacobians[:, first], jacobians[:, second])


def test_basis_composition_batch_is_reproducible_finite_and_well_shaped() -> None:
    first = make_basis_composition_transition_batch(2_000, seed=107)
    second = make_basis_composition_transition_batch(2_000, seed=107)

    assert first.state.shape == (2_000, 4)
    assert first.action.shape == (2_000, 2)
    assert first.next_state.shape == (2_000, 4)
    assert torch.isfinite(first.state).all()
    assert torch.isfinite(first.action).all()
    assert torch.isfinite(first.next_state).all()
    assert torch.all(first.state >= -0.5) and torch.all(first.state < 0.5)
    assert torch.all(first.action >= -1.0) and torch.all(first.action < 1.0)
    torch.testing.assert_close(first.state, second.state)
    torch.testing.assert_close(first.action, second.action)
    torch.testing.assert_close(first.next_state, second.next_state)
    torch.testing.assert_close(
        first.next_state,
        basis_composition_transition(first.state, first.action),
    )
