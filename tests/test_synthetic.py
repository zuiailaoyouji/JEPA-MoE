import torch

from jepa_moe import (
    ground_truth_expert_action_jacobians,
    ground_truth_expert_outputs,
    make_rollout_batch,
    make_transition_batch,
    synthetic_transition,
    true_routing_weights,
)
from jepa_moe.synthetic import action_is_heldout


def test_direct_next_state_experts_and_transition_match_definition() -> None:
    state = torch.tensor([[0.2, -0.3, 0.4, -0.1]], dtype=torch.float64)
    action = torch.tensor([[0.6, -0.8]], dtype=torch.float64)
    x0, x1, x2, x3 = state[0]
    a0, a1 = action[0]
    expected_expert_1 = torch.stack(
        (0.9 * x0 + 0.20 * a0, 0.8 * x1 + 0.20 * a1, 0.7 * x2, 0.7 * x3)
    )
    expected_expert_2 = torch.stack(
        (
            0.7 * x0,
            0.7 * x1,
            0.85 * x2 + 0.20 * (1.0 + 0.3 * torch.tanh(x0)) * a0,
            0.85 * x3 + 0.20 * (1.0 + 0.3 * torch.tanh(x1)) * a1,
        )
    )
    expected_expert_3 = torch.stack(
        (
            0.8 * x0 + 0.15 * a1,
            0.8 * x1 - 0.15 * a0,
            0.8 * x2 + 0.10 * (1.0 + 0.3 * torch.tanh(x2)) * a1,
            0.8 * x3 - 0.10 * (1.0 + 0.3 * torch.tanh(x3)) * a0,
        )
    )
    expert_outputs = ground_truth_expert_outputs(state, action)
    torch.testing.assert_close(
        expert_outputs[0],
        torch.stack((expected_expert_1, expected_expert_2, expected_expert_3)),
    )

    alpha = true_routing_weights(action)
    expected_alpha = torch.softmax(
        torch.tensor([[1.2, -1.6, 0.4]], dtype=torch.float64), dim=-1
    )
    torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(
        synthetic_transition(state, action),
        torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1),
    )


def test_analytic_ground_truth_jacobians_match_autograd() -> None:
    torch.manual_seed(7)
    state = torch.rand(5, 4, dtype=torch.float64) - 0.5
    action = (torch.rand(5, 2, dtype=torch.float64) * 2.0 - 1.0).requires_grad_()
    outputs = ground_truth_expert_outputs(state, action)
    autograd_jacobians = []
    for expert_index in range(3):
        output_gradients = []
        for state_index in range(4):
            output_gradients.append(
                torch.autograd.grad(
                    outputs[:, expert_index, state_index].sum(),
                    action,
                    retain_graph=True,
                )[0]
            )
        autograd_jacobians.append(torch.stack(output_gradients, dim=1))
    autograd_jacobians = torch.stack(autograd_jacobians, dim=1)

    torch.testing.assert_close(
        ground_truth_expert_action_jacobians(state), autograd_jacobians
    )


def test_action_regions_are_disjoint_and_deterministic() -> None:
    iid_first = make_transition_batch(2_000, "iid", seed=31)
    iid_second = make_transition_batch(2_000, "iid", seed=31)
    heldout = make_transition_batch(2_000, "heldout", seed=32)

    assert not action_is_heldout(iid_first.action).any()
    assert action_is_heldout(heldout.action).all()
    torch.testing.assert_close(iid_first.state, iid_second.state)
    torch.testing.assert_close(iid_first.action, iid_second.action)
    torch.testing.assert_close(iid_first.next_state, iid_second.next_state)


def test_full_iid_actions_cover_center_and_outer_regions() -> None:
    first = make_transition_batch(10_000, "full", seed=33)
    second = make_transition_batch(10_000, "full", seed=33)
    center = action_is_heldout(first.action)

    assert center.any()
    assert (~center).any()
    assert torch.all(first.action >= -1.0)
    assert torch.all(first.action <= 1.0)
    torch.testing.assert_close(first.state, second.state)
    torch.testing.assert_close(first.action, second.action)
    torch.testing.assert_close(first.next_state, second.next_state)


def test_rollout_targets_apply_transition_recurrently() -> None:
    rollout = make_rollout_batch(16, 25, "heldout", seed=41)
    state = rollout.initial_state
    for time_index in range(25):
        assert action_is_heldout(rollout.actions[:, time_index]).all()
        state = synthetic_transition(state, rollout.actions[:, time_index])
        torch.testing.assert_close(state, rollout.target_states[:, time_index])
