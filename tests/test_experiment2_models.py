import inspect

import pytest
import torch
from torch import nn

from jepa_moe import (
    HISTORY_CONTEXT_DIM,
    HISTORY_LENGTH,
    HistoryConditionedMoEPredictor,
    HistoryContextEncoder,
    HistoryRouter,
    build_experiment2_model,
    flatten_history,
    make_history_transition_batch,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = make_history_transition_batch(17, seed=401)
    return batch.state_history, batch.past_actions, batch.current_action


def _linear_shapes(module: nn.Module) -> list[tuple[int, int]]:
    return [
        (layer.in_features, layer.out_features)
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def test_context_encoder_and_router_architectures_are_exact() -> None:
    encoder = HistoryContextEncoder()
    router = HistoryRouter()

    assert HISTORY_LENGTH == 4
    assert HISTORY_CONTEXT_DIM == 64
    assert encoder.input_dim == 28
    assert _linear_shapes(encoder) == [(28, 128), (128, 128), (128, 64)]
    assert _linear_shapes(router) == [(64, 128), (128, 128), (128, 3)]
    assert sum(isinstance(layer, nn.SiLU) for layer in encoder.modules()) == 2
    assert sum(isinstance(layer, nn.SiLU) for layer in router.modules()) == 2


def test_experts_reuse_the_experiment1_architecture() -> None:
    model = build_experiment2_model(initialization_seed=400)

    assert len(model.experts) == 3
    for expert in model.experts:
        assert _linear_shapes(expert) == [(6, 128), (128, 128), (128, 4)]
        assert sum(isinstance(layer, nn.SiLU) for layer in expert.modules()) == 2


def test_history_flattening_uses_chronological_interleaving() -> None:
    states = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    actions = (100 + torch.arange(8, dtype=torch.float32)).reshape(1, 4, 2)
    expected = torch.cat(
        [
            states[:, index]
            if part == "state"
            else actions[:, index]
            for index in range(4)
            for part in ("state", "action")
        ]
        + [states[:, -1]],
        dim=-1,
    )

    flattened = flatten_history(states, actions)
    assert flattened.shape == (1, 28)
    torch.testing.assert_close(flattened, expected)


def test_history_predictor_outputs_have_expected_shapes_and_values() -> None:
    state_history, past_actions, current_action = _batch()
    model = build_experiment2_model(initialization_seed=402)
    context, routed_alpha = model.routing_weights(state_history, past_actions)
    output = model(state_history, past_actions, current_action)

    assert context.shape == (17, 64)
    assert output.pred.shape == (17, 4)
    assert output.alpha is not None
    assert output.expert_outputs is not None
    assert output.alpha.shape == (17, 3)
    assert output.expert_outputs.shape == (17, 3, 4)
    assert torch.all(output.alpha >= 0)
    torch.testing.assert_close(output.alpha.sum(dim=-1), torch.ones(17))
    torch.testing.assert_close(output.alpha, routed_alpha)
    torch.testing.assert_close(
        output.pred,
        torch.sum(output.alpha.unsqueeze(-1) * output.expert_outputs, dim=1),
    )
    assert all(torch.isfinite(value).all() for value in output)


def test_current_action_cannot_enter_context_or_router() -> None:
    state_history, past_actions, current_action = _batch()
    model = build_experiment2_model(initialization_seed=403)
    changed_action = current_action + 13.0

    first_context, first_alpha = model.routing_weights(state_history, past_actions)
    first_output = model(state_history, past_actions, current_action)
    second_context, second_alpha = model.routing_weights(state_history, past_actions)
    second_output = model(state_history, past_actions, changed_action)

    assert tuple(inspect.signature(model.context_encoder.forward).parameters) == (
        "state_history",
        "past_actions",
    )
    assert tuple(inspect.signature(model.router.forward).parameters) == ("context",)
    assert tuple(inspect.signature(model.routing_weights).parameters) == (
        "state_history",
        "past_actions",
    )
    torch.testing.assert_close(first_context, second_context, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_alpha, second_alpha, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first_output.alpha, second_output.alpha, rtol=0.0, atol=0.0
    )

    differentiable_action = current_action.clone().requires_grad_(True)
    alpha = model(
        state_history, past_actions, differentiable_action
    ).alpha
    assert alpha is not None
    action_gradient = torch.autograd.grad(
        alpha[:, 0].sum(), differentiable_action, allow_unused=True
    )[0]
    assert action_gradient is None


def test_predictor_action_jacobian_is_weighted_expert_jacobian() -> None:
    state_history, past_actions, current_action = _batch()
    current_action = current_action.to(torch.float64).requires_grad_(True)
    state_history = state_history.to(torch.float64)
    past_actions = past_actions.to(torch.float64)
    model = build_experiment2_model(initialization_seed=407).to(torch.float64)
    output = model(state_history, past_actions, current_action)
    assert output.alpha is not None and output.expert_outputs is not None

    direct = torch.stack(
        [
            torch.autograd.grad(
                output.pred[:, state_index].sum(),
                current_action,
                retain_graph=True,
            )[0]
            for state_index in range(4)
        ],
        dim=1,
    )
    expert_jacobians = []
    for expert_index in range(3):
        expert_jacobians.append(
            torch.stack(
                [
                    torch.autograd.grad(
                        output.expert_outputs[:, expert_index, state_index].sum(),
                        current_action,
                        retain_graph=True,
                    )[0]
                    for state_index in range(4)
                ],
                dim=1,
            )
        )
    expert_jacobians = torch.stack(expert_jacobians, dim=1)
    expected = torch.einsum("bk,bksa->bsa", output.alpha, expert_jacobians)

    torch.testing.assert_close(direct, expected, rtol=1e-12, atol=1e-12)


def test_history_tensors_enter_context_computation_graph() -> None:
    state_history, past_actions, _ = _batch()
    state_history = state_history.requires_grad_(True)
    past_actions = past_actions.requires_grad_(True)
    encoder = HistoryContextEncoder()
    context = encoder(state_history, past_actions)
    state_gradient, action_gradient = torch.autograd.grad(
        context.square().mean(), (state_history, past_actions)
    )

    assert torch.count_nonzero(state_gradient).item() > 0
    assert torch.count_nonzero(action_gradient).item() > 0
    assert torch.isfinite(state_gradient).all()
    assert torch.isfinite(action_gradient).all()


def test_seeded_model_initialization_is_exactly_reproducible() -> None:
    first = build_experiment2_model(initialization_seed=404)
    second = build_experiment2_model(initialization_seed=404)
    different = build_experiment2_model(initialization_seed=405)

    assert isinstance(first, HistoryConditionedMoEPredictor)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(
            first_parameter, second_parameter, rtol=0.0, atol=0.0
        )
    assert any(
        not torch.equal(first_parameter, different_parameter)
        for first_parameter, different_parameter in zip(
            first.parameters(), different.parameters(), strict=True
        )
    )


def test_history_predictor_rejects_invalid_inputs_and_has_no_system_id() -> None:
    state_history, past_actions, current_action = _batch()
    model = build_experiment2_model(initialization_seed=406)

    assert "system_id" not in inspect.signature(model.forward).parameters
    with pytest.raises(ValueError, match="state_history"):
        model(state_history[:, :-1], past_actions, current_action)
    with pytest.raises(ValueError, match="past_actions"):
        model(state_history, past_actions[:, :-1], current_action)
