import torch

from jepa_moe import (
    basis_composition_target_response,
    build_experiment1_model,
    control_response_loss,
    expert_action_jacobians,
    expert_action_jvp,
    mixture_action_response,
)


def _batch(
    batch_size: int = 11, *, seed: int = 301
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(batch_size, 4, generator=generator, dtype=torch.float64)
    action = torch.randn(batch_size, 2, generator=generator, dtype=torch.float64)
    direction = torch.randn(
        batch_size, 2, generator=generator, dtype=torch.float64
    )
    return state, action, direction


def test_expert_jvp_matches_full_expert_jacobians() -> None:
    state, action, direction = _batch(seed=302)
    model = build_experiment1_model("ours", initialization_seed=303).double()

    responses = expert_action_jvp(model, state, action, direction)
    full_jacobians = expert_action_jacobians(model, state, action)
    expected = torch.einsum("bksa,ba->bks", full_jacobians, direction)

    assert responses.shape == (11, 3, 4)
    torch.testing.assert_close(responses, expected, rtol=1e-12, atol=1e-12)


def test_state_only_mixture_response_matches_both_compositions() -> None:
    state, action, direction = _batch(seed=304)
    model = build_experiment1_model("ours", initialization_seed=305).double()
    output = model(state, action)
    assert output.alpha is not None

    expert_responses = expert_action_jvp(model, state, action, direction)
    composed = torch.sum(output.alpha.unsqueeze(-1) * expert_responses, dim=1)
    mixture = mixture_action_response(model, state, action, direction)

    def prediction(candidate_action: torch.Tensor) -> torch.Tensor:
        return model(state, candidate_action).pred

    _, direct_predictor_jvp = torch.func.jvp(
        prediction, (action,), (direction,)
    )
    assert mixture.shape == (11, 4)
    torch.testing.assert_close(mixture, composed, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        mixture, direct_predictor_jvp, rtol=1e-12, atol=1e-12
    )


def test_target_response_has_expected_shape_and_no_gradient_graph() -> None:
    state, action, direction = _batch(seed=306)
    state.requires_grad_(True)
    action.requires_grad_(True)
    direction.requires_grad_(True)

    target = basis_composition_target_response(state, action, direction)

    assert target.shape == (11, 4)
    assert not target.requires_grad
    assert target.grad_fn is None
    assert torch.isfinite(target).all()


def test_control_response_loss_matches_manual_batch_reduction() -> None:
    predicted = torch.tensor([[1.0, 2.0], [4.0, -1.0]])
    target = torch.tensor([[0.0, 4.0], [1.0, 1.0]])
    expected = torch.tensor((1.0**2 + (-2.0) ** 2 + 3.0**2 + (-2.0) ** 2) / 2)

    torch.testing.assert_close(control_response_loss(predicted, target), expected)


def test_control_response_loss_backpropagates_to_experts_and_router() -> None:
    state, action, direction = _batch(seed=307)
    model = build_experiment1_model("ours", initialization_seed=308).double()
    predicted = mixture_action_response(model, state, action, direction)
    target = basis_composition_target_response(state, action, direction)

    loss = control_response_loss(predicted, target)
    loss.backward()

    for expert in model.experts:
        gradients = [
            parameter.grad
            for parameter in expert.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)

    router_gradients = [parameter.grad for parameter in model.router.parameters()]
    assert all(gradient is not None for gradient in router_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in router_gradients
        if gradient is not None
    )
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in router_gradients
        if gradient is not None
    )
    assert not target.requires_grad


def test_control_response_loss_is_zero_for_identical_responses() -> None:
    response = torch.randn(7, 4)
    loss = control_response_loss(response, response.clone())
    torch.testing.assert_close(loss, torch.zeros(()), rtol=0.0, atol=0.0)
