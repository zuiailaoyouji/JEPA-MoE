import pytest
import torch

from jepa_moe import (
    JacobianMoEPredictor,
    VanillaMoEPredictor,
    control_jacobian_decomposition,
    control_jacobian_specialization_terms,
    expert_action_jacobians,
    router_action_jacobian,
)


@pytest.mark.parametrize("model_class", [VanillaMoEPredictor, JacobianMoEPredictor])
def test_jacobian_shapes_and_decomposition(model_class: type) -> None:
    torch.manual_seed(101)
    model = model_class().double()
    state = torch.randn(5, 4, dtype=torch.float64)
    action = torch.randn(5, 2, dtype=torch.float64)

    decomposition = control_jacobian_decomposition(model, state, action)

    assert decomposition.expert_jacobians.shape == (5, 3, 4, 2)
    assert decomposition.router_jacobian.shape == (5, 3, 2)
    assert decomposition.j_within.shape == (5, 4, 2)
    assert decomposition.j_switch.shape == (5, 4, 2)
    assert decomposition.direct.shape == (5, 4, 2)
    torch.testing.assert_close(
        decomposition.direct,
        decomposition.reconstructed,
        rtol=1e-10,
        atol=1e-12,
    )


def test_standalone_jacobian_utilities_have_required_shapes() -> None:
    torch.manual_seed(102)
    model = VanillaMoEPredictor()
    state = torch.randn(4, 4)
    action = torch.randn(4, 2)

    expert_jacobians = expert_action_jacobians(model, state, action)
    router_jacobian = router_action_jacobian(model, state, action)

    assert expert_jacobians.shape == (4, 3, 4, 2)
    assert router_jacobian.shape == (4, 3, 2)


def test_specialization_formula_and_detached_router_weights() -> None:
    torch.manual_seed(103)
    model = JacobianMoEPredictor()
    state = torch.randn(6, 4)
    action = torch.randn(6, 2)

    terms = control_jacobian_specialization_terms(model, state, action)

    jacobians = expert_action_jacobians(model, state, action)
    flattened = jacobians.flatten(start_dim=2)
    norms = torch.linalg.vector_norm(flattened, ord=2, dim=-1)
    normalized = torch.nn.functional.normalize(flattened, dim=-1, eps=1e-8)
    alpha = model(state, action).alpha
    assert alpha is not None
    weighted_losses = []
    weights = []
    expected_cosines = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        cosine = (normalized[:, first] * normalized[:, second]).sum(dim=-1)
        weight = (alpha[:, first] * alpha[:, second]).detach()
        expected_cosines.append(cosine.mean())
        weighted_losses.append(weight * torch.relu(cosine.abs() - 0.3).square())
        weights.append(weight)
    expected_diversity = torch.stack(weighted_losses).sum() / (
        torch.stack(weights).sum() + 1e-8
    )
    expected_activity = torch.relu(0.05 - norms).square().mean()

    assert terms.pairwise_cosines.shape == (3,)
    assert terms.expert_jacobian_norms.shape == (3,)
    assert terms.routing_weights.shape == (3,)
    torch.testing.assert_close(
        terms.total, terms.diversity + 0.1 * terms.activity
    )
    torch.testing.assert_close(terms.diversity, expected_diversity)
    torch.testing.assert_close(terms.activity, expected_activity)
    torch.testing.assert_close(
        terms.pairwise_cosines, torch.stack(expected_cosines)
    )
    torch.testing.assert_close(terms.expert_jacobian_norms, norms.mean(dim=0))
    torch.testing.assert_close(terms.routing_weights, alpha.mean(dim=0))
    torch.testing.assert_close(terms.routing_weights.sum(), torch.tensor(1.0))

    terms.total.backward()
    assert all(parameter.grad is None for parameter in model.router.parameters())
    assert any(
        parameter.grad is not None
        for expert in model.experts
        for parameter in expert.parameters()
    )
