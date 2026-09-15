import pytest
import torch

from jepa_moe.experiments.synthetic_basis import ExperimentConfig
from jepa_moe.experiments.synthetic_warmup import (
    _training_loss,
    expert_gradient_probe,
    specialization_weight,
)
from jepa_moe.losses import control_jacobian_specialization_terms
from jepa_moe.models import JacobianMoEPredictor, next_state_mse
from jepa_moe.synthetic import make_transition_batch


@pytest.mark.parametrize("warmup", (250, 500, 1000))
def test_warmup_boundaries_and_linear_ramp(warmup: int) -> None:
    condition = f"warmup_{warmup}"
    assert specialization_weight(condition, 0) == 0.0
    assert specialization_weight(condition, warmup) == 0.0
    assert specialization_weight(condition, warmup + 1) == pytest.approx(0.0002)
    assert specialization_weight(condition, warmup + 250) == pytest.approx(0.05)
    assert specialization_weight(condition, warmup + 500) == pytest.approx(0.1)
    assert specialization_weight(condition, 5000) == pytest.approx(0.1)


def test_immediate_and_vanilla_weights() -> None:
    assert specialization_weight("vanilla", 0) == 0.0
    assert specialization_weight("vanilla", 5000) == 0.0
    assert specialization_weight("immediate", 0) == pytest.approx(0.1)
    assert specialization_weight("immediate", 1) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="unknown condition"):
        specialization_weight("invalid", 1)


def test_probe_matches_independent_expert_gradients_and_does_not_write_grads() -> None:
    torch.manual_seed(42)
    model = JacobianMoEPredictor()
    batch = make_transition_batch(9, "full", seed=43)
    config = ExperimentConfig()
    metrics = expert_gradient_probe(
        model,
        batch.state,
        batch.action,
        batch.next_state,
        condition="immediate",
        step=1,
        config=config,
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    experts = tuple(model.experts.parameters())
    router = tuple(model.router.parameters())
    prediction = next_state_mse(
        model(batch.state, batch.action).pred, batch.next_state
    )
    terms = control_jacobian_specialization_terms(
        model, batch.state, batch.action, create_graph=True
    )
    pred_grads = torch.autograd.grad(prediction, experts)
    all_spec_grads = torch.autograd.grad(
        terms.total, experts + router, allow_unused=True
    )
    spec_grads = all_spec_grads[: len(experts)]
    router_spec_grads = all_spec_grads[len(experts) :]
    pred_norm = torch.stack([grad.square().sum() for grad in pred_grads]).sum().sqrt()
    spec_norm = torch.stack(
        [
            grad.square().sum() if grad is not None else terms.total.new_zeros(())
            for grad in spec_grads
        ]
    ).sum().sqrt()

    assert metrics["expert_prediction_grad_norm"] == pytest.approx(pred_norm.item())
    assert metrics["expert_specialization_grad_norm_unweighted"] == pytest.approx(
        spec_norm.item()
    )
    assert metrics["r_grad"] == pytest.approx(
        0.1 * spec_norm.item() / (pred_norm.item() + 1e-12)
    )
    assert all(grad is None for grad in router_spec_grads)
    assert all(parameter.grad is None for parameter in model.router.parameters())


def test_prediction_only_phase_keeps_exact_vanilla_objective() -> None:
    model = JacobianMoEPredictor()
    batch = make_transition_batch(9, "full", seed=44)
    config = ExperimentConfig()
    total, prediction = _training_loss(
        model,
        batch.state,
        batch.action,
        batch.next_state,
        weight=specialization_weight("warmup_500", 500),
        config=config,
    )
    torch.testing.assert_close(total, prediction)
    torch.testing.assert_close(
        prediction, next_state_mse(model(batch.state, batch.action).pred, batch.next_state)
    )
