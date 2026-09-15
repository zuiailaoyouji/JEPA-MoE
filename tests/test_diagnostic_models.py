import torch

from jepa_moe import OracleExpertMoEPredictor, OracleRouterMoEPredictor
from jepa_moe.models import next_state_mse
from jepa_moe.synthetic import (
    ground_truth_expert_outputs,
    synthetic_transition,
    true_routing_weights,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(71)
    return torch.randn(13, 4), torch.randn(13, 2)


def test_oracle_router_is_parameter_free_and_exact() -> None:
    state, action = _batch()
    model = OracleRouterMoEPredictor()
    output = model(state, action)

    assert not hasattr(model, "router")
    assert all(name.startswith("experts.") for name, _ in model.named_parameters())
    torch.testing.assert_close(output.alpha, true_routing_weights(action))
    torch.testing.assert_close(output.alpha.sum(dim=-1), torch.ones(len(state)))
    assert output.expert_outputs.shape == (len(state), 3, 4)


def test_oracle_experts_are_parameter_free_and_exact() -> None:
    state, action = _batch()
    model = OracleExpertMoEPredictor()
    output = model(state, action)

    assert not hasattr(model, "experts")
    assert all(name.startswith("router.") for name, _ in model.named_parameters())
    torch.testing.assert_close(
        output.expert_outputs, ground_truth_expert_outputs(state, action)
    )
    torch.testing.assert_close(output.alpha.sum(dim=-1), torch.ones(len(state)))


def test_oracle_expert_prediction_loss_trains_every_router_parameter() -> None:
    state, action = _batch()
    target = synthetic_transition(state, action)
    model = OracleExpertMoEPredictor()

    loss = next_state_mse(model(state, action).pred, target)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all()

