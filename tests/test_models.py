import pytest
import torch
from torch import nn

from jepa_moe import (
    DensePredictor,
    JacobianMoEPredictor,
    VanillaMoEPredictor,
)


BATCH_SIZE = 7


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    state = torch.randn(BATCH_SIZE, 4)
    action = torch.randn(BATCH_SIZE, 2)
    next_state = torch.randn(BATCH_SIZE, 4)
    return state, action, next_state


@pytest.mark.parametrize(
    "model_class", [DensePredictor, VanillaMoEPredictor, JacobianMoEPredictor]
)
def test_forward_output_shapes(model_class: type[nn.Module]) -> None:
    state, action, _ = _batch()
    output = model_class()(state, action)

    assert output.pred.shape == (BATCH_SIZE, 4)
    if model_class is DensePredictor:
        assert output.alpha is None
        assert output.expert_outputs is None
    else:
        assert output.alpha is not None
        assert output.expert_outputs is not None
        assert output.alpha.shape == (BATCH_SIZE, 3)
        assert output.expert_outputs.shape == (BATCH_SIZE, 3, 4)


@pytest.mark.parametrize(
    "model_class", [DensePredictor, VanillaMoEPredictor, JacobianMoEPredictor]
)
def test_all_parameters_receive_gradients(model_class: type[nn.Module]) -> None:
    state, action, next_state = _batch()
    model = model_class()

    loss = model.training_loss(state, action, next_state).total
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"


@pytest.mark.parametrize(
    "model_class", [DensePredictor, VanillaMoEPredictor, JacobianMoEPredictor]
)
def test_all_models_use_exact_absolute_next_state_mse(
    model_class: type[nn.Module],
) -> None:
    state, action, next_state = _batch()
    model = model_class()
    output = model(state, action)

    loss = model.training_loss(state, action, next_state)
    expected = ((output.pred - next_state) ** 2).mean()

    torch.testing.assert_close(loss.prediction, expected)
    with pytest.raises(ValueError, match="next_state must have shape"):
        model.training_loss(state, action, torch.randn(BATCH_SIZE, 1))


@pytest.mark.parametrize("model_class", [VanillaMoEPredictor, JacobianMoEPredictor])
def test_softmax_weights_are_normalized_per_sample(
    model_class: type[nn.Module],
) -> None:
    state, action, _ = _batch()
    alpha = model_class()(state, action).alpha

    assert alpha is not None
    torch.testing.assert_close(alpha.sum(dim=-1), torch.ones(BATCH_SIZE))
    assert torch.all(alpha >= 0)
    assert torch.all(alpha <= 1)


def test_moe_architectures_are_identical_and_exact() -> None:
    vanilla = VanillaMoEPredictor()
    jacobian = JacobianMoEPredictor()

    vanilla_shapes = {
        name: tuple(parameter.shape) for name, parameter in vanilla.named_parameters()
    }
    jacobian_shapes = {
        name: tuple(parameter.shape) for name, parameter in jacobian.named_parameters()
    }
    assert vanilla_shapes == jacobian_shapes

    for expert in vanilla.experts:
        linear_layers = [module for module in expert if isinstance(module, nn.Linear)]
        activations = [module for module in expert if isinstance(module, nn.SiLU)]
        assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
            (6, 128),
            (128, 128),
            (128, 4),
        ]
        assert len(activations) == 2

    router_linears = [
        module for module in vanilla.router if isinstance(module, nn.Linear)
    ]
    router_activations = [
        module for module in vanilla.router if isinstance(module, nn.SiLU)
    ]
    assert [(layer.in_features, layer.out_features) for layer in router_linears] == [
        (6, 64),
        (64, 64),
        (64, 3),
    ]
    assert len(router_activations) == 2

    forbidden = (nn.Dropout, nn.BatchNorm1d, nn.LayerNorm)
    assert not any(isinstance(module, forbidden) for module in vanilla.modules())

    expert_parameter_ids = [
        {id(parameter) for parameter in expert.parameters()} for expert in vanilla.experts
    ]
    for first in range(len(expert_parameter_ids)):
        for second in range(first + 1, len(expert_parameter_ids)):
            assert expert_parameter_ids[first].isdisjoint(expert_parameter_ids[second])


def test_vanilla_and_jacobian_only_differ_in_training_loss() -> None:
    torch.manual_seed(23)
    vanilla = VanillaMoEPredictor()
    jacobian = JacobianMoEPredictor()
    jacobian.load_state_dict(vanilla.state_dict())
    state, action, next_state = _batch()

    vanilla_output = vanilla(state, action)
    jacobian_output = jacobian(state, action)
    for vanilla_tensor, jacobian_tensor in zip(vanilla_output, jacobian_output):
        assert vanilla_tensor is not None
        assert jacobian_tensor is not None
        torch.testing.assert_close(vanilla_tensor, jacobian_tensor)

    vanilla_loss = vanilla.training_loss(state, action, next_state)
    jacobian_loss = jacobian.training_loss(
        state, action, next_state, lambda_jac=0.5
    )
    torch.testing.assert_close(vanilla_loss.prediction, jacobian_loss.prediction)
    torch.testing.assert_close(vanilla_loss.total, vanilla_loss.prediction)
    assert vanilla_loss.specialization.item() >= 0.0
    assert jacobian_loss.specialization.item() >= 0.0
    torch.testing.assert_close(
        jacobian_loss.total,
        jacobian_loss.prediction + 0.5 * jacobian_loss.specialization,
    )

    expected_metrics = {
        "total_loss",
        "prediction_loss",
        "control_jacobian_specialization_loss",
        "jacobian_diversity_loss",
        "activity_loss",
        "jacobian_cosine_0_1",
        "jacobian_cosine_0_2",
        "jacobian_cosine_1_2",
        "jacobian_norm_0",
        "jacobian_norm_1",
        "jacobian_norm_2",
        "routing_weight_0",
        "routing_weight_1",
        "routing_weight_2",
    }
    assert set(vanilla_loss.logging_metrics()) == expected_metrics
    assert set(jacobian_loss.logging_metrics()) == expected_metrics


def test_inputs_are_strictly_state_and_action() -> None:
    model = VanillaMoEPredictor()
    with pytest.raises(ValueError, match="state must have shape"):
        model(torch.randn(2, 5), torch.randn(2, 2))
    with pytest.raises(ValueError, match="action must have shape"):
        model(torch.randn(2, 4), torch.randn(2, 3))
    with pytest.raises(ValueError, match="same batch size"):
        model(torch.randn(2, 4), torch.randn(3, 2))
