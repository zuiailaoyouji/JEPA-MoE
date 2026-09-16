import torch
from torch import nn

from jepa_moe import MoEPredictor, build_experiment1_model


def _first_linear(module: nn.Sequential) -> nn.Linear:
    layer = module[0]
    assert isinstance(layer, nn.Linear)
    return layer


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(201)
    state = torch.randn(13, 4, generator=generator)
    action = torch.randn(13, 2, generator=generator)
    return state, action


def test_router_input_dimensions_are_explicit() -> None:
    moe_a = build_experiment1_model("moe_a", initialization_seed=202)
    moe_s = build_experiment1_model("moe_s", initialization_seed=202)
    assert isinstance(moe_a, MoEPredictor)
    assert isinstance(moe_s, MoEPredictor)

    assert moe_a.router_input == "state_action"
    assert moe_s.router_input == "state"
    assert _first_linear(moe_a.router).in_features == 6
    assert _first_linear(moe_s.router).in_features == 4


def test_moe_output_shapes_normalization_and_weighted_prediction() -> None:
    state, action = _batch()
    for name in ("moe_a", "moe_s", "ours"):
        model = build_experiment1_model(name, initialization_seed=203)
        output = model(state, action)
        assert output.alpha is not None
        assert output.expert_outputs is not None
        assert output.pred.shape == (13, 4)
        assert output.alpha.shape == (13, 3)
        assert output.expert_outputs.shape == (13, 3, 4)
        assert torch.all(output.alpha >= 0)
        torch.testing.assert_close(
            output.alpha.sum(dim=-1), torch.ones(13, dtype=state.dtype)
        )
        torch.testing.assert_close(
            output.pred,
            torch.sum(output.alpha.unsqueeze(-1) * output.expert_outputs, dim=1),
        )


def test_state_only_routing_is_exactly_action_independent() -> None:
    state, action = _batch()
    model = build_experiment1_model("moe_s", initialization_seed=204)
    first = model(state, action).alpha
    second = model(state, action + 17.0).alpha
    assert first is not None and second is not None
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_moe_a_router_computation_depends_on_action() -> None:
    state, action = _batch()
    action = action.requires_grad_(True)
    model = build_experiment1_model("moe_a", initialization_seed=205)
    alpha = model(state, action).alpha
    assert alpha is not None
    action_gradient = torch.autograd.grad(alpha[:, 0].sum(), action)[0]
    assert torch.count_nonzero(action_gradient).item() > 0


def test_moe_s_and_ours_have_identical_architecture() -> None:
    moe_s = build_experiment1_model("moe_s", initialization_seed=206)
    ours = build_experiment1_model("ours", initialization_seed=207)
    moe_s_signature = [
        (name, tuple(parameter.shape)) for name, parameter in moe_s.named_parameters()
    ]
    ours_signature = [
        (name, tuple(parameter.shape)) for name, parameter in ours.named_parameters()
    ]
    assert type(moe_s) is type(ours) is MoEPredictor
    assert moe_s.router_input == ours.router_input == "state"
    assert moe_s_signature == ours_signature


def test_same_seed_gives_all_moe_conditions_identical_experts() -> None:
    models = {
        name: build_experiment1_model(name, initialization_seed=208)
        for name in ("moe_a", "moe_s", "ours")
    }
    reference = dict(models["moe_a"].experts.named_parameters())

    for name in ("moe_s", "ours"):
        candidate = dict(models[name].experts.named_parameters())
        assert candidate.keys() == reference.keys()
        for parameter_name, reference_parameter in reference.items():
            torch.testing.assert_close(
                candidate[parameter_name],
                reference_parameter,
                rtol=0.0,
                atol=0.0,
            )
