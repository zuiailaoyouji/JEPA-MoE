import math

import torch
from torch import nn

from jepa_moe import (
    PredictorOutput,
    basis_composition_action_jacobian,
    basis_composition_transition,
    basis_subspace_metrics,
    build_experiment1_model,
    evaluate_experiment1_model,
    experiment1_response_metrics,
    expert_action_jacobians,
    ground_truth_expert_action_jacobians,
    make_basis_composition_rollout_batch,
    make_basis_composition_transition_batch,
    predictor_action_jacobian,
    sample_unit_action_directions,
)


class PerfectDensePredictor(nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        return PredictorOutput(
            pred=basis_composition_transition(state, action),
            alpha=None,
            expert_outputs=None,
        )


def test_perfect_dense_predictor_has_zero_errors_and_no_routing_metrics() -> None:
    iid_test = make_basis_composition_transition_batch(
        48, seed=601, dtype=torch.float64
    )
    rollout = make_basis_composition_rollout_batch(
        16, 25, seed=602, dtype=torch.float64
    )
    metrics = evaluate_experiment1_model(
        PerfectDensePredictor(),
        iid_test=iid_test,
        rollout_25=rollout,
        device=torch.device("cpu"),
        prediction_batch_size=13,
        jacobian_batch_size=11,
        direction_seed=603,
    )

    assert metrics["iid_one_step_mse"] == 0.0
    assert metrics["rollout_25_mse"] == 0.0
    assert metrics["directional_response_mse"] < 1e-30
    assert metrics["jacobian_frobenius_mse"] < 1e-30
    for key in (
        "mean_expert_usage",
        "routing_entropy",
        "normalized_routing_entropy",
        "routing_effective_experts",
        "load_balance_loss",
    ):
        assert metrics[key] is None


def test_state_only_full_jacobian_equals_weighted_expert_dictionary() -> None:
    data = make_basis_composition_transition_batch(
        17, seed=604, dtype=torch.float64
    )
    model = build_experiment1_model("moe_s", initialization_seed=605).double()
    output = model(data.state, data.action)
    assert output.alpha is not None

    direct = predictor_action_jacobian(model, data.state, data.action)
    experts = expert_action_jacobians(model, data.state, data.action)
    composed = torch.einsum("bk,bksa->bsa", output.alpha, experts)
    direction = sample_unit_action_directions(
        len(data),
        2,
        generator=torch.Generator().manual_seed(606),
        dtype=torch.float64,
    )

    torch.testing.assert_close(direct, composed, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.einsum("bsa,ba->bs", direct, direction),
        torch.einsum("bsa,ba->bs", composed, direction),
        rtol=1e-12,
        atol=1e-12,
    )


def test_moe_a_full_jacobian_includes_nonzero_router_contribution() -> None:
    data = make_basis_composition_transition_batch(
        19, seed=607, dtype=torch.float64
    )
    model = build_experiment1_model("moe_a", initialization_seed=608).double()
    action = data.action.detach().requires_grad_(True)
    output = model(data.state, action)
    assert output.alpha is not None and output.expert_outputs is not None
    full = predictor_action_jacobian(model, data.state, data.action)
    experts = expert_action_jacobians(model, data.state, data.action)
    within = torch.einsum("bk,bksa->bsa", output.alpha, experts)

    router_rows = [
        torch.autograd.grad(
            output.alpha[:, index].sum(), action, retain_graph=True
        )[0]
        for index in range(output.alpha.shape[-1])
    ]
    router_jacobian = torch.stack(router_rows, dim=1)
    switching = torch.einsum(
        "bks,bka->bsa", output.expert_outputs, router_jacobian
    )

    assert torch.count_nonzero(switching).item() > 0
    assert not torch.allclose(full, within, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(full, within + switching, rtol=1e-12, atol=1e-12)


def test_moe_a_directional_metric_uses_complete_predictor_jacobian() -> None:
    data = make_basis_composition_transition_batch(
        13, seed=609, dtype=torch.float64
    )
    model = build_experiment1_model("moe_a", initialization_seed=610).double()
    metrics = experiment1_response_metrics(
        model,
        data,
        device=torch.device("cpu"),
        batch_size=len(data),
        direction_seed=611,
    )
    directions = sample_unit_action_directions(
        len(data),
        2,
        generator=torch.Generator().manual_seed(611),
        dtype=torch.float64,
    )
    full = predictor_action_jacobian(model, data.state, data.action)
    target = basis_composition_action_jacobian(data.state, data.action)
    expected = (
        torch.einsum("bsa,ba->bs", full - target, directions)
        .square()
        .sum()
        / len(data)
    )
    torch.testing.assert_close(
        torch.tensor(metrics["directional_response_mse"], dtype=torch.float64),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_subspace_metrics_are_invariant_to_expert_permutation() -> None:
    data = make_basis_composition_transition_batch(
        31, seed=612, dtype=torch.float64
    )
    ground_truth = ground_truth_expert_action_jacobians(data.state)
    permuted = ground_truth[:, [2, 0, 1]]
    original_metrics = basis_subspace_metrics(ground_truth, ground_truth)
    permuted_metrics = basis_subspace_metrics(permuted, ground_truth)

    assert (
        original_metrics["dynamics_subspace_principal_cosine_similarity"]
        > 1.0 - 1e-12
    )
    assert (
        original_metrics["dynamics_subspace_projection_reconstruction_error"]
        < 1e-24
    )
    torch.testing.assert_close(
        torch.tensor(
            permuted_metrics["dynamics_subspace_principal_angle_cosines"]
        ),
        torch.tensor(
            original_metrics["dynamics_subspace_principal_angle_cosines"]
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    assert (
        abs(
            permuted_metrics[
                "dynamics_subspace_projection_reconstruction_error"
            ]
            - original_metrics[
                "dynamics_subspace_projection_reconstruction_error"
            ]
        )
        < 1e-24
    )


def test_moe_evaluation_routing_statistics_are_finite_on_cpu() -> None:
    data = make_basis_composition_transition_batch(29, seed=613)
    rollout = make_basis_composition_rollout_batch(7, 25, seed=614)
    model = build_experiment1_model("ours", initialization_seed=615)
    metrics = evaluate_experiment1_model(
        model,
        iid_test=data,
        rollout_25=rollout,
        device=torch.device("cpu"),
        prediction_batch_size=8,
        jacobian_batch_size=7,
        direction_seed=616,
    )

    assert len(metrics["mean_expert_usage"]) == 3
    assert abs(sum(metrics["mean_expert_usage"]) - 1.0) < 1e-6
    for key, value in metrics.items():
        if value is None or isinstance(value, list):
            continue
        assert math.isfinite(value), key
