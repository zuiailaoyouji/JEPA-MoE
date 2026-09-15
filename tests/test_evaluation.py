import torch
from torch import nn

from jepa_moe import PredictorOutput
from jepa_moe.evaluation import expert_jacobian_metrics, one_step_mse, rollout_mse
from jepa_moe.synthetic import (
    ground_truth_expert_outputs,
    make_rollout_batch,
    make_transition_batch,
    true_routing_weights,
)


class PermutedGroundTruthMoE(nn.Module):
    def __init__(self, signs: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.register_buffer("permutation", torch.tensor([2, 0, 1]))
        self.register_buffer("signs", torch.tensor(signs))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        expert_outputs = ground_truth_expert_outputs(state, action)[
            :, self.permutation
        ]
        expert_outputs = expert_outputs * self.signs[None, :, None]
        alpha = true_routing_weights(action)[:, self.permutation]
        pred = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
        return PredictorOutput(pred, alpha, expert_outputs)


class CollapsedGroundTruthMoE(nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> PredictorOutput:
        first = ground_truth_expert_outputs(state, action)[:, :1]
        expert_outputs = first.expand(-1, 3, -1)
        alpha = torch.full(
            (len(state), 3), 1.0 / 3.0, device=state.device, dtype=state.dtype
        )
        pred = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
        return PredictorOutput(pred, alpha, expert_outputs)


def test_absolute_cosine_recovery_ignores_permutation_and_sign() -> None:
    model = PermutedGroundTruthMoE(signs=(-1.0, 1.0, -1.0))
    data = make_transition_batch(128, "iid", seed=51)

    metrics = expert_jacobian_metrics(
        model, data, device=torch.device("cpu"), batch_size=31
    )

    assert metrics["basis_matching_learned_to_ground_truth"] == [2, 0, 1]
    torch.testing.assert_close(
        torch.tensor(metrics["basis_matched_cosines"]), torch.ones(3)
    )
    assert abs(metrics["basis_recovery_score"] - 1.0) < 1e-6
    assert abs(sum(metrics["mean_routing_weights"]) - 1.0) < 1e-6
    assert metrics["router_weight_mse"] > 0.0
    assert metrics["matched_router_weight_mse"] < 1e-14
    assert abs(metrics["dynamics_subspace_principal_cosine_similarity"] - 1.0) < 1e-12
    assert metrics["dynamics_subspace_projection_reconstruction_error"] < 1e-12


def test_subspace_metrics_penalize_a_rank_one_expert_dictionary() -> None:
    data = make_transition_batch(128, "full", seed=54)
    metrics = expert_jacobian_metrics(
        CollapsedGroundTruthMoE(),
        data,
        device=torch.device("cpu"),
        batch_size=31,
    )

    assert metrics["expert_redundancy_mean_abs_cosine"] > 0.999999
    assert metrics["dynamics_subspace_principal_cosine_similarity"] < 0.34
    assert metrics["dynamics_subspace_projection_reconstruction_error"] > 0.65
    assert abs(metrics["normalized_mean_sample_routing_entropy"] - 1.0) < 1e-6
    assert abs(metrics["normalized_routing_usage_entropy"] - 1.0) < 1e-6
    assert abs(metrics["routing_effective_experts"] - 3.0) < 1e-6


def test_perfect_model_has_zero_prediction_and_rollout_error() -> None:
    model = PermutedGroundTruthMoE()
    transition_data = make_transition_batch(64, "heldout", seed=52)
    rollout_data = make_rollout_batch(32, 25, "heldout", seed=53)

    assert (
        one_step_mse(
            model, transition_data, device=torch.device("cpu"), batch_size=17
        )
        < 1e-14
    )
    assert (
        rollout_mse(model, rollout_data, device=torch.device("cpu"), batch_size=11)
        < 1e-14
    )
