import torch

from jepa_moe import JacobianMoEPredictor, VanillaMoEPredictor
from jepa_moe.experiments.synthetic_basis import (
    ExperimentConfig,
    _seed_everything,
    summarize_results,
)


def _result(
    model: str,
    seed: int,
    *,
    iid: float,
    rollout: float,
    redundancy: float | None = None,
    principal_similarity: float | None = None,
    projection_error: float | None = None,
    usage_entropy: float | None = None,
    minimum_usage: float | None = None,
) -> dict:
    result = {
        "model": model,
        "seed": seed,
        "best_validation_prediction_mse": iid,
        "iid_one_step_mse": iid,
        "iid_rollout_mse": rollout,
    }
    if redundancy is not None:
        assert principal_similarity is not None
        assert projection_error is not None
        assert usage_entropy is not None
        assert minimum_usage is not None
        result.update(
            {
                "expert_redundancy_mean_abs_cosine": redundancy,
                "dynamics_subspace_principal_cosine_similarity": principal_similarity,
                "dynamics_subspace_projection_reconstruction_error": projection_error,
                "normalized_mean_sample_routing_entropy": 0.7,
                "normalized_routing_usage_entropy": usage_entropy,
                "routing_effective_experts": 2.8,
                "minimum_mean_routing_weight": minimum_usage,
                "basis_recovery_score": principal_similarity,
                "expert_pair_mean_abs_cosines": [redundancy] * 3,
                "mean_routing_weights": [0.2, 0.3, 0.5],
                "dynamics_subspace_principal_angle_cosines": [
                    principal_similarity
                ]
                * 3,
            }
        )
    return result


def test_preliminary_support_uses_mean_metrics_and_keeps_paired_wins() -> None:
    results = [
        _result("dense", 0, iid=0.2, rollout=0.2),
        _result(
            "vanilla_moe",
            0,
            iid=1.0,
            rollout=1.0,
            redundancy=0.8,
            principal_similarity=0.4,
            projection_error=0.6,
            usage_entropy=0.95,
            minimum_usage=0.2,
        ),
        _result(
            "jacobian_moe",
            0,
            iid=1.04,
            rollout=1.03,
            redundancy=0.2,
            principal_similarity=0.9,
            projection_error=0.1,
            usage_entropy=0.95,
            minimum_usage=0.2,
        ),
    ]

    support = summarize_results(
        results, ExperimentConfig(seeds=(0,))
    )["support"]

    assert support is not None
    assert support["provides_preliminary_support"]
    assert support["redundancy_wins"] == 1
    assert support["principal_similarity_wins"] == 1
    assert support["projection_error_wins"] == 1
    assert set(support["conditions"]) == {
        "iid_one_step_not_more_than_5_percent_worse",
        "iid_rollout_not_more_than_5_percent_worse",
        "lower_mean_expert_redundancy",
        "higher_mean_subspace_principal_similarity",
        "lower_mean_subspace_projection_error",
        "no_obvious_routing_collapse",
    }


def test_vanilla_and_jacobian_conditions_start_from_identical_weights() -> None:
    _seed_everything(19)
    vanilla = VanillaMoEPredictor()
    _seed_everything(19)
    jacobian = JacobianMoEPredictor()

    for name, vanilla_tensor in vanilla.state_dict().items():
        torch.testing.assert_close(vanilla_tensor, jacobian.state_dict()[name])
