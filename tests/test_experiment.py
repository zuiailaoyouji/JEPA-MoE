from jepa_moe.experiments.synthetic_basis import (
    ExperimentConfig,
    summarize_results,
)


def _result(
    model: str,
    seed: int,
    *,
    iid: float,
    heldout: float,
    heldout_rollout: float,
    redundancy: float | None = None,
    recovery: float | None = None,
) -> dict:
    result = {
        "model": model,
        "seed": seed,
        "best_validation_prediction_mse": iid,
        "iid_one_step_mse": iid,
        "heldout_composition_mse": heldout,
        "iid_rollout_mse": heldout_rollout,
        "heldout_composition_rollout_mse": heldout_rollout,
    }
    if redundancy is not None and recovery is not None:
        result.update(
            {
                "expert_redundancy_mean_abs_cosine": redundancy,
                "basis_recovery_score": recovery,
                "expert_pair_mean_abs_cosines": [redundancy] * 3,
                "mean_routing_weights": [0.2, 0.3, 0.5],
            }
        )
    return result


def test_preliminary_support_uses_mean_metrics_and_keeps_paired_wins() -> None:
    results = [
        _result("dense", 0, iid=0.2, heldout=0.2, heldout_rollout=0.2),
        _result(
            "vanilla_moe",
            0,
            iid=1.0,
            heldout=1.0,
            heldout_rollout=1.0,
            redundancy=0.8,
            recovery=0.4,
        ),
        _result(
            "jacobian_moe",
            0,
            iid=1.04,
            heldout=0.8,
            heldout_rollout=0.7,
            redundancy=0.2,
            recovery=0.9,
        ),
    ]

    support = summarize_results(
        results, ExperimentConfig(seeds=(0,))
    )["support"]

    assert support is not None
    assert support["provides_preliminary_support"]
    assert support["heldout_one_step_wins"] == 1
    assert support["heldout_rollout_wins"] == 1
    assert set(support["conditions"]) == {
        "iid_not_more_than_5_percent_worse",
        "lower_mean_expert_redundancy",
        "higher_mean_basis_recovery",
        "lower_mean_heldout_one_step_mse",
        "lower_mean_heldout_rollout_mse",
    }
