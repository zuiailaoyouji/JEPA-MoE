import inspect

import torch

from jepa_moe import (
    HistoryConditionedMoEPredictor,
    MoEPredictor,
    make_history_transition_batch,
)
from jepa_moe.experiments.experiment2_history_sanity import (
    HistorySanityConfig,
    ShuffledHistoryBatchStream,
    build_sanity_model,
    evaluate_validation,
    history_routing_diagnostics,
    history_sanity_loss,
    make_sanity_data,
)
from jepa_moe.experiments.experiment2_history_horizon import (
    _capture_resume_state,
)


def test_model_interfaces_exclude_forbidden_inputs() -> None:
    state_only = build_sanity_model("state_only", initialization_seed=501)
    history = build_sanity_model("history", initialization_seed=501)
    assert isinstance(state_only, MoEPredictor)
    assert isinstance(history, HistoryConditionedMoEPredictor)

    assert tuple(inspect.signature(state_only.forward).parameters) == (
        "state",
        "action",
    )
    assert "system_id" not in inspect.signature(state_only.forward).parameters
    assert tuple(inspect.signature(history.context_encoder.forward).parameters) == (
        "state_history",
        "past_actions",
    )
    assert tuple(inspect.signature(history.router.forward).parameters) == ("context",)
    assert "system_id" not in inspect.signature(history.forward).parameters


def test_models_share_dataset_stream_and_expert_initialization() -> None:
    config = HistorySanityConfig(
        train_samples=128,
        validation_samples=64,
        test_samples=64,
        batch_size=32,
        evaluation_batch_size=64,
        training_steps=1,
        validation_interval=1,
        diagnostic_samples=4,
    )
    data = make_sanity_data(config)
    first_stream = ShuffledHistoryBatchStream(data.train, 32, seed=777)
    second_stream = ShuffledHistoryBatchStream(data.train, 32, seed=777)
    first_batch = first_stream.next()
    second_batch = second_stream.next()
    for field in first_batch.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(first_batch, field), getattr(second_batch, field)
        )

    state_only = build_sanity_model("state_only", initialization_seed=502)
    history = build_sanity_model("history", initialization_seed=502)
    for first, second in zip(
        state_only.experts.parameters(), history.experts.parameters(), strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_batch_stream_state_restores_the_exact_next_batch() -> None:
    data = make_history_transition_batch(128, seed=506)
    first = ShuffledHistoryBatchStream(data, 32, seed=507)
    first.next()
    first.next()
    saved_state = first.state_dict()
    expected = first.next()

    restored = ShuffledHistoryBatchStream(data, 32, seed=999)
    restored.load_state_dict(saved_state)
    actual = restored.next()
    for field in expected.__dataclass_fields__:
        torch.testing.assert_close(getattr(expected, field), getattr(actual, field))


def test_both_losses_and_gradients_are_finite_without_control_response() -> None:
    batch = make_history_transition_batch(32, seed=503)
    for model_name in ("state_only", "history"):
        model = build_sanity_model(model_name, initialization_seed=503)
        total, prediction, balance = history_sanity_loss(
            model_name, model, batch, lambda_bal=0.01
        )
        torch.testing.assert_close(total, prediction + 0.01 * balance)
        assert torch.isfinite(torch.stack((total, prediction, balance))).all()
        total.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_validation_metric_is_prediction_mse_only() -> None:
    batch = make_history_transition_batch(64, seed=504)
    model = build_sanity_model("history", initialization_seed=504)
    validation = evaluate_validation(
        "history",
        model,
        batch,
        lambda_bal=0.01,
        device=torch.device("cpu"),
        batch_size=64,
    )

    assert set(validation) == {
        "prediction_loss",
        "load_balance_loss",
        "total_loss",
    }
    assert validation["total_loss"] >= validation["prediction_loss"]


def test_matched_history_routing_diagnostic_runs_with_valid_histories() -> None:
    model = build_sanity_model("history", initialization_seed=505)
    assert isinstance(model, HistoryConditionedMoEPredictor)
    diagnostic = history_routing_diagnostics(
        model,
        sample_count=8,
        seed=505,
        device=torch.device("cpu"),
    )

    assert diagnostic["history_transition_max_abs_error"] < 1e-8
    assert len(diagnostic["pairwise_mean_alpha_distance"]) == 4
    assert torch.isfinite(
        torch.tensor(diagnostic["pairwise_mean_alpha_distance"])
    ).all()


def test_full_resume_snapshot_contains_every_required_state() -> None:
    config = HistorySanityConfig(
        train_samples=128,
        validation_samples=64,
        test_samples=64,
        batch_size=32,
        evaluation_batch_size=64,
        training_steps=1,
        validation_interval=1,
        diagnostic_samples=4,
    )
    data = make_sanity_data(config)
    model = build_sanity_model("history", initialization_seed=508)
    assert isinstance(model, HistoryConditionedMoEPredictor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stream = ShuffledHistoryBatchStream(data.train, 32, seed=508)
    snapshot = _capture_resume_state(
        model,
        optimizer,
        stream,
        current_step=25,
        config=config,
        best_validation_mse=0.1,
        best_step=20,
    )

    assert snapshot["step"] == 25
    assert snapshot["direction_rng_state"] is None
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "batch_stream_state_dict",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
    } <= snapshot.keys()
