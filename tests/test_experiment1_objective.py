import torch

from jepa_moe import (
    Experiment1ObjectiveConfig,
    build_experiment1_model,
    experiment1_training_loss,
    load_balance_loss,
    make_basis_composition_transition_batch,
    sample_unit_action_directions,
)


def test_direction_sampler_shape_unit_norm_and_seed_reproducibility() -> None:
    first = sample_unit_action_directions(
        64, 2, generator=torch.Generator().manual_seed(401)
    )
    second = sample_unit_action_directions(
        64, 2, generator=torch.Generator().manual_seed(401)
    )
    different = sample_unit_action_directions(
        64, 2, generator=torch.Generator().manual_seed(402)
    )

    assert first.shape == (64, 2)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first, dim=-1), torch.ones(64)
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, different)


def test_direction_generator_is_independent_of_data_sampling() -> None:
    data_generator = torch.Generator().manual_seed(403)
    expected_first = torch.rand(8, generator=data_generator)
    expected_second = torch.rand(8, generator=data_generator)

    data_generator = torch.Generator().manual_seed(403)
    actual_first = torch.rand(8, generator=data_generator)
    sample_unit_action_directions(
        16, 2, generator=torch.Generator().manual_seed(404)
    )
    actual_second = torch.rand(8, generator=data_generator)

    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(actual_second, expected_second)


def test_load_balance_loss_uniform_and_collapsed_values() -> None:
    uniform = torch.full((17, 3), 1.0 / 3.0)
    collapsed = torch.zeros(17, 3)
    collapsed[:, 0] = 1.0

    torch.testing.assert_close(
        load_balance_loss(uniform), torch.zeros(()), atol=1e-12, rtol=0.0
    )
    assert load_balance_loss(collapsed) > 0


def test_load_balance_loss_backpropagates_to_router() -> None:
    batch = make_basis_composition_transition_batch(23, seed=405)
    model = build_experiment1_model("moe_s", initialization_seed=406)
    alpha = model(batch.state, batch.action).alpha
    assert alpha is not None

    load_balance_loss(alpha).backward()

    gradients = [parameter.grad for parameter in model.router.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
        if gradient is not None
    )
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
        if gradient is not None
    )


def test_all_four_total_loss_formulas_and_raw_logging() -> None:
    batch = make_basis_composition_transition_batch(19, seed=407)
    config = Experiment1ObjectiveConfig(lambda_cr=0.7, lambda_bal=0.03)
    direction = sample_unit_action_directions(
        19, 2, generator=torch.Generator().manual_seed(408)
    )

    for name in ("dense", "moe_a", "moe_s", "ours"):
        model = build_experiment1_model(name, initialization_seed=409)
        loss = experiment1_training_loss(
            name,
            model,
            batch.state,
            batch.action,
            batch.next_state,
            config=config,
            direction=direction if name == "ours" else None,
        )
        expected = loss.prediction
        if name != "dense":
            expected = expected + config.lambda_bal * loss.balance
        if name == "ours":
            expected = expected + config.lambda_cr * loss.control_response
        torch.testing.assert_close(loss.total, expected)
        assert set(loss.logging_metrics()) == {
            "total_loss",
            "prediction_loss",
            "control_response_loss",
            "load_balance_loss",
        }


def test_ours_total_loss_has_finite_expert_and_router_gradients() -> None:
    batch = make_basis_composition_transition_batch(17, seed=410)
    model = build_experiment1_model("ours", initialization_seed=411)
    direction = sample_unit_action_directions(
        17, 2, generator=torch.Generator().manual_seed(412)
    )
    loss = experiment1_training_loss(
        "ours",
        model,
        batch.state,
        batch.action,
        batch.next_state,
        config=Experiment1ObjectiveConfig(),
        direction=direction,
    )

    loss.total.backward()

    for group in (model.experts.parameters(), model.router.parameters()):
        gradients = [parameter.grad for parameter in group]
        assert all(gradient is not None for gradient in gradients)
        assert all(
            torch.isfinite(gradient).all()
            for gradient in gradients
            if gradient is not None
        )


def test_moe_s_and_dense_have_inactive_raw_terms_without_gradient_paths() -> None:
    batch = make_basis_composition_transition_batch(13, seed=413)
    config = Experiment1ObjectiveConfig()

    dense = build_experiment1_model("dense", initialization_seed=414)
    dense_loss = experiment1_training_loss(
        "dense",
        dense,
        batch.state,
        batch.action,
        batch.next_state,
        config=config,
    )
    assert dense_loss.control_response.item() == 0.0
    assert dense_loss.balance.item() == 0.0
    assert not dense_loss.control_response.requires_grad
    assert not dense_loss.balance.requires_grad
    torch.testing.assert_close(dense_loss.total, dense_loss.prediction)

    moe_s = build_experiment1_model("moe_s", initialization_seed=414)
    moe_s_loss = experiment1_training_loss(
        "moe_s",
        moe_s,
        batch.state,
        batch.action,
        batch.next_state,
        config=config,
    )
    assert moe_s_loss.control_response.item() == 0.0
    assert not moe_s_loss.control_response.requires_grad
    torch.testing.assert_close(
        moe_s_loss.total,
        moe_s_loss.prediction + config.lambda_bal * moe_s_loss.balance,
    )
