import inspect

import torch

from jepa_moe import (
    ACTION_DIM,
    HISTORY_LENGTH,
    NUM_SYSTEMS,
    STATE_DIM,
    HistoryTransitionBatch,
    make_history_transition_batch,
    system_true_transition,
)


def _expected_transition(
    state: torch.Tensor, action: torch.Tensor, system_id: torch.Tensor
) -> torch.Tensor:
    expected = torch.empty_like(state)
    for selected_system in range(NUM_SYSTEMS):
        mask = system_id == selected_system
        if torch.any(mask):
            expected[mask] = system_true_transition(
                state[mask], action[mask], selected_system
            )
    return expected


def test_history_batch_shapes_and_fixed_horizon() -> None:
    batch_size = 127
    batch = make_history_transition_batch(batch_size, seed=301)

    assert isinstance(batch, HistoryTransitionBatch)
    assert HISTORY_LENGTH == 4
    assert "horizon" not in inspect.signature(
        make_history_transition_batch
    ).parameters
    assert len(batch) == batch_size
    assert batch.state_history.shape == (
        batch_size,
        HISTORY_LENGTH + 1,
        STATE_DIM,
    )
    assert batch.past_actions.shape == (
        batch_size,
        HISTORY_LENGTH,
        ACTION_DIM,
    )
    assert batch.current_action.shape == (batch_size, ACTION_DIM)
    assert batch.next_state.shape == (batch_size, STATE_DIM)
    assert batch.system_id.shape == (batch_size,)
    assert batch.system_id.dtype == torch.long


def test_every_history_transition_uses_one_fixed_system() -> None:
    batch = make_history_transition_batch(256, seed=302, dtype=torch.float64)

    for history_index in range(HISTORY_LENGTH):
        expected = _expected_transition(
            batch.state_history[:, history_index],
            batch.past_actions[:, history_index],
            batch.system_id,
        )
        torch.testing.assert_close(
            batch.state_history[:, history_index + 1], expected
        )

    expected_next_state = _expected_transition(
        batch.state_history[:, -1], batch.current_action, batch.system_id
    )
    torch.testing.assert_close(batch.next_state, expected_next_state)


def test_current_action_is_separate_from_history_and_past_actions() -> None:
    batch = make_history_transition_batch(512, seed=303, dtype=torch.float64)

    assert batch.state_history.shape[-1] == STATE_DIM
    assert batch.past_actions.shape[1] == HISTORY_LENGTH
    assert batch.current_action.untyped_storage().data_ptr() != (
        batch.past_actions.untyped_storage().data_ptr()
    )
    current_action_in_past = torch.all(
        batch.past_actions == batch.current_action.unsqueeze(1), dim=-1
    )
    assert not torch.any(current_action_in_past)


def test_history_generation_is_seeded_and_reproducible() -> None:
    first = make_history_transition_batch(1_024, seed=304)
    second = make_history_transition_batch(1_024, seed=304)
    different = make_history_transition_batch(1_024, seed=305)

    for field_name in (
        "state_history",
        "past_actions",
        "current_action",
        "next_state",
        "system_id",
    ):
        first_value = getattr(first, field_name)
        second_value = getattr(second, field_name)
        different_value = getattr(different, field_name)
        torch.testing.assert_close(first_value, second_value)
        assert not torch.equal(first_value, different_value)


def test_history_batch_is_finite_and_system_sampling_is_balanced() -> None:
    batch = make_history_transition_batch(20_000, seed=306)

    assert torch.isfinite(batch.state_history).all()
    assert torch.isfinite(batch.past_actions).all()
    assert torch.isfinite(batch.current_action).all()
    assert torch.isfinite(batch.next_state).all()
    assert torch.all((batch.system_id >= 0) & (batch.system_id < NUM_SYSTEMS))

    usage = torch.bincount(batch.system_id, minlength=NUM_SYSTEMS).float()
    usage = usage / len(batch)
    torch.testing.assert_close(
        usage,
        torch.full((NUM_SYSTEMS,), 1.0 / NUM_SYSTEMS),
        rtol=0.0,
        atol=0.02,
    )
