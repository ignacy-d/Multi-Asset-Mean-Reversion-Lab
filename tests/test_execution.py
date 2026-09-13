from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from mr_lab.execution import (
    CancelAcknowledged,
    EnsureProtectionCommand,
    EntryAccepted,
    EntryFill,
    EntryRejected,
    ExecutionClosed,
    ExecutionEngine,
    ExecutionInvariantError,
    ExecutionLifecycle,
    FakeBroker,
    FakeBrokerScenario,
    FakeFill,
    ProtectionAcknowledged,
    ProtectionRejected,
    ProtectionStatus,
    SubmitEntryCommand,
)
from mr_lab.portfolio import PlanReference, ProposalProvenance
from mr_lab.risk import ProtectiveBoundary, SizedExecutionIntent

NOW = datetime(2024, 7, 1, 9, tzinfo=UTC)
D = Decimal


def intent(*, suffix="mr", sleeve="mean-reversion", instrument="EURUSD"):
    return SizedExecutionIntent(
        f"execution-{suffix}",
        f"risk-{suffix}",
        f"proposal-{suffix}",
        f"opportunity-{suffix}",
        sleeve,
        instrument,
        "LONG",
        D("10000"),
        f"{sleeve}-policy",
        PlanReference("strategy-entry", f"entry-{suffix}"),
        PlanReference("strategy-protection", f"protect-{suffix}"),
        ProposalProvenance("synthetic-test", "1"),
        "risk-specification-v1",
        ProtectiveBoundary(D("1.1000"), D("1.0900"), "risk-specification-v1"),
        NOW - timedelta(seconds=1),
        NOW,
        ("fx",),
    )


def submitted(item=None):
    item = item or intent()
    state, commands = ExecutionEngine().handle_intent(item)
    return item, state, commands[0]


def acknowledged(item=None, *, atomic=False):
    item, state, command = submitted(item)
    event = EntryAccepted(
        "accepted-1",
        item.execution_intent_id,
        command.command_id,
        "order-1",
        NOW,
        atomic,
        "protection-atomic" if atomic else None,
    )
    state, _ = ExecutionEngine().handle_event(state, event)
    return item, state, command


def apply(state, event):
    return ExecutionEngine().handle_event(state, event)


def test_new_state_and_submission_are_deterministic_and_preserve_plans():
    item = intent()
    engine = ExecutionEngine()
    first = engine.create_state(item)
    second = engine.create_state(item)
    assert first == second
    assert first.lifecycle is ExecutionLifecycle.NEW

    submitted_state, commands = engine.handle_intent(item, first)
    replay_state, replay_commands = engine.handle_intent(item, second)
    assert (submitted_state, commands) == (replay_state, replay_commands)
    assert len(commands) == 1
    assert isinstance(commands[0], SubmitEntryCommand)
    assert commands[0].entry_plan == item.entry_plan
    assert commands[0].protective_plan == item.protective_plan
    assert commands[0].protective_boundary == item.protective_boundary
    assert commands[0].command_id == replay_commands[0].command_id
    assert (
        commands[0].client_idempotency_key == replay_commands[0].client_idempotency_key
    )


def test_duplicate_intent_is_idempotent_but_conflicting_payload_fails_closed():
    item, state, _ = submitted()
    assert ExecutionEngine().handle_intent(item, state) == (state, ())
    with pytest.raises(ExecutionInvariantError, match="different payload"):
        ExecutionEngine().handle_intent(replace(item, quantity=D("9000")), state)


def test_acceptance_and_normal_entry_rejection():
    item, state, command = submitted()
    accepted, no_commands = apply(
        state,
        EntryAccepted(
            "accepted", item.execution_intent_id, command.command_id, "order", NOW
        ),
    )
    assert accepted.lifecycle is ExecutionLifecycle.ACKNOWLEDGED
    assert accepted.broker_order_ref == "order"
    assert no_commands == ()

    _, other_state, other_command = submitted(intent(suffix="reject"))
    rejected, _ = apply(
        other_state,
        EntryRejected(
            "rejected",
            "execution-reject",
            other_command.command_id,
            "margin unavailable",
            NOW,
        ),
    )
    assert rejected.lifecycle is ExecutionLifecycle.REJECTED
    assert rejected.last_failure_reason == "margin unavailable"


def test_partial_fills_weight_average_duplicate_replay_and_protection_path():
    item, state, _ = acknowledged()
    first = EntryFill(
        "fill-1", item.execution_intent_id, "order-1", D("4000"), D("1.1000"), NOW
    )
    state, commands = apply(state, first)
    assert state.filled_quantity == D("4000")
    assert state.remaining_quantity == D("6000")
    assert state.average_fill_price == D("1.1000")
    assert state.lifecycle is ExecutionLifecycle.PROTECTION_PENDING
    assert isinstance(commands[0], EnsureProtectionCommand)

    duplicate_state, duplicate_commands = apply(state, first)
    assert duplicate_state == state
    assert duplicate_commands == ()
    with pytest.raises(ExecutionInvariantError, match="different payload"):
        apply(state, replace(first, price=D("1.2")))

    protection = ProtectionAcknowledged(
        "protected-1",
        item.execution_intent_id,
        commands[0].command_id,
        "order-1",
        "protection-1",
        NOW,
    )
    state, _ = apply(state, protection)
    assert state.lifecycle is ExecutionLifecycle.PARTIALLY_FILLED
    assert state.protection_status is ProtectionStatus.ACTIVE

    state, commands = apply(
        state,
        EntryFill(
            "fill-2",
            item.execution_intent_id,
            "order-1",
            D("6000"),
            D("1.1010"),
            NOW + timedelta(microseconds=1),
        ),
    )
    assert state.lifecycle is ExecutionLifecycle.PROTECTION_PENDING
    assert state.average_fill_price == D("1.1006")
    state, _ = apply(
        state,
        ProtectionAcknowledged(
            "protected-2",
            item.execution_intent_id,
            commands[0].command_id,
            "order-1",
            "protection-2",
            NOW + timedelta(microseconds=1),
        ),
    )
    assert state.lifecycle is ExecutionLifecycle.PROTECTED


def test_full_fill_is_not_safe_until_post_fill_protection_is_acknowledged():
    item, state, _ = acknowledged()
    state, commands = apply(
        state,
        EntryFill(
            "fill", item.execution_intent_id, "order-1", D("10000"), D("1.1"), NOW
        ),
    )
    assert state.lifecycle is ExecutionLifecycle.PROTECTION_PENDING
    assert state.lifecycle is not ExecutionLifecycle.PROTECTED
    state, _ = apply(
        state,
        ProtectionAcknowledged(
            "protected",
            item.execution_intent_id,
            commands[0].command_id,
            "order-1",
            "p-1",
            NOW,
        ),
    )
    assert state.lifecycle is ExecutionLifecycle.PROTECTED


def test_atomic_protection_reaches_protected_without_ensure_command():
    item, state, _ = acknowledged(atomic=True)
    state, commands = apply(
        state,
        EntryFill(
            "fill", item.execution_intent_id, "order-1", D("10000"), D("1.1"), NOW
        ),
    )
    assert state.lifecycle is ExecutionLifecycle.PROTECTED
    assert commands == ()


def test_protection_rejection_is_explicit_unsafe_terminal_error():
    item, state, _ = acknowledged()
    state, commands = apply(
        state,
        EntryFill(
            "fill", item.execution_intent_id, "order-1", D("10000"), D("1.1"), NOW
        ),
    )
    state, _ = apply(
        state,
        ProtectionRejected(
            "protect-reject",
            item.execution_intent_id,
            commands[0].command_id,
            "order-1",
            "invalid stop",
            NOW,
        ),
    )
    assert state.lifecycle is ExecutionLifecycle.ERROR
    assert state.protection_status is ProtectionStatus.FAILED
    assert state.last_failure_reason == "invalid stop"


def test_overfill_wrong_link_and_out_of_order_fill_fail_closed():
    item, submitted_state, _ = submitted()
    fill = EntryFill(
        "fill", item.execution_intent_id, "order-1", D("10001"), D("1.1"), NOW
    )
    with pytest.raises(ExecutionInvariantError, match="out of order"):
        apply(submitted_state, fill)

    _, state, _ = acknowledged()
    with pytest.raises(ExecutionInvariantError, match="exceed"):
        apply(state, fill)
    with pytest.raises(ExecutionInvariantError, match="unrelated broker order"):
        apply(state, replace(fill, quantity=D("1"), broker_order_ref="other"))
    with pytest.raises(ExecutionInvariantError, match="another execution"):
        apply(state, replace(fill, execution_intent_id="execution-other"))


def test_event_contract_rejects_invalid_fill_values_and_non_utc_times():
    with pytest.raises(ValueError, match="positive"):
        EntryFill("fill", "execution", "order", D("0"), D("1"), NOW)
    with pytest.raises(ValueError, match="finite"):
        EntryFill("fill", "execution", "order", D("1"), D("NaN"), NOW)
    with pytest.raises(ValueError, match="UTC"):
        EntryFill(
            "fill", "execution", "order", D("1"), D("1"), NOW.replace(tzinfo=None)
        )


def test_unfilled_cancel_and_terminal_restart_do_not_resubmit():
    item, state, _ = acknowledged()
    state, commands = ExecutionEngine().request_cancel(state, NOW)
    assert state.lifecycle is ExecutionLifecycle.CANCEL_PENDING
    state, _ = apply(
        state,
        CancelAcknowledged(
            "cancelled",
            item.execution_intent_id,
            commands[0].command_id,
            "order-1",
            NOW,
        ),
    )
    assert state.lifecycle is ExecutionLifecycle.CANCELLED
    assert ExecutionEngine().handle_intent(item, state) == (state, ())

    rejected_item, rejected_state, rejected_command = submitted(intent(suffix="r"))
    rejected_state, _ = apply(
        rejected_state,
        EntryRejected(
            "r",
            rejected_item.execution_intent_id,
            rejected_command.command_id,
            "no",
            NOW,
        ),
    )
    assert ExecutionEngine().handle_intent(rejected_item, rejected_state) == (
        rejected_state,
        (),
    )


def test_fully_filled_entry_cannot_be_cancelled_or_have_exposure_erased():
    item, state, _ = acknowledged(atomic=True)
    state, _ = apply(
        state,
        EntryFill("fill", item.execution_intent_id, "order-1", D("10000"), D("1"), NOW),
    )
    with pytest.raises(ExecutionInvariantError, match="cannot be cancelled"):
        ExecutionEngine().request_cancel(state, NOW)
    with pytest.raises(ExecutionInvariantError, match="out of order"):
        apply(
            state,
            CancelAcknowledged("cancel", item.execution_intent_id, "x", "order-1", NOW),
        )


def test_close_is_only_valid_after_exposure_and_terminal_events_do_not_reopen():
    item, state, _ = acknowledged(atomic=True)
    with pytest.raises(ExecutionInvariantError, match="without exposure"):
        apply(
            state,
            ExecutionClosed("close", item.execution_intent_id, "order-1", NOW, "done"),
        )
    state, _ = apply(
        state,
        EntryFill("fill", item.execution_intent_id, "order-1", D("10000"), D("1"), NOW),
    )
    state, _ = apply(
        state,
        ExecutionClosed("close", item.execution_intent_id, "order-1", NOW, "stop"),
    )
    assert state.lifecycle is ExecutionLifecycle.CLOSED
    with pytest.raises(ExecutionInvariantError, match="terminal"):
        apply(
            state,
            EntryFill(
                "later", item.execution_intent_id, "order-1", D("1"), D("1"), NOW
            ),
        )


def test_reconstructed_progress_does_not_submit_duplicate_entry():
    item, acknowledged_state, _ = acknowledged()
    reconstructed = replace(acknowledged_state)
    assert ExecutionEngine().handle_intent(item, reconstructed) == (reconstructed, ())

    item, filled_state, _ = acknowledged(atomic=True)
    filled_state, _ = apply(
        filled_state,
        EntryFill("full", item.execution_intent_id, "order-1", D("10000"), D("1"), NOW),
    )
    for persisted in (replace(filled_state),):
        assert ExecutionEngine().handle_intent(item, persisted) == (persisted, ())


def test_same_instrument_multi_alpha_is_independent_and_order_invariant():
    mr = intent(suffix="mr", sleeve="mean-reversion")
    trend = intent(suffix="trend", sleeve="trend")
    engine = ExecutionEngine()
    forward = [engine.handle_intent(item) for item in (mr, trend)]
    reverse = [engine.handle_intent(item) for item in (trend, mr)]
    assert {state.execution_intent_id for state, _ in forward} == {
        "execution-mr",
        "execution-trend",
    }
    assert {state for state, _ in forward} == {state for state, _ in reverse}
    assert forward[0][1][0].command_id != forward[1][1][0].command_id


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        (FakeBrokerScenario(accept_entry=False), ExecutionLifecycle.REJECTED),
        (
            FakeBrokerScenario(
                fills=(FakeFill(D("10000"), D("1.1")),), atomic_protection=True
            ),
            ExecutionLifecycle.PROTECTED,
        ),
    ],
)
def test_fake_broker_entry_scenarios_are_deterministic(scenario, expected):
    item, state, command = submitted()
    broker = FakeBroker(scenario)
    events = broker.handle(command)
    assert events == FakeBroker(scenario).handle(command)
    for event in events:
        state, followups = apply(state, event)
        for followup in followups:
            for response in broker.handle(followup):
                state, _ = apply(state, response)
    assert state.lifecycle is expected


def test_fake_broker_partial_delayed_fill_and_protection_success_or_failure():
    scenario = FakeBrokerScenario(
        fills=(
            FakeFill(D("4000"), D("1.1000")),
            FakeFill(D("6000"), D("1.1010"), delayed=True),
        )
    )
    item, state, command = submitted()
    broker = FakeBroker(scenario)
    immediate = broker.handle(command)
    assert len(immediate) == 2
    for event in immediate:
        state, commands = apply(state, event)
    assert state.filled_quantity == D("4000")
    protection_event = broker.handle(commands[0])[0]
    state, _ = apply(state, protection_event)
    delayed = broker.release_delayed()
    assert len(delayed) == 1
    state, commands = apply(state, delayed[0])
    assert state.lifecycle is ExecutionLifecycle.PROTECTION_PENDING
    state, _ = apply(state, broker.handle(commands[0])[0])
    assert state.lifecycle is ExecutionLifecycle.PROTECTED

    failing = FakeBroker(FakeBrokerScenario(protection_succeeds=False))
    rejection = failing.handle(
        EnsureProtectionCommand(
            "protect-command",
            item.execution_intent_id,
            "order-1",
            D("1"),
            item.protective_plan,
            item.protective_boundary,
            NOW,
        )
    )[0]
    assert isinstance(rejection, ProtectionRejected)


def test_fake_broker_cancel_success():
    item, state, command = submitted()
    broker = FakeBroker(FakeBrokerScenario())
    state, _ = apply(state, broker.handle(command)[0])
    state, cancel_commands = ExecutionEngine().request_cancel(state, NOW)
    state, _ = apply(state, broker.handle(cancel_commands[0])[0])
    assert state.lifecycle is ExecutionLifecycle.CANCELLED
