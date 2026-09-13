from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from mr_lab.execution import (
    EntryAccepted,
    ExecutionEngine,
    ExecutionLifecycle,
)
from mr_lab.portfolio import PlanReference, ProposalProvenance
from mr_lab.portfolio.identity import stable_id
from mr_lab.risk import ProtectiveBoundary, SizedExecutionIntent
from mr_lab.runtime import (
    BrokerEntryStatus,
    BrokerExecutionRecord,
    BrokerRuntimeSnapshot,
    DataWatermark,
    FreshnessPolicy,
    InMemoryDurableStore,
    ReconciliationCategory,
    RuntimeController,
    RuntimeLifecycle,
    RuntimeReasonCode,
    RuntimeState,
    UnsupportedCheckpointSchema,
    dumps_checkpoint,
    loads_checkpoint,
    make_checkpoint,
    reconcile,
)

NOW = datetime(2024, 7, 1, 9, tzinfo=UTC)
D = Decimal


def intent(suffix: str = "mr", sleeve: str = "mean-reversion") -> SizedExecutionIntent:
    return SizedExecutionIntent(
        f"execution-{suffix}",
        f"risk-{suffix}",
        f"proposal-{suffix}",
        f"opportunity-{suffix}",
        sleeve,
        "EURUSD",
        "LONG",
        D("10000"),
        f"{sleeve}-policy",
        PlanReference("strategy-entry", f"entry-{suffix}"),
        PlanReference("strategy-protection", f"protect-{suffix}"),
        ProposalProvenance("synthetic-test", "1"),
        "risk-v1",
        ProtectiveBoundary(D("1.1000"), D("1.0900"), "risk-v1"),
        NOW - timedelta(seconds=1),
        NOW,
        ("fx",),
    )


def controller(store=None, required=()) -> RuntimeController:
    return RuntimeController(
        "account/runtime",
        store or InMemoryDurableStore(),
        FreshnessPolicy(tuple(required), timedelta(minutes=1), timedelta(minutes=1)),
        NOW,
    )


def submitting(suffix="mr", sleeve="mean-reversion"):
    item = intent(suffix, sleeve)
    state, commands = ExecutionEngine().handle_intent(item)
    return item, state, commands[0]


def accepted(suffix="mr", sleeve="mean-reversion"):
    item, state, command = submitting(suffix, sleeve)
    state, _ = ExecutionEngine().handle_event(
        state,
        EntryAccepted(
            f"accept-{suffix}",
            item.execution_intent_id,
            command.command_id,
            f"order-{suffix}",
            NOW,
        ),
    )
    return item, state


def record(state, **changes):
    values = dict(
        execution_intent_id=state.execution_intent_id,
        observed_at=NOW,
        client_idempotency_key=stable_id(
            "execution-client-key", state.execution_intent_id
        ),
        instrument=state.instrument,
        direction=state.direction,
        broker_order_ref=state.broker_order_ref,
        entry_status=BrokerEntryStatus.OPEN,
        cumulative_filled_quantity=state.filled_quantity,
        average_fill_price=state.average_fill_price,
        exposure_exists=state.filled_quantity > 0,
    )
    values.update(changes)
    return BrokerExecutionRecord(**values)


def test_clean_start_requires_explicit_live_and_persisted_live_is_not_trusted():
    store = InMemoryDurableStore()
    runtime = controller(store)
    assert runtime.state.lifecycle is RuntimeLifecycle.BOOTSTRAP
    runtime.restore()
    assert runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), (), NOW).healthy
    assert runtime.state.lifecycle is RuntimeLifecycle.SYNCED
    assert not runtime.can_open_new_entries
    runtime.enable_live(NOW)
    assert runtime.can_open_new_entries

    restarted = controller(store)
    restarted.restore()
    assert restarted.state.lifecycle is RuntimeLifecycle.BOOTSTRAP
    assert not restarted.can_open_new_entries


def test_checkpoint_round_trip_is_exact_canonical_and_economic():
    _, first, _ = submitting()
    _, second, _ = submitting("trend", "trend")
    base = RuntimeState(
        "key",
        RuntimeLifecycle.LIVE,
        NOW,
        NOW,
        executions=(first, second),
        watermarks=(DataWatermark("z", NOW), DataWatermark("a", NOW)),
    )
    checkpoint = make_checkpoint(base, NOW)
    assert loads_checkpoint(dumps_checkpoint(checkpoint)) == checkpoint
    reordered = make_checkpoint(
        replace(
            base,
            executions=(second, first),
            watermarks=tuple(reversed(base.watermarks)),
        ),
        NOW,
    )
    assert checkpoint.checkpoint_identity == reordered.checkpoint_identity
    changed = make_checkpoint(
        replace(
            base, executions=(replace(first, requested_quantity=D("9000")), second)
        ),
        NOW,
    )
    assert changed.checkpoint_identity != checkpoint.checkpoint_identity


def test_corrupt_and_future_checkpoint_fail_closed():
    store = InMemoryDurableStore()
    runtime = controller(store)
    store.values[runtime.checkpoint_key] = "not-json"
    runtime.restore()
    assert runtime.state.lifecycle is RuntimeLifecycle.HALTED
    assert runtime.state.reason.code is RuntimeReasonCode.CHECKPOINT_CORRUPT

    good = dumps_checkpoint(
        make_checkpoint(RuntimeState("key", RuntimeLifecycle.SYNCED, NOW, NOW), NOW)
    )
    with pytest.raises(UnsupportedCheckpointSchema):
        loads_checkpoint(good.replace('"schema_version":1', '"schema_version":2'))


def test_orphan_missing_terminal_and_conflicts_fail_closed():
    orphan = BrokerExecutionRecord("unknown", NOW, entry_status=BrokerEntryStatus.OPEN)
    result = reconcile((), BrokerRuntimeSnapshot(NOW, (orphan,)))
    assert not result.healthy
    assert result.items[0].category is ReconciliationCategory.ORPHAN_BROKER_EXECUTION

    _, live = accepted()
    assert (
        reconcile((live,), BrokerRuntimeSnapshot(NOW)).items[0].reason_code
        is RuntimeReasonCode.BROKER_EXECUTION_MISSING
    )
    terminal = replace(live, lifecycle=ExecutionLifecycle.CANCELLED)
    assert (
        reconcile((terminal,), BrokerRuntimeSnapshot(NOW, (record(terminal),)))
        .items[0]
        .reason_code
        is RuntimeReasonCode.EXECUTION_CONFLICT
    )
    filled = replace(live, filled_quantity=D("100"), average_fill_price=D("1.1"))
    for internal, bad in (
        (
            live,
            record(
                live, cumulative_filled_quantity=D("11000"), average_fill_price=D("1.1")
            ),
        ),
        (
            filled,
            record(
                filled,
                cumulative_filled_quantity=D("50"),
                average_fill_price=D("1.1"),
            ),
        ),
        (live, record(live, broker_order_ref="other")),
    ):
        assert not reconcile((internal,), BrokerRuntimeSnapshot(NOW, (bad,))).healthy


def test_inflight_is_ambiguous_but_matching_entry_and_fill_recover():
    _, state, _ = submitting()
    absent = reconcile((state,), BrokerRuntimeSnapshot(NOW))
    assert absent.items[0].category is ReconciliationCategory.AMBIGUOUS_IN_FLIGHT
    broker = record(state, broker_order_ref="order-mr")
    recovered = reconcile((state,), BrokerRuntimeSnapshot(NOW, (broker,)))
    assert recovered.healthy
    assert (
        recovered.recovered_executions[0].lifecycle is ExecutionLifecycle.ACKNOWLEDGED
    )

    _, state = accepted()
    ahead = record(
        state,
        cumulative_filled_quantity=D("4000"),
        average_fill_price=D("1.101"),
        exposure_exists=True,
    )
    first = reconcile((state,), BrokerRuntimeSnapshot(NOW, (ahead,)))
    second = reconcile(first.recovered_executions, BrokerRuntimeSnapshot(NOW, (ahead,)))
    assert first.recovered_executions == second.recovered_executions


def test_same_instrument_multi_alpha_and_record_order_are_independent():
    _, mr = accepted()
    _, trend = accepted("trend", "trend")
    records = (record(mr), record(trend))
    one = reconcile((mr, trend), BrokerRuntimeSnapshot(NOW, records))
    two = reconcile((trend, mr), BrokerRuntimeSnapshot(NOW, tuple(reversed(records))))
    assert one == two
    assert len(one.items) == 2


def test_freshness_stale_recovery_pause_halt_and_status():
    runtime = controller(required=("EURUSD:M1",))
    runtime.restore()
    runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), (), NOW)
    assert runtime.state.lifecycle is RuntimeLifecycle.STALE
    runtime.refresh(BrokerRuntimeSnapshot(NOW), (DataWatermark("EURUSD:M1", NOW),), NOW)
    assert runtime.state.lifecycle is RuntimeLifecycle.SYNCED
    runtime.enable_live(NOW)
    runtime.check_freshness(NOW + timedelta(minutes=2))
    assert runtime.state.lifecycle is RuntimeLifecycle.STALE
    assert not runtime.can_open_new_entries
    runtime.refresh(
        BrokerRuntimeSnapshot(NOW + timedelta(minutes=2)),
        (DataWatermark("EURUSD:M1", NOW + timedelta(minutes=2)),),
        NOW + timedelta(minutes=2),
    )
    assert runtime.state.lifecycle is RuntimeLifecycle.SYNCED
    runtime.enable_live(NOW + timedelta(minutes=2))
    runtime.pause_new_entries(NOW + timedelta(minutes=2))
    assert runtime.state.reason.code is RuntimeReasonCode.MANUAL_PAUSE
    runtime.halt("operator stop", NOW + timedelta(minutes=2))
    assert runtime.state.lifecycle is RuntimeLifecycle.HALTED
    assert runtime.status(NOW + timedelta(minutes=2)).active_execution_count == 0


@pytest.mark.parametrize("failure", ["fail_write", "fail_flush"])
def test_persist_before_dispatch_fails_closed(failure):
    store = InMemoryDurableStore()
    runtime = controller(store)
    runtime.restore()
    runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), (), NOW)
    runtime.enable_live(NOW)
    setattr(store, failure, True)
    commands = runtime.submit_intent(ExecutionEngine(), intent(), NOW)
    assert commands == ()
    assert runtime.state.lifecycle is RuntimeLifecycle.HALTED
    assert runtime.state.reason.code is RuntimeReasonCode.PERSISTENCE_FAILURE


def test_state_with_command_identity_is_committed_before_dispatch():
    store = InMemoryDurableStore()
    runtime = controller(store)
    runtime.restore()
    runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), (), NOW)
    runtime.enable_live(NOW)
    commands = runtime.submit_intent(ExecutionEngine(), intent(), NOW)
    persisted = loads_checkpoint(store.read_text(runtime.checkpoint_key))
    assert persisted.executions[0].entry_command_id == commands[0].command_id
    assert persisted.executions[0].lifecycle is ExecutionLifecycle.SUBMITTING
