from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from mr_lab.execution import (
    EntryAccepted,
    EntryFill,
    ExecutionEngine,
    ExecutionLifecycle,
    ProtectionAcknowledged,
    ProtectionStatus,
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
    if "cumulative_filled_quantity" in changes and "exposure_exists" not in changes:
        values["exposure_exists"] = changes["cumulative_filled_quantity"] > 0
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
        loads_checkpoint(good.replace('"schema_version":3', '"schema_version":4'))


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


def test_cancel_pending_racing_fill_recovers_complete_truth_and_is_idempotent():
    _, state = accepted()
    state, _ = ExecutionEngine().request_cancel(state, NOW)
    broker = record(
        state,
        entry_status=BrokerEntryStatus.CANCELLED,
        cumulative_filled_quantity=D("4000"),
        average_fill_price=D("1.1015"),
        exposure_exists=True,
    )
    first = reconcile((state,), BrokerRuntimeSnapshot(NOW, (broker,)))
    recovered = first.recovered_executions[0]
    assert not first.healthy  # exposure still needs proven protection
    assert recovered.filled_quantity == D("4000")
    assert recovered.average_fill_price == D("1.1015")
    assert recovered.entry_order_status.value == "CANCELLED"
    assert recovered.live_entry_quantity == 0
    assert recovered.lifecycle is ExecutionLifecycle.PROTECTION_PENDING
    assert recovered.lifecycle is not ExecutionLifecycle.CANCELLED
    replay = reconcile((recovered,), BrokerRuntimeSnapshot(NOW, (broker,)))
    assert replay.recovered_executions == (recovered,)


def test_protection_pending_broker_ahead_fill_requires_complete_protection():
    item, state = accepted()
    state, _ = ExecutionEngine().handle_event(
        state,
        EntryFill(
            "fill-1", item.execution_intent_id, "order-mr", D("4000"), D("1.1"), NOW
        ),
    )
    partial = record(
        state,
        entry_status=BrokerEntryStatus.FILLED,
        cumulative_filled_quantity=D("10000"),
        average_fill_price=D("1.12"),
        exposure_exists=True,
        protection_active=True,
        protected_quantity=D("4000"),
        protection_ref="protection-1",
    )
    result = reconcile((state,), BrokerRuntimeSnapshot(NOW, (partial,)))
    assert not result.healthy
    recovered = result.recovered_executions[0]
    assert recovered.filled_quantity == D("10000")
    assert recovered.protected_quantity == D("4000")
    assert recovered.pending_protection_quantity == D("10000")
    assert reconcile(
        (recovered,), BrokerRuntimeSnapshot(NOW, (partial,))
    ).recovered_executions == (recovered,)

    complete = replace(partial, protected_quantity=D("10000"))
    healthy = reconcile((state,), BrokerRuntimeSnapshot(NOW, (complete,)))
    assert healthy.healthy
    recovered = healthy.recovered_executions[0]
    assert recovered.filled_quantity == recovered.protected_quantity == D("10000")
    assert recovered.protection_status is ProtectionStatus.ACTIVE
    assert recovered.lifecycle is ExecutionLifecycle.PROTECTED
    assert reconcile(
        (recovered,), BrokerRuntimeSnapshot(NOW, (complete,))
    ).recovered_executions == (recovered,)

    missing_price = replace(complete, average_fill_price=None)
    assert not reconcile((state,), BrokerRuntimeSnapshot(NOW, (missing_price,))).healthy


def test_matched_requires_order_exposure_price_and_protection_equality():
    item, state = accepted()
    state, commands = ExecutionEngine().handle_event(
        state,
        EntryFill(
            "fill", item.execution_intent_id, "order-mr", D("10000"), D("1.1"), NOW
        ),
    )
    state, _ = ExecutionEngine().handle_event(
        state,
        ProtectionAcknowledged(
            "protected",
            item.execution_intent_id,
            commands[0].command_id,
            "order-mr",
            "protection",
            NOW,
        ),
    )
    matching = record(
        state,
        entry_status=BrokerEntryStatus.FILLED,
        exposure_exists=True,
        protection_active=True,
        protected_quantity=D("10000"),
        protection_ref="protection",
    )
    assert reconcile((state,), BrokerRuntimeSnapshot(NOW, (matching,))).healthy
    mismatches = (
        replace(matching, entry_status=BrokerEntryStatus.CANCELLED),
        replace(matching, exposure_exists=False),
        replace(matching, average_fill_price=D("1.1001")),
        replace(matching, protected_quantity=D("4000")),
        replace(matching, protection_ref="different"),
    )
    assert all(
        not reconcile((state,), BrokerRuntimeSnapshot(NOW, (value,))).healthy
        for value in mismatches
    )

    _, open_state = accepted("open")
    cancelled = record(open_state, entry_status=BrokerEntryStatus.CANCELLED)
    assert not reconcile(
        (open_state,), BrokerRuntimeSnapshot(NOW, (cancelled,))
    ).healthy
    internally_cancelled = replace(
        open_state, entry_order_status=open_state.entry_order_status.CANCELLED
    )
    assert not reconcile(
        (internally_cancelled,), BrokerRuntimeSnapshot(NOW, (record(open_state),))
    ).healthy


def test_terminal_records_are_state_specific():
    _, state, _ = submitting("rejected")
    rejected = replace(
        state,
        lifecycle=ExecutionLifecycle.REJECTED,
        entry_order_status=state.entry_order_status.REJECTED,
    )
    broker = record(rejected, entry_status=BrokerEntryStatus.REJECTED)
    assert reconcile((rejected,), BrokerRuntimeSnapshot(NOW, (broker,))).healthy
    filled_terminal = replace(
        broker,
        cumulative_filled_quantity=D("1"),
        average_fill_price=D("1.1"),
        exposure_exists=False,
    )
    assert not reconcile(
        (rejected,), BrokerRuntimeSnapshot(NOW, (filled_terminal,))
    ).healthy


def test_controller_uses_authoritative_state_and_unknown_event_halts():
    store = InMemoryDurableStore()
    runtime = controller(store)
    runtime.restore()
    runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), (), NOW)
    runtime.enable_live(NOW)
    command = runtime.submit_intent(ExecutionEngine(), intent(), NOW)[0]
    runtime.process_event(
        ExecutionEngine(),
        EntryAccepted("accept", "execution-mr", command.command_id, "order-mr", NOW),
        NOW,
    )
    stale = runtime.state.executions[0]
    runtime.process_event(
        ExecutionEngine(),
        EntryFill("fill-1", "execution-mr", "order-mr", D("4000"), D("1.1"), NOW),
        NOW,
    )
    runtime.process_event(
        ExecutionEngine(),
        EntryFill("fill-2", "execution-mr", "order-mr", D("1000"), D("1.2"), NOW),
        NOW,
    )
    current = runtime.state.executions[0]
    assert stale.filled_quantity == 0
    assert current.filled_quantity == D("5000")
    assert tuple(event.event_id for event in current.processed_events) == (
        "accept",
        "fill-1",
        "fill-2",
    )

    runtime.process_event(
        ExecutionEngine(),
        EntryAccepted("unknown", "not-owned", "command", "order", NOW),
        NOW,
    )
    assert runtime.state.lifecycle is RuntimeLifecycle.HALTED
    assert runtime.state.reason.code is RuntimeReasonCode.EXECUTION_CONFLICT


def test_manual_halt_is_a_persisted_operator_latch():
    store = InMemoryDurableStore()
    runtime = controller(store)
    runtime.restore()
    runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), (), NOW)
    runtime.halt("operator investigation", NOW)

    restarted = controller(store)
    restarted.restore()
    assert restarted.state.lifecycle is RuntimeLifecycle.BOOTSTRAP
    assert restarted.state.halt_latched
    assert restarted.state.reason.detail == "operator investigation"
    assert restarted.state.halt_reason.detail == "operator investigation"
    restarted.reconcile_startup(
        BrokerRuntimeSnapshot(NOW), (), NOW + timedelta(minutes=2)
    )
    assert restarted.state.lifecycle is RuntimeLifecycle.STALE
    assert restarted.state.reason.code is RuntimeReasonCode.STALE_BROKER_SNAPSHOT
    assert restarted.state.halt_reason.detail == "operator investigation"
    orphan_time = NOW + timedelta(minutes=2)
    restarted.reconcile_startup(
        BrokerRuntimeSnapshot(
            orphan_time,
            (
                BrokerExecutionRecord(
                    "orphan", orphan_time, entry_status=BrokerEntryStatus.OPEN
                ),
            ),
        ),
        (),
        orphan_time,
    )
    assert restarted.state.lifecycle is RuntimeLifecycle.HALTED
    assert restarted.state.halt_reason.detail == "operator investigation"
    restarted.reconcile_startup(BrokerRuntimeSnapshot(orphan_time), (), orphan_time)
    assert restarted.state.lifecycle is RuntimeLifecycle.HALTED
    status = restarted.status(orphan_time)
    assert status.halt_latched
    assert status.halt_reason.detail == "operator investigation"
    with pytest.raises(RuntimeError, match="not eligible"):
        restarted.enable_live(orphan_time)
    restarted.clear_halt(orphan_time)
    assert restarted.state.lifecycle is RuntimeLifecycle.SYNCED
    assert restarted.state.halt_reason is None
    assert not restarted.can_open_new_entries
    restarted.enable_live(orphan_time)
    assert restarted.can_open_new_entries


def test_future_freshness_is_stale_and_synced_freshness_loss_is_reported():
    runtime = controller(required=("feed",))
    future = NOW + timedelta(seconds=1)
    runtime.restore()
    runtime.reconcile_startup(
        BrokerRuntimeSnapshot(future), (DataWatermark("feed", future),), NOW
    )
    assert runtime.state.lifecycle is RuntimeLifecycle.STALE
    runtime.refresh(BrokerRuntimeSnapshot(NOW), (DataWatermark("feed", NOW),), NOW)
    assert runtime.state.lifecycle is RuntimeLifecycle.SYNCED
    runtime.check_freshness(NOW + timedelta(minutes=2))
    assert runtime.state.lifecycle is RuntimeLifecycle.STALE


def test_protection_and_cancel_commands_are_persisted_before_dispatch():
    store = InMemoryDurableStore()
    runtime = controller(store)
    runtime.restore()
    runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), (), NOW)
    runtime.enable_live(NOW)
    submit = runtime.submit_intent(ExecutionEngine(), intent(), NOW)[0]
    runtime.process_event(
        ExecutionEngine(),
        EntryAccepted("accepted", "execution-mr", submit.command_id, "order-mr", NOW),
        NOW,
    )
    protection = runtime.process_event(
        ExecutionEngine(),
        EntryFill("fill", "execution-mr", "order-mr", D("4000"), D("1.1"), NOW),
        NOW,
    )[0]
    persisted = loads_checkpoint(store.read_text(runtime.checkpoint_key)).executions[0]
    assert persisted.protection_command_id == protection.command_id
    assert persisted.pending_protection_quantity == D("4000")

    other = intent("cancel")
    submit = runtime.submit_intent(ExecutionEngine(), other, NOW)[0]
    runtime.process_event(
        ExecutionEngine(),
        EntryAccepted(
            "accepted-cancel",
            "execution-cancel",
            submit.command_id,
            "order-cancel",
            NOW,
        ),
        NOW,
    )
    cancel = runtime.request_cancel(ExecutionEngine(), "execution-cancel", NOW)[0]
    persisted = loads_checkpoint(store.read_text(runtime.checkpoint_key))
    cancelled = next(
        item
        for item in persisted.executions
        if item.execution_intent_id == "execution-cancel"
    )
    assert cancelled.cancel_command_id == cancel.command_id
    assert cancelled.lifecycle is ExecutionLifecycle.CANCEL_PENDING


def _live_runtime(
    *, required=(), source_age=timedelta(minutes=1), broker_age=timedelta(minutes=1)
):
    runtime = RuntimeController(
        "admission/runtime",
        InMemoryDurableStore(),
        FreshnessPolicy(tuple(required), source_age, broker_age),
        NOW,
    )
    watermarks = tuple(DataWatermark(source, NOW) for source in required)
    runtime.restore()
    runtime.reconcile_startup(BrokerRuntimeSnapshot(NOW), watermarks, NOW)
    runtime.enable_live(NOW)
    return runtime


def test_submit_rechecks_broker_freshness_without_heartbeat():
    runtime = _live_runtime()
    with pytest.raises(RuntimeError, match="stale"):
        runtime.submit_intent(
            ExecutionEngine(), intent("late-broker"), NOW + timedelta(minutes=2)
        )
    assert runtime.state.executions == ()
    assert runtime.state.lifecycle is RuntimeLifecycle.STALE
    assert runtime.state.reason.code is RuntimeReasonCode.STALE_BROKER_SNAPSHOT


@pytest.mark.parametrize("missing", [False, True])
def test_submit_rechecks_required_source_freshness_without_heartbeat(missing):
    runtime = _live_runtime(
        required=("feed",),
        source_age=timedelta(minutes=1),
        broker_age=timedelta(minutes=5),
    )
    if missing:
        runtime.state = replace(runtime.state, watermarks=())
    with pytest.raises(RuntimeError, match="stale"):
        runtime.submit_intent(
            ExecutionEngine(),
            intent(f"late-source-{missing}"),
            NOW + timedelta(minutes=2),
        )
    assert runtime.state.executions == ()
    assert runtime.state.lifecycle is RuntimeLifecycle.STALE
    assert runtime.state.reason.code is RuntimeReasonCode.STALE_REQUIRED_SOURCE


def test_submit_rejects_future_freshness_and_fresh_submission_still_commits():
    future = NOW + timedelta(seconds=1)
    runtime = _live_runtime(required=("feed",))
    runtime.state = replace(
        runtime.state,
        broker_snapshot_time=future,
        watermarks=(DataWatermark("feed", future),),
    )
    with pytest.raises(RuntimeError, match="stale"):
        runtime.submit_intent(ExecutionEngine(), intent("future"), NOW)
    assert runtime.state.executions == ()
    assert runtime.state.reason.code is RuntimeReasonCode.STALE_BROKER_SNAPSHOT

    fresh = _live_runtime()
    commands = fresh.submit_intent(ExecutionEngine(), intent("fresh"), NOW)
    checkpoint = loads_checkpoint(fresh.store.read_text(fresh.checkpoint_key))
    assert checkpoint.executions[0].entry_command_id == commands[0].command_id


def test_runtime_status_reports_effective_freshness_without_mutating():
    runtime = _live_runtime()
    status = runtime.status(NOW + timedelta(minutes=2))
    assert runtime.state.lifecycle is RuntimeLifecycle.LIVE
    assert not status.can_open_new_entries


def test_terminal_entry_protection_cannot_exceed_actual_exposure():
    with pytest.raises(ValueError, match="cannot exceed cumulative exposure"):
        BrokerExecutionRecord(
            "cancelled-fill",
            NOW,
            entry_status=BrokerEntryStatus.CANCELLED,
            cumulative_filled_quantity=D("4000"),
            average_fill_price=D("1.1"),
            exposure_exists=True,
            protection_active=True,
            protected_quantity=D("10000"),
            protection_ref="protection",
        )
    open_record = BrokerExecutionRecord(
        "open-atomic",
        NOW,
        entry_status=BrokerEntryStatus.OPEN,
        cumulative_filled_quantity=D("4000"),
        average_fill_price=D("1.1"),
        exposure_exists=True,
        protection_active=True,
        protected_quantity=D("10000"),
        protection_ref="atomic-protection",
    )
    assert open_record.protected_quantity == D("10000")
