"""Synchronous lifecycle and persist-before-dispatch orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from mr_lab.execution import (
    BrokerEvent,
    ExecutionCommand,
    ExecutionEngine,
    ExecutionInvariantError,
    ExecutionState,
)
from mr_lab.risk import SizedExecutionIntent

from .checkpoint import (
    CheckpointError,
    UnsupportedCheckpointSchema,
    dumps_checkpoint,
    loads_checkpoint,
    make_checkpoint,
)
from .contracts import (
    BrokerRuntimeSnapshot,
    DataWatermark,
    FreshnessPolicy,
    RuntimeLifecycle,
    RuntimeReason,
    RuntimeReasonCode,
    RuntimeState,
    RuntimeStatus,
    execution_counts,
)
from .persistence import DurableStore
from .reconcile import ReconciliationResult, reconcile


class RuntimeController:
    def __init__(
        self,
        runtime_key: str,
        store: DurableStore,
        policy: FreshnessPolicy,
        started_at: datetime,
        *,
        checkpoint_key: str | None = None,
    ) -> None:
        self.store, self.policy = store, policy
        self.checkpoint_key = checkpoint_key or f"runtime:{runtime_key}:checkpoint"
        self.state = RuntimeState(
            runtime_key, RuntimeLifecycle.BOOTSTRAP, started_at, started_at
        )
        self.reconciliation: ReconciliationResult | None = None
        self.persistence_healthy = True

    def can_open_new_entries(self, now: datetime) -> bool:
        """Return the complete, time-aware entry admission decision."""
        return self._entry_eligible(now)

    def restore(self) -> RuntimeState:
        try:
            payload = self.store.read_text(self.checkpoint_key)
            if payload is not None:
                checkpoint = loads_checkpoint(payload)
                if checkpoint.runtime_key != self.state.runtime_key:
                    raise CheckpointError("runtime key mismatch")
                self.state = replace(
                    self.state,
                    checkpoint_sequence=checkpoint.checkpoint_sequence,
                    reason=checkpoint.runtime_reason,
                    halt_latched=checkpoint.halt_latched,
                    halt_reason=checkpoint.halt_reason,
                    executions=checkpoint.executions,
                    watermarks=checkpoint.watermarks,
                )
        except UnsupportedCheckpointSchema as exc:
            self._halt_without_persist(
                RuntimeReasonCode.CHECKPOINT_SCHEMA_UNSUPPORTED, str(exc)
            )
        except Exception as exc:
            self._halt_without_persist(RuntimeReasonCode.CHECKPOINT_CORRUPT, str(exc))
        return self.state

    def reconcile_startup(
        self,
        snapshot: BrokerRuntimeSnapshot,
        watermarks: tuple[DataWatermark, ...],
        now: datetime,
    ) -> ReconciliationResult:
        return self._reconcile(snapshot, watermarks, now, preserve_live=False)

    def _reconcile(
        self,
        snapshot: BrokerRuntimeSnapshot,
        watermarks: tuple[DataWatermark, ...],
        now: datetime,
        *,
        preserve_live: bool,
    ) -> ReconciliationResult:
        result = reconcile(self.state.executions, snapshot)
        self.reconciliation = result
        self.state = replace(
            self.state,
            executions=result.recovered_executions,
            broker_snapshot_time=snapshot.observed_at,
            watermarks=watermarks,
            updated_at=now,
        )
        if not result.healthy:
            first = next(item for item in result.items if item.reason_code is not None)
            diagnostic = RuntimeReason(first.reason_code, first.detail)
            self._transition(
                RuntimeLifecycle.HALTED,
                now,
                diagnostic,
                halt_latched=True,
                halt_reason=self.state.halt_reason or diagnostic,
            )
        else:
            reason = self._freshness_reason(now)
            if reason:
                self._transition(RuntimeLifecycle.STALE, now, reason)
            elif self.state.halt_latched:
                lifecycle = (
                    RuntimeLifecycle.HALTED
                    if self.state.lifecycle is RuntimeLifecycle.HALTED
                    else RuntimeLifecycle.BOOTSTRAP
                )
                self._transition(lifecycle, now, None)
            else:
                lifecycle = (
                    RuntimeLifecycle.LIVE if preserve_live else RuntimeLifecycle.SYNCED
                )
                self._transition(lifecycle, now, None, last_synced_at=now)
        return result

    def enable_live(self, now: datetime) -> None:
        if (
            self.state.lifecycle is not RuntimeLifecycle.SYNCED
            or not self.reconciliation
            or not self.reconciliation.healthy
            or self._freshness_reason(now)
            or not self.persistence_healthy
            or self.state.halt_latched
        ):
            raise RuntimeError("runtime is not eligible for LIVE")
        self._transition(RuntimeLifecycle.LIVE, now, None, last_live_at=now)

    def refresh(
        self,
        snapshot: BrokerRuntimeSnapshot,
        watermarks: tuple[DataWatermark, ...],
        now: datetime,
    ) -> None:
        if (
            self.state.lifecycle is RuntimeLifecycle.HALTED
            and not self.state.halt_latched
        ):
            return
        self._reconcile(
            snapshot,
            watermarks,
            now,
            preserve_live=self.state.lifecycle is RuntimeLifecycle.LIVE,
        )

    def check_freshness(self, now: datetime) -> None:
        reason = self._freshness_reason(now)
        if reason and self.state.lifecycle in {
            RuntimeLifecycle.LIVE,
            RuntimeLifecycle.SYNCED,
        }:
            self._transition(RuntimeLifecycle.STALE, now, reason)

    def pause_new_entries(self, now: datetime) -> None:
        if self.state.lifecycle is RuntimeLifecycle.LIVE:
            self._transition(
                RuntimeLifecycle.SYNCED,
                now,
                RuntimeReason(RuntimeReasonCode.MANUAL_PAUSE, "manual new-entry pause"),
            )

    def halt(self, reason: str, now: datetime) -> None:
        self._transition(
            RuntimeLifecycle.HALTED,
            now,
            RuntimeReason(RuntimeReasonCode.MANUAL_HALT, reason),
            halt_latched=True,
            halt_reason=RuntimeReason(RuntimeReasonCode.MANUAL_HALT, reason),
        )

    def clear_halt(self, now: datetime) -> None:
        if (
            not self.state.halt_latched
            or not self.reconciliation
            or not self.reconciliation.healthy
            or self._freshness_reason(now)
            or not self.persistence_healthy
        ):
            raise RuntimeError("halt cannot be cleared before healthy reconciliation")
        self._transition(
            RuntimeLifecycle.SYNCED,
            now,
            None,
            halt_latched=False,
            halt_reason=None,
            last_synced_at=now,
        )

    def _commit_execution_transition(
        self,
        current: ExecutionState | None,
        transition: tuple[ExecutionState, tuple[ExecutionCommand, ...]],
        now: datetime,
    ) -> tuple[ExecutionCommand, ...]:
        new_state, commands = transition
        if (
            current is not None
            and new_state.execution_intent_id != current.execution_intent_id
        ):
            self._transition(
                RuntimeLifecycle.HALTED,
                now,
                RuntimeReason(
                    RuntimeReasonCode.EXECUTION_CONFLICT,
                    "execution transition changed identity",
                ),
                halt_latched=True,
                halt_reason=self.state.halt_reason
                or RuntimeReason(
                    RuntimeReasonCode.EXECUTION_CONFLICT,
                    "execution transition changed identity",
                ),
            )
            return ()
        executions = {item.execution_intent_id: item for item in self.state.executions}
        executions[new_state.execution_intent_id] = new_state
        candidate = replace(
            self.state, executions=tuple(executions.values()), updated_at=now
        )
        if not self._persist(candidate, now):
            return ()
        return commands

    def submit_intent(
        self, engine: ExecutionEngine, intent: SizedExecutionIntent, now: datetime
    ) -> tuple[ExecutionCommand, ...]:
        freshness = self._freshness_reason(now)
        if self.state.lifecycle is RuntimeLifecycle.LIVE and freshness is not None:
            self._transition(RuntimeLifecycle.STALE, now, freshness)
            raise RuntimeError("new entries are disabled by stale runtime data")
        if not self._entry_eligible(now):
            raise RuntimeError("new entries are disabled")
        existing = next(
            (
                item
                for item in self.state.executions
                if item.execution_intent_id == intent.execution_intent_id
            ),
            None,
        )
        try:
            transition = engine.handle_intent(intent, existing)
        except ExecutionInvariantError as exc:
            self._halt_execution_invariant(exc, now)
            raise
        return self._commit_execution_transition(existing, transition, now)

    def process_event(
        self,
        engine: ExecutionEngine,
        event: BrokerEvent,
        now: datetime,
    ) -> tuple[ExecutionCommand, ...]:
        current = next(
            (
                item
                for item in self.state.executions
                if item.execution_intent_id == event.execution_intent_id
            ),
            None,
        )
        if current is None:
            self._transition(
                RuntimeLifecycle.HALTED,
                now,
                RuntimeReason(
                    RuntimeReasonCode.EXECUTION_CONFLICT,
                    "broker event has no owned execution",
                ),
                halt_latched=True,
                halt_reason=self.state.halt_reason
                or RuntimeReason(
                    RuntimeReasonCode.EXECUTION_CONFLICT,
                    "broker event has no owned execution",
                ),
            )
            return ()
        try:
            transition = engine.handle_event(current, event)
        except ExecutionInvariantError as exc:
            self._halt_execution_invariant(exc, now)
            raise
        return self._commit_execution_transition(current, transition, now)

    def request_cancel(
        self, engine: ExecutionEngine, execution_intent_id: str, now: datetime
    ) -> tuple[ExecutionCommand, ...]:
        current = next(
            (
                item
                for item in self.state.executions
                if item.execution_intent_id == execution_intent_id
            ),
            None,
        )
        if current is None:
            raise RuntimeError("execution is not owned by runtime")
        try:
            transition = engine.request_cancel(current, now)
        except ExecutionInvariantError as exc:
            self._halt_execution_invariant(exc, now)
            raise
        return self._commit_execution_transition(current, transition, now)

    def _halt_execution_invariant(
        self, error: ExecutionInvariantError, now: datetime
    ) -> None:
        reason = RuntimeReason(RuntimeReasonCode.EXECUTION_CONFLICT, str(error))
        self._transition(
            RuntimeLifecycle.HALTED,
            now,
            reason,
            halt_latched=True,
            halt_reason=self.state.halt_reason or reason,
        )

    def status(self, now: datetime) -> RuntimeStatus:
        active, unsafe, inflight = execution_counts(self.state.executions)
        marks = {item.source_id: item.event_time for item in self.state.watermarks}
        source_freshness = tuple(
            (
                source,
                source in marks
                and marks[source] <= now
                and now - marks[source] <= self.policy.maximum_source_age,
            )
            for source in self.policy.required_source_ids
        )
        age = (
            None
            if self.state.broker_snapshot_time is None
            else now - self.state.broker_snapshot_time
        )
        return RuntimeStatus(
            self.state.lifecycle,
            self._entry_eligible(now),
            self.state.last_synced_at,
            age,
            self.state.reason,
            self.state.halt_latched,
            self.state.halt_reason,
            active,
            unsafe,
            inflight,
            source_freshness,
        )

    def _entry_eligible(self, now: datetime) -> bool:
        return (
            self.state.lifecycle is RuntimeLifecycle.LIVE
            and not self.state.halt_latched
            and self.persistence_healthy
            and self.reconciliation is not None
            and self.reconciliation.healthy
            and self._freshness_reason(now) is None
        )

    def _freshness_reason(self, now: datetime) -> RuntimeReason | None:
        if (
            self.state.broker_snapshot_time is None
            or self.state.broker_snapshot_time > now
            or now - self.state.broker_snapshot_time
            > self.policy.maximum_broker_snapshot_age
        ):
            return RuntimeReason(
                RuntimeReasonCode.STALE_BROKER_SNAPSHOT,
                "broker snapshot is missing or stale",
            )
        marks = {item.source_id: item.event_time for item in self.state.watermarks}
        stale = [
            source
            for source in self.policy.required_source_ids
            if source not in marks
            or marks[source] > now
            or now - marks[source] > self.policy.maximum_source_age
        ]
        return (
            RuntimeReason(
                RuntimeReasonCode.STALE_REQUIRED_SOURCE,
                f"missing or stale sources: {','.join(stale)}",
            )
            if stale
            else None
        )

    def _transition(
        self,
        lifecycle: RuntimeLifecycle,
        now: datetime,
        reason: RuntimeReason | None,
        **changes: object,
    ) -> None:
        candidate = replace(
            self.state, lifecycle=lifecycle, updated_at=now, reason=reason, **changes
        )
        self._persist(candidate, now)

    def _persist(self, candidate: RuntimeState, now: datetime) -> bool:
        try:
            checkpoint = make_checkpoint(candidate, now)
            self.store.write_text(self.checkpoint_key, dumps_checkpoint(checkpoint))
            self.store.flush()
            self.state = replace(
                candidate, checkpoint_sequence=checkpoint.checkpoint_sequence
            )
            return True
        except Exception as exc:
            self.persistence_healthy = False
            self._halt_without_persist(
                RuntimeReasonCode.PERSISTENCE_FAILURE, str(exc), candidate
            )
            return False

    def _halt_without_persist(
        self,
        code: RuntimeReasonCode,
        detail: str,
        candidate: RuntimeState | None = None,
    ) -> None:
        basis = candidate or self.state
        self.state = replace(
            basis,
            lifecycle=RuntimeLifecycle.HALTED,
            reason=RuntimeReason(code, detail),
            halt_latched=True,
            halt_reason=basis.halt_reason or RuntimeReason(code, detail),
        )
