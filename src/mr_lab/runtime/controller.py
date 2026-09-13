"""Synchronous lifecycle and persist-before-dispatch orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from mr_lab.execution import (
    BrokerEvent,
    ExecutionCommand,
    ExecutionEngine,
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

    @property
    def can_open_new_entries(self) -> bool:
        return self.state.can_open_new_entries

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
        if self.state.lifecycle is RuntimeLifecycle.HALTED:
            return self.reconciliation or ReconciliationResult(
                (), self.state.executions
            )
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
            self._transition(
                RuntimeLifecycle.HALTED,
                now,
                RuntimeReason(first.reason_code, first.detail),
            )
        else:
            reason = self._freshness_reason(now)
            if reason:
                self._transition(RuntimeLifecycle.STALE, now, reason)
            else:
                self._transition(RuntimeLifecycle.SYNCED, now, None, last_synced_at=now)
        return result

    def enable_live(self, now: datetime) -> None:
        if (
            self.state.lifecycle is not RuntimeLifecycle.SYNCED
            or not self.reconciliation
            or not self.reconciliation.healthy
            or self._freshness_reason(now)
            or not self.persistence_healthy
        ):
            raise RuntimeError("runtime is not eligible for LIVE")
        self._transition(RuntimeLifecycle.LIVE, now, None, last_live_at=now)

    def refresh(
        self,
        snapshot: BrokerRuntimeSnapshot,
        watermarks: tuple[DataWatermark, ...],
        now: datetime,
    ) -> None:
        if self.state.lifecycle is RuntimeLifecycle.HALTED:
            return
        prior = self.state.lifecycle
        self.reconcile_startup(snapshot, watermarks, now)
        if (
            prior is RuntimeLifecycle.LIVE
            and self.state.lifecycle is RuntimeLifecycle.SYNCED
        ):
            self._transition(RuntimeLifecycle.LIVE, now, None)

    def check_freshness(self, now: datetime) -> None:
        reason = self._freshness_reason(now)
        if reason and self.state.lifecycle is RuntimeLifecycle.LIVE:
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
        )

    def apply_transition(
        self,
        transition: Callable[[], tuple[ExecutionState, tuple[ExecutionCommand, ...]]],
        now: datetime,
    ) -> tuple[ExecutionCommand, ...]:
        new_state, commands = transition()
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
        if not self.can_open_new_entries:
            raise RuntimeError("new entries are disabled")
        existing = next(
            (
                item
                for item in self.state.executions
                if item.execution_intent_id == intent.execution_intent_id
            ),
            None,
        )
        return self.apply_transition(
            lambda: engine.handle_intent(intent, existing), now
        )

    def process_event(
        self,
        engine: ExecutionEngine,
        state: ExecutionState,
        event: BrokerEvent,
        now: datetime,
    ) -> tuple[ExecutionCommand, ...]:
        return self.apply_transition(lambda: engine.handle_event(state, event), now)

    def status(self, now: datetime) -> RuntimeStatus:
        active, unsafe, inflight = execution_counts(self.state.executions)
        marks = {item.source_id: item.event_time for item in self.state.watermarks}
        source_freshness = tuple(
            (
                source,
                source in marks
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
            self.can_open_new_entries,
            self.state.last_synced_at,
            age,
            self.state.reason,
            active,
            unsafe,
            inflight,
            source_freshness,
        )

    def _freshness_reason(self, now: datetime) -> RuntimeReason | None:
        if (
            self.state.broker_snapshot_time is None
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
            basis, lifecycle=RuntimeLifecycle.HALTED, reason=RuntimeReason(code, detail)
        )
