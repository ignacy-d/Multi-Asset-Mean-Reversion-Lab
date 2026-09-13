"""Pure, fail-closed comparison of persisted and normalized broker state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from mr_lab.execution import (
    EntryOrderStatus,
    ExecutionLifecycle,
    ExecutionState,
    ProtectionStatus,
)
from mr_lab.portfolio.identity import stable_id

from .contracts import (
    BrokerEntryStatus,
    BrokerExecutionRecord,
    BrokerRuntimeSnapshot,
    RuntimeReasonCode,
)


class ReconciliationCategory(StrEnum):
    MATCHED = "MATCHED"
    BROKER_AHEAD = "BROKER_AHEAD"
    BROKER_MISSING = "BROKER_MISSING"
    ORPHAN_BROKER_EXECUTION = "ORPHAN_BROKER_EXECUTION"
    CONFLICT = "CONFLICT"
    AMBIGUOUS_IN_FLIGHT = "AMBIGUOUS_IN_FLIGHT"


@dataclass(frozen=True, slots=True)
class ReconciliationItem:
    execution_intent_id: str
    category: ReconciliationCategory
    reason_code: RuntimeReasonCode | None
    detail: str


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    items: tuple[ReconciliationItem, ...]
    recovered_executions: tuple[ExecutionState, ...]

    @property
    def healthy(self) -> bool:
        return all(
            item.category
            in {ReconciliationCategory.MATCHED, ReconciliationCategory.BROKER_AHEAD}
            for item in self.items
        )


_IN_FLIGHT = {
    ExecutionLifecycle.SUBMITTING,
    ExecutionLifecycle.PROTECTION_PENDING,
    ExecutionLifecycle.CANCEL_PENDING,
}


def _conflict(
    state: ExecutionState, detail: str
) -> tuple[ReconciliationItem, ExecutionState]:
    return ReconciliationItem(
        state.execution_intent_id,
        ReconciliationCategory.CONFLICT,
        RuntimeReasonCode.EXECUTION_CONFLICT,
        detail,
    ), state


def _compare(
    state: ExecutionState, broker: BrokerExecutionRecord | None
) -> tuple[ReconciliationItem, ExecutionState]:
    if broker is None:
        if state.lifecycle is ExecutionLifecycle.NEW or state.is_terminal:
            return ReconciliationItem(
                state.execution_intent_id,
                ReconciliationCategory.MATCHED,
                None,
                "no broker state expected",
            ), state
        if state.lifecycle in _IN_FLIGHT:
            return ReconciliationItem(
                state.execution_intent_id,
                ReconciliationCategory.AMBIGUOUS_IN_FLIGHT,
                RuntimeReasonCode.AMBIGUOUS_IN_FLIGHT,
                "persisted command has no conclusive broker proof",
            ), state
        return ReconciliationItem(
            state.execution_intent_id,
            ReconciliationCategory.BROKER_MISSING,
            RuntimeReasonCode.BROKER_EXECUTION_MISSING,
            "broker execution is missing",
        ), state
    if state.is_terminal and broker.is_live:
        return _conflict(state, "terminal internal execution is live at broker")
    if broker.instrument is not None and broker.instrument != state.instrument:
        return _conflict(state, "instrument differs for execution identity")
    if broker.direction is not None and broker.direction != state.direction:
        return _conflict(state, "direction differs for execution identity")
    if broker.cumulative_filled_quantity > state.requested_quantity:
        return _conflict(state, "broker fill exceeds requested quantity")
    if broker.cumulative_filled_quantity < state.filled_quantity:
        return _conflict(state, "broker fill regresses persisted quantity")
    if state.broker_order_ref and broker.broker_order_ref != state.broker_order_ref:
        return _conflict(state, "broker order identity conflicts")
    if (
        state.protection_status is ProtectionStatus.ACTIVE
        and not broker.protection_active
    ):
        return _conflict(state, "broker reports persisted protection absent")
    if state.lifecycle is ExecutionLifecycle.SUBMITTING:
        expected_key = stable_id("execution-client-key", state.execution_intent_id)
        if (
            broker.client_idempotency_key != expected_key
            or broker.entry_status is not BrokerEntryStatus.OPEN
            or not broker.broker_order_ref
        ):
            return ReconciliationItem(
                state.execution_intent_id,
                ReconciliationCategory.AMBIGUOUS_IN_FLIGHT,
                RuntimeReasonCode.AMBIGUOUS_IN_FLIGHT,
                "entry operation is not conclusively applied",
            ), state
        recovered = replace(
            state,
            lifecycle=ExecutionLifecycle.ACKNOWLEDGED,
            entry_order_status=EntryOrderStatus.OPEN,
            broker_order_ref=broker.broker_order_ref,
            updated_at=max(state.updated_at, broker.observed_at),
        )
        return ReconciliationItem(
            state.execution_intent_id,
            ReconciliationCategory.BROKER_AHEAD,
            None,
            "matching deterministic entry is accepted",
        ), recovered
    if state.lifecycle in {
        ExecutionLifecycle.PROTECTION_PENDING,
        ExecutionLifecycle.CANCEL_PENDING,
    }:
        applied = (
            state.lifecycle is ExecutionLifecycle.PROTECTION_PENDING
            and broker.protection_active
            and broker.protected_quantity >= state.pending_protection_quantity
        ) or (
            state.lifecycle is ExecutionLifecycle.CANCEL_PENDING
            and broker.entry_status is BrokerEntryStatus.CANCELLED
        )
        if not applied:
            return ReconciliationItem(
                state.execution_intent_id,
                ReconciliationCategory.AMBIGUOUS_IN_FLIGHT,
                RuntimeReasonCode.AMBIGUOUS_IN_FLIGHT,
                "in-flight operation is not conclusively applied",
            ), state
        if state.lifecycle is ExecutionLifecycle.PROTECTION_PENDING:
            recovered = replace(
                state,
                lifecycle=(
                    ExecutionLifecycle.PROTECTED
                    if state.filled_quantity == state.requested_quantity
                    else ExecutionLifecycle.PARTIALLY_FILLED
                ),
                protection_status=ProtectionStatus.ACTIVE,
                protection_ref=broker.protection_ref,
                protected_quantity=broker.protected_quantity,
                protection_command_id=None,
                pending_protection_quantity=None,
                updated_at=max(state.updated_at, broker.observed_at),
            )
        else:
            recovered = replace(
                state,
                lifecycle=(
                    ExecutionLifecycle.CANCELLED
                    if state.filled_quantity == 0
                    else ExecutionLifecycle.PARTIALLY_FILLED
                ),
                entry_order_status=EntryOrderStatus.CANCELLED,
                updated_at=max(state.updated_at, broker.observed_at),
            )
        return ReconciliationItem(
            state.execution_intent_id,
            ReconciliationCategory.BROKER_AHEAD,
            None,
            "matching in-flight operation is applied",
        ), recovered
    recovered = state
    if broker.cumulative_filled_quantity > state.filled_quantity:
        if broker.average_fill_price is None:
            return _conflict(state, "broker-ahead fill has no average price")
        recovered = replace(
            state,
            filled_quantity=broker.cumulative_filled_quantity,
            average_fill_price=broker.average_fill_price,
            broker_order_ref=broker.broker_order_ref or state.broker_order_ref,
            updated_at=max(state.updated_at, broker.observed_at),
        )
        return ReconciliationItem(
            state.execution_intent_id,
            ReconciliationCategory.BROKER_AHEAD,
            None,
            "broker cumulative fill is ahead",
        ), recovered
    return ReconciliationItem(
        state.execution_intent_id,
        ReconciliationCategory.MATCHED,
        None,
        "economic state matches",
    ), recovered


def reconcile(
    executions: tuple[ExecutionState, ...], snapshot: BrokerRuntimeSnapshot
) -> ReconciliationResult:
    internal = {item.execution_intent_id: item for item in executions}
    broker = {item.execution_intent_id: item for item in snapshot.executions}
    pairs = [_compare(internal[key], broker.get(key)) for key in sorted(internal)]
    for key in sorted(set(broker) - set(internal)):
        record = broker[key]
        if record.is_live:
            pairs.append(
                (
                    ReconciliationItem(
                        key,
                        ReconciliationCategory.ORPHAN_BROKER_EXECUTION,
                        RuntimeReasonCode.ORPHAN_BROKER_EXECUTION,
                        "unowned/orphan broker execution",
                    ),
                    None,
                )
            )
    return ReconciliationResult(
        tuple(item for item, _ in pairs),
        tuple(state for _, state in pairs if state is not None),
    )
