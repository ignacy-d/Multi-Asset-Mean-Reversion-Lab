"""Pure, complete, fail-closed comparison of internal and broker state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
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
_TERMINAL_BROKER_STATUS = {BrokerEntryStatus.CANCELLED, BrokerEntryStatus.REJECTED}


def _item(
    state: ExecutionState,
    category: ReconciliationCategory,
    detail: str,
    code: RuntimeReasonCode | None = None,
) -> ReconciliationItem:
    return ReconciliationItem(state.execution_intent_id, category, code, detail)


def _conflict(
    state: ExecutionState, detail: str, recovered: ExecutionState | None = None
) -> tuple[ReconciliationItem, ExecutionState]:
    return _item(
        state,
        ReconciliationCategory.CONFLICT,
        detail,
        RuntimeReasonCode.EXECUTION_CONFLICT,
    ), recovered or state


def _entry_status(status: BrokerEntryStatus) -> EntryOrderStatus:
    return {
        BrokerEntryStatus.OPEN: EntryOrderStatus.OPEN,
        BrokerEntryStatus.FILLED: EntryOrderStatus.FILLED,
        BrokerEntryStatus.CANCELLED: EntryOrderStatus.CANCELLED,
        BrokerEntryStatus.REJECTED: EntryOrderStatus.REJECTED,
        BrokerEntryStatus.ABSENT: EntryOrderStatus.NOT_SUBMITTED,
    }[status]


def _protection_pending(
    state: ExecutionState, broker: BrokerExecutionRecord
) -> ExecutionState:
    """Canonical M3-equivalent state needing protection for all exposure."""
    target = broker.cumulative_filled_quantity
    return replace(
        state,
        lifecycle=ExecutionLifecycle.PROTECTION_PENDING,
        entry_order_status=_entry_status(broker.entry_status),
        filled_quantity=target,
        average_fill_price=broker.average_fill_price,
        broker_order_ref=broker.broker_order_ref or state.broker_order_ref,
        protection_status=ProtectionStatus.PENDING,
        protection_ref=broker.protection_ref,
        protected_quantity=broker.protected_quantity,
        protection_command_id=stable_id(
            "execution-command", state.execution_intent_id, "ensure-protection", target
        ),
        pending_protection_quantity=target,
        updated_at=max(state.updated_at, broker.observed_at),
    )


def _fully_protected(
    state: ExecutionState, broker: BrokerExecutionRecord
) -> ExecutionState:
    entry = _entry_status(broker.entry_status)
    no_live_entry = entry not in {
        EntryOrderStatus.OPEN,
        EntryOrderStatus.SUBMITTING,
        EntryOrderStatus.CANCEL_PENDING,
    }
    return replace(
        state,
        lifecycle=ExecutionLifecycle.PROTECTED
        if no_live_entry
        else ExecutionLifecycle.PARTIALLY_FILLED,
        entry_order_status=entry,
        filled_quantity=broker.cumulative_filled_quantity,
        average_fill_price=broker.average_fill_price,
        broker_order_ref=broker.broker_order_ref or state.broker_order_ref,
        protection_status=ProtectionStatus.ACTIVE,
        protection_ref=broker.protection_ref,
        protected_quantity=broker.protected_quantity,
        protection_command_id=None,
        pending_protection_quantity=None,
        updated_at=max(state.updated_at, broker.observed_at),
    )


def _recover_cancel(
    state: ExecutionState, broker: BrokerExecutionRecord
) -> tuple[ReconciliationItem, ExecutionState]:
    if broker.entry_status is not BrokerEntryStatus.CANCELLED:
        return _item(
            state,
            ReconciliationCategory.AMBIGUOUS_IN_FLIGHT,
            "cancellation is not conclusively applied",
            RuntimeReasonCode.AMBIGUOUS_IN_FLIGHT,
        ), state
    if broker.cumulative_filled_quantity == 0:
        if broker.protection_active:
            return _conflict(
                state, "cancelled unfilled entry retains broker protection"
            )
        recovered = replace(
            state,
            lifecycle=ExecutionLifecycle.CANCELLED,
            entry_order_status=EntryOrderStatus.CANCELLED,
            protection_status=ProtectionStatus.NOT_REQUESTED,
            protection_ref=None,
            protected_quantity=Decimal(0),
            protection_command_id=None,
            pending_protection_quantity=None,
            updated_at=max(state.updated_at, broker.observed_at),
        )
    elif (
        broker.protection_active
        and broker.protected_quantity >= broker.cumulative_filled_quantity
    ):
        recovered = _fully_protected(state, broker)
    else:
        recovered = _protection_pending(state, broker)
        return _conflict(
            state,
            "cancelled remainder has exposure not proven fully protected",
            recovered,
        )
    return _item(
        state,
        ReconciliationCategory.BROKER_AHEAD,
        "cancelled remainder and racing fill recovered together",
    ), recovered


def _recover_protection(
    state: ExecutionState, broker: BrokerExecutionRecord
) -> tuple[ReconciliationItem, ExecutionState]:
    target = state.pending_protection_quantity
    if (
        not broker.protection_active
        or target is None
        or broker.protected_quantity < target
    ):
        return _item(
            state,
            ReconciliationCategory.AMBIGUOUS_IN_FLIGHT,
            "protection operation is not conclusively applied",
            RuntimeReasonCode.AMBIGUOUS_IN_FLIGHT,
        ), state
    if (
        broker.cumulative_filled_quantity > state.filled_quantity
        and broker.average_fill_price is None
    ):
        return _conflict(state, "broker-ahead fill has no average price")
    if broker.protected_quantity >= broker.cumulative_filled_quantity:
        return _item(
            state,
            ReconciliationCategory.BROKER_AHEAD,
            "complete broker-ahead fill and protection recovered",
        ), _fully_protected(state, broker)
    recovered = _protection_pending(state, broker)
    return _conflict(
        state, "broker-ahead exposure is only partially protected", recovered
    )


def _terminal_consistency(
    state: ExecutionState, broker: BrokerExecutionRecord
) -> tuple[ReconciliationItem, ExecutionState]:
    if broker.is_live:
        return _conflict(state, "terminal internal execution is live at broker")
    if broker.protection_active:
        return _conflict(state, "terminal execution retains active broker protection")
    if state.lifecycle in {ExecutionLifecycle.REJECTED, ExecutionLifecycle.CANCELLED}:
        if broker.cumulative_filled_quantity != 0 or broker.exposure_exists:
            return _conflict(state, "unexposed terminal execution has broker fills")
        expected = (
            BrokerEntryStatus.REJECTED
            if state.lifecycle is ExecutionLifecycle.REJECTED
            else BrokerEntryStatus.CANCELLED
        )
        if broker.entry_status is not expected:
            return _conflict(state, "terminal entry status differs")
    elif state.lifecycle is ExecutionLifecycle.CLOSED:
        if broker.cumulative_filled_quantity != state.filled_quantity:
            return _conflict(state, "closed execution fill history differs")
        if broker.average_fill_price != state.average_fill_price:
            return _conflict(state, "closed execution average fill price differs")
        if broker.entry_status is BrokerEntryStatus.OPEN:
            return _conflict(state, "closed execution still has a live entry order")
    return _item(state, ReconciliationCategory.MATCHED, "terminal state matches"), state


def _compare(
    state: ExecutionState, broker: BrokerExecutionRecord | None
) -> tuple[ReconciliationItem, ExecutionState]:
    if broker is None:
        if state.lifecycle is ExecutionLifecycle.NEW or state.is_terminal:
            return _item(
                state, ReconciliationCategory.MATCHED, "no broker state expected"
            ), state
        if state.lifecycle in _IN_FLIGHT:
            return _item(
                state,
                ReconciliationCategory.AMBIGUOUS_IN_FLIGHT,
                "persisted command has no conclusive broker proof",
                RuntimeReasonCode.AMBIGUOUS_IN_FLIGHT,
            ), state
        return _item(
            state,
            ReconciliationCategory.BROKER_MISSING,
            "broker execution is missing",
            RuntimeReasonCode.BROKER_EXECUTION_MISSING,
        ), state
    if broker.instrument is not None and broker.instrument != state.instrument:
        return _conflict(state, "instrument differs for execution identity")
    if broker.direction is not None and broker.direction != state.direction:
        return _conflict(state, "direction differs for execution identity")
    if broker.cumulative_filled_quantity > state.requested_quantity:
        return _conflict(state, "broker fill exceeds requested quantity")
    if broker.protected_quantity > state.requested_quantity:
        return _conflict(state, "broker protection exceeds requested quantity")
    if broker.cumulative_filled_quantity < state.filled_quantity:
        return _conflict(state, "broker fill regresses persisted quantity")
    if (
        broker.cumulative_filled_quantity > state.filled_quantity
        and broker.average_fill_price is None
    ):
        return _conflict(state, "broker-ahead fill has no average price")
    if state.broker_order_ref and broker.broker_order_ref != state.broker_order_ref:
        return _conflict(state, "broker order identity conflicts")
    if state.is_terminal:
        return _terminal_consistency(state, broker)
    if state.lifecycle is ExecutionLifecycle.SUBMITTING:
        expected_key = stable_id("execution-client-key", state.execution_intent_id)
        if (
            broker.client_idempotency_key != expected_key
            or broker.entry_status is not BrokerEntryStatus.OPEN
            or not broker.broker_order_ref
            or broker.cumulative_filled_quantity != 0
        ):
            return _item(
                state,
                ReconciliationCategory.AMBIGUOUS_IN_FLIGHT,
                "entry operation is not conclusively accepted",
                RuntimeReasonCode.AMBIGUOUS_IN_FLIGHT,
            ), state
        protection_status = (
            ProtectionStatus.ACTIVE
            if broker.protection_active
            else ProtectionStatus.NOT_REQUESTED
        )
        protected_quantity = (
            broker.protected_quantity if broker.protection_active else Decimal(0)
        )
        return _item(
            state,
            ReconciliationCategory.BROKER_AHEAD,
            "matching deterministic entry is accepted",
        ), replace(
            state,
            lifecycle=ExecutionLifecycle.ACKNOWLEDGED,
            entry_order_status=EntryOrderStatus.OPEN,
            broker_order_ref=broker.broker_order_ref,
            protection_status=protection_status,
            protection_ref=broker.protection_ref,
            protected_quantity=protected_quantity,
            updated_at=max(state.updated_at, broker.observed_at),
        )
    if state.lifecycle is ExecutionLifecycle.CANCEL_PENDING:
        return _recover_cancel(state, broker)
    if state.lifecycle is ExecutionLifecycle.PROTECTION_PENDING:
        return _recover_protection(state, broker)
    if broker.cumulative_filled_quantity > state.filled_quantity:
        if broker.average_fill_price is None:
            return _conflict(state, "broker-ahead fill has no average price")
        if (
            not broker.protection_active
            or broker.protected_quantity < broker.cumulative_filled_quantity
        ):
            return _conflict(
                state,
                "broker-ahead fill is not proven fully protected",
                _protection_pending(state, broker),
            )
        return _item(
            state,
            ReconciliationCategory.BROKER_AHEAD,
            "fully protected broker-ahead fill recovered",
        ), _fully_protected(state, broker)
    expected_status = state.entry_order_status
    if _entry_status(broker.entry_status) is not expected_status:
        return _conflict(state, "entry order status differs")
    internal_exposure = (
        state.filled_quantity > 0 and state.lifecycle is not ExecutionLifecycle.CLOSED
    )
    if broker.exposure_exists != internal_exposure:
        return _conflict(state, "exposure existence differs")
    if (
        state.filled_quantity > 0
        and broker.average_fill_price != state.average_fill_price
    ):
        return _conflict(state, "average fill price differs")
    if state.protection_status is ProtectionStatus.ACTIVE:
        if (
            not broker.protection_active
            or broker.protected_quantity < state.protected_quantity
        ):
            return _conflict(state, "active protection quantity regresses")
        if state.protection_ref and broker.protection_ref != state.protection_ref:
            return _conflict(state, "protection identity conflicts")
        if broker.protected_quantity > state.protected_quantity:
            return _item(
                state,
                ReconciliationCategory.BROKER_AHEAD,
                "broker protection quantity is ahead",
            ), replace(
                state,
                protected_quantity=broker.protected_quantity,
                updated_at=max(state.updated_at, broker.observed_at),
            )
    elif broker.protection_active:
        return _conflict(state, "broker protection is not represented internally")
    return _item(
        state, ReconciliationCategory.MATCHED, "complete economic state matches"
    ), state


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
