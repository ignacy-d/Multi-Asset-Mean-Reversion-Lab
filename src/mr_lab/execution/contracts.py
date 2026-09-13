"""Immutable broker-neutral contracts for the execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from mr_lab.portfolio import PlanReference, ProposalProvenance
from mr_lab.portfolio.contracts import require_text, require_utc
from mr_lab.risk import ProtectiveBoundary
from mr_lab.risk.contracts import require_direction, require_finite


class ExecutionLifecycle(StrEnum):
    NEW = "NEW"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    PROTECTION_PENDING = "PROTECTION_PENDING"
    PROTECTED = "PROTECTED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"
    UNSAFE = "UNSAFE"
    ERROR = "ERROR"

    @property
    def terminal(self) -> bool:
        return self in {
            ExecutionLifecycle.CANCELLED,
            ExecutionLifecycle.REJECTED,
            ExecutionLifecycle.CLOSED,
            ExecutionLifecycle.ERROR,
        }


class ProtectionStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class EntryOrderStatus(StrEnum):
    """Broker-neutral liveness of the entry order, separate from exposure state."""

    NOT_SUBMITTED = "NOT_SUBMITTED"
    SUBMITTING = "SUBMITTING"
    OPEN = "OPEN"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class ProcessedEvent:
    event_id: str
    fingerprint: str

    def __post_init__(self) -> None:
        require_text("event_id", self.event_id)
        require_text("fingerprint", self.fingerprint)


@dataclass(frozen=True, slots=True)
class ExecutionState:
    """Persistable execution aggregate; it contains no native broker objects."""

    execution_intent_id: str
    intent_fingerprint: str
    proposal_id: str
    instrument: str
    direction: str
    requested_quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    lifecycle: ExecutionLifecycle
    entry_order_status: EntryOrderStatus
    entry_plan: PlanReference
    protective_plan: PlanReference
    protective_boundary: ProtectiveBoundary
    strategy_policy_id: str
    risk_decision_id: str
    proposal_provenance: ProposalProvenance
    created_at: datetime
    updated_at: datetime
    entry_command_id: str | None = None
    broker_order_ref: str | None = None
    protection_status: ProtectionStatus = ProtectionStatus.NOT_REQUESTED
    protection_ref: str | None = None
    protected_quantity: Decimal = Decimal(0)
    protection_command_id: str | None = None
    pending_protection_quantity: Decimal | None = None
    cancel_command_id: str | None = None
    processed_events: tuple[ProcessedEvent, ...] = ()
    last_failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "execution_intent_id",
            "intent_fingerprint",
            "proposal_id",
            "instrument",
            "strategy_policy_id",
            "risk_decision_id",
        ):
            require_text(name, getattr(self, name))
        require_direction(self.direction)
        require_finite("requested_quantity", self.requested_quantity, positive=True)
        require_finite("filled_quantity", self.filled_quantity)
        require_finite("protected_quantity", self.protected_quantity)
        if not Decimal(0) <= self.filled_quantity <= self.requested_quantity:
            raise ValueError("filled_quantity must be within requested quantity")
        atomic_reservation = (
            self.protection_status is ProtectionStatus.ACTIVE
            and self.protected_quantity == self.requested_quantity
        )
        if (
            not Decimal(0) <= self.protected_quantity <= self.filled_quantity
            and not atomic_reservation
        ):
            raise ValueError("protected_quantity must cover existing exposure only")
        if (self.average_fill_price is None) != (self.filled_quantity == 0):
            raise ValueError("average_fill_price must correspond to filled quantity")
        if self.average_fill_price is not None:
            require_finite("average_fill_price", self.average_fill_price, positive=True)
        if self.pending_protection_quantity is not None:
            require_finite(
                "pending_protection_quantity",
                self.pending_protection_quantity,
                positive=True,
            )
            if self.pending_protection_quantity > self.filled_quantity:
                raise ValueError("pending protection cannot exceed filled quantity")
        if (self.protection_command_id is None) != (
            self.pending_protection_quantity is None
        ):
            raise ValueError(
                "pending protection command identity and quantity must coexist"
            )
        require_utc("created_at", self.created_at)
        require_utc("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if len({item.event_id for item in self.processed_events}) != len(
            self.processed_events
        ):
            raise ValueError("processed event IDs must be unique")

    @property
    def remaining_quantity(self) -> Decimal:
        return self.requested_quantity - self.filled_quantity

    @property
    def live_entry_quantity(self) -> Decimal:
        """Quantity that may still execute, independent of economic remainder."""
        if self.entry_order_status in {
            EntryOrderStatus.SUBMITTING,
            EntryOrderStatus.OPEN,
            EntryOrderStatus.CANCEL_PENDING,
        }:
            return self.remaining_quantity
        return Decimal(0)

    @property
    def is_terminal(self) -> bool:
        return self.lifecycle.terminal


@dataclass(frozen=True, slots=True)
class SubmitEntryCommand:
    command_id: str
    client_idempotency_key: str
    execution_intent_id: str
    instrument: str
    direction: str
    quantity: Decimal
    entry_plan: PlanReference
    protective_plan: PlanReference
    protective_boundary: ProtectiveBoundary
    strategy_policy_id: str
    risk_decision_id: str
    proposal_id: str
    proposal_provenance: ProposalProvenance
    proposal_timestamp: datetime
    approved_at: datetime
    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "command_id",
            "client_idempotency_key",
            "execution_intent_id",
            "instrument",
            "strategy_policy_id",
            "risk_decision_id",
            "proposal_id",
        ):
            require_text(name, getattr(self, name))
        require_direction(self.direction)
        require_finite("quantity", self.quantity, positive=True)
        self.protective_boundary.validate_for(self.direction)
        for name in ("proposal_timestamp", "approved_at", "created_at"):
            require_utc(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class EnsureProtectionCommand:
    command_id: str
    execution_intent_id: str
    broker_order_ref: str
    protected_quantity: Decimal
    protective_plan: PlanReference
    protective_boundary: ProtectiveBoundary
    created_at: datetime

    def __post_init__(self) -> None:
        for name in ("command_id", "execution_intent_id", "broker_order_ref"):
            require_text(name, getattr(self, name))
        require_finite("protected_quantity", self.protected_quantity, positive=True)
        require_utc("created_at", self.created_at)


@dataclass(frozen=True, slots=True)
class CancelEntryCommand:
    command_id: str
    execution_intent_id: str
    broker_order_ref: str
    created_at: datetime

    def __post_init__(self) -> None:
        for name in ("command_id", "execution_intent_id", "broker_order_ref"):
            require_text(name, getattr(self, name))
        require_utc("created_at", self.created_at)


type ExecutionCommand = (
    SubmitEntryCommand | EnsureProtectionCommand | CancelEntryCommand
)


def _validate_event(
    event_id: str, execution_intent_id: str, event_time: datetime
) -> None:
    require_text("event_id", event_id)
    require_text("execution_intent_id", execution_intent_id)
    require_utc("event_time", event_time)


@dataclass(frozen=True, slots=True)
class EntryAccepted:
    event_id: str
    execution_intent_id: str
    command_id: str
    broker_order_ref: str
    event_time: datetime
    protection_active: bool = False
    protection_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_event(self.event_id, self.execution_intent_id, self.event_time)
        require_text("command_id", self.command_id)
        require_text("broker_order_ref", self.broker_order_ref)
        if self.protection_active != (self.protection_ref is not None):
            raise ValueError("active atomic protection requires a protection reference")


@dataclass(frozen=True, slots=True)
class EntryRejected:
    event_id: str
    execution_intent_id: str
    command_id: str
    reason: str
    event_time: datetime

    def __post_init__(self) -> None:
        _validate_event(self.event_id, self.execution_intent_id, self.event_time)
        require_text("command_id", self.command_id)
        require_text("reason", self.reason)


@dataclass(frozen=True, slots=True)
class EntryFill:
    event_id: str
    execution_intent_id: str
    broker_order_ref: str
    quantity: Decimal
    price: Decimal
    event_time: datetime

    def __post_init__(self) -> None:
        _validate_event(self.event_id, self.execution_intent_id, self.event_time)
        require_text("broker_order_ref", self.broker_order_ref)
        require_finite("quantity", self.quantity, positive=True)
        require_finite("price", self.price, positive=True)


@dataclass(frozen=True, slots=True)
class CancelAcknowledged:
    event_id: str
    execution_intent_id: str
    command_id: str
    broker_order_ref: str
    event_time: datetime

    def __post_init__(self) -> None:
        _validate_event(self.event_id, self.execution_intent_id, self.event_time)
        require_text("command_id", self.command_id)
        require_text("broker_order_ref", self.broker_order_ref)


@dataclass(frozen=True, slots=True)
class ProtectionAcknowledged:
    event_id: str
    execution_intent_id: str
    command_id: str
    broker_order_ref: str
    protection_ref: str
    event_time: datetime

    def __post_init__(self) -> None:
        _validate_event(self.event_id, self.execution_intent_id, self.event_time)
        for name in ("command_id", "broker_order_ref", "protection_ref"):
            require_text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ProtectionRejected:
    event_id: str
    execution_intent_id: str
    command_id: str
    broker_order_ref: str
    reason: str
    event_time: datetime

    def __post_init__(self) -> None:
        _validate_event(self.event_id, self.execution_intent_id, self.event_time)
        for name in ("command_id", "broker_order_ref", "reason"):
            require_text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ExecutionClosed:
    event_id: str
    execution_intent_id: str
    broker_order_ref: str
    event_time: datetime
    reason: str

    def __post_init__(self) -> None:
        _validate_event(self.event_id, self.execution_intent_id, self.event_time)
        require_text("broker_order_ref", self.broker_order_ref)
        require_text("reason", self.reason)


type BrokerEvent = (
    EntryAccepted
    | EntryRejected
    | EntryFill
    | CancelAcknowledged
    | ProtectionAcknowledged
    | ProtectionRejected
    | ExecutionClosed
)
