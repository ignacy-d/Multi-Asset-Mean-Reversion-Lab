"""Immutable, broker-neutral contracts for the production runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from mr_lab.execution import EntryOrderStatus, ExecutionLifecycle, ExecutionState
from mr_lab.portfolio.contracts import require_text, require_utc
from mr_lab.risk.contracts import require_direction, require_finite


class RuntimeLifecycle(StrEnum):
    BOOTSTRAP = "BOOTSTRAP"
    SYNCED = "SYNCED"
    LIVE = "LIVE"
    STALE = "STALE"
    HALTED = "HALTED"


class RuntimeReasonCode(StrEnum):
    CHECKPOINT_CORRUPT = "CHECKPOINT_CORRUPT"
    CHECKPOINT_SCHEMA_UNSUPPORTED = "CHECKPOINT_SCHEMA_UNSUPPORTED"
    ORPHAN_BROKER_EXECUTION = "ORPHAN_BROKER_EXECUTION"
    BROKER_EXECUTION_MISSING = "BROKER_EXECUTION_MISSING"
    EXECUTION_CONFLICT = "EXECUTION_CONFLICT"
    AMBIGUOUS_IN_FLIGHT = "AMBIGUOUS_IN_FLIGHT"
    STALE_BROKER_SNAPSHOT = "STALE_BROKER_SNAPSHOT"
    STALE_REQUIRED_SOURCE = "STALE_REQUIRED_SOURCE"
    PERSISTENCE_FAILURE = "PERSISTENCE_FAILURE"
    MANUAL_HALT = "MANUAL_HALT"
    MANUAL_PAUSE = "MANUAL_PAUSE"


@dataclass(frozen=True, slots=True, order=True)
class DataWatermark:
    source_id: str
    event_time: datetime

    def __post_init__(self) -> None:
        require_text("source_id", self.source_id)
        require_utc("event_time", self.event_time)


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    required_source_ids: tuple[str, ...]
    maximum_source_age: timedelta
    maximum_broker_snapshot_age: timedelta

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required_source_ids", tuple(sorted(set(self.required_source_ids)))
        )
        if self.maximum_source_age < timedelta(
            0
        ) or self.maximum_broker_snapshot_age < timedelta(0):
            raise ValueError("freshness ages cannot be negative")


@dataclass(frozen=True, slots=True)
class RuntimeReason:
    code: RuntimeReasonCode
    detail: str


@dataclass(frozen=True, slots=True)
class RuntimeState:
    runtime_key: str
    lifecycle: RuntimeLifecycle
    started_at: datetime
    updated_at: datetime
    last_synced_at: datetime | None = None
    last_live_at: datetime | None = None
    reason: RuntimeReason | None = None
    halt_latched: bool = False
    halt_reason: RuntimeReason | None = None
    checkpoint_sequence: int = 0
    broker_snapshot_time: datetime | None = None
    watermarks: tuple[DataWatermark, ...] = ()
    executions: tuple[ExecutionState, ...] = ()

    def __post_init__(self) -> None:
        require_text("runtime_key", self.runtime_key)
        require_utc("started_at", self.started_at)
        require_utc("updated_at", self.updated_at)
        if self.checkpoint_sequence < 0:
            raise ValueError("checkpoint sequence cannot be negative")
        if self.halt_latched != (self.halt_reason is not None):
            raise ValueError("halt latch and halt reason must coexist")
        for name in ("last_synced_at", "last_live_at", "broker_snapshot_time"):
            value = getattr(self, name)
            if value is not None:
                require_utc(name, value)
        executions = tuple(
            sorted(self.executions, key=lambda value: value.execution_intent_id)
        )
        if len({item.execution_intent_id for item in executions}) != len(executions):
            raise ValueError("execution intent IDs must be unique")
        watermarks = tuple(sorted(self.watermarks, key=lambda value: value.source_id))
        if len({item.source_id for item in watermarks}) != len(watermarks):
            raise ValueError("watermark source IDs must be unique")
        object.__setattr__(self, "executions", executions)
        object.__setattr__(self, "watermarks", watermarks)

    @property
    def can_open_new_entries(self) -> bool:
        return self.lifecycle is RuntimeLifecycle.LIVE


class BrokerEntryStatus(StrEnum):
    ABSENT = "ABSENT"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class BrokerExecutionRecord:
    execution_intent_id: str
    observed_at: datetime
    client_idempotency_key: str | None = None
    instrument: str | None = None
    direction: str | None = None
    broker_order_ref: str | None = None
    entry_status: BrokerEntryStatus = BrokerEntryStatus.ABSENT
    cumulative_filled_quantity: Decimal = Decimal(0)
    average_fill_price: Decimal | None = None
    exposure_exists: bool = False
    protection_active: bool = False
    protected_quantity: Decimal = Decimal(0)
    protection_ref: str | None = None

    def __post_init__(self) -> None:
        require_text("execution_intent_id", self.execution_intent_id)
        require_utc("observed_at", self.observed_at)
        for name in (
            "client_idempotency_key",
            "instrument",
            "broker_order_ref",
            "protection_ref",
        ):
            value = getattr(self, name)
            if value is not None:
                require_text(name, value)
        if self.direction is not None:
            require_direction(self.direction)
        require_finite("cumulative_filled_quantity", self.cumulative_filled_quantity)
        require_finite("protected_quantity", self.protected_quantity)
        if self.cumulative_filled_quantity < 0 or self.protected_quantity < 0:
            raise ValueError("broker quantities cannot be negative")
        if self.average_fill_price is not None:
            require_finite("average_fill_price", self.average_fill_price, positive=True)
        if self.average_fill_price is not None and self.cumulative_filled_quantity == 0:
            raise ValueError("average fill price requires a cumulative fill")
        if not self.protection_active and (
            self.protected_quantity != 0 or self.protection_ref is not None
        ):
            raise ValueError("inactive protection cannot claim quantity or identity")
        if self.protection_active and (
            self.protected_quantity <= 0 or self.protection_ref is None
        ):
            raise ValueError("active protection requires quantity and identity")
        if self.exposure_exists and self.cumulative_filled_quantity == 0:
            raise ValueError("live exposure requires a positive cumulative fill")
        if (
            self.protection_active
            and self.entry_status
            in {
                BrokerEntryStatus.CANCELLED,
                BrokerEntryStatus.FILLED,
                BrokerEntryStatus.REJECTED,
            }
            and self.protected_quantity > self.cumulative_filled_quantity
        ):
            raise ValueError(
                "terminal entry protection cannot exceed cumulative exposure"
            )

    @property
    def is_live(self) -> bool:
        return self.entry_status is BrokerEntryStatus.OPEN or self.exposure_exists


@dataclass(frozen=True, slots=True)
class BrokerRuntimeSnapshot:
    observed_at: datetime
    executions: tuple[BrokerExecutionRecord, ...] = ()
    account_reference: str | None = None

    def __post_init__(self) -> None:
        require_utc("observed_at", self.observed_at)
        records = tuple(
            sorted(self.executions, key=lambda value: value.execution_intent_id)
        )
        if len({item.execution_intent_id for item in records}) != len(records):
            raise ValueError("broker execution IDs must be unique")
        if any(item.observed_at > self.observed_at for item in records):
            raise ValueError("broker record cannot be newer than its snapshot")
        object.__setattr__(self, "executions", records)


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    lifecycle: RuntimeLifecycle
    can_open_new_entries: bool
    last_synced_at: datetime | None
    broker_snapshot_age: timedelta | None
    reason: RuntimeReason | None
    halt_latched: bool
    halt_reason: RuntimeReason | None
    active_execution_count: int
    unsafe_execution_count: int
    in_flight_command_count: int
    source_freshness: tuple[tuple[str, bool], ...]


def execution_counts(executions: tuple[ExecutionState, ...]) -> tuple[int, int, int]:
    active = sum(not item.is_terminal for item in executions)
    unsafe = sum(
        item.lifecycle in {ExecutionLifecycle.UNSAFE, ExecutionLifecycle.ERROR}
        for item in executions
    )
    inflight = sum(
        item.lifecycle
        in {
            ExecutionLifecycle.SUBMITTING,
            ExecutionLifecycle.PROTECTION_PENDING,
            ExecutionLifecycle.CANCEL_PENDING,
        }
        or item.entry_order_status
        in {EntryOrderStatus.SUBMITTING, EntryOrderStatus.CANCEL_PENDING}
        for item in executions
    )
    return active, unsafe, inflight
