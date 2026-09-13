"""Immutable values copied from cTrader at the adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from mr_lab.portfolio.contracts import require_text, require_utc
from mr_lab.risk.contracts import require_direction, require_finite


class ObservationLifecycle(StrEnum):
    BOOTSTRAP = "BOOTSTRAP"
    SYNCED = "SYNCED"
    STALE = "STALE"
    HALTED = "HALTED"


@dataclass(frozen=True, slots=True)
class AccountObservation:
    observed_at: datetime
    broker_name: str
    account_type: str
    currency: str
    is_live: bool
    balance: Decimal
    equity: Decimal
    free_margin: Decimal | None
    runtime_id: str

    def __post_init__(self) -> None:
        require_utc("observed_at", self.observed_at)
        for name in ("broker_name", "account_type", "currency", "runtime_id"):
            require_text(name, getattr(self, name))
        require_finite("balance", self.balance)
        require_finite("equity", self.equity)
        if self.free_margin is not None:
            require_finite("free_margin", self.free_margin)


@dataclass(frozen=True, slots=True)
class MarketQuote:
    source_id: str
    instrument: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal

    def __post_init__(self) -> None:
        require_text("source_id", self.source_id)
        require_text("instrument", self.instrument)
        require_utc("timestamp", self.timestamp)
        require_finite("bid", self.bid, positive=True)
        require_finite("ask", self.ask, positive=True)
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")


@dataclass(frozen=True, slots=True)
class ClosedBar:
    source_id: str
    instrument: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        for name in ("source_id", "instrument", "timeframe"):
            require_text(name, getattr(self, name))
        require_utc("open_time", self.open_time)
        require_utc("close_time", self.close_time)
        if self.close_time <= self.open_time:
            raise ValueError("bar close must follow bar open")
        for name in ("open", "high", "low", "close"):
            require_finite(name, getattr(self, name), positive=True)
        if self.low > min(self.open, self.close) or self.high < max(
            self.open, self.close
        ):
            raise ValueError("OHLC values are inconsistent")


@dataclass(frozen=True, slots=True)
class WouldExecuteObservation:
    observation_id: str
    timestamp: datetime
    execution_intent_id: str
    instrument: str
    direction: str
    quantity: Decimal
    sleeve_id: str
    strategy_policy_id: str
    risk_policy_id: str
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "execution_intent_id",
            "instrument",
            "sleeve_id",
            "strategy_policy_id",
            "risk_policy_id",
            "reason",
        ):
            require_text(name, getattr(self, name))
        require_utc("timestamp", self.timestamp)
        require_direction(self.direction)
        require_finite("quantity", self.quantity, positive=True)


class SignalOnlyPipeline(Protocol):
    def on_bar_closed(self, bar: ClosedBar) -> tuple[WouldExecuteObservation, ...]: ...
