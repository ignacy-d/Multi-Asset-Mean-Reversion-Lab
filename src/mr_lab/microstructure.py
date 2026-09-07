"""Future-data boundary for venue-explicit L1/L2/L3 observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum

from mr_lab.identity import methodology_id


class BookDepth(Enum):
    L1 = "l1"
    L2_MARKET_BY_PRICE = "l2_market_by_price"
    L3_MARKET_BY_ORDER = "l3_market_by_order"


@dataclass(frozen=True, slots=True)
class MicrostructureSourceSpec:
    venue: str
    instrument: str
    depth: BookDepth
    relationship_to_signal_market: str
    version: str = "microstructure-source-v1"

    def __post_init__(self) -> None:
        if not all(
            (
                self.venue.strip(),
                self.instrument.strip(),
                self.relationship_to_signal_market.strip(),
            )
        ):
            raise ValueError("source semantics must be explicit")

    @property
    def identity(self) -> str:
        return methodology_id("microstructure-spec-v1", asdict(self))


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: float
    quantity: float


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    source: MicrostructureSourceSpec
    event_time: datetime
    available_at: datetime
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("event_time", self.event_time),
            ("available_at", self.available_at),
        ):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.available_at < self.event_time:
            raise ValueError("availability cannot precede the market event")
        if not self.bids or not self.asks:
            raise ValueError("a snapshot requires both sides")
        if self.bids[0].price >= self.asks[0].price:
            raise ValueError("crossed/non-positive top of book")
