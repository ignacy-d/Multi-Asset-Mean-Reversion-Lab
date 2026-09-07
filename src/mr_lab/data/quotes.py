"""Fail-closed synchronization of native BID and ASK observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite

from mr_lab.data.models import Bar, PriceBasis
from mr_lab.identity import methodology_id


class QuoteSynchronizationError(ValueError):
    """Quote sides cannot be interpreted without guessing."""


class QuoteStatus(Enum):
    SYNCHRONIZED = "synchronized"
    BID_ONLY = "bid_only"
    ASK_ONLY = "ask_only"
    CROSSED = "crossed"
    NON_POSITIVE = "non_positive_spread"


@dataclass(frozen=True, slots=True)
class SynchronizedQuote:
    instrument: str
    observed_at: datetime
    available_at: datetime
    bid: float | None
    ask: float | None
    status: QuoteStatus
    pip_size: float

    @property
    def spread_price(self) -> float | None:
        if self.status is not QuoteStatus.SYNCHRONIZED:
            return None
        assert self.bid is not None and self.ask is not None
        return self.ask - self.bid

    @property
    def spread_pips(self) -> float | None:
        value = self.spread_price
        return None if value is None else value / self.pip_size

    @property
    def spread_bps(self) -> float | None:
        value = self.spread_price
        if value is None:
            return None
        assert self.bid is not None and self.ask is not None
        return value / ((self.bid + self.ask) / 2) * 10_000


@dataclass(frozen=True, slots=True)
class SynchronizationDiagnostics:
    synchronized: int
    bid_only: int
    ask_only: int
    crossed: int
    non_positive: int
    gap_count: int


SPREAD_METHODOLOGY_ID = methodology_id(
    "spread-methodology-v1",
    {"join": "exact_open_time", "price": "completed_bar_close", "fill": "none"},
)


def synchronize_quotes(
    bids: tuple[Bar, ...], asks: tuple[Bar, ...], *, pip_size: float
) -> tuple[tuple[SynchronizedQuote, ...], SynchronizationDiagnostics]:
    """Outer-join exact timestamps; never forward-fill or discard anomalies."""
    if not isfinite(pip_size) or pip_size <= 0:
        raise QuoteSynchronizationError("pip_size must be finite and positive")
    for values, side in ((bids, PriceBasis.BID), (asks, PriceBasis.ASK)):
        if any(bar.price_basis is not side for bar in values):
            raise QuoteSynchronizationError(f"{side.value} input contains another side")
        times = [bar.open_time for bar in values]
        if times != sorted(set(times)):
            raise QuoteSynchronizationError(
                f"{side.value} timestamps must be unique/sorted"
            )
    bid_map, ask_map = ({x.open_time: x for x in bids}, {x.open_time: x for x in asks})
    rows: list[SynchronizedQuote] = []
    counts = {status: 0 for status in QuoteStatus}
    previous: datetime | None = None
    gaps = 0
    for timestamp in sorted(bid_map.keys() | ask_map.keys()):
        bid, ask = bid_map.get(timestamp), ask_map.get(timestamp)
        if bid and ask:
            if bid.instrument != ask.instrument or bid.timeframe != ask.timeframe:
                raise QuoteSynchronizationError("paired quote identity mismatch")
            if bid.close_time != ask.close_time:
                raise QuoteSynchronizationError("paired quote interval mismatch")
            difference = ask.close - bid.close
            status = (
                QuoteStatus.CROSSED
                if difference < 0
                else QuoteStatus.NON_POSITIVE
                if difference == 0
                else QuoteStatus.SYNCHRONIZED
            )
        else:
            status = QuoteStatus.BID_ONLY if bid else QuoteStatus.ASK_ONLY
        bar = bid or ask
        assert bar is not None
        if previous is not None and timestamp - previous > bar.timeframe.duration:
            gaps += 1
        previous = timestamp
        counts[status] += 1
        rows.append(
            SynchronizedQuote(
                bar.instrument,
                timestamp,
                max(x.available_at for x in (bid, ask) if x is not None),
                None if bid is None else bid.close,
                None if ask is None else ask.close,
                status,
                pip_size,
            )
        )
    diagnostics = SynchronizationDiagnostics(
        counts[QuoteStatus.SYNCHRONIZED],
        counts[QuoteStatus.BID_ONLY],
        counts[QuoteStatus.ASK_ONLY],
        counts[QuoteStatus.CROSSED],
        counts[QuoteStatus.NON_POSITIVE],
        gaps,
    )
    return tuple(rows), diagnostics
