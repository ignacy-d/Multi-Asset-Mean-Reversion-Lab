"""Causal state/signal engine for VOL-COMPRESSION-BREAKOUT-2024-v1.

This discovery engine deliberately stops at event generation.  It does not read
data, calculate outcomes or economics, or integrate with execution.  Possible
follow-up models, only if this simple v1 survives, include HAR-RV for
multi-horizon volatility persistence, GARCH-family conditional variance, and
Markov/HMM latent calm/volatile state transitions.  They are not part of v1.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
from math import log
from typing import Literal

from mr_lab.data.models import Bar, Timeframe, VolumeSemantics

STUDY_ID = "VOL-COMPRESSION-BREAKOUT-2024-v1"
SCIENTIFIC_STATUS = "2024_DISCOVERY_NOT_CONFIRMATION"


@dataclass(frozen=True, slots=True)
class VolCompressionConfig:
    """Frozen v1 parameters, exposed so the methodology is explicit."""

    rv_return_count: int = 4
    reference_count: int = 1920
    compression_percentile: float = 0.20
    cooldown: timedelta = timedelta(minutes=60)
    timeframe: Timeframe = field(default_factory=lambda: Timeframe("15m"))

    def __post_init__(self) -> None:
        if self.rv_return_count != 4:
            raise ValueError("v1 requires exactly four prior M15 returns")
        if self.reference_count != 1920:
            raise ValueError("v1 requires exactly 1920 prior RV states")
        if self.compression_percentile != 0.20:
            raise ValueError("v1 requires the prior empirical 20th percentile")
        if self.cooldown != timedelta(minutes=60):
            raise ValueError("v1 requires a 60-minute cooldown")
        if self.timeframe != Timeframe("15m"):
            raise ValueError("v1 requires canonical M15 bars")


@dataclass(frozen=True, slots=True)
class VolCompressionEvent:
    """Audit-ready event values; raw prices retain later pip conversion ability."""

    study_id: str
    event_id: str
    timestamp: datetime
    instrument: str
    direction: Literal["LONG", "SHORT"]
    rv_1h_prior: float
    rv_p20_prior: float
    rv_percentile: float
    box_high: float
    box_low: float
    box_width_raw: float
    breakout_close: float
    breakout_distance_raw: float
    breakout_bar_log_return: float
    current_quote_activity: float | None
    prior_quote_activity: tuple[float | None, ...]


def deterministic_percentile(values: Iterable[float], quantile: float) -> float:
    """Return a Type-7 linearly interpolated percentile in deterministic order.

    With sorted values ``x`` and zero-based rank ``(n - 1) * quantile``, the
    result linearly interpolates between the surrounding ranks.  Endpoints are
    inclusive.  This definition is intentionally local rather than delegated
    to a dependency whose defaults could change.
    """

    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    fraction = rank - lower
    if fraction == 0.0:
        return ordered[lower]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def prior_realized_variance(bars: Iterable[Bar]) -> float:
    """Sum squared close-to-close log returns across exactly five prior bars."""

    prior_bars = tuple(bars)
    if len(prior_bars) != 5:
        raise ValueError("RV_1H requires five bars defining four returns")
    if any(bar.close <= 0.0 for bar in prior_bars):
        raise ValueError("RV_1H requires positive closes")
    return sum(
        log(prior_bars[index].close / prior_bars[index - 1].close) ** 2
        for index in range(1, len(prior_bars))
    )


def _event_id(bar: Bar, direction: str) -> str:
    identity = "|".join(
        (STUDY_ID, bar.instrument, bar.open_time.isoformat(), direction)
    )
    return f"sha256:{sha256(identity.encode('utf-8')).hexdigest()}"


def _validate_canonical_m15(bar: Bar, expected_timeframe: Timeframe) -> None:
    """Reject bars whose declared M15 semantics do not match their timestamps."""

    if bar.timeframe != expected_timeframe:
        raise ValueError("all input bars must be canonical M15 bars")
    for name in ("open_time", "close_time", "available_at"):
        timestamp = getattr(bar, name)
        if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
            raise ValueError(f"{name} must be a timezone-aware UTC timestamp")
    if bar.close_time - bar.open_time != timedelta(minutes=15):
        raise ValueError("M15 bars must span exactly 15 minutes")
    if (
        bar.open_time.minute % 15 != 0
        or bar.open_time.second != 0
        or bar.open_time.microsecond != 0
    ):
        raise ValueError("M15 bar open_time must be epoch-aligned")


def generate_events(
    bars: Iterable[Bar], config: VolCompressionConfig | None = None
) -> tuple[VolCompressionEvent, ...]:
    """Generate causal breakout events from in-memory canonical M15 bars.

    Input may interleave instruments, but must be globally nondecreasing by
    ``available_at`` and strictly chronological within each instrument.  A gap
    breaks return/state continuity, so observations separated by missing M15
    bars are never treated as adjacent.  Earlier valid RV states remain eligible
    for the prior-state reference window.  Volume is diagnostic only and never
    affects state or signals.
    """

    config = config or VolCompressionConfig()
    histories: dict[str, deque[Bar]] = defaultdict(
        lambda: deque(maxlen=config.rv_return_count + 1)
    )
    rv_references: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=config.reference_count)
    )
    last_seen: dict[str, Bar] = {}
    cooldown_until: dict[str, datetime] = {}
    events: list[VolCompressionEvent] = []
    last_available_at: datetime | None = None

    for bar in bars:
        _validate_canonical_m15(bar, config.timeframe)
        if last_available_at is not None and bar.available_at < last_available_at:
            raise ValueError("bars must be nondecreasing by available_at")
        last_available_at = bar.available_at

        previous = last_seen.get(bar.instrument)
        if previous is not None:
            if bar.open_time <= previous.open_time:
                raise ValueError("instrument bars must be strictly chronological")
            if bar.open_time != previous.close_time:
                histories[bar.instrument].clear()
        last_seen[bar.instrument] = bar

        history = histories[bar.instrument]
        references = rv_references[bar.instrument]
        if min(bar.open, bar.high, bar.low, bar.close) <= 0.0:
            # Log returns and price geometry are undefined; do not emit or let
            # this invalid bar participate in a later contiguous state.
            history.clear()
            continue
        if len(history) == config.rv_return_count + 1:
            prior_bars = tuple(history)
            rv_prior = prior_realized_variance(prior_bars)

            # Evaluate before appending rv_prior: the current state is excluded.
            if len(references) == config.reference_count:
                p20 = deterministic_percentile(
                    references, config.compression_percentile
                )
                box_bars = prior_bars[-config.rv_return_count :]
                box_high = max(item.high for item in box_bars)
                box_low = min(item.low for item in box_bars)
                is_long = bar.close > box_high
                is_short = bar.close < box_low
                compressed = rv_prior <= p20
                cooldown_complete = bar.available_at >= cooldown_until.get(
                    bar.instrument, bar.available_at
                )

                # Invalid/contradictory geometry fails closed.
                if (
                    compressed
                    and cooldown_complete
                    and box_high >= box_low
                    and is_long != is_short
                ):
                    direction: Literal["LONG", "SHORT"] = "LONG" if is_long else "SHORT"
                    distance = bar.close - box_high if is_long else box_low - bar.close
                    event = VolCompressionEvent(
                        study_id=STUDY_ID,
                        event_id=_event_id(bar, direction),
                        timestamp=bar.available_at,
                        instrument=bar.instrument,
                        direction=direction,
                        rv_1h_prior=rv_prior,
                        rv_p20_prior=p20,
                        rv_percentile=sum(value <= rv_prior for value in references)
                        / len(references),
                        box_high=box_high,
                        box_low=box_low,
                        box_width_raw=box_high - box_low,
                        breakout_close=bar.close,
                        breakout_distance_raw=distance,
                        breakout_bar_log_return=log(bar.close / prior_bars[-1].close),
                        current_quote_activity=(
                            bar.volume
                            if bar.volume_semantics is VolumeSemantics.QUOTE_ACTIVITY
                            else None
                        ),
                        prior_quote_activity=tuple(
                            item.volume
                            if item.volume_semantics is VolumeSemantics.QUOTE_ACTIVITY
                            else None
                            for item in box_bars
                        ),
                    )
                    events.append(event)
                    cooldown_until[bar.instrument] = bar.available_at + config.cooldown

            references.append(rv_prior)
        history.append(bar)

    return tuple(events)
