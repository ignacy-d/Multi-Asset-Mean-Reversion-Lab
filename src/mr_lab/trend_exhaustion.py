"""PIT-causal Trend Exhaustion v1 events on complete canonical M15 bars."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from itertools import pairwise

from mr_lab.data import Bar, Timeframe
from mr_lab.research import Direction

FAMILY = "trend-exhaustion"
SPEC_VERSION = "trend-exhaustion-m15-v1"
DISPLACEMENT_THRESHOLDS = (1.25, 1.50, 1.75)
PRIMARY_THRESHOLD = 1.50
TREND_BARS = 8
EXHAUSTION_BARS = 4
ATR_PERIOD = 20
MIN_EFFICIENCY = 0.60
MIN_EXTENSION_ATR = 0.25
MAX_RETAINED_ATR = 0.10


class TrendExhaustionError(ValueError):
    """Raised when inputs violate the frozen family contract."""


@dataclass(frozen=True, slots=True)
class TrendExhaustionSpec:
    displacement_threshold: float

    def __post_init__(self) -> None:
        if self.displacement_threshold not in DISPLACEMENT_THRESHOLDS:
            raise TrendExhaustionError("threshold must be in the frozen grid")

    @property
    def spec_id(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "sha256:" + sha256(value.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TrendExhaustionEvent:
    event_id: str
    family: str
    family_spec_id: str
    instrument: str
    signal_timestamp: datetime
    direction: Direction
    displacement_threshold: float
    trend_direction: int
    frozen_atr20: float
    trend_displacement: float
    normalized_displacement: float
    efficiency_ratio: float
    exhaustion_extension: float
    retained_progress: float
    upper_wick: float
    lower_wick: float
    body_size: float
    wick_body_ratio: float | None
    m15_range: float
    volume: float | None
    volume_change: float | None
    session_label: str | None = None
    minutes_from_session_open: int | None = None
    minutes_to_session_close: int | None = None

    def as_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["signal_timestamp"] = self.signal_timestamp.isoformat()
        row["direction"] = self.direction.name.lower()
        return row


def _true_range(bar: Bar, previous_close: float) -> float:
    return max(
        bar.high - bar.low,
        abs(bar.high - previous_close),
        abs(bar.low - previous_close),
    )


def _candidate(
    window: tuple[Bar, ...], spec: TrendExhaustionSpec
) -> TrendExhaustionEvent | None:
    phase_a = window[-(TREND_BARS + EXHAUSTION_BARS) : -EXHAUSTION_BARS]
    phase_b = window[-EXHAUSTION_BARS:]
    atr_bars = window[-(ATR_PERIOD + EXHAUSTION_BARS) : -EXHAUSTION_BARS]
    atr_previous = window[-(ATR_PERIOD + EXHAUSTION_BARS + 1)]
    ranges = [
        _true_range(bar, previous.close)
        for previous, bar in pairwise((atr_previous, *atr_bars))
    ]
    atr = sum(ranges) / ATR_PERIOD
    if not math.isfinite(atr) or atr <= 0:
        return None
    displacement = phase_a[-1].close - phase_a[0].close
    if displacement == 0:
        return None
    trend_direction = 1 if displacement > 0 else -1
    changes = sum(abs(right.close - left.close) for left, right in pairwise(phase_a))
    efficiency = abs(displacement) / changes if changes else 0.0
    normalized = abs(displacement) / atr
    terminal = phase_a[-1].close
    extension = max(
        trend_direction * (bar.high - terminal)
        if trend_direction > 0
        else trend_direction * (bar.low - terminal)
        for bar in phase_b
    )
    retained = trend_direction * (phase_b[-1].close - terminal)
    last_move = trend_direction * (phase_b[-1].close - phase_b[-2].close)
    eligible = (
        normalized > spec.displacement_threshold
        and efficiency >= MIN_EFFICIENCY
        and extension >= MIN_EXTENSION_ATR * atr
        and retained <= MAX_RETAINED_ATR * atr
        and last_move < 0
    )
    if not eligible:
        return None
    last = phase_b[-1]
    body = abs(last.close - last.open)
    upper = last.high - max(last.open, last.close)
    lower = min(last.open, last.close) - last.low
    prior_volume = phase_b[-2].volume
    volume_change = (
        None
        if last.volume is None or prior_volume is None
        else last.volume - prior_volume
    )
    payload = f"{last.instrument}|{last.available_at.isoformat()}|{spec.spec_id}"
    return TrendExhaustionEvent(
        "trend-exhaustion-" + sha256(payload.encode()).hexdigest()[:24],
        FAMILY,
        spec.spec_id,
        last.instrument,
        last.available_at,
        Direction.SHORT if trend_direction > 0 else Direction.LONG,
        spec.displacement_threshold,
        trend_direction,
        atr,
        displacement,
        normalized,
        efficiency,
        extension,
        retained,
        upper,
        lower,
        body,
        (upper + lower) / body if body else None,
        last.high - last.low,
        last.volume,
        volume_change,
    )


def detect_trend_exhaustion(
    m15_bars: tuple[Bar, ...] | list[Bar], spec: TrendExhaustionSpec
) -> tuple[TrendExhaustionEvent, ...]:
    """Emit once per episode; a false candidate window structurally re-arms."""
    bars = tuple(m15_bars)
    if any(bar.timeframe != Timeframe("15m") for bar in bars):
        raise TrendExhaustionError("detector requires canonical M15 bars")
    if any(
        left.instrument != right.instrument or left.available_at >= right.available_at
        for left, right in pairwise(bars)
    ):
        raise TrendExhaustionError(
            "M15 input must be one instrument and strictly ordered"
        )
    needed = ATR_PERIOD + EXHAUSTION_BARS + 1
    armed = True
    events = []
    segment_start = 0
    for end in range(1, len(bars) + 1):
        if end > 1 and bars[end - 2].close_time != bars[end - 1].open_time:
            segment_start = end - 1
            armed = True
        if end - segment_start < needed:
            continue
        event = _candidate(bars[segment_start:end], spec)
        if event is None:
            armed = True
        elif armed:
            events.append(event)
            armed = False
    return tuple(events)
