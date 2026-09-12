"""Causal failed-breakout discovery events for the 2024 research track.

The detector deliberately knows nothing about VWAP, OU eligibility, trade exits,
or transaction costs.  A structural level and its normalization scale must both
be known before the first bar that can qualify as a breakout.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256

from mr_lab.data import Bar, Timeframe
from mr_lab.research import Direction

FAILED_BREAKOUT_FAMILY = "failed-breakout"
FAILED_BREAKOUT_SPEC_VERSION = "failed-breakout-reclaim-v1"
UPPER = "upper"
LOWER = "lower"
SIDES = (UPPER, LOWER)


class FailedBreakoutError(ValueError):
    """Raised when failed-breakout inputs violate causal research contracts."""


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise FailedBreakoutError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class StructuralLevel:
    """One causal structural level with a pre-event normalization scale."""

    instrument: str
    level_id: str
    anchor_family: str
    side: str
    price: float
    scale: float
    available_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        for name in ("instrument", "level_id", "anchor_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise FailedBreakoutError(f"{name} must be a non-empty string")
        if self.side not in SIDES:
            raise FailedBreakoutError(f"side must be one of {SIDES!r}")
        for name in ("price", "scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise FailedBreakoutError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise FailedBreakoutError(f"{name} must be finite")
        if self.scale <= 0:
            raise FailedBreakoutError("scale must be positive")
        _require_utc("available_at", self.available_at)
        _require_utc("expires_at", self.expires_at)
        if self.available_at >= self.expires_at:
            raise FailedBreakoutError("level availability must precede expiry")


@dataclass(frozen=True, slots=True)
class FailedBreakoutSpec:
    """Pre-registered detector semantics for one breakout-depth cell."""

    minimum_depth_fraction: float
    reclaim_window_minutes: int = 30
    specification_version: str = FAILED_BREAKOUT_SPEC_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_depth_fraction, bool)
            or not isinstance(self.minimum_depth_fraction, int | float)
            or not math.isfinite(self.minimum_depth_fraction)
            or self.minimum_depth_fraction <= 0
        ):
            raise FailedBreakoutError(
                "minimum_depth_fraction must be finite and positive"
            )
        if (
            type(self.reclaim_window_minutes) is not int
            or self.reclaim_window_minutes < 1
        ):
            raise FailedBreakoutError("reclaim_window_minutes must be positive")
        if self.specification_version != FAILED_BREAKOUT_SPEC_VERSION:
            raise FailedBreakoutError("unsupported failed-breakout specification")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def spec_id(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return "sha256:" + sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FailedBreakoutEvent:
    """One completed reclaim event; no execution assumptions are embedded."""

    candidate_event_id: str
    family: str
    family_spec_id: str
    instrument: str
    signal_timestamp: datetime
    direction: Direction
    anchor_family: str
    level_id: str
    level_price: float
    scale: float
    breakout_timestamp: datetime
    reclaim_timestamp: datetime
    initial_depth_fraction: float
    max_depth_fraction: float
    outside_close_count: int
    minutes_to_reclaim: int
    reclaim_inside_fraction: float

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["direction"] = self.direction.name.lower()
        for field in ("signal_timestamp", "breakout_timestamp", "reclaim_timestamp"):
            value[field] = getattr(self, field).isoformat()
        return value


@dataclass(slots=True)
class _Episode:
    breakout_timestamp: datetime
    deadline: datetime
    initial_depth_fraction: float
    max_depth_fraction: float
    outside_close_count: int = 1
    timed_out: bool = False


def _outside_close(level: StructuralLevel, bar: Bar) -> bool:
    if level.side == UPPER:
        return bar.close > level.price
    return bar.close < level.price


def _inside_close(level: StructuralLevel, bar: Bar) -> bool:
    return not _outside_close(level, bar)


def _depth_fraction(level: StructuralLevel, bar: Bar) -> float:
    if level.side == UPPER:
        excursion = max(0.0, bar.high - level.price)
    else:
        excursion = max(0.0, level.price - bar.low)
    return excursion / level.scale


def _reclaim_inside_fraction(level: StructuralLevel, bar: Bar) -> float:
    if level.side == UPPER:
        return max(0.0, level.price - bar.close) / level.scale
    return max(0.0, bar.close - level.price) / level.scale


def _direction(level: StructuralLevel) -> Direction:
    return Direction.SHORT if level.side == UPPER else Direction.LONG


def _event_id(
    level: StructuralLevel,
    spec: FailedBreakoutSpec,
    episode: _Episode,
    reclaim_timestamp: datetime,
) -> str:
    payload = json.dumps(
        {
            "family": FAILED_BREAKOUT_FAMILY,
            "level_id": level.level_id,
            "breakout_timestamp": episode.breakout_timestamp.isoformat(),
            "reclaim_timestamp": reclaim_timestamp.isoformat(),
            "spec_id": spec.spec_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "failed-breakout-" + sha256(payload.encode()).hexdigest()[:24]


def _validate_bars(bars: tuple[Bar, ...]) -> str:
    if any(not isinstance(bar, Bar) for bar in bars):
        raise FailedBreakoutError("all observations must be canonical Bar instances")
    if not bars:
        raise FailedBreakoutError("bar validation requires at least one bar")
    instrument = bars[0].instrument
    for index, bar in enumerate(bars):
        if bar.instrument != instrument or bar.timeframe != Timeframe("1m"):
            raise FailedBreakoutError(
                "failed-breakout discovery requires one-instrument canonical M1 bars"
            )
        if index and bars[index - 1].available_at >= bar.available_at:
            raise FailedBreakoutError(
                "M1 bars must have strictly increasing availability timestamps"
            )
    return instrument


def detect_failed_breakouts(
    bars: Iterable[Bar],
    levels: Iterable[StructuralLevel],
    spec: FailedBreakoutSpec,
) -> tuple[FailedBreakoutEvent, ...]:
    """Detect causal breakout/reclaim episodes without inspecting future bars.

    A breakout begins only after a completed M1 close outside the level and an
    intrabar excursion at least ``minimum_depth_fraction * scale``.  A signal is
    emitted only if a later completed M1 close reclaims the old range within the
    fixed window.  Gaps cancel open episodes and require a fresh inside close
    before that level is armed again.
    """
    if not isinstance(spec, FailedBreakoutSpec):
        raise FailedBreakoutError("spec must be a FailedBreakoutSpec")
    items = tuple(bars)
    if not items:
        return ()
    instrument = _validate_bars(items)

    ordered_levels = tuple(
        sorted(levels, key=lambda item: (item.available_at, item.level_id))
    )
    if any(not isinstance(level, StructuralLevel) for level in ordered_levels):
        raise FailedBreakoutError("all levels must be StructuralLevel instances")
    if any(level.instrument != instrument for level in ordered_levels):
        raise FailedBreakoutError("bar and level instruments must match")
    level_ids = [level.level_id for level in ordered_levels]
    if len(level_ids) != len(set(level_ids)):
        raise FailedBreakoutError("level_id values must be unique")

    active: dict[str, StructuralLevel] = {}
    episodes: dict[str, _Episode] = {}
    blocked: set[str] = set()
    events: list[FailedBreakoutEvent] = []
    next_level = 0
    previous_at: datetime | None = None

    for bar in items:
        if previous_at is not None and bar.available_at - previous_at != timedelta(
            minutes=1
        ):
            episodes.clear()
            blocked.update(active)

        for level_id, level in tuple(active.items()):
            if level.expires_at <= bar.available_at:
                active.pop(level_id, None)
                episodes.pop(level_id, None)
                blocked.discard(level_id)

        while (
            next_level < len(ordered_levels)
            and ordered_levels[next_level].available_at < bar.available_at
        ):
            level = ordered_levels[next_level]
            next_level += 1
            if level.expires_at <= bar.available_at:
                continue
            active[level.level_id] = level
            if bar.available_at - level.available_at > timedelta(minutes=1):
                blocked.add(level.level_id)

        for level_id in sorted(active):
            level = active[level_id]
            if level_id in blocked:
                if _inside_close(level, bar):
                    blocked.remove(level_id)
                continue

            episode = episodes.get(level_id)
            if episode is None:
                depth = _depth_fraction(level, bar)
                if _outside_close(level, bar) and depth >= spec.minimum_depth_fraction:
                    episodes[level_id] = _Episode(
                        breakout_timestamp=bar.available_at,
                        deadline=bar.available_at
                        + timedelta(minutes=spec.reclaim_window_minutes),
                        initial_depth_fraction=depth,
                        max_depth_fraction=depth,
                    )
                continue

            episode.max_depth_fraction = max(
                episode.max_depth_fraction, _depth_fraction(level, bar)
            )
            if _outside_close(level, bar):
                episode.outside_close_count += 1
            if bar.available_at > episode.deadline:
                episode.timed_out = True

            if _inside_close(level, bar):
                if not episode.timed_out and bar.available_at <= episode.deadline:
                    minutes = int(
                        (bar.available_at - episode.breakout_timestamp).total_seconds()
                        // 60
                    )
                    event_id = _event_id(level, spec, episode, bar.available_at)
                    events.append(
                        FailedBreakoutEvent(
                            candidate_event_id=event_id,
                            family=FAILED_BREAKOUT_FAMILY,
                            family_spec_id=spec.spec_id,
                            instrument=instrument,
                            signal_timestamp=bar.available_at,
                            direction=_direction(level),
                            anchor_family=level.anchor_family,
                            level_id=level.level_id,
                            level_price=level.price,
                            scale=level.scale,
                            breakout_timestamp=episode.breakout_timestamp,
                            reclaim_timestamp=bar.available_at,
                            initial_depth_fraction=episode.initial_depth_fraction,
                            max_depth_fraction=episode.max_depth_fraction,
                            outside_close_count=episode.outside_close_count,
                            minutes_to_reclaim=minutes,
                            reclaim_inside_fraction=_reclaim_inside_fraction(level, bar),
                        )
                    )
                episodes.pop(level_id, None)

        previous_at = bar.available_at

    return tuple(events)


__all__ = [
    "FAILED_BREAKOUT_FAMILY",
    "FAILED_BREAKOUT_SPEC_VERSION",
    "LOWER",
    "UPPER",
    "FailedBreakoutError",
    "FailedBreakoutEvent",
    "FailedBreakoutSpec",
    "StructuralLevel",
    "detect_failed_breakouts",
]
