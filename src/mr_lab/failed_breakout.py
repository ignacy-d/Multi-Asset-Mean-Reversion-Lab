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
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo

from mr_lab.data import Bar, Timeframe
from mr_lab.research import Direction
from mr_lab.sessions import DEFAULT_SESSION_SPEC, SessionSpec, TimeWindow

FAILED_BREAKOUT_FAMILY = "failed-breakout"
FAILED_BREAKOUT_SPEC_VERSION = "failed-breakout-reclaim-v1"
UPPER = "upper"
LOWER = "lower"
SIDES = (UPPER, LOWER)
PREREGISTERED_MINIMUM_DEPTHS = (0.05, 0.15)
PRE_EVENT_SCALE = "previous-complete-utc-day-range"


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
class StructuralLevelSpec:
    """Configuration for the three preregistered causal anchor families.

    All anchors use the range of the immediately preceding complete UTC day.
    This scale is deliberately simple, fixed before event detection, and
    independent of Module A features.
    """

    session_spec: SessionSpec = DEFAULT_SESSION_SPEC
    asia_session: str = "asia"
    london_session: str = "london"
    london_opening_range_minutes: int = 60
    scale_definition: str = PRE_EVENT_SCALE
    discovery_year: int = 2024

    def __post_init__(self) -> None:
        if not isinstance(self.session_spec, SessionSpec):
            raise FailedBreakoutError("session_spec must be a SessionSpec")
        windows = {window.name for window in self.session_spec.major_sessions}
        if self.asia_session not in windows or self.london_session not in windows:
            raise FailedBreakoutError("configured anchor sessions must exist")
        if type(self.london_opening_range_minutes) is not int or not (
            1 <= self.london_opening_range_minutes < 24 * 60
        ):
            raise FailedBreakoutError("opening-range minutes must be positive")
        if self.scale_definition != PRE_EVENT_SCALE:
            raise FailedBreakoutError("unsupported pre-event scale definition")
        if type(self.discovery_year) is not int:
            raise FailedBreakoutError("discovery_year must be an integer")


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
            or self.minimum_depth_fraction not in PREREGISTERED_MINIMUM_DEPTHS
        ):
            raise FailedBreakoutError(
                "minimum_depth_fraction must be in the preregistered grid"
            )
        if self.reclaim_window_minutes != 30:
            raise FailedBreakoutError("reclaim_window_minutes is frozen at 30")
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


def _window_bounds(window: TimeWindow, instance: date) -> tuple[datetime, datetime]:
    """Map one session-local instance to UTC with historical timezone rules."""
    zone = ZoneInfo(window.timezone)
    start = datetime.combine(instance, window.start, zone)
    end_date = instance + timedelta(days=window.start > window.end)
    end = datetime.combine(end_date, window.end, zone)
    return start.astimezone(UTC), end.astimezone(UTC)


def _complete_window(
    by_open: dict[datetime, Bar], start: datetime, end: datetime
) -> tuple[Bar, ...] | None:
    """Return exact M1 coverage or fail closed without synthesising observations."""
    minutes = int((end - start).total_seconds() // 60)
    if minutes <= 0:
        return None
    result = tuple(
        by_open.get(start + timedelta(minutes=index)) for index in range(minutes)
    )
    if any(bar is None for bar in result):
        return None
    return result  # type: ignore[return-value]


def _level_pair(
    *,
    instrument: str,
    family: str,
    instance: date,
    observations: tuple[Bar, ...],
    scale: float,
    available_at: datetime,
    expires_at: datetime,
) -> tuple[StructuralLevel, StructuralLevel]:
    slug = instance.isoformat()
    return (
        StructuralLevel(
            instrument,
            f"{family}-high-{slug}",
            family,
            UPPER,
            max(bar.high for bar in observations),
            scale,
            available_at,
            expires_at,
        ),
        StructuralLevel(
            instrument,
            f"{family}-low-{slug}",
            family,
            LOWER,
            min(bar.low for bar in observations),
            scale,
            available_at,
            expires_at,
        ),
    )


def generate_structural_levels(
    bars: Iterable[Bar], spec: StructuralLevelSpec | None = None
) -> tuple[StructuralLevel, ...]:
    """Generate PIT PDH/PDL, Asia, and London OR60 levels from canonical M1 bars.

    A window contributes only when every expected M1 observation is present.
    Delayed bars delay ``available_at``; they are never back-filled into an
    earlier research instant. Inputs are materialised but never modified.
    """
    if spec is None:
        spec = StructuralLevelSpec()
    if not isinstance(spec, StructuralLevelSpec):
        raise FailedBreakoutError("spec must be a StructuralLevelSpec")
    items = tuple(bars)
    if not items:
        return ()
    instrument = _validate_bars(items)
    if any(bar.open_time.year != spec.discovery_year for bar in items):
        raise FailedBreakoutError("all discovery bars must belong to discovery_year")
    by_open = {bar.open_time: bar for bar in items}
    if len(by_open) != len(items):
        raise FailedBreakoutError("M1 open timestamps must be unique")

    utc_dates = sorted({bar.open_time.date() for bar in items})
    daily: dict[date, tuple[Bar, ...]] = {}
    for day in utc_dates:
        start = datetime.combine(day, time(), UTC)
        complete = _complete_window(by_open, start, start + timedelta(days=1))
        if complete is not None:
            daily[day] = complete

    windows = {window.name: window for window in spec.session_spec.major_sessions}
    asia = windows[spec.asia_session]
    london = windows[spec.london_session]
    local_dates: set[date] = set()
    for bar in items:
        local_dates.add(bar.open_time.astimezone(ZoneInfo(asia.timezone)).date())
        local_dates.add(bar.open_time.astimezone(ZoneInfo(london.timezone)).date())

    levels: list[StructuralLevel] = []
    for day in utc_dates:
        previous = daily.get(day - timedelta(days=1))
        if previous is None:
            continue
        start = datetime.combine(day, time(), UTC)
        end = start + timedelta(days=1)
        scale = max(bar.high for bar in previous) - min(bar.low for bar in previous)
        if scale <= 0:
            continue
        available = max(start, max(bar.available_at for bar in previous))
        if available < end:
            levels.extend(
                _level_pair(
                    instrument=instrument,
                    family="previous-day",
                    instance=day,
                    observations=previous,
                    scale=scale,
                    available_at=available,
                    expires_at=end,
                )
            )

    for local_day in sorted(local_dates):
        for family, window, width in (
            ("asia-session", asia, None),
            ("london-or60", london, spec.london_opening_range_minutes),
        ):
            start, session_end = _window_bounds(window, local_day)
            anchor_end = (
                session_end if width is None else start + timedelta(minutes=width)
            )
            observations = _complete_window(by_open, start, anchor_end)
            scale_bars = daily.get(start.date() - timedelta(days=1))
            if observations is None or scale_bars is None:
                continue
            scale = max(bar.high for bar in scale_bars) - min(
                bar.low for bar in scale_bars
            )
            if scale <= 0:
                continue
            available = max(
                anchor_end,
                *(bar.available_at for bar in observations),
                *(bar.available_at for bar in scale_bars),
            )
            if family == "asia-session":
                next_start, _ = _window_bounds(window, local_day + timedelta(days=1))
                expires = next_start
            else:
                expires = session_end
            if available < expires:
                levels.extend(
                    _level_pair(
                        instrument=instrument,
                        family=family,
                        instance=local_day,
                        observations=observations,
                        scale=scale,
                        available_at=available,
                        expires_at=expires,
                    )
                )
    return tuple(sorted(levels, key=lambda item: (item.available_at, item.level_id)))


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
                            reclaim_inside_fraction=_reclaim_inside_fraction(
                                level, bar
                            ),
                        )
                    )
                episodes.pop(level_id, None)

        previous_at = bar.available_at

    return tuple(events)


__all__ = [
    "FAILED_BREAKOUT_FAMILY",
    "FAILED_BREAKOUT_SPEC_VERSION",
    "LOWER",
    "PREREGISTERED_MINIMUM_DEPTHS",
    "PRE_EVENT_SCALE",
    "UPPER",
    "FailedBreakoutError",
    "FailedBreakoutEvent",
    "FailedBreakoutSpec",
    "StructuralLevel",
    "StructuralLevelSpec",
    "detect_failed_breakouts",
    "generate_structural_levels",
]
