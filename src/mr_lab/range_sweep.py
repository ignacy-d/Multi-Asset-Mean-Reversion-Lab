"""Causal structural reference levels and M5 sweep/reclaim events.

This module implements the deliberately simple engine for the
``RANGE-SWEEP-2024-v1`` discovery hypothesis.  It does not perform data I/O,
economic analysis, cost modelling, or execution.  More advanced first-passage,
survival/hazard, and semi-Markov models are possible later model families only
if this unfiltered structural hypothesis first demonstrates economic value.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from hashlib import sha256
from zoneinfo import ZoneInfo

from mr_lab.data import Bar, DataContractError, Timeframe, validate_dataset
from mr_lab.sessions import DEFAULT_SESSION_SPEC, TimeWindow, classify_bar

STUDY_ID = "RANGE-SWEEP-2024-v1"
SCIENTIFIC_STATUS = "2024_DISCOVERY_NOT_CONFIRMATION"

_M5 = Timeframe("5m")
_FX_TIMEZONE = "America/New_York"
_FX_DAY_BOUNDARY = time(17)
_ASIA_SESSION_NAME = "asia"


class ReferenceFamily(Enum):
    """The two preregistered structural-reference families."""

    PREVIOUS_FX_DAY = "previous_fx_day"
    COMPLETED_ASIA_SESSION = "completed_asia_session"


class ReferenceSide(Enum):
    """The side represented by a structural level."""

    HIGH = "high"
    LOW = "low"


class SignalDirection(Enum):
    """Direction implied by a reclaim back into the prior range."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True, slots=True)
class StructuralReference:
    """A high or low whose complete source interval is already known."""

    family: ReferenceFamily
    instance: str
    side: ReferenceSide
    instrument: str
    price: float
    valid_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class RangeSweepEvent:
    """One first-sweep/reclaim event, including non-filtering diagnostics."""

    event_id: str
    reference_family: ReferenceFamily
    reference_instance: str
    reference_side: ReferenceSide
    reference_price: float
    signal_direction: SignalDirection
    instrument: str
    signal_timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    overshoot_raw_price: float
    overshoot_pips: float | None
    minutes_since_reference_valid: float
    session_labels: tuple[str, ...]
    volume: float | None


@dataclass(frozen=True, slots=True)
class RangeSweepConfig:
    """Optional diagnostic metadata; v1 methodology itself is frozen."""

    pip_sizes: Mapping[str, float] | None = None


def _window_bounds(local_day: date, window: TimeWindow) -> tuple[datetime, datetime]:
    timezone = ZoneInfo(window.timezone)
    start = datetime.combine(local_day, window.start, timezone)
    end_day = local_day + timedelta(days=window.start >= window.end)
    end = datetime.combine(end_day, window.end, timezone)
    return start.astimezone(UTC), end.astimezone(UTC)


def _fx_bounds(timestamp: datetime) -> tuple[datetime, datetime]:
    timezone = ZoneInfo(_FX_TIMEZONE)
    local = timestamp.astimezone(timezone)
    boundary = datetime.combine(local.date(), _FX_DAY_BOUNDARY, timezone)
    if local < boundary:
        boundary = datetime.combine(
            local.date() - timedelta(days=1), _FX_DAY_BOUNDARY, timezone
        )
    end = datetime.combine(
        boundary.date() + timedelta(days=1), _FX_DAY_BOUNDARY, timezone
    )
    return boundary.astimezone(UTC), end.astimezone(UTC)


def _asia_window() -> TimeWindow:
    matches = tuple(
        window
        for window in DEFAULT_SESSION_SPEC.major_sessions
        if window.name == _ASIA_SESSION_NAME
    )
    if len(matches) != 1:
        raise ValueError("asia_session_name must identify exactly one major session")
    return matches[0]


def _asia_bounds(timestamp: datetime, window: TimeWindow) -> tuple[datetime, datetime]:
    local = timestamp.astimezone(ZoneInfo(window.timezone))
    local_day = local.date()
    start, end = _window_bounds(local_day, window)
    if timestamp < start and window.start > window.end:
        start, end = _window_bounds(local_day - timedelta(days=1), window)
    return start, end


def _is_provider_padding(bar: Bar) -> bool:
    """Apply the repository's zero-activity, flat-OHLC padding policy."""
    return bar.volume == 0 and bar.open == bar.high == bar.low == bar.close


def _next_fx_boundary(end: datetime) -> datetime:
    timezone = ZoneInfo(_FX_TIMEZONE)
    local_end = end.astimezone(timezone)
    return datetime.combine(
        local_end.date() + timedelta(days=1), _FX_DAY_BOUNDARY, timezone
    ).astimezone(UTC)


def _next_asia_start(end: datetime, window: TimeWindow) -> datetime:
    local_end = end.astimezone(ZoneInfo(window.timezone))
    start, _ = _window_bounds(local_end.date() + timedelta(days=1), window)
    return start


def _is_complete_interval(
    components: list[Bar], start: datetime, end: datetime
) -> bool:
    expected = (end - start) // _M5.duration
    if len(components) != expected:
        return False
    return all(
        bar.open_time == start + index * _M5.duration
        and bar.close_time == start + (index + 1) * _M5.duration
        and bar.available_at <= end
        and not _is_provider_padding(bar)
        for index, bar in enumerate(components)
    )


def _references(bars: tuple[Bar, ...]) -> tuple[StructuralReference, ...]:
    asia = _asia_window()
    grouped: dict[tuple[ReferenceFamily, datetime, datetime], list[Bar]] = {}
    for bar in bars:
        fx_start, fx_end = _fx_bounds(bar.open_time)
        if bar.open_time >= fx_start and bar.close_time <= fx_end:
            grouped.setdefault(
                (ReferenceFamily.PREVIOUS_FX_DAY, fx_start, fx_end), []
            ).append(bar)

        asia_start, asia_end = _asia_bounds(bar.open_time, asia)
        active, _ = asia.contains(bar.open_time)
        if active and bar.open_time >= asia_start and bar.close_time <= asia_end:
            grouped.setdefault(
                (ReferenceFamily.COMPLETED_ASIA_SESSION, asia_start, asia_end), []
            ).append(bar)

    references: list[StructuralReference] = []
    for (family, start, end), components in grouped.items():
        if not _is_complete_interval(components, start, end):
            continue
        instance = f"{start.isoformat()}/{end.isoformat()}"
        expires_at = (
            _next_fx_boundary(end)
            if family is ReferenceFamily.PREVIOUS_FX_DAY
            else _next_asia_start(end, asia)
        )
        for side, price in (
            (ReferenceSide.HIGH, max(bar.high for bar in components)),
            (ReferenceSide.LOW, min(bar.low for bar in components)),
        ):
            references.append(
                StructuralReference(
                    family,
                    instance,
                    side,
                    components[0].instrument,
                    price,
                    end,
                    expires_at,
                )
            )
    return tuple(
        sorted(
            references,
            key=lambda item: (item.valid_at, item.family.value, item.side.value),
        )
    )


def _event_id(reference: StructuralReference, signal_timestamp: datetime) -> str:
    identity = "|".join(
        (
            STUDY_ID,
            reference.family.value,
            reference.instance,
            reference.side.value,
            reference.instrument,
            signal_timestamp.isoformat(),
        )
    )
    return f"sha256:{sha256(identity.encode()).hexdigest()}"


def range_sweep_events(
    bars: Sequence[Bar], config: RangeSweepConfig | None = None
) -> tuple[RangeSweepEvent, ...]:
    """Return causal first-sweep events from canonical complete M5 bars.

    Each family always uses its most recently completed instance.  Each side of
    each instance can fire once; a touch or a close exactly at the level cannot.
    Volume is copied only as metadata and never participates in qualification.
    """
    effective = config or RangeSweepConfig()
    report = validate_dataset(list(bars))
    observations = report.bars
    if any(bar.timeframe != _M5 for bar in observations):
        raise DataContractError("range sweep signals require canonical 5m bars")
    references = _references(observations)
    consumed: set[tuple[ReferenceFamily, str, ReferenceSide, str]] = set()
    events: list[RangeSweepEvent] = []
    for bar in observations:
        latest: dict[tuple[ReferenceFamily, ReferenceSide], StructuralReference] = {}
        for reference in references:
            if (
                reference.valid_at <= bar.open_time
                and reference.valid_at <= bar.available_at < reference.expires_at
            ):
                latest[(reference.family, reference.side)] = reference
        classification = classify_bar(bar, DEFAULT_SESSION_SPEC)
        labels = (*classification.active_sessions, *classification.active_named_windows)
        for reference in latest.values():
            key = (
                reference.family,
                reference.instance,
                reference.side,
                reference.instrument,
            )
            if key in consumed:
                continue
            if reference.side is ReferenceSide.HIGH:
                qualifies = bar.high > reference.price and bar.close < reference.price
                direction = SignalDirection.SHORT
                overshoot = bar.high - reference.price
            else:
                qualifies = bar.low < reference.price and bar.close > reference.price
                direction = SignalDirection.LONG
                overshoot = reference.price - bar.low
            if not qualifies:
                continue
            consumed.add(key)
            pip_size = (effective.pip_sizes or {}).get(bar.instrument)
            events.append(
                RangeSweepEvent(
                    event_id=_event_id(reference, bar.available_at),
                    reference_family=reference.family,
                    reference_instance=reference.instance,
                    reference_side=reference.side,
                    reference_price=reference.price,
                    signal_direction=direction,
                    instrument=bar.instrument,
                    signal_timestamp=bar.available_at,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    overshoot_raw_price=overshoot,
                    overshoot_pips=None if pip_size is None else overshoot / pip_size,
                    minutes_since_reference_valid=(
                        bar.available_at - reference.valid_at
                    ).total_seconds()
                    / 60,
                    session_labels=labels,
                    volume=bar.volume,
                )
            )
    return tuple(events)


__all__ = [
    "SCIENTIFIC_STATUS",
    "STUDY_ID",
    "RangeSweepConfig",
    "RangeSweepEvent",
    "ReferenceFamily",
    "ReferenceSide",
    "SignalDirection",
    "StructuralReference",
    "range_sweep_events",
]
