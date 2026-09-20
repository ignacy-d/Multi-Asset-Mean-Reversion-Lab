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
from mr_lab.sessions import DEFAULT_SESSION_SPEC, SessionSpec, TimeWindow, classify_bar


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
    """Configuration for reference construction and optional pip diagnostics."""

    session_spec: SessionSpec = DEFAULT_SESSION_SPEC
    asia_session_name: str = "asia"
    fx_timezone: str = "America/New_York"
    fx_day_boundary: time = time(17)
    pip_sizes: Mapping[str, float] | None = None


def _window_bounds(local_day: date, window: TimeWindow) -> tuple[datetime, datetime]:
    timezone = ZoneInfo(window.timezone)
    start = datetime.combine(local_day, window.start, timezone)
    end_day = local_day + timedelta(days=window.start >= window.end)
    end = datetime.combine(end_day, window.end, timezone)
    return start.astimezone(UTC), end.astimezone(UTC)


def _fx_bounds(
    timestamp: datetime, config: RangeSweepConfig
) -> tuple[datetime, datetime]:
    timezone = ZoneInfo(config.fx_timezone)
    local = timestamp.astimezone(timezone)
    boundary = datetime.combine(local.date(), config.fx_day_boundary, timezone)
    if local < boundary:
        boundary = datetime.combine(
            local.date() - timedelta(days=1), config.fx_day_boundary, timezone
        )
    end = datetime.combine(
        boundary.date() + timedelta(days=1), config.fx_day_boundary, timezone
    )
    return boundary.astimezone(UTC), end.astimezone(UTC)


def _asia_window(config: RangeSweepConfig) -> TimeWindow:
    matches = tuple(
        window
        for window in config.session_spec.major_sessions
        if window.name == config.asia_session_name
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


def _references(
    bars: tuple[Bar, ...], config: RangeSweepConfig
) -> tuple[StructuralReference, ...]:
    asia = _asia_window(config)
    grouped: dict[tuple[ReferenceFamily, datetime, datetime], list[Bar]] = {}
    for bar in bars:
        fx_start, fx_end = _fx_bounds(bar.open_time, config)
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
        # Delayed observations unavailable at finalization cannot revise a level.
        known = tuple(bar for bar in components if bar.available_at <= end)
        if not known:
            continue
        instance = f"{start.isoformat()}/{end.isoformat()}"
        for side, price in (
            (ReferenceSide.HIGH, max(bar.high for bar in known)),
            (ReferenceSide.LOW, min(bar.low for bar in known)),
        ):
            references.append(
                StructuralReference(
                    family, instance, side, known[0].instrument, price, end
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
    if any(bar.timeframe != Timeframe("5m") for bar in observations):
        raise DataContractError("range sweep signals require canonical 5m bars")
    references = _references(observations, effective)
    consumed: set[tuple[ReferenceFamily, str, ReferenceSide, str]] = set()
    events: list[RangeSweepEvent] = []
    for bar in observations:
        latest: dict[tuple[ReferenceFamily, ReferenceSide], StructuralReference] = {}
        for reference in references:
            if reference.valid_at <= bar.open_time:
                latest[(reference.family, reference.side)] = reference
        classification = classify_bar(bar, effective.session_spec)
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
                    event_id=_event_id(reference, bar.close_time),
                    reference_family=reference.family,
                    reference_instance=reference.instance,
                    reference_side=reference.side,
                    reference_price=reference.price,
                    signal_direction=direction,
                    instrument=bar.instrument,
                    signal_timestamp=bar.close_time,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    overshoot_raw_price=overshoot,
                    overshoot_pips=None if pip_size is None else overshoot / pip_size,
                    minutes_since_reference_valid=(
                        bar.close_time - reference.valid_at
                    ).total_seconds()
                    / 60,
                    session_labels=labels,
                    volume=bar.volume,
                )
            )
    return tuple(events)


__all__ = [
    "RangeSweepConfig",
    "RangeSweepEvent",
    "ReferenceFamily",
    "ReferenceSide",
    "SignalDirection",
    "StructuralReference",
    "range_sweep_events",
]
