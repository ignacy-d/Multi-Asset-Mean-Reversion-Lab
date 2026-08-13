"""Historical market-session and named research-window classification."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from hashlib import sha256
from types import MappingProxyType
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mr_lab.data import Bar


class SessionSpecError(ValueError):
    """Raised when a session specification or timestamp is invalid."""


def _validate_name(name: str, field: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise SessionSpecError(f"{field} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """An immutable half-open local-clock interval, optionally weekday-limited.

    Weekdays use ``datetime.weekday()`` values (Monday=0 through Sunday=6).
    For a window crossing midnight, the active weekday is the local date on
    which the interval starts.
    """

    name: str
    timezone: str
    start: time
    end: time
    active_weekdays: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "window name")
        _validate_name(self.timezone, "timezone")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise SessionSpecError(f"unknown IANA timezone: {self.timezone}") from error
        if not isinstance(self.start, time) or not isinstance(self.end, time):
            raise SessionSpecError("start and end must be datetime.time values")
        if self.start.tzinfo is not None or self.end.tzinfo is not None:
            raise SessionSpecError("start and end must be timezone-naive local clocks")
        if self.start == self.end:
            raise SessionSpecError("start and end must differ")
        if self.active_weekdays is not None:
            try:
                weekdays = tuple(sorted(set(self.active_weekdays)))
            except TypeError as error:
                raise SessionSpecError("active weekdays must be integers") from error
            if any(type(day) is not int or not 0 <= day <= 6 for day in weekdays):
                raise SessionSpecError("active weekdays must be integers from 0 to 6")
            if not weekdays:
                raise SessionSpecError("active weekdays cannot be empty")
            object.__setattr__(self, "active_weekdays", weekdays)

    def contains(self, timestamp: datetime) -> tuple[bool, datetime]:
        """Return membership and the corresponding timezone-aware local time."""
        _require_canonical_utc(timestamp)
        local = timestamp.astimezone(ZoneInfo(self.timezone))
        clock = local.timetz().replace(tzinfo=None)
        crosses_midnight = self.start > self.end
        active = (
            self.start <= clock < self.end
            if not crosses_midnight
            else clock >= self.start or clock < self.end
        )
        weekday = local.weekday()
        if crosses_midnight and clock < self.end:
            weekday = (weekday - 1) % 7
        if self.active_weekdays is not None:
            active = active and weekday in self.active_weekdays
        return active, local

    def as_dict(self) -> dict[str, object]:
        """Return stable semantic data suitable for canonical JSON."""
        return {
            "active_weekdays": (
                list(self.active_weekdays) if self.active_weekdays is not None else None
            ),
            "end": self.end.isoformat(),
            "name": self.name,
            "start": self.start.isoformat(),
            "timezone": self.timezone,
        }


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Versioned collection of major sessions and named hypothesis windows."""

    version: str
    major_sessions: tuple[TimeWindow, ...]
    named_windows: tuple[TimeWindow, ...] = ()

    def __post_init__(self) -> None:
        _validate_name(self.version, "spec version")
        for field in ("major_sessions", "named_windows"):
            windows = getattr(self, field)
            if not isinstance(windows, tuple) or not all(
                isinstance(window, TimeWindow) for window in windows
            ):
                raise SessionSpecError(f"{field} must be a tuple of TimeWindow values")
        names = [window.name for window in (*self.major_sessions, *self.named_windows)]
        if len(names) != len(set(names)):
            raise SessionSpecError("window names must be globally unique")

    def as_dict(self) -> dict[str, object]:
        """Return canonical, declaration-order-independent semantic data."""
        return {
            "major_sessions": [
                window.as_dict()
                for window in sorted(self.major_sessions, key=lambda item: item.name)
            ],
            "named_windows": [
                window.as_dict()
                for window in sorted(self.named_windows, key=lambda item: item.name)
            ],
            "version": self.version,
        }

    def to_json(self) -> str:
        """Serialize the specification as deterministic compact canonical JSON."""
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def session_spec_id(self) -> str:
        """Return the SHA-256 identity of semantic specification inputs."""
        return f"sha256:{sha256(self.to_json().encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SessionClassification:
    """Deterministic labels for one canonical UTC timestamp."""

    timestamp: datetime
    session_membership: Mapping[str, bool]
    active_sessions: tuple[str, ...]
    regime: str
    active_named_windows: tuple[str, ...]
    local_times: Mapping[str, datetime]
    session_spec_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_membership", MappingProxyType(dict(self.session_membership))
        )
        object.__setattr__(
            self, "local_times", MappingProxyType(dict(self.local_times))
        )


def _require_canonical_utc(timestamp: datetime) -> None:
    if (
        not isinstance(timestamp, datetime)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != timedelta(0)
    ):
        raise SessionSpecError(
            "timestamp must be a timezone-aware canonical UTC datetime"
        )


def classify_timestamp(timestamp: datetime, spec: SessionSpec) -> SessionClassification:
    """Classify one canonical UTC instant using historical IANA timezone rules."""
    _require_canonical_utc(timestamp)
    if not isinstance(spec, SessionSpec):
        raise SessionSpecError("spec must be a SessionSpec")
    canonical_timestamp = timestamp.astimezone(UTC)
    membership: dict[str, bool] = {}
    local_times: dict[str, datetime] = {}
    for window in sorted(spec.major_sessions, key=lambda item: item.name):
        membership[window.name], local_times[window.name] = window.contains(
            canonical_timestamp
        )
    active_sessions = tuple(name for name, active in membership.items() if active)
    if not active_sessions:
        regime = "outside_major_sessions"
    elif len(active_sessions) == 1:
        regime = f"{active_sessions[0]}_only"
    else:
        regime = f"{'_'.join(active_sessions)}_overlap"

    named = []
    for window in sorted(spec.named_windows, key=lambda item: item.name):
        active, local = window.contains(canonical_timestamp)
        local_times[window.name] = local
        if active:
            named.append(window.name)
    return SessionClassification(
        timestamp=canonical_timestamp,
        session_membership=membership,
        active_sessions=active_sessions,
        regime=regime,
        active_named_windows=tuple(named),
        local_times=local_times,
        session_spec_id=spec.session_spec_id,
    )


def classify_bar(bar: Bar, spec: SessionSpec) -> SessionClassification:
    """Classify a completed canonical bar solely by its canonical ``open_time``."""
    if not isinstance(bar, Bar):
        raise SessionSpecError("bar must be a canonical Bar")
    return classify_timestamp(bar.open_time, spec)


def classify_bars(
    bars: Iterable[Bar], spec: SessionSpec
) -> tuple[SessionClassification, ...]:
    """Classify existing bars independently, without filling absent observations."""
    return tuple(classify_bar(bar, spec) for bar in bars)


DEFAULT_SESSION_SPEC = SessionSpec(
    version="stage-1c-research-v1",
    major_sessions=(
        TimeWindow("asia", "Asia/Tokyo", time(9), time(18)),
        TimeWindow("london", "Europe/London", time(8), time(17)),
        TimeWindow("new_york", "America/New_York", time(8), time(17)),
    ),
    named_windows=(
        TimeWindow("asian_kz_20_00_et", "America/New_York", time(20), time(0)),
        TimeWindow("london_kz_02_05_et", "America/New_York", time(2), time(5)),
        TimeWindow("london_core_02_04_et", "America/New_York", time(2), time(4)),
        TimeWindow("new_york_kz_07_10_et", "America/New_York", time(7), time(10)),
        TimeWindow(
            "new_york_kz_0830_1100_et",
            "America/New_York",
            time(8, 30),
            time(11),
        ),
        TimeWindow("london_close_10_12_et", "America/New_York", time(10), time(12)),
        TimeWindow("new_york_lunch_11_13_et", "America/New_York", time(11), time(13)),
    ),
)


__all__ = [
    "DEFAULT_SESSION_SPEC",
    "SessionClassification",
    "SessionSpec",
    "SessionSpecError",
    "TimeWindow",
    "classify_bar",
    "classify_bars",
    "classify_timestamp",
]
