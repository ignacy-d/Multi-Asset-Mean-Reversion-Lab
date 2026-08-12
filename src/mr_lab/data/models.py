"""Canonical semantic contracts for completed historical market-data bars."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite


class DataContractError(ValueError):
    """Raised when canonical market data violates its semantic contract."""


class PriceBasis(Enum):
    """The kind of price represented by a bar's OHLC values."""

    BID = "bid"
    ASK = "ask"
    MID = "mid"
    TRADE = "trade"
    OTHER = "other"
    UNKNOWN = "unknown"


class VolumeSemantics(Enum):
    """The meaning of a bar's optional volume value."""

    TRADED = "traded"
    TICK = "tick"
    QUOTE_ACTIVITY = "quote_activity"
    UNKNOWN = "unknown"
    NONE = "none"


_TIMEFRAME_PATTERN = re.compile(r"([1-9][0-9]*)([smhd])")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400}


@dataclass(frozen=True, slots=True)
class Timeframe:
    """A canonical, provider-independent fixed duration."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _TIMEFRAME_PATTERN.fullmatch(
            self.value
        ):
            raise DataContractError(
                "timeframe must use lowercase fixed-duration syntax such as '15m'"
            )

    @classmethod
    def parse(cls, value: str) -> "Timeframe":
        """Parse canonical notation without accepting provider aliases."""
        return cls(value)

    @property
    def duration(self) -> timedelta:
        """Return the represented fixed duration."""
        match = _TIMEFRAME_PATTERN.fullmatch(self.value)
        assert match is not None  # Construction validates this invariant.
        amount, unit = match.groups()
        return timedelta(seconds=int(amount) * _UNIT_SECONDS[unit])

    def __str__(self) -> str:
        return self.value


def _require_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"{name} must be a non-empty string")


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise DataContractError(f"{name} must be a {enum_type.__name__}")


def _require_utc(name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DataContractError(f"{name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise DataContractError(f"{name} must use the canonical UTC timezone")


def _require_finite_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
    ):
        raise DataContractError(f"{name} must be a finite number")


@dataclass(frozen=True, slots=True)
class Bar:
    """One immutable completed bar; this does not prescribe dataset storage."""

    instrument: str
    timeframe: Timeframe
    open_time: datetime
    close_time: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    price_basis: PriceBasis
    volume: float | None = None
    volume_semantics: VolumeSemantics = VolumeSemantics.NONE

    def __post_init__(self) -> None:
        _require_non_empty_string("instrument", self.instrument)
        if not isinstance(self.timeframe, Timeframe):
            raise DataContractError("timeframe must be a Timeframe")
        for name in ("open_time", "close_time", "available_at"):
            _require_utc(name, getattr(self, name))
        if self.open_time >= self.close_time:
            raise DataContractError("open_time must be before close_time")
        if self.available_at < self.close_time:
            raise DataContractError("available_at must be at or after close_time")

        for name in ("open", "high", "low", "close"):
            _require_finite_number(name, getattr(self, name))
        if self.high < max(self.open, self.close, self.low):
            raise DataContractError("high must be at least open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise DataContractError("low must be at most open, close, and high")

        _require_enum("price_basis", self.price_basis, PriceBasis)
        _require_enum("volume_semantics", self.volume_semantics, VolumeSemantics)
        if self.volume is None:
            if self.volume_semantics is not VolumeSemantics.NONE:
                raise DataContractError("volume=None requires NONE volume semantics")
        else:
            _require_finite_number("volume", self.volume)
            if self.volume < 0:
                raise DataContractError("volume must be non-negative")
            if self.volume_semantics is VolumeSemantics.NONE:
                raise DataContractError("numeric volume cannot use NONE semantics")

    def is_available_at(self, research_time: datetime) -> bool:
        """Return whether this complete bar is legally usable at research time."""
        _require_utc("research_time", research_time)
        return self.available_at <= research_time


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Immutable provenance and semantic metadata for a canonical dataset."""

    source: str
    instrument: str
    price_basis: PriceBasis
    volume_semantics: VolumeSemantics
    schema_version: str
    dataset_id: str
    native_timeframe: Timeframe | None = None
    source_timezone: str | None = None

    def __post_init__(self) -> None:
        for name in ("source", "instrument", "schema_version", "dataset_id"):
            _require_non_empty_string(name, getattr(self, name))
        _require_enum("price_basis", self.price_basis, PriceBasis)
        _require_enum("volume_semantics", self.volume_semantics, VolumeSemantics)
        if self.native_timeframe is not None and not isinstance(
            self.native_timeframe, Timeframe
        ):
            raise DataContractError("native_timeframe must be a Timeframe or None")
        if self.source_timezone is not None:
            _require_non_empty_string("source_timezone", self.source_timezone)
