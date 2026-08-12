"""Canonical market-data semantics, independent of providers and storage."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite


class DataContractError(ValueError):
    """Raised when canonical market data violates its semantic contract."""


class PriceBasis(StrEnum):
    """The market observation from which prices originate."""

    BID = "bid"
    ASK = "ask"
    MID = "mid"
    TRADE = "trade"
    OTHER = "other"
    UNKNOWN = "unknown"


class VolumeType(StrEnum):
    """The economic meaning of a volume value, not its unit."""

    TRADED = "traded"
    TICK = "tick"
    QUOTE_ACTIVITY = "quote_activity"
    UNKNOWN = "unknown"
    NONE = "none"


_TIMEFRAME_PATTERN = re.compile(r"(?P<count>[1-9][0-9]*)(?P<unit>[smhd])")
_TIMEFRAME_SECONDS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400}


@dataclass(frozen=True, slots=True)
class Timeframe:
    """Provider-independent fixed-duration bar interval, such as ``5m`` or ``1h``."""

    duration: timedelta

    def __post_init__(self) -> None:
        seconds = self.duration.total_seconds()
        if not isfinite(seconds) or seconds <= 0 or not seconds.is_integer():
            raise DataContractError(
                "timeframe must be a positive whole number of seconds"
            )

    @classmethod
    def parse(cls, value: str) -> "Timeframe":
        """Parse an unambiguous lowercase fixed-duration representation."""
        if not isinstance(value, str):
            raise DataContractError("timeframe must be a string")
        match = _TIMEFRAME_PATTERN.fullmatch(value)
        if match is None:
            raise DataContractError(
                "timeframe must use '<positive integer><s|m|h|d>', for example '15m'"
            )
        seconds = int(match["count"]) * _TIMEFRAME_SECONDS[match["unit"]]
        return cls(timedelta(seconds=seconds))


def _require_utc(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DataContractError(f"{field} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise DataContractError(f"{field} must be UTC")


def _require_finite(value: float, field: str) -> None:
    valid_number = not isinstance(value, bool) and isinstance(value, int | float)
    if not valid_number or not isfinite(value):
        raise DataContractError(f"{field} must be a finite number")


@dataclass(frozen=True, slots=True)
class Bar:
    """One complete canonical OHLC observation.

    ``open_time`` begins the observation interval and ``close_time`` ends it.
    ``available_at`` is the earliest instant at which the complete values may be
    consumed by point-in-time research; it is never inferred from row order.
    """

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
    volume_type: VolumeType = VolumeType.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise DataContractError("instrument must be a non-empty string")
        if not isinstance(self.timeframe, Timeframe):
            raise DataContractError("timeframe must be a Timeframe")
        if not isinstance(self.price_basis, PriceBasis):
            raise DataContractError("price_basis must be a PriceBasis")
        if not isinstance(self.volume_type, VolumeType):
            raise DataContractError("volume_type must be a VolumeType")

        for field in ("open_time", "close_time", "available_at"):
            _require_utc(getattr(self, field), field)
        if self.open_time >= self.close_time:
            raise DataContractError("open_time must be earlier than close_time")
        if self.available_at < self.close_time:
            raise DataContractError("available_at cannot be earlier than close_time")

        for field in ("open", "high", "low", "close"):
            _require_finite(getattr(self, field), field)
        if self.high < max(self.open, self.low, self.close):
            raise DataContractError("high cannot be below another OHLC value")
        if self.low > min(self.open, self.high, self.close):
            raise DataContractError("low cannot be above another OHLC value")

        if self.volume is None:
            if self.volume_type is not VolumeType.NONE:
                raise DataContractError("absent volume requires volume_type 'none'")
        else:
            _require_finite(self.volume, "volume")
            if self.volume < 0:
                raise DataContractError("volume cannot be negative")
            if self.volume_type is VolumeType.NONE:
                raise DataContractError(
                    "present volume requires a non-'none' volume_type"
                )

    def is_available_at(self, research_time: datetime) -> bool:
        """Return whether this complete observation is safe to use at that UTC time."""
        _require_utc(research_time, "research_time")
        return self.available_at <= research_time


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Identity and semantics of a source dataset, without adapter behavior."""

    source: str
    instrument: str
    price_basis: PriceBasis
    volume_type: VolumeType
    schema_version: str
    dataset_id: str
    native_timeframe: Timeframe | None = None
    source_timezone: str | None = None

    def __post_init__(self) -> None:
        for field in ("source", "instrument", "schema_version", "dataset_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise DataContractError(f"{field} must be a non-empty string")
        if not isinstance(self.price_basis, PriceBasis):
            raise DataContractError("price_basis must be a PriceBasis")
        if not isinstance(self.volume_type, VolumeType):
            raise DataContractError("volume_type must be a VolumeType")
        if self.native_timeframe is not None and not isinstance(
            self.native_timeframe, Timeframe
        ):
            raise DataContractError("native_timeframe must be a Timeframe or None")
        if self.source_timezone is not None and (
            not isinstance(self.source_timezone, str)
            or not self.source_timezone.strip()
        ):
            raise DataContractError(
                "source_timezone must be a non-empty string or None"
            )
