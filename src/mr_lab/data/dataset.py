"""Collection validation and point-in-time views for canonical bars."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from mr_lab.data.models import (
    Bar,
    DataContractError,
    DatasetMetadata,
    Timeframe,
    _require_utc,
)


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """Identity of an observation within a particular source dataset."""

    dataset_id: str
    instrument: str
    timeframe: Timeframe
    open_time: datetime
    price_basis: str


@dataclass(frozen=True, slots=True)
class Gap:
    """An informational discontinuity between otherwise valid observations."""

    previous_close: datetime
    next_open: datetime


@dataclass(frozen=True, slots=True)
class DatasetValidation:
    """Successful validation report; gaps are informational rather than errors."""

    observation_count: int
    gaps: tuple[Gap, ...]


def observation_identity(bar: Bar, dataset_id: str) -> ObservationIdentity:
    """Build an identity that cannot merge providers or price bases silently."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise DataContractError("dataset_id must be a non-empty string")
    return ObservationIdentity(
        dataset_id=dataset_id,
        instrument=bar.instrument,
        timeframe=bar.timeframe,
        open_time=bar.open_time,
        price_basis=bar.price_basis.value,
    )


def validate_dataset(
    bars: Iterable[Bar], metadata: DatasetMetadata
) -> DatasetValidation:
    """Validate one logically homogeneous, chronologically ordered dataset."""
    observations = tuple(bars)
    identities: set[ObservationIdentity] = set()
    intervals: set[tuple[datetime, datetime]] = set()
    gaps: list[Gap] = []

    previous: Bar | None = None
    for bar in observations:
        if not isinstance(bar, Bar):
            raise DataContractError("dataset values must be Bar instances")
        if bar.instrument != metadata.instrument:
            raise DataContractError("dataset contains inconsistent instruments")
        if bar.price_basis is not metadata.price_basis:
            raise DataContractError("dataset contains inconsistent price bases")
        if bar.volume_type is not metadata.volume_type:
            raise DataContractError("dataset contains inconsistent volume semantics")
        if (
            metadata.native_timeframe is not None
            and bar.timeframe != metadata.native_timeframe
        ):
            raise DataContractError("dataset contains inconsistent timeframes")

        identity = observation_identity(bar, metadata.dataset_id)
        if identity in identities:
            raise DataContractError("dataset contains a duplicate observation identity")
        identities.add(identity)

        interval = (bar.open_time, bar.close_time)
        if interval in intervals:
            raise DataContractError("dataset contains a duplicate interval")
        intervals.add(interval)

        if previous is not None:
            if bar.open_time < previous.open_time:
                raise DataContractError("dataset is not in chronological order")
            if bar.open_time < previous.close_time:
                raise DataContractError("dataset contains overlapping intervals")
            if bar.open_time > previous.close_time:
                gaps.append(Gap(previous.close_time, bar.open_time))
        previous = bar

    return DatasetValidation(len(observations), tuple(gaps))


def available_bars(bars: Iterable[Bar], research_time: datetime) -> tuple[Bar, ...]:
    """Return observations whose complete information was available by UTC time."""
    _require_utc(research_time, "research_time")
    return tuple(bar for bar in bars if bar.available_at <= research_time)
