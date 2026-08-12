"""Storage-neutral collection semantics for canonical bars."""

from dataclasses import dataclass
from datetime import datetime

from mr_lab.data.models import (
    Bar,
    DataContractError,
    DatasetMetadata,
    PriceBasis,
    Timeframe,
)


class DatasetValidationError(DataContractError):
    """Raised when a bar collection has a structural inconsistency."""


@dataclass(frozen=True, slots=True)
class Gap:
    """An informational, not necessarily erroneous, absent interval."""

    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Validated observations and informational gaps in chronological order."""

    bars: tuple[Bar, ...]
    gaps: tuple[Gap, ...]


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """Stable identity within a named logical dataset."""

    dataset_id: str
    instrument: str
    timeframe: Timeframe
    open_time: datetime
    price_basis: PriceBasis


def observation_identity(dataset_id: str, bar: Bar) -> ObservationIdentity:
    """Build the deterministic identity of one observation."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise DatasetValidationError("dataset_id must be a non-empty string")
    if not isinstance(bar, Bar):
        raise DatasetValidationError("observation must be a canonical Bar")
    return ObservationIdentity(
        dataset_id, bar.instrument, bar.timeframe, bar.open_time, bar.price_basis
    )


def validate_dataset(
    bars: list[Bar] | tuple[Bar, ...],
    metadata: DatasetMetadata | None = None,
) -> DatasetValidationReport:
    """Validate one homogeneous, chronologically ordered logical dataset."""
    observations = tuple(bars)
    if any(not isinstance(bar, Bar) for bar in observations):
        raise DatasetValidationError("all observations must be canonical Bar instances")
    if not observations:
        return DatasetValidationReport((), ())

    first = observations[0]
    if metadata is not None:
        if not isinstance(metadata, DatasetMetadata):
            raise DatasetValidationError("metadata must be DatasetMetadata")
        expected = (
            metadata.instrument,
            metadata.price_basis,
            metadata.volume_semantics,
        )
        actual = (first.instrument, first.price_basis, first.volume_semantics)
        if actual != expected:
            raise DatasetValidationError("bars do not match dataset metadata semantics")
        if metadata.native_timeframe not in (None, first.timeframe):
            raise DatasetValidationError("bars do not match metadata native_timeframe")

    gaps: list[Gap] = []
    for index, bar in enumerate(observations):
        if (
            bar.instrument != first.instrument
            or bar.price_basis is not first.price_basis
            or bar.volume_semantics is not first.volume_semantics
            or bar.timeframe != first.timeframe
        ):
            raise DatasetValidationError("dataset bars must have homogeneous semantics")
        if bar.close_time - bar.open_time != bar.timeframe.duration:
            raise DatasetValidationError(
                "bar interval must equal its canonical timeframe"
            )
        if index == 0:
            continue
        previous = observations[index - 1]
        if bar == previous:
            raise DatasetValidationError("duplicate observation")
        if (
            bar.open_time == previous.open_time
            and bar.close_time == previous.close_time
        ):
            raise DatasetValidationError("duplicate interval")
        if bar.open_time < previous.open_time:
            raise DatasetValidationError("bars must be in chronological order")
        if bar.open_time < previous.close_time:
            raise DatasetValidationError("bar intervals must not overlap")
        if bar.open_time > previous.close_time:
            gaps.append(Gap(previous.close_time, bar.open_time))
    return DatasetValidationReport(observations, tuple(gaps))


def available_bars(
    bars: list[Bar] | tuple[Bar, ...], research_time: datetime
) -> tuple[Bar, ...]:
    """Return only bars whose explicit availability is at or before research time."""
    # Bar performs the shared Stage 0B UTC validation, including UTC-equivalent tzinfo.
    return tuple(bar for bar in bars if bar.is_available_at(research_time))
