"""Deterministic Stage 1B assembly of canonical Dukascopy daily payloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime

from mr_lab.data import Bar, DatasetMetadata, Timeframe, resample_bars, validate_dataset
from mr_lab.providers.dukascopy_bi5 import (
    CANONICAL_SCHEMA_VERSION,
    INSTRUMENT,
    M1,
    PARSER_SCHEMA_VERSION,
    PROVIDER,
    SOURCE_TIMEZONE,
    VOLUME_SEMANTICS,
    build_dataset_metadata,
    parse_m1_bid_bars,
)
from mr_lab.providers.instruments import get_instrument_spec

MANIFEST_SCHEMA_VERSION = "dukascopy-multiday-manifest-v1"


class MultiDayAssemblyError(ValueError):
    """Raised when daily inputs cannot form one Stage 1B logical dataset."""


@dataclass(frozen=True, slots=True)
class DailyPayload:
    """One immutable raw payload explicitly associated with its requested UTC day."""

    requested_day: date
    payload: bytes
    instrument: str = INSTRUMENT

    def __post_init__(self) -> None:
        if not isinstance(self.requested_day, date) or isinstance(
            self.requested_day, datetime
        ):
            raise MultiDayAssemblyError("requested_day must be a date")
        if not isinstance(self.payload, bytes):
            raise MultiDayAssemblyError("payload must be immutable bytes")
        try:
            spec = get_instrument_spec(self.instrument)
        except ValueError as error:
            raise MultiDayAssemblyError(str(error)) from error
        object.__setattr__(self, "instrument", spec.instrument)


@dataclass(frozen=True, slots=True)
class ComponentDay:
    """Stable identity and provenance for one successfully canonicalized day."""

    requested_day: date
    raw_sha256: str
    daily_dataset_id: str

    def as_dict(self) -> dict[str, str]:
        """Return this component in the stable manifest representation."""
        return {
            "requested_day": self.requested_day.isoformat(),
            "raw_sha256": self.raw_sha256,
            "daily_dataset_id": self.daily_dataset_id,
        }


@dataclass(frozen=True, slots=True)
class MultiDayManifest:
    """Deterministic, serializable audit for an assembled canonical dataset."""

    dataset_id: str
    metadata: DatasetMetadata
    components: tuple[ComponentDay, ...]
    first_open_time: datetime
    final_close_time: datetime
    m1_bar_count: int
    gap_count: int
    resample_counts: tuple[tuple[str, int], ...]
    incomplete_window_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit with deterministic field contents."""
        return {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "assembled_dataset_id": self.dataset_id,
            "provider": self.metadata.source,
            "instrument": self.metadata.instrument,
            "native_timeframe": str(self.metadata.native_timeframe),
            "price_basis": self.metadata.price_basis.value,
            "volume_semantics": self.metadata.volume_semantics.value,
            "source_timezone": self.metadata.source_timezone,
            "canonical_schema_version": self.metadata.schema_version,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "first_open_time": self.first_open_time.isoformat(),
            "final_close_time": self.final_close_time.isoformat(),
            "m1_bar_count": self.m1_bar_count,
            "component_day_count": len(self.components),
            "component_dates": [
                component.requested_day.isoformat() for component in self.components
            ],
            "components": [component.as_dict() for component in self.components],
            "gap_count": self.gap_count,
            "resample_counts": dict(self.resample_counts),
            "incomplete_window_counts": dict(self.incomplete_window_counts),
        }

    def to_json(self) -> str:
        """Serialize the audit as canonical sorted compact JSON."""
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )


@dataclass(frozen=True, slots=True)
class MultiDayDataset:
    """An assembled logical dataset and its deterministic provenance manifest."""

    bars: tuple[Bar, ...]
    metadata: DatasetMetadata
    manifest: MultiDayManifest


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def assemble_daily_payloads(
    daily_payloads: list[DailyPayload] | tuple[DailyPayload, ...],
) -> MultiDayDataset:
    """Canonicalize and assemble explicitly dated raw days in date/time order."""
    inputs = tuple(daily_payloads)
    if not inputs:
        raise MultiDayAssemblyError("at least one daily payload is required")
    if any(not isinstance(item, DailyPayload) for item in inputs):
        raise MultiDayAssemblyError("all inputs must be DailyPayload instances")

    instruments = {item.instrument for item in inputs}
    if len(instruments) != 1:
        raise MultiDayAssemblyError(
            "daily payloads must contain exactly one instrument"
        )
    spec = get_instrument_spec(instruments.pop())

    ordered = sorted(inputs, key=lambda item: item.requested_day)
    days = [item.requested_day for item in ordered]
    if len(days) != len(set(days)):
        raise MultiDayAssemblyError("duplicate requested day")

    components: list[ComponentDay] = []
    bars: list[Bar] = []
    for item in ordered:
        # Stage 1A remains the sole BI5 decoder and daily identity implementation.
        daily_bars = parse_m1_bid_bars(
            item.payload, item.requested_day, item.instrument
        )
        daily_metadata = build_dataset_metadata(
            item.payload, item.requested_day, item.instrument
        )
        validate_dataset(daily_bars, daily_metadata)
        components.append(
            ComponentDay(
                requested_day=item.requested_day,
                raw_sha256=hashlib.sha256(item.payload).hexdigest(),
                daily_dataset_id=daily_metadata.dataset_id,
            )
        )
        bars.extend(daily_bars)

    identity_inputs = {
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "components": [component.as_dict() for component in components],
        "instrument": spec.instrument,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "native_timeframe": M1.value,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "price_basis": "bid",
        "provider": PROVIDER,
        "source_timezone": SOURCE_TIMEZONE,
        "volume_semantics": VOLUME_SEMANTICS.value,
    }
    if spec.instrument != INSTRUMENT:
        identity_inputs["instrument_spec"] = spec.as_dict()
    dataset_id = (
        f"sha256:{hashlib.sha256(_canonical_json(identity_inputs)).hexdigest()}"
    )
    metadata = DatasetMetadata(
        source=spec.provider,
        instrument=spec.instrument,
        price_basis=daily_metadata.price_basis,
        volume_semantics=VOLUME_SEMANTICS,
        schema_version=CANONICAL_SCHEMA_VERSION,
        dataset_id=dataset_id,
        native_timeframe=M1,
        source_timezone=SOURCE_TIMEZONE,
    )
    report = validate_dataset(bars, metadata)
    resamples = tuple(
        (name, resample_bars(report.bars, Timeframe(timeframe)))
        for name, timeframe in (("m5", "5m"), ("m15", "15m"), ("h1", "1h"))
    )
    manifest = MultiDayManifest(
        dataset_id=dataset_id,
        metadata=metadata,
        components=tuple(components),
        first_open_time=report.bars[0].open_time,
        final_close_time=report.bars[-1].close_time,
        m1_bar_count=len(report.bars),
        gap_count=len(report.gaps),
        resample_counts=tuple((name, len(result.bars)) for name, result in resamples),
        incomplete_window_counts=tuple(
            (name, len(result.incomplete_windows)) for name, result in resamples
        ),
    )
    return MultiDayDataset(report.bars, metadata, manifest)
