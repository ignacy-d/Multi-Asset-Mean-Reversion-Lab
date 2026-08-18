"""Bounded Stage 1D acquisition and audit for an explicit UTC date range."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from mr_lab.data import PriceBasis
from mr_lab.providers.dukascopy import ProviderNoData, acquire
from mr_lab.providers.dukascopy_bi5 import (
    CANONICAL_SCHEMA_VERSION,
    INSTRUMENT,
    PARSER_SCHEMA_VERSION,
    SOURCE_TIMEZONE,
    VOLUME_SEMANTICS,
)
from mr_lab.providers.dukascopy_multiday import (
    DailyPayload,
    MultiDayDataset,
    assemble_daily_payloads,
)
from mr_lab.providers.instruments import get_instrument_spec

CORPUS_SCHEMA_VERSION = "dukascopy-eurusd-corpus-v1"
GENERIC_CORPUS_SCHEMA_VERSION = "dukascopy-multi-asset-corpus-v1"
MAX_CALENDAR_DAYS = 370
DISCOVERY_START = date(2024, 1, 1)
DISCOVERY_END = date(2024, 12, 31)
HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2025, 12, 31)


class RangeAcquisitionError(ValueError):
    """Raised when a range request or its acquired components are invalid."""


def load_offline_corpus(corpus_dir: Path) -> MultiDayDataset:
    """Reconstruct a Stage 1B dataset from manifest-declared Stage 1D dates only."""
    manifest_path = corpus_dir / "corpus-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = tuple(
            date.fromisoformat(value)
            for value in manifest["successful_component_dates"]
        )
        components = manifest["components"]
        expected_dataset_id = manifest["assembled_dataset_id"]
        instrument = get_instrument_spec(
            manifest.get("instrument", INSTRUMENT)
        ).instrument
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RangeAcquisitionError(
            "invalid or missing offline corpus manifest"
        ) from error
    if dates != tuple(sorted(set(dates))) or len(components) != len(dates):
        raise RangeAcquisitionError(
            "manifest component dates must be unique and ordered"
        )
    declared = {
        item.get("requested_day"): item for item in components if isinstance(item, dict)
    }
    payloads = []
    for day in dates:
        stem = f"{instrument}-{day.isoformat()}-M1-BID"
        try:
            raw = (corpus_dir / f"{stem}.bi5").read_bytes()
            provenance = json.loads(
                (corpus_dir / f"{stem}.json").read_text(encoding="utf-8")
            )
            component = declared[day.isoformat()]
        except (OSError, json.JSONDecodeError, KeyError) as error:
            raise RangeAcquisitionError(
                f"missing or corrupt offline component for {day}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if (
            provenance.get("requested_date") != day.isoformat()
            or provenance.get("canonical_instrument", INSTRUMENT) != instrument
            or provenance.get("sha256") != digest
        ):
            raise RangeAcquisitionError(f"provenance mismatch for {day}")
        if component.get("raw_sha256") != digest:
            raise RangeAcquisitionError(f"manifest payload hash mismatch for {day}")
        payloads.append(DailyPayload(day, raw, instrument))
    try:
        dataset = assemble_daily_payloads(payloads)
    except (OSError, ValueError) as error:
        raise RangeAcquisitionError("offline corpus canonicalization failed") from error
    if dataset.metadata.dataset_id != expected_dataset_id:
        raise RangeAcquisitionError(
            "reconstructed dataset identity does not match manifest"
        )
    return dataset


def enumerate_dates(start_date: date, end_date: date) -> tuple[date, ...]:
    """Return an explicit inclusive range in ascending calendar-date order."""
    if any(
        not isinstance(value, date) or isinstance(value, datetime)
        for value in (start_date, end_date)
    ):
        raise RangeAcquisitionError("start_date and end_date must be dates")
    if start_date > end_date:
        raise RangeAcquisitionError("start_date must be on or before end_date")
    count = (end_date - start_date).days + 1
    if count > MAX_CALENDAR_DAYS:
        raise RangeAcquisitionError(
            f"requested range exceeds the {MAX_CALENDAR_DAYS}-day safety limit"
        )
    return tuple(start_date + timedelta(days=offset) for offset in range(count))


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Deterministic request-level identity layered above Stage 1B identity."""

    requested_start_date: date
    requested_end_date: date
    confirmed_absent_dates: tuple[date, ...]
    dataset: MultiDayDataset

    def identity_inputs(self) -> dict[str, object]:
        """Return all and only stable inputs to the corpus identity."""
        assembled = self.dataset.manifest
        spec = get_instrument_spec(self.dataset.metadata.instrument)
        values = {
            "corpus_schema_version": (
                CORPUS_SCHEMA_VERSION
                if spec.instrument == INSTRUMENT
                else GENERIC_CORPUS_SCHEMA_VERSION
            ),
            "provider": spec.provider,
            "instrument": spec.instrument,
            "requested_start_date": self.requested_start_date.isoformat(),
            "requested_end_date": self.requested_end_date.isoformat(),
            "requested_calendar_day_count": len(
                enumerate_dates(self.requested_start_date, self.requested_end_date)
            ),
            "successful_component_dates": [
                component.requested_day.isoformat()
                for component in assembled.components
            ],
            "confirmed_absent_dates": [
                day.isoformat() for day in self.confirmed_absent_dates
            ],
            "successful_component_count": len(assembled.components),
            "absent_component_count": len(self.confirmed_absent_dates),
            "components": [component.as_dict() for component in assembled.components],
            "assembled_dataset_id": assembled.dataset_id,
            "first_canonical_bar_open_time": assembled.first_open_time.isoformat(),
            "final_canonical_bar_close_time": assembled.final_close_time.isoformat(),
            "m1_bar_count": assembled.m1_bar_count,
            "gap_count": assembled.gap_count,
            "resample_counts": dict(assembled.resample_counts),
            "incomplete_window_counts": dict(assembled.incomplete_window_counts),
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "native_timeframe": "1m",
            "price_basis": PriceBasis.BID.value,
            "volume_semantics": VOLUME_SEMANTICS.value,
            "source_timezone": SOURCE_TIMEZONE,
        }
        if spec.instrument != INSTRUMENT:
            values["instrument_spec"] = spec.as_dict()
        return values

    @property
    def corpus_id(self) -> str:
        """Hash canonical compact JSON without operational acquisition metadata."""
        digest = hashlib.sha256(self.identity_json().encode()).hexdigest()
        return f"sha256:{digest}"

    def identity_json(self) -> str:
        """Serialize stable corpus identity inputs as canonical compact JSON."""
        return _canonical_json(self.identity_inputs())

    def as_dict(self) -> dict[str, object]:
        """Return the complete deterministic audit, including its identity."""
        return {"corpus_id": self.corpus_id, **self.identity_inputs()}

    def to_json(self) -> str:
        """Serialize the complete deterministic audit as canonical compact JSON."""
        return _canonical_json(self.as_dict())


def build_corpus_manifest(
    start_date: date,
    end_date: date,
    successful_payloads: Iterable[DailyPayload],
    confirmed_absent_dates: Iterable[date],
    instrument: str = INSTRUMENT,
) -> CorpusManifest:
    """Assemble successful days and bind them to the complete range request."""
    requested = enumerate_dates(start_date, end_date)
    spec = get_instrument_spec(instrument)
    payloads = tuple(successful_payloads)
    if any(item.instrument != spec.instrument for item in payloads):
        raise RangeAcquisitionError("successful payload instrument mismatch")
    absent = tuple(sorted(confirmed_absent_dates))
    successful_dates = tuple(sorted(item.requested_day for item in payloads))
    if len(absent) != len(set(absent)):
        raise RangeAcquisitionError("duplicate confirmed absent date")
    if set(absent) & set(successful_dates):
        raise RangeAcquisitionError("a date cannot be both successful and absent")
    if set(absent) | set(successful_dates) != set(requested):
        raise RangeAcquisitionError("every requested date must be successful or absent")
    dataset = assemble_daily_payloads(payloads)
    return CorpusManifest(start_date, end_date, absent, dataset)


@dataclass(frozen=True, slots=True)
class RangeAcquisitionResult:
    """Paths and logical outputs produced by one completed range acquisition."""

    manifest: CorpusManifest
    raw_paths: tuple[Path, ...]
    provenance_paths: tuple[Path, ...]
    manifest_path: Path


def acquire_range(
    output_dir: Path,
    start_date: date,
    end_date: date,
    *,
    timeout: float = 30.0,
    retries: int = 6,
    delay_seconds: float = 1.0,
    acquire_day: Callable[..., tuple[Path, Path]] = acquire,
    sleeper: Callable[[float], None] = time.sleep,
    logger: Callable[[str], None] = print,
    instrument: str = INSTRUMENT,
) -> RangeAcquisitionResult:
    """Sequentially acquire, validate, assemble, and audit an inclusive range.

    Only ``ProviderNoData`` (the provider's HTTP 404) is an absent day. Every
    transport, server, payload, parsing, or assembly error propagates and fails
    the request. No bars or dates are synthesized.
    """
    spec = get_instrument_spec(instrument)
    if start_date < DISCOVERY_START or end_date > DISCOVERY_END:
        raise RangeAcquisitionError(
            "Stage 3 acquisition accepts only 2024 discovery dates"
        )
    days = enumerate_dates(start_date, end_date)
    if delay_seconds < 0:
        raise RangeAcquisitionError("delay_seconds must be non-negative")
    raw_paths: list[Path] = []
    provenance_paths: list[Path] = []
    payloads: list[DailyPayload] = []
    absent: list[date] = []
    for index, day in enumerate(days):
        progress = f"[{index + 1}/{len(days)}]"
        logger(f"{progress} acquiring {day.isoformat()}")
        try:
            kwargs = {"timeout": timeout, "retries": retries}
            if spec.instrument != INSTRUMENT:
                kwargs["instrument"] = spec.instrument
            raw_path, provenance_path = acquire_day(output_dir, day, **kwargs)
        except ProviderNoData:
            absent.append(day)
            logger(f"{progress} absent {day.isoformat()}")
        else:
            raw_paths.append(raw_path)
            provenance_paths.append(provenance_path)
            payloads.append(DailyPayload(day, raw_path.read_bytes(), spec.instrument))
        if delay_seconds and index != len(days) - 1:
            sleeper(delay_seconds)

    manifest = build_corpus_manifest(
        start_date, end_date, payloads, absent, spec.instrument
    )
    manifest_path = output_dir / "corpus-manifest.json"
    if manifest_path.exists():
        raise RangeAcquisitionError("refusing to overwrite an existing corpus manifest")
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return RangeAcquisitionResult(
        manifest,
        tuple(raw_paths),
        tuple(provenance_paths),
        manifest_path,
    )


def main() -> None:
    """Acquire one explicitly bounded range; dates never default to today."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instrument", default=INSTRUMENT)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    result = acquire_range(
        args.output_dir,
        args.start_date,
        args.end_date,
        delay_seconds=args.delay_seconds,
        instrument=args.instrument,
    )
    print(f"corpus_manifest={result.manifest_path}")
    print(f"corpus_id={result.manifest.corpus_id}")
    print(f"assembled_dataset_id={result.manifest.dataset.metadata.dataset_id}")


if __name__ == "__main__":
    main()
