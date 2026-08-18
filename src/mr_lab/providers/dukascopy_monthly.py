"""Checkpointed monthly acquisition and deterministic full-year assembly."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mr_lab.providers.dukascopy_bi5 import INSTRUMENT
from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import (
    RangeAcquisitionError,
    RangeAcquisitionResult,
    acquire_range,
    build_corpus_manifest,
)
from mr_lab.providers.instruments import get_instrument_spec

DISCOVERY_YEAR = 2024


def month_bounds(year: int, month: int) -> tuple[date, date]:
    """Return the exact calendar bounds for one supported discovery month."""
    if year != DISCOVERY_YEAR:
        raise RangeAcquisitionError("monthly acquisition accepts only 2024")
    if month not in range(1, 13):
        raise RangeAcquisitionError("month must be between 1 and 12")
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def acquire_month(
    output_dir: Path,
    year: int,
    month: int,
    *,
    instrument: str = INSTRUMENT,
    delay_seconds: float = 2.0,
    **kwargs: object,
) -> RangeAcquisitionResult:
    """Acquire one independently checkpointable month, sequentially."""
    start, end = month_bounds(year, month)
    return acquire_range(
        output_dir,
        start,
        end,
        instrument=instrument,
        delay_seconds=delay_seconds,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class YearAssemblyResult:
    """The assembled manifest and its immutable corpus directory."""

    manifest_path: Path
    corpus_id: str
    dataset_id: str


def _read_month(
    directory: Path, instrument: str, expected_start: date, expected_end: date
) -> tuple[list[DailyPayload], list[date], list[tuple[Path, Path]]]:
    try:
        manifest = json.loads((directory / "corpus-manifest.json").read_text())
        start = date.fromisoformat(manifest["requested_start_date"])
        end = date.fromisoformat(manifest["requested_end_date"])
        declared_instrument = manifest["instrument"]
        components = manifest["components"]
        absent = [
            date.fromisoformat(value) for value in manifest["confirmed_absent_dates"]
        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RangeAcquisitionError(f"invalid monthly corpus at {directory}") from error
    if (start, end) != (expected_start, expected_end):
        raise RangeAcquisitionError(
            f"missing or incorrect month {expected_start:%Y-%m}"
        )
    if declared_instrument != instrument:
        raise RangeAcquisitionError("mixed-instrument monthly corpora")
    payloads: list[DailyPayload] = []
    paths: list[tuple[Path, Path]] = []
    seen: set[date] = set()
    for component in components:
        try:
            day = date.fromisoformat(component["requested_day"])
            expected_hash = component["raw_sha256"]
        except (KeyError, TypeError, ValueError) as error:
            raise RangeAcquisitionError(
                "invalid monthly component declaration"
            ) from error
        if day in seen:
            raise RangeAcquisitionError(f"duplicate day {day}")
        seen.add(day)
        stem = f"{instrument}-{day.isoformat()}-M1-BID"
        raw_path, provenance_path = (
            directory / f"{stem}.bi5",
            directory / f"{stem}.json",
        )
        try:
            raw = raw_path.read_bytes()
            provenance = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RangeAcquisitionError(f"missing component for {day}") from error
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != expected_hash
            or provenance.get("sha256") != digest
            or provenance.get("requested_date") != day.isoformat()
            or provenance.get("canonical_instrument", INSTRUMENT) != instrument
        ):
            raise RangeAcquisitionError(f"raw provenance mismatch for {day}")
        payloads.append(DailyPayload(day, raw, instrument))
        paths.append((raw_path, provenance_path))
    expected_days = {
        date.fromordinal(value)
        for value in range(start.toordinal(), end.toordinal() + 1)
    }
    if seen & set(absent) or seen | set(absent) != expected_days:
        raise RangeAcquisitionError(f"missing or duplicated day in {start:%Y-%m}")
    if len(absent) != len(set(absent)):
        raise RangeAcquisitionError(f"duplicate absent day in {start:%Y-%m}")
    return payloads, absent, paths


def assemble_year(
    chunks_root: Path,
    output_dir: Path,
    *,
    year: int = DISCOVERY_YEAR,
    instrument: str = INSTRUMENT,
) -> YearAssemblyResult:
    """Verify exactly 12 monthly checkpoints and reproduce full-year semantics."""
    spec = get_instrument_spec(instrument)
    if year != DISCOVERY_YEAR:
        raise RangeAcquisitionError("year assembly accepts only 2024")
    manifests = sorted(chunks_root.rglob("corpus-manifest.json"))
    if len(manifests) != 12:
        raise RangeAcquisitionError("exactly 12 monthly corpora are required")
    by_month: dict[int, Path] = {}
    for path in manifests:
        try:
            value = json.loads(path.read_text())
            start = date.fromisoformat(value["requested_start_date"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise RangeAcquisitionError(f"invalid monthly manifest {path}") from error
        if start.year != year or start.month in by_month:
            raise RangeAcquisitionError("duplicate or out-of-year monthly corpus")
        by_month[start.month] = path.parent
    payloads: list[DailyPayload] = []
    absent: list[date] = []
    source_paths: list[tuple[Path, Path]] = []
    for month in range(1, 13):
        start, end = month_bounds(year, month)
        if month not in by_month:
            raise RangeAcquisitionError(f"missing month {year}-{month:02d}")
        month_payloads, month_absent, paths = _read_month(
            by_month[month], spec.instrument, start, end
        )
        payloads.extend(month_payloads)
        absent.extend(month_absent)
        source_paths.extend(paths)
    manifest = build_corpus_manifest(
        date(year, 1, 1), date(year, 12, 31), payloads, absent, spec.instrument
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RangeAcquisitionError("refusing to overwrite an assembled corpus")
    output_dir.mkdir(parents=True, exist_ok=True)
    for raw, provenance in source_paths:
        shutil.copyfile(raw, output_dir / raw.name)
        shutil.copyfile(provenance, output_dir / provenance.name)
    manifest_path = output_dir / "corpus-manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return YearAssemblyResult(
        manifest_path, manifest.corpus_id, manifest.dataset.metadata.dataset_id
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire_parser = subparsers.add_parser("acquire-month")
    acquire_parser.add_argument("--year", type=int, default=DISCOVERY_YEAR)
    acquire_parser.add_argument("--month", type=int, required=True)
    acquire_parser.add_argument("--instrument", default=INSTRUMENT)
    acquire_parser.add_argument("--output-dir", type=Path, required=True)
    assembly_parser = subparsers.add_parser("assemble-year")
    assembly_parser.add_argument("--year", type=int, default=DISCOVERY_YEAR)
    assembly_parser.add_argument("--instrument", default=INSTRUMENT)
    assembly_parser.add_argument("--chunks-root", type=Path, required=True)
    assembly_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire-month":
        result = acquire_month(
            args.output_dir, args.year, args.month, instrument=args.instrument
        )
        print(f"month_manifest={result.manifest_path}")
    else:
        result = assemble_year(
            args.chunks_root,
            args.output_dir,
            year=args.year,
            instrument=args.instrument,
        )
        print(f"corpus_manifest={result.manifest_path}")
        print(f"corpus_id={result.corpus_id}")
        print(f"assembled_dataset_id={result.dataset_id}")


if __name__ == "__main__":
    main()
