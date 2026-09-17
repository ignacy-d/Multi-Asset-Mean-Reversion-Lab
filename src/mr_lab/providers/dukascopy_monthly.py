"""Checkpointed monthly acquisition and deterministic full-year assembly."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict, Unpack

from mr_lab.providers.dukascopy import (
    HOST,
    PROVIDER,
    AcquisitionError,
    ProviderNoData,
    acquire,
    build_url,
    validate_payload,
)
from mr_lab.providers.dukascopy_bi5 import (
    CANONICAL_SCHEMA_VERSION,
    INSTRUMENT,
    PARSER_SCHEMA_VERSION,
    SOURCE_TIMEZONE,
)
from mr_lab.providers.dukascopy_multiday import (
    DailyPayload,
    MultiDayDataset,
    assemble_daily_payloads,
)
from mr_lab.providers.dukascopy_range import (
    CORPUS_SCHEMA_VERSION,
    GENERIC_CORPUS_SCHEMA_VERSION,
    RangeAcquisitionError,
    RangeAcquisitionResult,
    acquire_range,
    build_corpus_manifest,
    enumerate_dates,
    load_offline_corpus,
)
from mr_lab.providers.instruments import get_instrument_spec

DISCOVERY_YEAR = 2024


class _AcquisitionOptions(TypedDict, total=False):
    timeout: float
    retries: int
    acquire_day: Callable[..., tuple[Path, Path]]
    sleeper: Callable[[float], None]
    logger: Callable[[str], None]


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
    resume: bool = False,
    **kwargs: Unpack[_AcquisitionOptions],
) -> RangeAcquisitionResult:
    """Acquire one independently checkpointable month, sequentially."""
    start, end = month_bounds(year, month)
    if resume:
        return _resume_month(
            output_dir,
            start,
            end,
            instrument=instrument,
            delay_seconds=delay_seconds,
            **kwargs,
        )
    return acquire_range(
        output_dir,
        start,
        end,
        instrument=instrument,
        delay_seconds=delay_seconds,
        **kwargs,
    )


def _validate_existing_day(
    output_dir: Path, day: date, instrument: str
) -> tuple[DailyPayload, Path, Path] | None:
    """Load one exact immutable daily pair, or fail closed on partial/corrupt state."""
    spec = get_instrument_spec(instrument)
    stem = f"{spec.instrument}-{day.isoformat()}-M1-BID"
    raw_path = output_dir / f"{stem}.bi5"
    provenance_path = output_dir / f"{stem}.json"
    raw_exists, provenance_exists = raw_path.exists(), provenance_path.exists()
    if not raw_exists and not provenance_exists:
        return None
    if raw_exists != provenance_exists:
        raise RangeAcquisitionError(f"orphaned raw/provenance snapshot for {day}")
    try:
        raw = raw_path.read_bytes()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        decoded_length = validate_payload(raw)
    except (OSError, json.JSONDecodeError, ValueError, AcquisitionError) as error:
        raise RangeAcquisitionError(f"invalid existing snapshot for {day}") from error
    digest = hashlib.sha256(raw).hexdigest()
    expected = {
        "provider": PROVIDER,
        "host": HOST,
        "requested_url": build_url(spec.instrument, day),
        "canonical_instrument": spec.instrument,
        "provider_symbol": spec.provider_symbol,
        "price_scale": spec.price_scale,
        "price_precision": spec.price_precision,
        "source_timeframe": "M1",
        "price_side": "BID",
        "requested_date": day.isoformat(),
        "http_status": 200,
        "compressed_byte_length": len(raw),
        "decoded_byte_length": decoded_length,
        "sha256": digest,
    }
    if not isinstance(provenance, dict) or any(
        provenance.get(key) != value for key, value in expected.items()
    ):
        raise RangeAcquisitionError(f"existing snapshot provenance mismatch for {day}")
    try:
        retrieved_at = datetime.fromisoformat(provenance["retrieved_at"])
        item = DailyPayload(day, raw, spec.instrument)
        assemble_daily_payloads([item])
    except (KeyError, TypeError, ValueError) as error:
        raise RangeAcquisitionError(f"invalid existing snapshot for {day}") from error
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() != UTC.utcoffset(None):
        raise RangeAcquisitionError(f"existing snapshot provenance mismatch for {day}")
    return item, raw_path, provenance_path


def _validate_checkpoint_inventory(
    output_dir: Path, days: tuple[date, ...], instrument: str
) -> None:
    """Reject unexpected snapshot-shaped files in this exact monthly checkpoint."""
    if not output_dir.exists():
        return
    expected = {
        f"{instrument}-{day.isoformat()}-M1-BID{suffix}"
        for day in days
        for suffix in (".bi5", ".json")
    }
    expected.add("corpus-manifest.json")
    unexpected = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and (path.suffix in {".bi5", ".json"})
        and path.name not in expected
    )
    if unexpected:
        raise RangeAcquisitionError(
            "mismatched, ambiguous, or duplicate snapshot files: "
            + ", ".join(unexpected)
        )


def _completed_month(
    output_dir: Path, start: date, end: date, instrument: str
) -> RangeAcquisitionResult:
    """Authenticate an existing completed month and reproduce its logical result."""
    payloads, absent, paths = _read_month(output_dir, instrument, start, end)
    validated = [
        _validate_existing_day(output_dir, item.requested_day, instrument)
        for item in payloads
    ]
    if any(
        item is None for item in validated
    ):  # pragma: no cover - _read_month read it
        raise RangeAcquisitionError("completed monthly snapshot disappeared")
    rebuilt = build_corpus_manifest(start, end, payloads, absent, instrument)
    manifest_path = output_dir / "corpus-manifest.json"
    try:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RangeAcquisitionError("invalid completed monthly manifest") from error
    if stored != rebuilt.as_dict():
        raise RangeAcquisitionError("completed monthly manifest identity mismatch")
    return RangeAcquisitionResult(
        rebuilt,
        tuple(raw for raw, _ in paths),
        tuple(provenance for _, provenance in paths),
        manifest_path,
    )


def _resume_month(
    output_dir: Path,
    start: date,
    end: date,
    *,
    instrument: str,
    timeout: float = 30.0,
    retries: int = 6,
    delay_seconds: float = 2.0,
    acquire_day: Callable[..., tuple[Path, Path]] = acquire,
    sleeper: Callable[[float], None] = time.sleep,
    logger: Callable[[str], None] = print,
) -> RangeAcquisitionResult:
    """Resume an interrupted month without weakening immutable daily writes."""
    spec = get_instrument_spec(instrument)
    manifest_path = output_dir / "corpus-manifest.json"
    days = enumerate_dates(start, end)
    _validate_checkpoint_inventory(output_dir, days, spec.instrument)
    if manifest_path.exists():
        return _completed_month(output_dir, start, end, spec.instrument)
    if delay_seconds < 0:
        raise RangeAcquisitionError("delay_seconds must be non-negative")

    payloads: list[DailyPayload] = []
    absent: list[date] = []
    raw_paths: list[Path] = []
    provenance_paths: list[Path] = []
    for index, day in enumerate(days):
        progress = f"[{index + 1}/{len(days)}]"
        existing = _validate_existing_day(output_dir, day, spec.instrument)
        if existing is not None:
            item, raw_path, provenance_path = existing
            logger(f"{progress} reusing {day.isoformat()}")
        else:
            logger(f"{progress} acquiring {day.isoformat()}")
            try:
                call_kwargs: dict[str, object] = {
                    "timeout": timeout,
                    "retries": retries,
                }
                if spec.instrument != INSTRUMENT:
                    call_kwargs["instrument"] = spec.instrument
                raw_path, provenance_path = acquire_day(output_dir, day, **call_kwargs)
            except ProviderNoData:
                absent.append(day)
                logger(f"{progress} absent {day.isoformat()}")
                if delay_seconds and index != len(days) - 1:
                    sleeper(delay_seconds)
                continue
            validated = _validate_existing_day(output_dir, day, spec.instrument)
            if validated is None:  # pragma: no cover - defensive provider contract
                raise RangeAcquisitionError(
                    f"acquisition did not write snapshot for {day}"
                )
            item, raw_path, provenance_path = validated
        payloads.append(item)
        raw_paths.append(raw_path)
        provenance_paths.append(provenance_path)
        if delay_seconds and index != len(days) - 1:
            sleeper(delay_seconds)

    manifest = build_corpus_manifest(start, end, payloads, absent, spec.instrument)
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise RangeAcquisitionError("refusing to overwrite an existing corpus manifest")
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return RangeAcquisitionResult(
        manifest, tuple(raw_paths), tuple(provenance_paths), manifest_path
    )


@dataclass(frozen=True, slots=True)
class YearAssemblyResult:
    """The assembled manifest and its immutable corpus directory."""

    manifest_path: Path
    corpus_id: str
    dataset_id: str
    dataset: MultiDayDataset | None = None


def verify_full_year_corpus(
    corpus_dir: Path,
    *,
    year: int = DISCOVERY_YEAR,
    instrument: str = INSTRUMENT,
) -> YearAssemblyResult:
    """Fail closed unless a published corpus has the exact immutable contract."""
    spec = get_instrument_spec(instrument)
    if year != DISCOVERY_YEAR:
        raise RangeAcquisitionError("full-year verification accepts only 2024")
    manifest_path = corpus_dir / "corpus-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RangeAcquisitionError("invalid or missing full-year manifest") from error

    expected = {
        "corpus_schema_version": (
            CORPUS_SCHEMA_VERSION
            if spec.instrument == INSTRUMENT
            else GENERIC_CORPUS_SCHEMA_VERSION
        ),
        "provider": spec.provider,
        "instrument": spec.instrument,
        "requested_start_date": f"{year}-01-01",
        "requested_end_date": f"{year}-12-31",
        "requested_calendar_day_count": 366,
        "native_timeframe": spec.native_timeframe.value,
        "price_basis": spec.price_basis.value,
        "volume_semantics": spec.volume_semantics.value,
        "source_timezone": SOURCE_TIMEZONE,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
    }
    if spec.instrument != INSTRUMENT:
        expected["instrument_spec"] = spec.as_dict()
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise RangeAcquisitionError("full-year manifest contract mismatch")

    requested = {
        date.fromordinal(value)
        for value in range(
            date(year, 1, 1).toordinal(),
            date(year, 12, 31).toordinal() + 1,
        )
    }
    try:
        successful = [
            date.fromisoformat(value)
            for value in manifest["successful_component_dates"]
        ]
        absent = [
            date.fromisoformat(value) for value in manifest["confirmed_absent_dates"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise RangeAcquisitionError("invalid full-year date declarations") from error
    if (
        len(successful) != len(set(successful))
        or len(absent) != len(set(absent))
        or set(successful) & set(absent)
        or set(successful) | set(absent) != requested
    ):
        raise RangeAcquisitionError("full-year dates do not exactly partition 2024")

    claimed_corpus_id = manifest.get("corpus_id")
    identity = {key: value for key, value in manifest.items() if key != "corpus_id"}
    actual_corpus_id = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
    )
    if claimed_corpus_id != actual_corpus_id:
        raise RangeAcquisitionError("full-year corpus identity mismatch")
    dataset = load_offline_corpus(corpus_dir)
    return YearAssemblyResult(
        manifest_path, actual_corpus_id, dataset.metadata.dataset_id, dataset
    )


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
    month_directory_template: str = "month-{month:02d}",
) -> YearAssemblyResult:
    """Verify 12 explicit monthly paths and reproduce full-year semantics."""
    spec = get_instrument_spec(instrument)
    if year != DISCOVERY_YEAR:
        raise RangeAcquisitionError("year assembly accepts only 2024")
    by_month: dict[int, Path] = {}
    for month in range(1, 13):
        try:
            directory_name = month_directory_template.format(
                instrument=spec.instrument, year=year, month=month
            )
        except (KeyError, ValueError) as error:
            raise RangeAcquisitionError("invalid month directory template") from error
        relative = Path(directory_name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name in {"", ".", ".."}
        ):
            raise RangeAcquisitionError(
                "month directory template must produce one explicit child name"
            )
        directory = chunks_root / relative
        path = directory / "corpus-manifest.json"
        try:
            value = json.loads(path.read_text())
            start = date.fromisoformat(value["requested_start_date"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise RangeAcquisitionError(
                f"missing or invalid month {year}-{month:02d}"
            ) from error
        if start != date(year, month, 1):
            raise RangeAcquisitionError(
                "monthly corpus is bound to the wrong explicit path"
            )
        by_month[month] = directory
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
        manifest_path,
        manifest.corpus_id,
        manifest.dataset.metadata.dataset_id,
        manifest.dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire_parser = subparsers.add_parser("acquire-month")
    acquire_parser.add_argument("--year", type=int, default=DISCOVERY_YEAR)
    acquire_parser.add_argument("--month", type=int, required=True)
    acquire_parser.add_argument("--instrument", default=INSTRUMENT)
    acquire_parser.add_argument("--output-dir", type=Path, required=True)
    acquire_parser.add_argument("--delay-seconds", type=float, default=2.0)
    acquire_parser.add_argument("--resume", action="store_true")
    assembly_parser = subparsers.add_parser("assemble-year")
    assembly_parser.add_argument("--year", type=int, default=DISCOVERY_YEAR)
    assembly_parser.add_argument("--instrument", default=INSTRUMENT)
    assembly_parser.add_argument("--chunks-root", type=Path, required=True)
    assembly_parser.add_argument("--output-dir", type=Path, required=True)
    assembly_parser.add_argument(
        "--month-directory-template", default="month-{month:02d}"
    )
    verify_parser = subparsers.add_parser("verify-year")
    verify_parser.add_argument("--year", type=int, default=DISCOVERY_YEAR)
    verify_parser.add_argument("--instrument", required=True)
    verify_parser.add_argument("--corpus-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire-month":
        month_result = acquire_month(
            args.output_dir,
            args.year,
            args.month,
            instrument=args.instrument,
            delay_seconds=args.delay_seconds,
            resume=args.resume,
        )
        print(f"month_manifest={month_result.manifest_path}")
    elif args.command == "assemble-year":
        year_result = assemble_year(
            args.chunks_root,
            args.output_dir,
            year=args.year,
            instrument=args.instrument,
            month_directory_template=args.month_directory_template,
        )
        print(f"corpus_manifest={year_result.manifest_path}")
        print(f"corpus_id={year_result.corpus_id}")
        print(f"assembled_dataset_id={year_result.dataset_id}")
    else:
        verification_result = verify_full_year_corpus(
            args.corpus_dir, year=args.year, instrument=args.instrument
        )
        print(f"corpus_manifest={verification_result.manifest_path}")
        print(f"corpus_id={verification_result.corpus_id}")
        print(f"assembled_dataset_id={verification_result.dataset_id}")


if __name__ == "__main__":
    main()
