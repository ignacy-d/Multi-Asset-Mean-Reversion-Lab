"""Authenticated registry and semantic validation for the nine-pair 2024 universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from itertools import combinations, pairwise
from pathlib import Path

from mr_lab.providers.dukascopy import HOST, PROVIDER, build_url, validate_payload
from mr_lab.providers.dukascopy_bi5 import candidate_canonicalization_audit
from mr_lab.providers.dukascopy_monthly import verify_full_year_corpus
from mr_lab.providers.instruments import (
    get_candidate_instrument_spec,
    get_instrument_spec,
)
from mr_lab.providers.verify_instruments import (
    PLAUSIBLE_PRICE_BOUNDS,
    VERIFICATION_DAYS,
    VERIFICATION_REPORT_SCHEMA_VERSION,
    build_url_for_candidate,
)

REGISTRY_SCHEMA_VERSION = "fx-universe-2024-registry-v1"
VALIDATION_SCHEMA_VERSION = "fx-universe-2024-semantic-validation-v1"
REGISTRY_INSTRUMENTS = (
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDJPY",
    "USDCAD",
    "USDCHF",
    "AUDJPY",
    "EURGBP",
)
HISTORICAL_INSTRUMENTS = ("EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "AUDJPY")
EXPANDED_INSTRUMENTS = ("NZDUSD", "USDCAD", "USDCHF", "EURGBP")
REQUESTED_START = "2024-01-01"
REQUESTED_END = "2024-12-31"
EXPECTED_CALENDAR_MINUTES = 366 * 24 * 60
DEFAULT_LARGE_GAP_MINUTES = 3 * 24 * 60
CANONICAL_HISTORICAL_ROOT = Path("/mnt/e/mr-lab/frozen-2024")
CANONICAL_EXPANDED_ROOT = Path("/mnt/e/mr-lab/corpora/2024/fx-universe-v2")
CANONICAL_HISTORICAL_REGISTRY = Path("configs/stage4a-2024-corpus-registry.json")
CANONICAL_REGISTRY = Path("configs/fx-universe-2024-registry-v1.json")
HISTORICAL_REGISTRY_SCHEMA = "stage-4a-2024-corpus-registry-v1"
SHA256_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
REPOSITORY_ROOT = Path(__file__).parents[3]


class FxUniverseError(ValueError):
    """Raised when the explicit expanded-universe contract is unsafe."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_identity(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _canonical_path(path: Path) -> Path:
    """Normalize spelling without resolving symlinks or discovering substitutes."""
    return Path(os.path.abspath(path.expanduser()))


def _expected_path(path: Path) -> Path:
    return _canonical_path(path if path.is_absolute() else REPOSITORY_ROOT / path)


def _identity_path(path: Path, canonical_label: Path | None = None) -> str:
    """Return one stable label for a path included in a deterministic identity."""
    normalized = _canonical_path(path)
    if canonical_label is not None and normalized == _expected_path(canonical_label):
        return canonical_label.as_posix()
    return str(normalized)


def _same_explicit_path(given: Path, expected: Path) -> bool:
    """Compare normalized labels without resolving symlinks or discovering state."""
    return _canonical_path(given) == _expected_path(expected)


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FxUniverseError(f"invalid or missing {description}") from error
    if not isinstance(value, dict):
        raise FxUniverseError(f"{description} must be a JSON object")
    return value


def _validate_raw_provenance(
    raw_path: Path,
    provenance_path: Path,
    instrument: str,
    requested_day: date,
    *,
    require_instrument_spec: bool,
) -> tuple[bytes, dict[str, object]]:
    """Reauthenticate one explicit raw/provenance pair without discovery."""
    try:
        payload = raw_path.read_bytes()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        decoded_byte_length = validate_payload(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise FxUniverseError(
            f"invalid raw provenance for {instrument} on {requested_day}"
        ) from error
    if not isinstance(provenance, dict):
        raise FxUniverseError(
            f"invalid raw provenance for {instrument} on {requested_day}"
        )
    spec = get_instrument_spec(instrument)
    expected = {
        "provider": PROVIDER,
        "host": HOST,
        "requested_url": build_url(instrument, requested_day),
        "canonical_instrument": instrument,
        "source_timeframe": "M1",
        "price_side": "BID",
        "requested_date": requested_day.isoformat(),
        "http_status": 200,
        "compressed_byte_length": len(payload),
        "decoded_byte_length": decoded_byte_length,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    instrument_spec = {
        "provider_symbol": spec.provider_symbol,
        "price_scale": spec.price_scale,
        "price_precision": spec.price_precision,
    }
    if require_instrument_spec:
        expected.update(instrument_spec)
    elif any(
        key in provenance and provenance.get(key) != value
        for key, value in instrument_spec.items()
    ):
        raise FxUniverseError(
            f"raw provenance instrument spec mismatch for {instrument} on "
            f"{requested_day}"
        )
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise FxUniverseError(
            f"raw provenance mismatch for {instrument} on {requested_day}"
        )
    try:
        retrieved_at = datetime.fromisoformat(provenance["retrieved_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise FxUniverseError(
            f"invalid retrieval timestamp for {instrument} on {requested_day}"
        ) from error
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() != UTC.utcoffset(None):
        raise FxUniverseError(
            f"retrieval timestamp must be UTC for {instrument} on {requested_day}"
        )
    return payload, provenance


def _verify_report(path: Path, instrument: str) -> dict[str, object]:
    report = _read_json(path, f"verification report for {instrument}")
    report_id = report.get("verification_report_id")
    body = {
        key: value for key, value in report.items() if key != "verification_report_id"
    }
    if report_id != _sha256_identity(_canonical_json(body).encode()):
        raise FxUniverseError(f"verification report identity mismatch for {instrument}")
    candidate_spec = get_candidate_instrument_spec(instrument)
    declared = report.get("candidate_spec")
    if not isinstance(declared, dict):
        raise FxUniverseError(f"missing candidate spec for {instrument}")
    if declared != candidate_spec.as_dict():
        raise FxUniverseError(
            f"verification report candidate spec mismatch for {instrument}"
        )
    spec = get_instrument_spec(instrument)
    lower_price, upper_price = PLAUSIBLE_PRICE_BOUNDS[instrument]
    if (
        report.get("report_schema_version") != VERIFICATION_REPORT_SCHEMA_VERSION
        or report.get("verification_passed") is not True
        or report.get("authorized_dates")
        != [day.isoformat() for day in VERIFICATION_DAYS]
        or report.get("sample_count") != len(VERIFICATION_DAYS)
        or report.get("plausible_price_bounds") != [lower_price, upper_price]
    ):
        raise FxUniverseError(f"incomplete verification evidence for {instrument}")
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != len(VERIFICATION_DAYS):
        raise FxUniverseError(f"invalid verification samples for {instrument}")
    for day, sample in zip(VERIFICATION_DAYS, samples, strict=True):
        if (
            not isinstance(sample, dict)
            or sample.get("resampling_consistent") is not True
        ):
            raise FxUniverseError(f"resampling evidence failed for {instrument}")
        audit = sample.get("audit")
        response = sample.get("provider_response")
        if not isinstance(audit, dict) or not isinstance(response, dict):
            raise FxUniverseError(f"invalid verification evidence for {instrument}")
        stem = f"{instrument}-{day.isoformat()}-M1-BID"
        raw_path = path.parent / f"{stem}.bi5"
        payload, provenance = _validate_raw_provenance(
            raw_path,
            path.parent / f"{stem}.json",
            instrument,
            day,
            require_instrument_spec=True,
        )
        expected_audit = candidate_canonicalization_audit(payload, day, instrument)
        minimum_price = expected_audit.get("minimum_price")
        maximum_price = expected_audit.get("maximum_price")
        expected_response = {
            "canonical_instrument": instrument,
            "provider_symbol": spec.provider_symbol,
            "price_scale": spec.price_scale,
            "price_precision": spec.price_precision,
            "source_timeframe": "M1",
            "price_side": "BID",
            "requested_date": day.isoformat(),
            "requested_url": build_url_for_candidate(instrument, day),
            "http_status": 200,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if (
            audit != expected_audit
            or response != expected_response
            or not isinstance(provenance, dict)
            or not isinstance(minimum_price, int | float)
            or isinstance(minimum_price, bool)
            or not isinstance(maximum_price, int | float)
            or isinstance(maximum_price, bool)
            or not lower_price <= minimum_price <= maximum_price <= upper_price
            or any(
                provenance.get(key) != value for key, value in expected_response.items()
            )
        ):
            raise FxUniverseError(f"verification sample mismatch for {instrument}")
    return report


def _load_historical_registry(path: Path) -> dict[str, object]:
    registry = _read_json(path, "historical frozen registry")
    entries = registry.get("instruments")
    if (
        registry.get("registry_schema_version") != HISTORICAL_REGISTRY_SCHEMA
        or not isinstance(entries, dict)
        or set(entries) != set(HISTORICAL_INSTRUMENTS)
    ):
        raise FxUniverseError("historical frozen registry contract mismatch")
    for instrument in HISTORICAL_INSTRUMENTS:
        entry = entries[instrument]
        if not isinstance(entry, dict) or (
            entry.get("instrument"),
            entry.get("verification_status"),
            entry.get("requested_start_date"),
            entry.get("requested_end_date"),
        ) != (instrument, "verified", REQUESTED_START, REQUESTED_END):
            raise FxUniverseError(
                f"historical frozen registry mismatch for {instrument}"
            )
    return registry


def _corpus_entry(
    corpus_path: Path,
    instrument: str,
    verification: Mapping[str, object],
) -> dict[str, object]:
    verified = verify_full_year_corpus(corpus_path, instrument=instrument)
    manifest_path = corpus_path / "corpus-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    spec = get_instrument_spec(instrument)
    if verified.dataset is None:  # pragma: no cover - verifier contract
        raise FxUniverseError("full-year verifier did not return the dataset")
    if (
        manifest.get("instrument") != instrument
        or manifest.get("provider") != spec.provider
        or manifest.get("native_timeframe") != "1m"
        or manifest.get("price_basis") != "bid"
        or manifest.get("source_timezone") != "UTC"
        or manifest.get("requested_start_date") != REQUESTED_START
        or manifest.get("requested_end_date") != REQUESTED_END
        or manifest.get("corpus_id") != verified.corpus_id
        or manifest.get("assembled_dataset_id") != verified.dataset_id
    ):
        raise FxUniverseError(f"full-year corpus contract mismatch for {instrument}")
    component_dates = manifest.get("successful_component_dates")
    if not isinstance(component_dates, list):
        raise FxUniverseError(f"invalid component dates for {instrument}")
    for value in component_dates:
        try:
            requested_day = date.fromisoformat(value)
        except (TypeError, ValueError) as error:
            raise FxUniverseError(f"invalid component date for {instrument}") from error
        stem = f"{instrument}-{requested_day.isoformat()}-M1-BID"
        _validate_raw_provenance(
            corpus_path / f"{stem}.bi5",
            corpus_path / f"{stem}.json",
            instrument,
            requested_day,
            require_instrument_spec=instrument in EXPANDED_INSTRUMENTS,
        )
    return {
        "instrument": instrument,
        "provider": spec.provider,
        "provider_symbol": spec.provider_symbol,
        "price_basis": spec.price_basis.value,
        "native_timeframe": spec.native_timeframe.value,
        "source_timezone": "UTC",
        "price_scale": spec.price_scale,
        "price_precision": spec.price_precision,
        "requested_start_date": REQUESTED_START,
        "requested_end_date": REQUESTED_END,
        "corpus_id": verified.corpus_id,
        "assembled_dataset_id": verified.dataset_id,
        "manifest_sha256": _sha256_identity(manifest_bytes),
        "verification_status": "verified",
        "verification_evidence": dict(verification),
        "corpus_path": _identity_path(corpus_path),
    }


def build_registry(
    historical_root: Path,
    expanded_root: Path,
    historical_registry_path: Path,
) -> dict[str, object]:
    """Build the registry from nine explicit instrument paths; never discover."""
    historical_registry = _load_historical_registry(historical_registry_path)
    historical_entries = historical_registry["instruments"]
    if not isinstance(historical_entries, dict):  # pragma: no cover - loader contract
        raise FxUniverseError("invalid historical registry")
    historical_registry_sha = _sha256_identity(historical_registry_path.read_bytes())
    entries: dict[str, object] = {}
    for instrument in REGISTRY_INSTRUMENTS:
        evidence: dict[str, object]
        if instrument in HISTORICAL_INSTRUMENTS:
            frozen = historical_entries[instrument]
            if (
                not isinstance(frozen, dict)
                or frozen.get("verification_status") != "verified"
            ):
                raise FxUniverseError(
                    f"historical instrument is unverified: {instrument}"
                )
            corpus_path = historical_root / instrument
            evidence = {
                "kind": "historical-frozen-registry",
                "registry_path": _identity_path(
                    historical_registry_path, CANONICAL_HISTORICAL_REGISTRY
                ),
                "registry_sha256": historical_registry_sha,
            }
            entry = _corpus_entry(corpus_path, instrument, evidence)
            for field in ("corpus_id", "assembled_dataset_id"):
                if entry[field] != frozen.get(field):
                    raise FxUniverseError(
                        f"historical frozen identity mismatch for {instrument}"
                    )
        else:
            corpus_path = expanded_root / instrument
            report_path = (
                expanded_root / "verification" / instrument / "verification-report.json"
            )
            report = _verify_report(report_path, instrument)
            evidence = {
                "kind": "bounded-real-provider",
                "report_path": _identity_path(report_path),
                "verification_report_id": report["verification_report_id"],
            }
            entry = _corpus_entry(corpus_path, instrument, evidence)
        entries[instrument] = entry
    body = {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "requested_start_date": REQUESTED_START,
        "requested_end_date": REQUESTED_END,
        "instruments": entries,
    }
    return {"registry_id": _sha256_identity(_canonical_json(body).encode()), **body}


def validate_registry(registry: Mapping[str, object]) -> dict[str, object]:
    """Validate complete registry semantics without touching corpus paths."""
    if registry.get("registry_schema_version") != REGISTRY_SCHEMA_VERSION:
        raise FxUniverseError("unsupported or missing expanded registry schema")
    body = {key: value for key, value in registry.items() if key != "registry_id"}
    if registry.get("registry_id") != _sha256_identity(_canonical_json(body).encode()):
        raise FxUniverseError("expanded registry identity mismatch")
    if (registry.get("requested_start_date"), registry.get("requested_end_date")) != (
        REQUESTED_START,
        REQUESTED_END,
    ):
        raise FxUniverseError("expanded registry is outside exact 2024 bounds")
    entries = registry.get("instruments")
    if not isinstance(entries, dict) or set(entries) != set(REGISTRY_INSTRUMENTS):
        raise FxUniverseError("registry must contain the exact nine-pair universe")
    for instrument in REGISTRY_INSTRUMENTS:
        entry = entries[instrument]
        spec = get_instrument_spec(instrument)
        expected = {
            "instrument": instrument,
            "provider": spec.provider,
            "provider_symbol": spec.provider_symbol,
            "price_basis": "bid",
            "native_timeframe": "1m",
            "source_timezone": "UTC",
            "price_scale": spec.price_scale,
            "price_precision": spec.price_precision,
            "requested_start_date": REQUESTED_START,
            "requested_end_date": REQUESTED_END,
            "verification_status": "verified",
        }
        if not isinstance(entry, dict) or any(
            entry.get(field) != value for field, value in expected.items()
        ):
            raise FxUniverseError(f"registry semantic mismatch for {instrument}")
        for field in (
            "corpus_id",
            "assembled_dataset_id",
            "manifest_sha256",
        ):
            value = entry.get(field)
            if not isinstance(value, str) or not SHA256_IDENTITY.fullmatch(value):
                raise FxUniverseError(f"missing {field} for {instrument}")
        if not isinstance(entry.get("verification_evidence"), dict):
            raise FxUniverseError(f"missing verification evidence for {instrument}")
        corpus_path = entry.get("corpus_path")
        if not isinstance(corpus_path, str) or not Path(corpus_path).is_absolute():
            raise FxUniverseError(f"corpus path must be explicit for {instrument}")
    return dict(registry)


def _write_immutable_json(path: Path, value: Mapping[str, object]) -> Path:
    encoded = (_canonical_json(value) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FxUniverseError(f"refusing to overwrite different artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def write_registry(
    output_path: Path,
    historical_root: Path,
    expanded_root: Path,
    historical_registry_path: Path,
) -> Path:
    registry = build_registry(historical_root, expanded_root, historical_registry_path)
    validate_registry(registry)
    return _write_immutable_json(output_path, registry)


def _validate_entry_path(entry: Mapping[str, object], instrument: str) -> Path:
    value = entry["corpus_path"]
    assert isinstance(value, str)
    path = _canonical_path(Path(value))
    root = (
        CANONICAL_HISTORICAL_ROOT
        if instrument in HISTORICAL_INSTRUMENTS
        else CANONICAL_EXPANDED_ROOT
    )
    expected = root / instrument
    if not _same_explicit_path(path, expected):
        raise FxUniverseError(f"unexpected corpus path for {instrument}")
    return path


def validate_expanded_universe(
    registry_path: Path,
    output_path: Path,
    *,
    large_gap_minutes: int = DEFAULT_LARGE_GAP_MINUTES,
) -> Path:
    """Authenticate and semantically audit all nine explicitly registered corpora."""
    if large_gap_minutes <= 0:
        raise FxUniverseError("large_gap_minutes must be positive")
    registry = validate_registry(_read_json(registry_path, "expanded registry"))
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    minute_sets: dict[str, set[int]] = {}
    instrument_reports: dict[str, object] = {}
    quality_warnings: list[str] = []
    range_start = datetime(2024, 1, 1, tzinfo=UTC)
    range_end = datetime(2025, 1, 1, tzinfo=UTC)

    for instrument in REGISTRY_INSTRUMENTS:
        entry = entries[instrument]
        assert isinstance(entry, dict)
        corpus_path = _validate_entry_path(entry, instrument)
        verified = verify_full_year_corpus(corpus_path, instrument=instrument)
        dataset = verified.dataset
        if dataset is None:  # pragma: no cover - verifier contract
            raise FxUniverseError("full-year verifier did not return the dataset")
        manifest_bytes = verified.manifest_path.read_bytes()
        if (
            verified.corpus_id != entry["corpus_id"]
            or verified.dataset_id != entry["assembled_dataset_id"]
            or _sha256_identity(manifest_bytes) != entry["manifest_sha256"]
        ):
            raise FxUniverseError(f"registry identity mismatch for {instrument}")

        bars = dataset.bars
        timestamps: set[int] = set()
        invalid_prices = nonfinite_prices = impossible_ohlc = duplicate_count = 0
        for bar in bars:
            if (
                bar.instrument != instrument
                or bar.price_basis.value != "bid"
                or bar.timeframe.value != "1m"
                or bar.open_time.tzinfo is None
                or bar.open_time.utcoffset() != UTC.utcoffset(None)
                or not range_start <= bar.open_time < range_end
                or bar.close_time != bar.open_time + bar.timeframe.duration
            ):
                raise FxUniverseError(f"bar semantic mismatch for {instrument}")
            values = (bar.open, bar.high, bar.low, bar.close)
            nonfinite_prices += sum(not math.isfinite(value) for value in values)
            invalid_prices += sum(value <= 0 for value in values)
            impossible_ohlc += int(
                bar.low > min(bar.open, bar.close)
                or bar.high < max(bar.open, bar.close)
            )
            minute = int((bar.open_time - range_start).total_seconds() // 60)
            if minute in timestamps:
                duplicate_count += 1
            timestamps.add(minute)
        if nonfinite_prices or invalid_prices or impossible_ohlc or duplicate_count:
            raise FxUniverseError(f"invalid canonical bars for {instrument}")
        if len(timestamps) != len(bars):
            raise FxUniverseError(f"timestamp uniqueness mismatch for {instrument}")

        if not bars:  # pragma: no cover - full-year corpus verifier rejects this
            raise FxUniverseError(f"empty canonical dataset for {instrument}")
        gap_boundaries = [
            (range_start, bars[0].open_time),
            *(
                (previous.close_time, current.open_time)
                for previous, current in pairwise(bars)
            ),
            (bars[-1].close_time, range_end),
        ]
        large_gaps = []
        maximum_large_gap_minutes = 0
        for gap_start, gap_end in gap_boundaries:
            gap_minutes = int((gap_end - gap_start).total_seconds() // 60)
            if gap_minutes >= large_gap_minutes:
                maximum_large_gap_minutes = max(maximum_large_gap_minutes, gap_minutes)
                large_gaps.append(
                    {
                        "start": gap_start.isoformat(),
                        "end": gap_end.isoformat(),
                        "missing_minutes": gap_minutes,
                    }
                )
        if large_gaps:
            quality_warnings.append(
                f"{instrument} has {len(large_gaps)} gaps at least "
                f"{large_gap_minutes} minutes"
            )
        manifest = json.loads(manifest_bytes)
        minute_sets[instrument] = timestamps
        instrument_reports[instrument] = {
            "instrument": instrument,
            "provider": "Dukascopy",
            "provider_symbol": entry["provider_symbol"],
            "price_basis": "bid",
            "native_timeframe": "1m",
            "source_timezone": "UTC",
            "corpus_id": verified.corpus_id,
            "assembled_dataset_id": verified.dataset_id,
            "manifest_sha256": entry["manifest_sha256"],
            "m1_bar_count": len(bars),
            "unique_timestamp_count": len(timestamps),
            "duplicate_timestamp_count": duplicate_count,
            "timestamps_strictly_monotonic": True,
            "missing_calendar_minute_count": EXPECTED_CALENDAR_MINUTES - len(bars),
            "calendar_minute_coverage_ratio": len(bars) / EXPECTED_CALENDAR_MINUTES,
            "first_open_time": bars[0].open_time.isoformat(),
            "final_close_time": bars[-1].close_time.isoformat(),
            "manifest_gap_count": dataset.manifest.gap_count,
            "confirmed_absent_day_count": manifest["absent_component_count"],
            "successful_component_day_count": manifest["successful_component_count"],
            "nonfinite_price_count": nonfinite_prices,
            "invalid_nonpositive_price_count": invalid_prices,
            "impossible_ohlc_count": impossible_ohlc,
            "large_gap_threshold_minutes": large_gap_minutes,
            "large_gap_count": len(large_gaps),
            "maximum_large_gap_minutes": maximum_large_gap_minutes,
            "large_gaps": large_gaps,
        }

    intersection = set.intersection(
        *(minute_sets[name] for name in REGISTRY_INSTRUMENTS)
    )
    union = set.union(*(minute_sets[name] for name in REGISTRY_INSTRUMENTS))
    per_instrument_loss = {
        instrument: {
            "timestamps_not_in_nine_way_intersection": len(
                minute_sets[instrument] - intersection
            ),
            "nine_way_intersection_share": len(intersection)
            / len(minute_sets[instrument]),
        }
        for instrument in REGISTRY_INSTRUMENTS
    }
    pairwise_intersections = {
        f"{left}:{right}": len(minute_sets[left] & minute_sets[right])
        for left, right in combinations(REGISTRY_INSTRUMENTS, 2)
    }
    report_body: dict[str, object] = {
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "registry_id": registry["registry_id"],
        "registry_path": _identity_path(registry_path, CANONICAL_REGISTRY),
        "requested_start_date": REQUESTED_START,
        "requested_end_date": REQUESTED_END,
        "semantic_validation_passed": True,
        "instruments": instrument_reports,
        "synchronization": {
            "nine_way_intersection_count": len(intersection),
            "union_count": len(union),
            "union_minus_intersection_count": len(union - intersection),
            "intersection_to_union_ratio": len(intersection) / len(union),
            "per_instrument": per_instrument_loss,
            "pairwise_intersection_counts": pairwise_intersections,
        },
        "quality_warnings": quality_warnings,
        "repairs_applied": [],
    }
    report = {
        "validation_report_id": _sha256_identity(_canonical_json(report_body).encode()),
        **report_body,
    }
    return _write_immutable_json(output_path, report)


def _require_canonical_cli_paths(args: argparse.Namespace) -> None:
    checks: Sequence[tuple[Path, Path, str]] = (
        (args.historical_root, CANONICAL_HISTORICAL_ROOT, "historical root"),
        (args.expanded_root, CANONICAL_EXPANDED_ROOT, "expanded root"),
        (
            args.historical_registry,
            CANONICAL_HISTORICAL_REGISTRY,
            "historical registry",
        ),
        (args.output, CANONICAL_REGISTRY, "expanded registry output"),
    )
    for given, expected, name in checks:
        if not _same_explicit_path(given, expected):
            raise FxUniverseError(f"{name} must be the explicit path {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-registry")
    build.add_argument("--historical-root", type=Path, required=True)
    build.add_argument("--expanded-root", type=Path, required=True)
    build.add_argument("--historical-registry", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--registry", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument(
        "--large-gap-minutes", type=int, default=DEFAULT_LARGE_GAP_MINUTES
    )
    args = parser.parse_args()
    if args.command == "build-registry":
        _require_canonical_cli_paths(args)
        print(
            "registry="
            + str(
                write_registry(
                    args.output,
                    args.historical_root,
                    args.expanded_root,
                    args.historical_registry,
                )
            )
        )
    else:
        if not _same_explicit_path(args.registry, CANONICAL_REGISTRY):
            raise FxUniverseError(
                f"registry must be the explicit path {CANONICAL_REGISTRY}"
            )
        print(
            "validation="
            + str(
                validate_expanded_universe(
                    args.registry,
                    args.output,
                    large_gap_minutes=args.large_gap_minutes,
                )
            )
        )


if __name__ == "__main__":
    main()
