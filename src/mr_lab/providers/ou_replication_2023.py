"""Bounded, authenticated 2023 corpus acquisition for later OU replication.

This module is data infrastructure only.  It deliberately has no dependency on
strategy or research runners and never searches for alternative corpus paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from mr_lab.providers.dukascopy import (
    HOST,
    PROVIDER,
    ProviderNoData,
    acquire,
    build_url,
    validate_payload,
)
from mr_lab.providers.dukascopy_multiday import DailyPayload, assemble_daily_payloads
from mr_lab.providers.instruments import get_instrument_spec

AUTHORIZED_INSTRUMENTS = ("AUDUSD", "EURUSD", "GBPUSD")
AUTHORIZED_START = date(2023, 1, 1)
AUTHORIZED_END = date(2023, 12, 31)
CORPUS_SCHEMA_VERSION = "ou-replication-2023-corpus-v1"
REGISTRY_SCHEMA_VERSION = "ou-replication-2023-registry-v1"
DEFAULT_ROOT = Path("/mnt/e/mr-lab/corpora/2023/ou-replication-v1")


class OuReplicationCorpusError(ValueError):
    """Raised when the fixed 2023 acquisition contract cannot be authenticated."""


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _instrument(value: str) -> str:
    if not isinstance(value, str) or value not in AUTHORIZED_INSTRUMENTS:
        raise OuReplicationCorpusError(
            "instrument is not authorized for 2023 replication"
        )
    return value


def _range(start: date, end: date) -> tuple[date, ...]:
    if (start, end) != (AUTHORIZED_START, AUTHORIZED_END):
        raise OuReplicationCorpusError(
            "request must be exactly calendar year 2023 (2023-01-01 through 2023-12-31)"
        )
    return tuple(start + timedelta(days=n) for n in range(365))


def _paths(root: Path, instrument: str, day: date) -> tuple[Path, Path, Path]:
    stem = root / f"{instrument}-{day.isoformat()}-M1-BID"
    return (
        stem.with_suffix(".bi5"),
        stem.with_suffix(".json"),
        root / f"{stem.name}.absent.json",
    )


def _read_success(root: Path, instrument: str, day: date) -> DailyPayload | None:
    raw_path, provenance_path, absent_path = _paths(root, instrument, day)
    present = (raw_path.exists(), provenance_path.exists(), absent_path.exists())
    if present in {(False, False, False), (False, False, True)}:
        return None
    if present != (True, True, False):
        raise OuReplicationCorpusError(f"incompatible or partial artifacts for {day}")
    try:
        raw = raw_path.read_bytes()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        decoded_length = validate_payload(raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise OuReplicationCorpusError(
            f"corrupt payload/provenance for {day}"
        ) from error
    spec = get_instrument_spec(instrument)
    expected = {
        "provider": PROVIDER,
        "host": HOST,
        "requested_url": build_url(instrument, day),
        "canonical_instrument": instrument,
        "provider_symbol": spec.provider_symbol,
        "price_scale": spec.price_scale,
        "price_precision": spec.price_precision,
        "source_timeframe": "M1",
        "price_side": "BID",
        "requested_date": day.isoformat(),
        "http_status": 200,
        "compressed_byte_length": len(raw),
        "decoded_byte_length": decoded_length,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if not isinstance(provenance, dict) or any(
        provenance.get(k) != v for k, v in expected.items()
    ):
        raise OuReplicationCorpusError(f"payload provenance mismatch for {day}")
    try:
        retrieved = datetime.fromisoformat(provenance["retrieved_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise OuReplicationCorpusError(
            f"invalid retrieval timestamp for {day}"
        ) from error
    if retrieved.tzinfo is None or retrieved.utcoffset() != UTC.utcoffset(None):
        raise OuReplicationCorpusError(f"retrieval timestamp is not UTC for {day}")
    return DailyPayload(day, raw, instrument)


def _read_absence(root: Path, instrument: str, day: date) -> bool:
    raw_path, provenance_path, path = _paths(root, instrument, day)
    if raw_path.exists() or provenance_path.exists() or not path.exists():
        return False
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OuReplicationCorpusError(f"corrupt absence evidence for {day}") from error
    expected = {
        "provider": PROVIDER,
        "host": HOST,
        "canonical_instrument": instrument,
        "requested_date": day.isoformat(),
        "requested_url": build_url(instrument, day),
        "source_timeframe": "M1",
        "price_side": "BID",
        "http_status": 404,
        "status": "confirmed_absent",
    }
    if not isinstance(evidence, dict) or any(
        evidence.get(k) != v for k, v in expected.items()
    ):
        raise OuReplicationCorpusError(f"absence evidence mismatch for {day}")
    return True


def _manifest(
    root: Path,
    instrument: str,
    payloads: list[DailyPayload],
    absent: list[date],
) -> dict[str, object]:
    dataset = assemble_daily_payloads(payloads)
    components = [
        {
            "requested_day": item.requested_day.isoformat(),
            "raw_sha256": _sha(item.payload),
            "provenance_sha256": _sha(
                _paths(root, instrument, item.requested_day)[1].read_bytes()
            ),
        }
        for item in sorted(payloads, key=lambda item: item.requested_day)
    ]
    absence_components = [
        {
            "requested_day": day.isoformat(),
            "absence_evidence_sha256": _sha(
                _paths(root, instrument, day)[2].read_bytes()
            ),
        }
        for day in sorted(absent)
    ]
    body: dict[str, object] = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "provider": PROVIDER,
        "instrument": instrument,
        "requested_start_date": AUTHORIZED_START.isoformat(),
        "requested_end_date": AUTHORIZED_END.isoformat(),
        "requested_calendar_day_count": 365,
        "native_timeframe": "1m",
        "price_basis": "bid",
        "source_timezone": "UTC",
        "components": components,
        "successful_component_dates": [item["requested_day"] for item in components],
        "confirmed_absent_dates": [day.isoformat() for day in sorted(absent)],
        "absence_components": absence_components,
        "assembled_dataset_id": dataset.metadata.dataset_id,
    }
    return {"corpus_id": _sha(_json(body).encode()), **body}


@dataclass(frozen=True)
class AcquisitionResult:
    manifest_path: Path
    corpus_id: str
    assembled_dataset_id: str


def acquire_corpus(
    corpus_dir: Path,
    instrument: str,
    *,
    start_date: date = AUTHORIZED_START,
    end_date: date = AUTHORIZED_END,
    acknowledge_external_network: bool = False,
    delay_seconds: float = 1.0,
    acquire_day: Callable[..., tuple[Path, Path]] = acquire,
    sleeper: Callable[[float], None] = time.sleep,
    logger: Callable[[str], None] = print,
) -> AcquisitionResult:
    """Acquire/resume the exact year; network use requires explicit acknowledgement."""
    symbol = _instrument(instrument)
    days = _range(start_date, end_date)
    if not acknowledge_external_network:
        raise OuReplicationCorpusError(
            "external network acquisition must be explicitly acknowledged"
        )
    if delay_seconds < 0:
        raise OuReplicationCorpusError("delay_seconds must be non-negative")
    manifest_path = corpus_dir / "corpus-manifest.json"
    if manifest_path.exists():
        return authenticate_corpus(corpus_dir, symbol)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[DailyPayload] = []
    absent: list[date] = []
    for index, day in enumerate(days):
        existing = _read_success(corpus_dir, symbol, day)
        if existing is not None:
            payloads.append(existing)
        elif _read_absence(corpus_dir, symbol, day):
            absent.append(day)
        else:
            logger(f"[{index + 1}/365] acquiring {symbol} {day}")
            try:
                acquire_day(corpus_dir, day, instrument=symbol)
            except ProviderNoData:
                evidence = {
                    "provider": PROVIDER,
                    "host": HOST,
                    "canonical_instrument": symbol,
                    "requested_date": day.isoformat(),
                    "requested_url": build_url(symbol, day),
                    "source_timeframe": "M1",
                    "price_side": "BID",
                    "http_status": 404,
                    "status": "confirmed_absent",
                }
                _paths(corpus_dir, symbol, day)[2].write_text(
                    _json(evidence) + "\n", encoding="utf-8"
                )
                absent.append(day)
            else:
                item = _read_success(corpus_dir, symbol, day)
                if item is None:
                    raise OuReplicationCorpusError(
                        f"acquirer did not create exact artifacts for {day}"
                    )
                payloads.append(item)
        if delay_seconds and index != len(days) - 1:
            sleeper(delay_seconds)
    manifest = _manifest(corpus_dir, symbol, payloads, absent)
    if len(payloads) + len(absent) != 365:
        raise OuReplicationCorpusError(
            "daily evidence does not partition calendar year 2023"
        )
    if manifest_path.exists():
        raise OuReplicationCorpusError("refusing to overwrite an existing manifest")
    manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8")
    return authenticate_corpus(corpus_dir, symbol)


def authenticate_corpus(corpus_dir: Path, instrument: str) -> AcquisitionResult:
    """Reconstruct all manifest-declared identity from exact expected paths."""
    symbol = _instrument(instrument)
    try:
        stored = json.loads(
            (corpus_dir / "corpus-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise OuReplicationCorpusError("missing or corrupt corpus manifest") from error
    if not isinstance(stored, dict) or stored.get("instrument") != symbol:
        raise OuReplicationCorpusError("manifest instrument identity mismatch")
    payloads, absent = [], []
    for day in _range(AUTHORIZED_START, AUTHORIZED_END):
        item = _read_success(corpus_dir, symbol, day)
        if item is not None:
            payloads.append(item)
        elif _read_absence(corpus_dir, symbol, day):
            absent.append(day)
        else:
            raise OuReplicationCorpusError(f"missing authenticated component for {day}")
    rebuilt = _manifest(corpus_dir, symbol, payloads, absent)
    if stored != rebuilt:
        raise OuReplicationCorpusError(
            "manifest or assembled dataset identity mismatch"
        )
    return AcquisitionResult(
        corpus_dir / "corpus-manifest.json",
        rebuilt["corpus_id"],
        rebuilt["assembled_dataset_id"],
    )


def build_registry(root: Path, output_path: Path) -> Path:
    """Authenticate three explicit child paths and write a deterministic registry."""
    entries: dict[str, object] = {}
    for instrument in AUTHORIZED_INSTRUMENTS:
        corpus_path = root / instrument
        result = authenticate_corpus(corpus_path, instrument)
        manifest_bytes = result.manifest_path.read_bytes()
        entries[instrument] = {
            "instrument": instrument,
            "provider": PROVIDER,
            "requested_start_date": AUTHORIZED_START.isoformat(),
            "requested_end_date": AUTHORIZED_END.isoformat(),
            "native_timeframe": "M1",
            "price_basis": "BID",
            "source_timezone": "UTC",
            "corpus_path": str(corpus_path),
            "corpus_id": result.corpus_id,
            "assembled_dataset_id": result.assembled_dataset_id,
            "manifest_sha256": _sha(manifest_bytes),
            "verification_status": "authenticated",
            "authentication_status": "authenticated",
        }
    body: dict[str, object] = {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "instruments": entries,
    }
    registry = {**body, "registry_id": _sha(_json(body).encode())}
    encoded = _json(registry) + "\n"
    if output_path.exists() and output_path.read_text(encoding="utf-8") != encoded:
        raise OuReplicationCorpusError("refusing to overwrite an incompatible registry")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument(
        "--instrument", required=True, choices=AUTHORIZED_INSTRUMENTS
    )
    acquire_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    acquire_parser.add_argument(
        "--acknowledge-external-network", action="store_true", required=True
    )
    acquire_parser.add_argument("--delay-seconds", type=float, default=1.0)
    registry_parser = sub.add_parser("build-registry")
    registry_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    registry_parser.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "registry.json"
    )
    args = parser.parse_args()
    if args.command == "acquire":
        result = acquire_corpus(
            args.root / args.instrument,
            args.instrument,
            acknowledge_external_network=args.acknowledge_external_network,
            delay_seconds=args.delay_seconds,
        )
        print(f"corpus_manifest={result.manifest_path}")
        print(f"corpus_id={result.corpus_id}")
        print(f"assembled_dataset_id={result.assembled_dataset_id}")
    else:
        print(f"registry={build_registry(args.root, args.output)}")


if __name__ == "__main__":
    main()
