import hashlib
import json
import lzma
import struct
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import mr_lab.providers.ou_replication_2023 as corpus
from mr_lab.providers.dukascopy import ProviderNoData, build_url
from mr_lab.providers.instruments import get_instrument_spec

RECORD = struct.Struct(">5if")


def _payload() -> bytes:
    decoded = b"".join(
        RECORD.pack(offset, 100_000, 100_000, 100_000, 100_000, 1.0)
        for offset in range(0, 3600, 60)
    )
    return lzma.compress(decoded, format=lzma.FORMAT_ALONE)


def fake_acquire(output: Path, day: date, *, instrument: str) -> tuple[Path, Path]:
    if day != date(2023, 1, 3):
        raise ProviderNoData("synthetic absence")
    raw = _payload()
    spec = get_instrument_spec(instrument)
    stem = output / f"{instrument}-{day.isoformat()}-M1-BID"
    raw_path, provenance_path = stem.with_suffix(".bi5"), stem.with_suffix(".json")
    raw_path.write_bytes(raw)
    provenance_path.write_text(
        json.dumps(
            {
                "provider": "Dukascopy",
                "host": "datafeed.dukascopy.com",
                "requested_url": build_url(instrument, day),
                "canonical_instrument": instrument,
                "provider_symbol": spec.provider_symbol,
                "price_scale": spec.price_scale,
                "price_precision": spec.price_precision,
                "source_timeframe": "M1",
                "price_side": "BID",
                "requested_date": day.isoformat(),
                "retrieved_at": datetime(2023, 1, 4, tzinfo=UTC).isoformat(),
                "http_status": 200,
                "compressed_byte_length": len(raw),
                "decoded_byte_length": 60 * RECORD.size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return raw_path, provenance_path


def acquire_synthetic(root: Path, instrument: str) -> corpus.AcquisitionResult:
    return corpus.acquire_corpus(
        root / instrument,
        instrument,
        acknowledge_external_network=True,
        delay_seconds=0,
        acquire_day=fake_acquire,
        logger=lambda _: None,
    )


@pytest.mark.parametrize("instrument", corpus.AUTHORIZED_INSTRUMENTS)
def test_only_exact_authorized_instruments_are_accepted(
    tmp_path: Path, instrument: str
) -> None:
    result = acquire_synthetic(tmp_path, instrument)
    assert result.corpus_id.startswith("sha256:")
    assert corpus.authenticate_corpus(tmp_path / instrument, instrument) == result


@pytest.mark.parametrize("instrument", ["USDJPY", "audusd", "", "EURGBP"])
def test_other_instruments_fail_before_acquisition(
    tmp_path: Path, instrument: str
) -> None:
    with pytest.raises(corpus.OuReplicationCorpusError, match="not authorized"):
        acquire_synthetic(tmp_path, instrument)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2022, 1, 1), date(2022, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2023, 1, 2), date(2023, 12, 31)),
        (date(2023, 1, 1), date(2023, 12, 30)),
    ],
)
def test_only_exact_calendar_2023_fails_before_acquisition(
    tmp_path: Path, start: date, end: date
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> tuple[Path, Path]:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(
        corpus.OuReplicationCorpusError, match="exactly calendar year 2023"
    ):
        corpus.acquire_corpus(
            tmp_path,
            "AUDUSD",
            start_date=start,
            end_date=end,
            acknowledge_external_network=True,
            acquire_day=forbidden,
        )
    assert not called


def test_network_use_requires_explicit_local_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(
        corpus.OuReplicationCorpusError, match="explicitly acknowledged"
    ):
        corpus.acquire_corpus(tmp_path, "AUDUSD", acquire_day=fake_acquire)


def test_payload_and_manifest_hash_mismatches_fail_closed(tmp_path: Path) -> None:
    result = acquire_synthetic(tmp_path, "AUDUSD")
    raw_path = tmp_path / "AUDUSD" / "AUDUSD-2023-01-03-M1-BID.bi5"
    raw_path.write_bytes(raw_path.read_bytes() + b"corrupt")
    with pytest.raises(corpus.OuReplicationCorpusError, match="payload"):
        corpus.authenticate_corpus(tmp_path / "AUDUSD", "AUDUSD")

    raw_path.write_bytes(_payload())
    manifest = json.loads(result.manifest_path.read_text())
    manifest["corpus_id"] = "sha256:" + "0" * 64
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(corpus.OuReplicationCorpusError, match="identity mismatch"):
        corpus.authenticate_corpus(tmp_path / "AUDUSD", "AUDUSD")


def test_provenance_byte_mutation_fails_authentication(tmp_path: Path) -> None:
    acquire_synthetic(tmp_path, "AUDUSD")
    provenance = tmp_path / "AUDUSD" / "AUDUSD-2023-01-03-M1-BID.json"
    provenance.write_bytes(provenance.read_bytes() + b"\n")
    with pytest.raises(corpus.OuReplicationCorpusError, match="identity mismatch"):
        corpus.authenticate_corpus(tmp_path / "AUDUSD", "AUDUSD")


def test_absence_evidence_byte_mutation_fails_authentication(tmp_path: Path) -> None:
    acquire_synthetic(tmp_path, "EURUSD")
    evidence = tmp_path / "EURUSD" / "EURUSD-2023-02-01-M1-BID.absent.json"
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    with pytest.raises(corpus.OuReplicationCorpusError, match="identity mismatch"):
        corpus.authenticate_corpus(tmp_path / "EURUSD", "EURUSD")


def test_missing_and_corrupt_components_fail_closed(tmp_path: Path) -> None:
    acquire_synthetic(tmp_path, "EURUSD")
    missing = tmp_path / "EURUSD" / "EURUSD-2023-02-01-M1-BID.absent.json"
    missing.unlink()
    with pytest.raises(corpus.OuReplicationCorpusError, match="missing authenticated"):
        corpus.authenticate_corpus(tmp_path / "EURUSD", "EURUSD")


def test_registry_is_exact_and_deterministic(tmp_path: Path) -> None:
    for instrument in corpus.AUTHORIZED_INSTRUMENTS:
        acquire_synthetic(tmp_path, instrument)
    output = tmp_path / "registry.json"
    corpus.build_registry(tmp_path, output)
    first = output.read_bytes()
    corpus.build_registry(tmp_path, output)
    assert output.read_bytes() == first
    registry = json.loads(first)
    assert tuple(registry["instruments"]) == corpus.AUTHORIZED_INSTRUMENTS
    assert all(
        entry["manifest_sha256"].startswith("sha256:")
        and len(entry["manifest_sha256"]) == 71
        for entry in registry["instruments"].values()
    )
    assert all(
        entry["authentication_status"] == "authenticated"
        for entry in registry["instruments"].values()
    )


def test_pipeline_performs_no_discovery_or_strategy_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("filesystem discovery is forbidden")

    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    result = acquire_synthetic(tmp_path, "GBPUSD")
    assert result.manifest_path.is_file()
    source = Path(corpus.__file__).read_text(encoding="utf-8")
    assert "stage4b_runner" not in source
    assert "ou_longusd_tradable_runner" not in source
