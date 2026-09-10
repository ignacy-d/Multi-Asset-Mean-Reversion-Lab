import hashlib
import json
import lzma
import shutil
import struct
from datetime import date
from pathlib import Path

import pytest

from mr_lab.providers.dukascopy import AcquisitionError
from mr_lab.providers.dukascopy_monthly import (
    acquire_month,
    assemble_year,
    month_bounds,
    verify_full_year_corpus,
)
from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import (
    RangeAcquisitionError,
    build_corpus_manifest,
)


def payload(month: int) -> bytes:
    price = 110_000 + month
    return lzma.compress(struct.pack(">5if", 0, price, price, price, price, 1.0))


def write_month(root: Path, month: int, *, instrument: str = "EURUSD") -> None:
    start, end = month_bounds(2024, month)
    raw = payload(month)
    absent = [
        date.fromordinal(day)
        for day in range(start.toordinal() + 1, end.toordinal() + 1)
    ]
    manifest = build_corpus_manifest(
        start, end, [DailyPayload(start, raw, instrument)], absent, instrument
    )
    directory = root / f"checkpoint-{month:02d}"
    directory.mkdir(parents=True)
    stem = directory / f"{instrument}-{start.isoformat()}-M1-BID"
    stem.with_suffix(".bi5").write_bytes(raw)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "requested_date": start.isoformat(),
                "canonical_instrument": instrument,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    (directory / "corpus-manifest.json").write_text(manifest.to_json())


def complete_chunks(root: Path, *, instrument: str = "EURUSD") -> None:
    for month in range(1, 13):
        write_month(root, month, instrument=instrument)


def test_monthly_assembly_matches_direct_full_year_identity(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)

    result = assemble_year(chunks, tmp_path / "assembled")
    payloads = [
        DailyPayload(date(2024, month, 1), payload(month)) for month in range(1, 13)
    ]
    successful = {item.requested_day for item in payloads}
    absent = [
        date.fromordinal(day)
        for day in range(
            date(2024, 1, 1).toordinal(), date(2024, 12, 31).toordinal() + 1
        )
        if date.fromordinal(day) not in successful
    ]
    direct = build_corpus_manifest(
        date(2024, 1, 1), date(2024, 12, 31), payloads, absent
    )

    assert result.dataset_id == direct.dataset.metadata.dataset_id
    assert result.corpus_id == direct.corpus_id
    assert json.loads(result.manifest_path.read_text()) == direct.as_dict()


def test_missing_month_and_missing_day_are_rejected(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)
    (chunks / "checkpoint-12" / "corpus-manifest.json").unlink()
    with pytest.raises(RangeAcquisitionError, match="12 monthly"):
        assemble_year(chunks, tmp_path / "out")

    shutil.rmtree(chunks / "checkpoint-12")
    write_month(chunks, 12)
    manifest_path = chunks / "checkpoint-06" / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["confirmed_absent_dates"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RangeAcquisitionError, match="missing or duplicated day"):
        assemble_year(chunks, tmp_path / "out")


def test_duplicate_month_day_and_mixed_instrument_are_rejected(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)
    duplicate = chunks / "checkpoint-02" / "corpus-manifest.json"
    value = json.loads(duplicate.read_text())
    value["requested_start_date"] = "2024-01-01"
    duplicate.write_text(json.dumps(value))
    with pytest.raises(RangeAcquisitionError, match="duplicate"):
        assemble_year(chunks, tmp_path / "out")

    chunks = tmp_path / "mixed"
    complete_chunks(chunks)
    manifest_path = chunks / "checkpoint-03" / "corpus-manifest.json"
    value = json.loads(manifest_path.read_text())
    value["instrument"] = "GBPUSD"
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(RangeAcquisitionError, match="mixed-instrument"):
        assemble_year(chunks, tmp_path / "mixed-out")


def test_month_transport_failure_remains_hard_and_delay_is_two_seconds(
    tmp_path: Path,
) -> None:
    calls = []

    def failing(*_args, **_kwargs):
        raise AcquisitionError("HTTP 503 exhausted")

    with pytest.raises(AcquisitionError, match="503"):
        acquire_month(tmp_path, 2024, 1, acquire_day=failing)

    def mostly_absent(output_dir, day, **_kwargs):
        from mr_lab.providers.dukascopy import ProviderNoData

        if day.day != 1:
            raise ProviderNoData("404")
        raw = payload(1)
        output_dir.mkdir(parents=True)
        stem = output_dir / f"EURUSD-{day.isoformat()}-M1-BID"
        raw_path = stem.with_suffix(".bi5")
        provenance_path = stem.with_suffix(".json")
        raw_path.write_bytes(raw)
        provenance_path.write_text(
            json.dumps(
                {
                    "requested_date": day.isoformat(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        )
        return raw_path, provenance_path

    acquire_month(
        tmp_path / "absent", 2024, 1, acquire_day=mostly_absent, sleeper=calls.append
    )
    assert calls == [2.0] * 30


def test_raw_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)
    (chunks / "checkpoint-04" / "EURUSD-2024-04-01-M1-BID.bi5").write_bytes(b"changed")
    with pytest.raises(RangeAcquisitionError, match="provenance mismatch"):
        assemble_year(chunks, tmp_path / "out")


def test_gbpusd_full_year_publication_is_exact_and_identity_bound(
    tmp_path: Path,
) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks, instrument="GBPUSD")
    corpus = tmp_path / "assembled"
    assembled = assemble_year(chunks, corpus, instrument="GBPUSD")

    verified = verify_full_year_corpus(corpus, instrument="GBPUSD")

    manifest = json.loads(verified.manifest_path.read_text())
    assert (manifest["requested_start_date"], manifest["requested_end_date"]) == (
        "2024-01-01",
        "2024-12-31",
    )
    assert manifest["instrument"] == "GBPUSD"
    assert manifest["provider"] == "Dukascopy"
    assert manifest["price_basis"] == "bid"
    assert manifest["native_timeframe"] == "1m"
    assert verified.corpus_id == assembled.corpus_id
    assert verified.dataset_id == assembled.dataset_id


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("instrument", "EURUSD"),
        ("requested_start_date", "2024-01-02"),
        ("requested_end_date", "2024-12-30"),
        ("provider", "Other"),
        ("price_basis", "ask"),
        ("native_timeframe", "5m"),
        ("instrument_spec", {"instrument": "GBPUSD"}),
        ("corpus_id", "sha256:not-the-identity"),
    ],
)
def test_gbpusd_publication_refuses_malformed_or_mismatched_metadata(
    tmp_path: Path, field: str, bad: object
) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks, instrument="GBPUSD")
    corpus = tmp_path / "assembled"
    assemble_year(chunks, corpus, instrument="GBPUSD")
    manifest_path = corpus / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = bad
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RangeAcquisitionError):
        verify_full_year_corpus(corpus, instrument="GBPUSD")


def test_gbpusd_publication_refuses_non_2024_component_date(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks, instrument="GBPUSD")
    corpus = tmp_path / "assembled"
    assemble_year(chunks, corpus, instrument="GBPUSD")
    manifest_path = corpus / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["confirmed_absent_dates"][-1] = "2026-01-01"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RangeAcquisitionError, match="partition 2024"):
        verify_full_year_corpus(corpus, instrument="GBPUSD")
