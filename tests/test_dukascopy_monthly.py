import hashlib
import json
import lzma
import shutil
import struct
from datetime import date
from pathlib import Path

import pytest

from mr_lab.providers.dukascopy import AcquisitionError, acquire
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


def acquiring_with(calls: list[date], *, fail_on: date | None = None):
    def acquire_day(output_dir: Path, day: date, **kwargs):
        calls.append(day)
        if day == fail_on:
            raise AcquisitionError("injected transport failure")
        instrument = str(kwargs.pop("instrument", "EURUSD"))
        return acquire(
            output_dir,
            day,
            instrument=instrument,
            getter=lambda _url, _timeout: (200, payload(day.day)),
            **kwargs,
        )

    return acquire_day


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
    directory = root / f"month-{month:02d}"
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


def test_legacy_eurusd_full_year_publication_verifies_without_instrument_spec(
    tmp_path: Path,
) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks, instrument="EURUSD")
    corpus = tmp_path / "assembled"
    assembled = assemble_year(chunks, corpus, instrument="EURUSD")

    manifest = json.loads(assembled.manifest_path.read_text())
    assert "instrument_spec" not in manifest

    verified = verify_full_year_corpus(corpus, instrument="EURUSD")
    assert verified.corpus_id == assembled.corpus_id
    assert verified.dataset_id == assembled.dataset_id


def test_missing_month_and_missing_day_are_rejected(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)
    (chunks / "month-12" / "corpus-manifest.json").unlink()
    with pytest.raises(RangeAcquisitionError, match="month 2024-12"):
        assemble_year(chunks, tmp_path / "out")

    shutil.rmtree(chunks / "month-12")
    write_month(chunks, 12)
    manifest_path = chunks / "month-06" / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["confirmed_absent_dates"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RangeAcquisitionError, match="missing or duplicated day"):
        assemble_year(chunks, tmp_path / "out")


def test_year_assembly_never_discovers_a_substitute_month(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)
    expected = chunks / "month-12" / "corpus-manifest.json"
    decoy = chunks / "unexpected" / "nested"
    decoy.mkdir(parents=True)
    shutil.copyfile(expected, decoy / "corpus-manifest.json")
    expected.unlink()

    with pytest.raises(RangeAcquisitionError, match="month 2024-12"):
        assemble_year(chunks, tmp_path / "out")


def test_duplicate_month_day_and_mixed_instrument_are_rejected(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)
    duplicate = chunks / "month-02" / "corpus-manifest.json"
    value = json.loads(duplicate.read_text())
    value["requested_start_date"] = "2024-01-01"
    duplicate.write_text(json.dumps(value))
    with pytest.raises(RangeAcquisitionError, match="wrong explicit path"):
        assemble_year(chunks, tmp_path / "out")

    chunks = tmp_path / "mixed"
    complete_chunks(chunks)
    manifest_path = chunks / "month-03" / "corpus-manifest.json"
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


def test_interrupted_month_resumes_without_requesting_valid_days_and_matches_clean(
    tmp_path: Path,
) -> None:
    resumed_dir = tmp_path / "resumed"
    first_calls: list[date] = []
    with pytest.raises(AcquisitionError, match="injected"):
        acquire_month(
            resumed_dir,
            2024,
            1,
            instrument="GBPUSD",
            acquire_day=acquiring_with(first_calls, fail_on=date(2024, 1, 4)),
            delay_seconds=0,
        )
    assert first_calls == [date(2024, 1, day) for day in range(1, 5)]

    resume_calls: list[date] = []
    resumed = acquire_month(
        resumed_dir,
        2024,
        1,
        instrument="GBPUSD",
        resume=True,
        acquire_day=acquiring_with(resume_calls),
        delay_seconds=0,
    )
    assert resume_calls == [date(2024, 1, day) for day in range(4, 32)]

    clean_calls: list[date] = []
    clean = acquire_month(
        tmp_path / "clean",
        2024,
        1,
        instrument="GBPUSD",
        acquire_day=acquiring_with(clean_calls),
        delay_seconds=0,
    )
    assert resumed.manifest.corpus_id == clean.manifest.corpus_id
    assert (
        resumed.manifest.dataset.metadata.dataset_id
        == clean.manifest.dataset.metadata.dataset_id
    )


@pytest.mark.parametrize("damaged", ["raw", "provenance", "orphan_raw", "orphan_json"])
def test_resume_fails_closed_on_corrupt_or_orphaned_snapshot(
    tmp_path: Path, damaged: str
) -> None:
    acquire(tmp_path, date(2024, 1, 1), getter=lambda *_: (200, payload(1)))
    stem = tmp_path / "EURUSD-2024-01-01-M1-BID"
    raw_path, provenance_path = stem.with_suffix(".bi5"), stem.with_suffix(".json")
    if damaged == "raw":
        raw_path.write_bytes(b"corrupt")
    elif damaged == "provenance":
        provenance = json.loads(provenance_path.read_text())
        provenance["canonical_instrument"] = "GBPUSD"
        provenance_path.write_text(json.dumps(provenance))
    elif damaged == "orphan_raw":
        provenance_path.unlink()
    else:
        raw_path.unlink()

    getter_calls: list[date] = []
    with pytest.raises(RangeAcquisitionError, match="snapshot|provenance|orphaned"):
        acquire_month(
            tmp_path,
            2024,
            1,
            resume=True,
            acquire_day=acquiring_with(getter_calls),
            delay_seconds=0,
        )
    assert getter_calls == []


def test_resume_rejects_wrong_bounds_before_touching_acquisition(
    tmp_path: Path,
) -> None:
    calls: list[date] = []
    for year, month in ((2025, 1), (2024, 13)):
        with pytest.raises(RangeAcquisitionError):
            acquire_month(
                tmp_path,
                year,
                month,
                resume=True,
                acquire_day=acquiring_with(calls),
            )
    assert calls == []


@pytest.mark.parametrize(
    "name",
    [
        "GBPUSD-2024-01-01-M1-BID.bi5",
        "EURUSD-2024-02-01-M1-BID.json",
        "EURUSD-2024-01-01-M1-BID-copy.bi5",
    ],
)
def test_resume_rejects_wrong_instrument_month_or_duplicate_file(
    tmp_path: Path, name: str
) -> None:
    tmp_path.joinpath(name).write_bytes(b"do not inspect")
    calls: list[date] = []

    with pytest.raises(RangeAcquisitionError, match="mismatched|ambiguous|duplicate"):
        acquire_month(
            tmp_path,
            2024,
            1,
            resume=True,
            acquire_day=acquiring_with(calls),
            delay_seconds=0,
        )
    assert calls == []


def test_completed_month_resume_is_idempotent_and_makes_no_acquisition(
    tmp_path: Path,
) -> None:
    initial_calls: list[date] = []
    initial = acquire_month(
        tmp_path,
        2024,
        1,
        instrument="GBPUSD",
        acquire_day=acquiring_with(initial_calls),
        delay_seconds=0,
    )
    forbidden_calls: list[date] = []
    resumed = acquire_month(
        tmp_path,
        2024,
        1,
        instrument="GBPUSD",
        resume=True,
        acquire_day=acquiring_with(forbidden_calls),
        delay_seconds=0,
    )
    assert forbidden_calls == []
    assert resumed.manifest.corpus_id == initial.manifest.corpus_id


def test_raw_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks)
    (chunks / "month-04" / "EURUSD-2024-04-01-M1-BID.bi5").write_bytes(b"changed")
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
    assert manifest["instrument_spec"]["instrument"] == "GBPUSD"
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


def test_gbpusd_publication_refuses_missing_instrument_spec(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    complete_chunks(chunks, instrument="GBPUSD")
    corpus = tmp_path / "assembled"
    assemble_year(chunks, corpus, instrument="GBPUSD")
    manifest_path = corpus / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("instrument_spec")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RangeAcquisitionError, match="manifest contract mismatch"):
        verify_full_year_corpus(corpus, instrument="GBPUSD")
