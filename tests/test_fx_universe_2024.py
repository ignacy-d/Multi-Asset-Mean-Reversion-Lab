import hashlib
import json
import lzma
import struct
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

import mr_lab.providers.fx_universe_2024 as universe
from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import build_corpus_manifest, enumerate_dates
from mr_lab.providers.fx_universe_2024 import (
    EXPANDED_INSTRUMENTS,
    HISTORICAL_INSTRUMENTS,
    REGISTRY_INSTRUMENTS,
    FxUniverseError,
    build_registry,
    validate_expanded_universe,
    validate_registry,
)
from mr_lab.providers.instruments import (
    DECLARED_INSTRUMENTS,
    InstrumentSpecError,
    get_candidate_instrument_spec,
    get_instrument_spec,
    require_verified,
)
from mr_lab.providers.verify_instruments import (
    VERIFICATION_DAYS,
    VERIFICATION_INSTRUMENTS,
    CandidateVerificationError,
    build_url_for_candidate,
    verify_candidate,
)

RECORD = struct.Struct(">5if")
ENCODED_PRICES = {
    "USDCAD": 135_000,
    "USDCHF": 90_000,
    "NZDUSD": 62_000,
    "EURGBP": 86_000,
}


def payload(instrument: str) -> bytes:
    price = ENCODED_PRICES[instrument]
    decoded = b"".join(
        RECORD.pack(offset, price, price, price, price, 1.0)
        for offset in range(0, 3600, 60)
    )
    return lzma.compress(decoded, format=lzma.FORMAT_ALONE)


def synthetic_acquire(
    output_dir: Path, day: date, instrument: str
) -> tuple[Path, Path]:
    spec = get_candidate_instrument_spec(instrument)
    raw = payload(instrument)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{instrument}-{day.isoformat()}-M1-BID"
    raw_path = stem.with_suffix(".bi5")
    provenance_path = stem.with_suffix(".json")
    raw_path.write_bytes(raw)
    provenance_path.write_text(
        json.dumps(
            {
                "canonical_instrument": instrument,
                "provider_symbol": spec.provider_symbol,
                "price_scale": spec.price_scale,
                "price_precision": spec.price_precision,
                "source_timeframe": "M1",
                "price_side": "BID",
                "requested_date": day.isoformat(),
                "requested_url": build_url_for_candidate(instrument, day),
                "http_status": 200,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return raw_path, provenance_path


def test_candidate_declarations_are_exact_and_promotion_remains_explicit() -> None:
    assert DECLARED_INSTRUMENTS[-4:] == VERIFICATION_INSTRUMENTS
    for instrument in VERIFICATION_INSTRUMENTS:
        spec = get_candidate_instrument_spec(instrument)
        assert (spec.provider_symbol, spec.price_scale, spec.price_precision) == (
            instrument,
            100_000,
            5,
        )
        assert not spec.decoding_verified
        with pytest.raises(InstrumentSpecError, match="not production-verified"):
            require_verified(spec)
        promoted = get_instrument_spec(instrument)
        assert promoted.decoding_verified
        assert replace(promoted, decoding_verified=False) == spec


@pytest.mark.parametrize("instrument", VERIFICATION_INSTRUMENTS)
@pytest.mark.parametrize("day", VERIFICATION_DAYS)
def test_candidate_urls_use_exact_symbol_and_zero_based_2024_month(
    instrument: str, day: date
) -> None:
    expected = (
        f"https://datafeed.dukascopy.com/datafeed/{instrument}/2024/"
        f"{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    )
    assert build_url_for_candidate(instrument, day) == expected


def test_candidate_workflow_rejects_non_2024_and_substitute_dates() -> None:
    with pytest.raises(CandidateVerificationError, match="only 2024"):
        build_url_for_candidate("USDCAD", date(2026, 1, 2))
    with pytest.raises(CandidateVerificationError, match="expanded-universe"):
        build_url_for_candidate("EURUSD", date(2024, 1, 2))
    with pytest.raises(CandidateVerificationError, match="exact three"):
        verify_candidate(
            Path("unused"),
            "USDCAD",
            days=(date(2024, 1, 3),),
            acquire_sample=lambda *_: pytest.fail("must reject before acquisition"),
        )


def test_candidate_verification_report_is_deterministic_and_semantic(
    tmp_path: Path,
) -> None:
    path = verify_candidate(tmp_path, "USDCAD", acquire_sample=synthetic_acquire)
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["verification_passed"] is True
    assert report["candidate_spec"]["decoding_verified"] is False
    assert report["authorized_dates"] == [day.isoformat() for day in VERIFICATION_DAYS]
    assert report["sample_count"] == 3
    assert report["verification_report_id"].startswith("sha256:")
    for sample in report["samples"]:
        assert sample["resampling_consistent"] is True
        assert sample["audit"]["price_basis"] == "bid"
        assert sample["audit"]["source_timezone"] == "UTC"
        assert sample["audit"]["instrument"] == "USDCAD"
        assert sample["provider_response"]["http_status"] == 200
        assert "retrieved_at" not in sample["provider_response"]


def write_full_year_corpus(root: Path, instrument: str) -> dict[str, object]:
    day = date(2024, 1, 2)
    raw = payload(instrument) if instrument in ENCODED_PRICES else payload("USDCAD")
    absent = [
        requested
        for requested in enumerate_dates(date(2024, 1, 1), date(2024, 12, 31))
        if requested != day
    ]
    manifest = build_corpus_manifest(
        date(2024, 1, 1),
        date(2024, 12, 31),
        [DailyPayload(day, raw, instrument)],
        absent,
        instrument,
    )
    corpus_dir = root / instrument
    corpus_dir.mkdir(parents=True)
    stem = corpus_dir / f"{instrument}-{day.isoformat()}-M1-BID"
    stem.with_suffix(".bi5").write_bytes(raw)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "requested_date": day.isoformat(),
                "canonical_instrument": instrument,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (corpus_dir / "corpus-manifest.json").write_text(
        manifest.to_json() + "\n", encoding="utf-8"
    )
    return manifest.as_dict()


def synthetic_registry_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    historical_root = tmp_path / "historical"
    expanded_root = tmp_path / "expanded"
    historical_entries = {}
    for instrument in REGISTRY_INSTRUMENTS:
        root = (
            historical_root if instrument in HISTORICAL_INSTRUMENTS else expanded_root
        )
        manifest = write_full_year_corpus(root, instrument)
        if instrument in HISTORICAL_INSTRUMENTS:
            historical_entries[instrument] = {
                "instrument": instrument,
                "verification_status": "verified",
                "requested_start_date": "2024-01-01",
                "requested_end_date": "2024-12-31",
                "corpus_id": manifest["corpus_id"],
                "assembled_dataset_id": manifest["assembled_dataset_id"],
            }
        else:
            verify_candidate(
                expanded_root / "verification",
                instrument,
                acquire_sample=synthetic_acquire,
            )
    historical_registry = tmp_path / "historical-registry.json"
    historical_registry.write_text(
        json.dumps(
            {
                "registry_schema_version": "stage-4a-2024-corpus-registry-v1",
                "instruments": historical_entries,
            }
        ),
        encoding="utf-8",
    )
    return historical_root, expanded_root, historical_registry


def test_expanded_registry_is_exact_identity_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    historical_root, expanded_root, historical_registry = synthetic_registry_inputs(
        tmp_path
    )
    registry = build_registry(historical_root, expanded_root, historical_registry)
    validated = validate_registry(registry)

    assert set(validated["instruments"]) == set(REGISTRY_INSTRUMENTS)
    assert validated["registry_id"].startswith("sha256:")
    assert set(EXPANDED_INSTRUMENTS) <= set(validated["instruments"])
    for instrument, entry in validated["instruments"].items():
        assert entry["instrument"] == instrument
        assert entry["verification_status"] == "verified"
        assert entry["provider"] == "Dukascopy"
        assert entry["price_basis"] == "bid"
        assert entry["native_timeframe"] == "1m"
        assert entry["source_timezone"] == "UTC"
        assert Path(entry["corpus_path"]).is_absolute()

    changed = json.loads(json.dumps(registry))
    changed["instruments"]["EURGBP"]["price_basis"] = "ask"
    with pytest.raises(FxUniverseError, match="identity mismatch"):
        validate_registry(changed)


def test_nine_instrument_semantic_report_has_exact_synchronization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical_root, expanded_root, historical_registry = synthetic_registry_inputs(
        tmp_path
    )
    registry = build_registry(historical_root, expanded_root, historical_registry)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(universe, "CANONICAL_HISTORICAL_ROOT", historical_root)
    monkeypatch.setattr(universe, "CANONICAL_EXPANDED_ROOT", expanded_root)

    report_path = validate_expanded_universe(
        registry_path, tmp_path / "validation.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["semantic_validation_passed"] is True
    assert set(report["instruments"]) == set(REGISTRY_INSTRUMENTS)
    assert report["synchronization"]["nine_way_intersection_count"] == 60
    assert report["synchronization"]["union_count"] == 60
    assert report["repairs_applied"] == []
    assert all(
        item["instrument"] == instrument
        and item["provider"] == "Dukascopy"
        and item["provider_symbol"] == instrument
        and item["price_basis"] == "bid"
        and item["native_timeframe"] == "1m"
        and item["source_timezone"] == "UTC"
        and item["timestamps_strictly_monotonic"] is True
        and item["duplicate_timestamp_count"] == 0
        and item["nonfinite_price_count"] == 0
        and item["invalid_nonpositive_price_count"] == 0
        and item["impossible_ohlc_count"] == 0
        for instrument, item in report["instruments"].items()
    )
