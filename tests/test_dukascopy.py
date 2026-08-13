import json
import lzma
from datetime import date
from pathlib import Path

import pytest

from mr_lab.providers.dukascopy import (
    AcquisitionError,
    acquire,
    build_url,
    make_provenance,
    payload_sha256,
    validate_payload,
)


def bi5_payload(records: int = 1) -> bytes:
    return lzma.compress(bytes(24 * records), format=lzma.FORMAT_ALONE)


def test_url_uses_zero_based_january_and_get_datafeed_path() -> None:
    assert build_url(" eurusd ", date(2024, 1, 2)) == (
        "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/02/"
        "BID_candles_min_1.bi5"
    )


def test_stage_is_bounded_to_eurusd() -> None:
    with pytest.raises(ValueError, match="bounded to EURUSD"):
        build_url("GBPUSD", date(2024, 1, 2))


@pytest.mark.parametrize("payload", [b"", b"<html>error</html>", b"not lzma"])
def test_invalid_payload_is_rejected(payload: bytes) -> None:
    with pytest.raises(AcquisitionError):
        validate_payload(payload)


def test_structural_validation_counts_24_byte_records() -> None:
    assert validate_payload(bi5_payload(3)) == 3


def test_sha256_is_deterministic() -> None:
    assert payload_sha256(b"same") == payload_sha256(b"same")
    assert payload_sha256(b"same") != payload_sha256(b"different")


def test_provenance_records_source_without_credentials() -> None:
    payload = bi5_payload()
    metadata = make_provenance(
        payload=payload,
        requested_day=date(2024, 1, 2),
        url=build_url("EURUSD", date(2024, 1, 2)),
        http_status=200,
        row_count=1,
    )
    assert metadata["price_side"] == "BID"
    assert metadata["sha256"] == payload_sha256(payload)
    assert not any("credential" in key or "secret" in key for key in metadata)


def test_acquire_writes_payload_and_metadata_to_caller_directory(
    tmp_path: Path,
) -> None:
    payload = bi5_payload(2)
    calls = []

    def getter(url: str, timeout: float) -> tuple[int, bytes]:
        calls.append((url, timeout))
        return 200, payload

    raw, audit = acquire(tmp_path, date(2024, 1, 2), getter=getter)

    assert raw.read_bytes() == payload
    assert json.loads(audit.read_text())["decoded_record_count"] == 2
    assert calls == [(build_url("EURUSD", date(2024, 1, 2)), 30.0)]


def test_acquire_refuses_to_overwrite_raw_snapshot(tmp_path: Path) -> None:
    def getter(_url: str, _timeout: float) -> tuple[int, bytes]:
        return 200, bi5_payload()

    acquire(tmp_path, date(2024, 1, 2), getter=getter)
    with pytest.raises(AcquisitionError, match="overwrite"):
        acquire(tmp_path, date(2024, 1, 2), getter=getter)
