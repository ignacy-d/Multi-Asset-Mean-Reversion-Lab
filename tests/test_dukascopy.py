import json
import lzma
from datetime import date
from pathlib import Path
from urllib.error import URLError

import pytest

from mr_lab.providers.dukascopy import (
    AcquisitionError,
    ProviderNoData,
    acquire,
    build_url,
    make_provenance,
    payload_sha256,
    validate_payload,
)


def lzma_payload(decoded: bytes = b"opaque provider data") -> bytes:
    return lzma.compress(decoded, format=lzma.FORMAT_ALONE)


def test_url_uses_zero_based_january_and_get_datafeed_path() -> None:
    assert build_url(" eurusd ", date(2024, 1, 2)) == (
        "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/02/"
        "BID_candles_min_1.bi5"
    )


def test_stage_is_bounded_to_eurusd() -> None:
    with pytest.raises(ValueError, match="bounded to EURUSD"):
        build_url("GBPUSD", date(2024, 1, 2))


@pytest.mark.parametrize("payload", [b"", b"<html>error</html>"])
def test_empty_or_html_payload_is_rejected(payload: bytes) -> None:
    with pytest.raises(AcquisitionError):
        validate_payload(payload)


def test_valid_non_empty_lzma_payload_is_accepted() -> None:
    assert validate_payload(lzma_payload(b"uninterpreted bytes")) == 19


def test_empty_decoded_lzma_payload_is_rejected() -> None:
    with pytest.raises(AcquisitionError, match="decoded BI5 payload is empty"):
        validate_payload(lzma_payload(b""))


def test_corrupt_lzma_payload_is_rejected() -> None:
    with pytest.raises(AcquisitionError, match="not valid LZMA"):
        validate_payload(b"not lzma")


def test_sha256_is_deterministic() -> None:
    assert payload_sha256(b"same") == payload_sha256(b"same")
    assert payload_sha256(b"same") != payload_sha256(b"different")


def test_provenance_records_source_without_credentials() -> None:
    payload = lzma_payload()
    metadata = make_provenance(
        payload=payload,
        requested_day=date(2024, 1, 2),
        url=build_url("EURUSD", date(2024, 1, 2)),
        http_status=200,
        decoded_byte_length=len(b"opaque provider data"),
    )
    assert metadata["price_side"] == "BID"
    assert metadata["sha256"] == payload_sha256(payload)
    assert metadata["compressed_byte_length"] == len(payload)
    assert metadata["decoded_byte_length"] == len(b"opaque provider data")
    assert "decoded_record_count" not in metadata
    assert not any("credential" in key or "secret" in key for key in metadata)


def test_acquire_writes_payload_and_metadata_to_caller_directory(
    tmp_path: Path,
) -> None:
    decoded = b"synthetic opaque payload"
    payload = lzma_payload(decoded)
    calls = []

    def getter(url: str, timeout: float) -> tuple[int, bytes]:
        calls.append((url, timeout))
        return 200, payload

    raw, audit = acquire(tmp_path, date(2024, 1, 2), getter=getter)

    assert raw.read_bytes() == payload
    metadata = json.loads(audit.read_text())
    assert metadata["decoded_byte_length"] == len(decoded)
    assert "decoded_record_count" not in metadata
    assert calls == [(build_url("EURUSD", date(2024, 1, 2)), 30.0)]


def test_acquire_refuses_to_overwrite_raw_snapshot(tmp_path: Path) -> None:
    def getter(_url: str, _timeout: float) -> tuple[int, bytes]:
        return 200, lzma_payload()

    acquire(tmp_path, date(2024, 1, 2), getter=getter)
    with pytest.raises(AcquisitionError, match="overwrite"):
        acquire(tmp_path, date(2024, 1, 2), getter=getter)


def test_only_http_404_is_confirmed_provider_absence(tmp_path: Path) -> None:
    def getter(_url: str, _timeout: float) -> tuple[int, bytes]:
        return 404, b""

    with pytest.raises(ProviderNoData, match="no daily file"):
        acquire(tmp_path, date(2024, 1, 6), retries=0, getter=getter)


def test_server_failure_is_not_provider_absence(tmp_path: Path) -> None:
    def getter(_url: str, _timeout: float) -> tuple[int, bytes]:
        return 500, b""

    with pytest.raises(AcquisitionError, match="HTTP 500") as caught:
        acquire(tmp_path, date(2024, 1, 6), retries=0, getter=getter)

    assert not isinstance(caught.value, ProviderNoData)


def test_http_503_twice_then_success_retries_with_bounded_backoff(
    tmp_path: Path,
) -> None:
    payload = lzma_payload()
    responses = iter([(503, b""), (503, b""), (200, payload)])
    sleeps: list[float] = []

    raw, _ = acquire(
        tmp_path,
        date(2024, 2, 11),
        getter=lambda _url, _timeout: next(responses),
        sleeper=sleeps.append,
        logger=lambda _message: None,
    )

    assert raw.read_bytes() == payload
    assert sleeps == [1.0, 2.0]


def test_transient_http_exhaustion_is_fatal_and_names_date(tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []

    def getter(_url: str, _timeout: float) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        return 503, b""

    with pytest.raises(
        AcquisitionError, match=r"2024-02-11.*HTTP 503.*exhausted 7 attempts"
    ):
        acquire(
            tmp_path,
            date(2024, 2, 11),
            getter=getter,
            sleeper=sleeps.append,
            logger=lambda _message: None,
        )

    assert calls == 7
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]


def test_http_404_is_not_retried(tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []

    def getter(_url: str, _timeout: float) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        return 404, b""

    with pytest.raises(ProviderNoData):
        acquire(
            tmp_path,
            date(2024, 2, 11),
            getter=getter,
            sleeper=sleeps.append,
        )

    assert calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "failure", [TimeoutError("timed out"), URLError("temporary DNS failure")]
)
def test_transport_failure_can_recover_without_real_sleep(
    tmp_path: Path, failure: Exception
) -> None:
    payload = lzma_payload()
    responses: list[Exception | tuple[int, bytes]] = [failure, (200, payload)]
    sleeps: list[float] = []

    def getter(_url: str, _timeout: float) -> tuple[int, bytes]:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    raw, _ = acquire(
        tmp_path,
        date(2024, 2, 11),
        getter=getter,
        sleeper=sleeps.append,
        logger=lambda _message: None,
    )

    assert raw.read_bytes() == payload
    assert sleeps == [1.0]
