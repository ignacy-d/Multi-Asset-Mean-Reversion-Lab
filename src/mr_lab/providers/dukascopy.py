"""Bounded, credential-free acquisition from Dukascopy's public datafeed."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HOST = "datafeed.dukascopy.com"
PROVIDER = "Dukascopy"
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_BACKOFF_SECONDS = 30.0


class AcquisitionError(RuntimeError):
    """Raised when a provider response cannot be accepted as raw market data."""


class ProviderNoData(AcquisitionError):
    """Raised only when the provider explicitly reports that a day is absent."""


def build_url(instrument: str, day: date) -> str:
    """Build the public M1 BID candle URL (provider months are zero based)."""
    normalized = instrument.strip().upper()
    if normalized != "EURUSD":
        raise ValueError("Stage 1A-R acquisition is bounded to EURUSD")
    return (
        f"https://{HOST}/datafeed/{normalized}/{day.year:04d}/"
        f"{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    )


def payload_sha256(payload: bytes) -> str:
    """Return a deterministic content fingerprint."""
    return hashlib.sha256(payload).hexdigest()


def validate_payload(payload: bytes) -> int:
    """Return decoded size after validating a non-empty LZMA provider payload."""
    if not payload:
        raise AcquisitionError("provider returned an empty payload")
    prefix = payload[:256].lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html")):
        raise AcquisitionError("provider returned HTML instead of BI5 data")
    try:
        decoded = lzma.decompress(payload)
    except lzma.LZMAError as error:
        raise AcquisitionError(
            "payload is not valid LZMA-compressed BI5 data"
        ) from error
    if not decoded:
        raise AcquisitionError("decoded BI5 payload is empty")
    return len(decoded)


def make_provenance(
    *,
    payload: bytes,
    requested_day: date,
    url: str,
    http_status: int,
    decoded_byte_length: int,
) -> dict[str, object]:
    """Build audit metadata; retrieval time is provenance, not the SHA-256."""
    return {
        "provider": PROVIDER,
        "host": HOST,
        "requested_url": url,
        "canonical_instrument": "EURUSD",
        "source_timeframe": "M1",
        "price_side": "BID",
        "requested_date": requested_day.isoformat(),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "http_status": http_status,
        "compressed_byte_length": len(payload),
        "decoded_byte_length": decoded_byte_length,
        "sha256": payload_sha256(payload),
    }


def _get(url: str, timeout: float) -> tuple[int, bytes]:
    request = Request(url, method="GET", headers={"User-Agent": "mr-lab/Stage-1A-R"})
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def acquire(
    output_dir: Path,
    requested_day: date,
    *,
    timeout: float = 30.0,
    retries: int = 6,
    getter: Callable[[str, float], tuple[int, bytes]] = _get,
    sleeper: Callable[[float], None] = time.sleep,
    logger: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    """GET one frozen day and write its payload and provenance."""
    if retries < 0:
        raise ValueError("retries must be non-negative")
    url = build_url("EURUSD", requested_day)
    total_attempts = retries + 1
    for attempt in range(total_attempts):
        try:
            status, payload = getter(url, timeout)
            if status != 200:
                raise HTTPError(url, status, "unexpected HTTP status", {}, None)
            break
        except HTTPError as error:
            if error.code == 404:
                raise ProviderNoData(
                    f"Dukascopy has no daily file for {requested_day.isoformat()}"
                ) from error
            if error.code not in _TRANSIENT_STATUS:
                raise AcquisitionError(
                    f"Dukascopy GET failed for {requested_day.isoformat()}: "
                    f"HTTP {error.code} on attempt {attempt + 1}"
                ) from error
            if attempt == retries:
                raise AcquisitionError(
                    f"Dukascopy GET failed for {requested_day.isoformat()}: "
                    f"HTTP {error.code}; exhausted {total_attempts} attempts"
                ) from error
            failure = f"HTTP {error.code}"
        except (TimeoutError, URLError) as error:
            if attempt == retries:
                raise AcquisitionError(
                    f"Dukascopy GET failed for {requested_day.isoformat()}: "
                    f"transport error {error}; exhausted {total_attempts} attempts"
                ) from error
            failure = f"transport error {error}"
        backoff = min(float(2**attempt), _MAX_BACKOFF_SECONDS)
        logger(
            f"retry date={requested_day.isoformat()} error={failure} "
            f"attempt={attempt + 1}/{total_attempts} sleep={backoff:g}s"
        )
        sleeper(backoff)

    decoded_byte_length = validate_payload(payload)
    metadata = make_provenance(
        payload=payload,
        requested_day=requested_day,
        url=url,
        http_status=status,
        decoded_byte_length=decoded_byte_length,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"EURUSD-{requested_day.isoformat()}-M1-BID"
    raw_path = output_dir / f"{stem}.bi5"
    metadata_path = output_dir / f"{stem}.json"
    if raw_path.exists() or metadata_path.exists():
        raise AcquisitionError("refusing to overwrite an existing raw snapshot")
    raw_path.write_bytes(payload)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return raw_path, metadata_path


def main() -> None:
    """Run the frozen Stage 1A-R acquisition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=date.fromisoformat, default=date(2024, 1, 2))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw_path, metadata_path = acquire(args.output_dir, args.date)
    print(f"raw_payload={raw_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
