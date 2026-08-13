"""Parse the verified Dukascopy EURUSD M1 BID candle representation.

This provider-local module deliberately does not acquire data.  The 24-byte
record contract is frozen from the 2024-01-02 raw artifact; it is not claimed
to describe other Dukascopy instruments or files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import struct
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from mr_lab.data import (
    Bar,
    DatasetMetadata,
    PriceBasis,
    Timeframe,
    VolumeSemantics,
    resample_bars,
    validate_dataset,
)

PROVIDER = "Dukascopy"
INSTRUMENT = "EURUSD"
PARSER_SCHEMA_VERSION = "dukascopy-bi5-eurusd-m1-bid-v1"
CANONICAL_SCHEMA_VERSION = "bar-v1"
SOURCE_TIMEZONE = "UTC"
RECORD = struct.Struct(">5if")
PRICE_SCALE = 100_000
M1 = Timeframe("1m")


class Bi5ParseError(ValueError):
    """Raised when a BI5 payload violates the verified bounded contract."""


def decode_m1_bid_payload(payload: bytes) -> bytes:
    """LZMA-decompress raw BI5 bytes and require whole non-empty records."""
    if not isinstance(payload, bytes):
        raise Bi5ParseError("payload must be bytes")
    try:
        decoded = lzma.decompress(payload)
    except lzma.LZMAError as error:
        raise Bi5ParseError("payload is not valid LZMA-compressed BI5 data") from error
    if not decoded:
        raise Bi5ParseError("decoded BI5 payload is empty")
    if len(decoded) % RECORD.size:
        raise Bi5ParseError("decoded BI5 payload contains a partial 24-byte record")
    return decoded


def parse_decoded_m1_bid_bars(decoded: bytes, requested_day: date) -> tuple[Bar, ...]:
    """Convert complete verified records to immutable point-in-time bars."""
    if not isinstance(requested_day, date) or isinstance(requested_day, datetime):
        raise Bi5ParseError("requested_day must be a date")
    if not decoded:
        raise Bi5ParseError("decoded BI5 payload is empty")
    if len(decoded) % RECORD.size:
        raise Bi5ParseError("decoded BI5 payload contains a partial 24-byte record")

    day_start = datetime.combine(requested_day, datetime.min.time(), tzinfo=UTC)
    bars: list[Bar] = []
    previous_offset: int | None = None
    for index, values in enumerate(RECORD.iter_unpack(decoded)):
        offset, open_int, close_int, low_int, high_int, volume = values
        if offset < 0 or offset >= 86_400:
            raise Bi5ParseError(f"record {index} offset is outside its UTC day")
        if offset % 60:
            raise Bi5ParseError(f"record {index} offset is not M1-aligned")
        if previous_offset is not None and offset <= previous_offset:
            reason = "duplicate" if offset == previous_offset else "decreasing"
            raise Bi5ParseError(f"record {index} has {reason} offset")
        if not math.isfinite(volume):
            raise Bi5ParseError(f"record {index} volume is not finite")
        if volume < 0:
            raise Bi5ParseError(f"record {index} volume is negative")

        open_price = open_int / PRICE_SCALE
        close_price = close_int / PRICE_SCALE
        low_price = low_int / PRICE_SCALE
        high_price = high_int / PRICE_SCALE
        if not all(
            math.isfinite(value)
            for value in (open_price, close_price, low_price, high_price)
        ):
            raise Bi5ParseError(f"record {index} price is not finite")
        if low_price > min(open_price, close_price) or high_price < max(
            open_price, close_price
        ):
            raise Bi5ParseError(f"record {index} violates OHLC ordering")

        open_time = day_start + timedelta(seconds=offset)
        close_time = open_time + M1.duration
        bars.append(
            Bar(
                instrument=INSTRUMENT,
                timeframe=M1,
                open_time=open_time,
                close_time=close_time,
                available_at=close_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                price_basis=PriceBasis.BID,
                volume=float(volume),
                # JForex IBar documents only "volume of the bar"; it does not
                # establish centralized trades or a precise quote aggregation.
                volume_semantics=VolumeSemantics.UNKNOWN,
            )
        )
        previous_offset = offset
    return tuple(bars)


def parse_m1_bid_bars(payload: bytes, requested_day: date) -> tuple[Bar, ...]:
    """Decode raw BI5 bytes and return verified EURUSD M1 BID bars."""
    return parse_decoded_m1_bid_bars(decode_m1_bid_payload(payload), requested_day)


def build_dataset_metadata(payload: bytes, requested_day: date) -> DatasetMetadata:
    """Build stable canonical metadata whose identity excludes retrieval time."""
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    identity_inputs = {
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "instrument": INSTRUMENT,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "price_basis": PriceBasis.BID.value,
        "provider": PROVIDER,
        "raw_sha256": raw_sha256,
        "requested_day": requested_day.isoformat(),
        "timeframe": M1.value,
    }
    serialized = json.dumps(
        identity_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    dataset_id = f"sha256:{hashlib.sha256(serialized).hexdigest()}"
    return DatasetMetadata(
        source=PROVIDER,
        instrument=INSTRUMENT,
        price_basis=PriceBasis.BID,
        volume_semantics=VolumeSemantics.UNKNOWN,
        schema_version=CANONICAL_SCHEMA_VERSION,
        dataset_id=dataset_id,
        native_timeframe=M1,
        source_timezone=SOURCE_TIMEZONE,
    )


def canonicalization_audit(payload: bytes, requested_day: date) -> dict[str, object]:
    """Validate, resample, and summarize a payload without storing canonical bars."""
    decoded = decode_m1_bid_payload(payload)
    bars = parse_decoded_m1_bid_bars(decoded, requested_day)
    metadata = build_dataset_metadata(payload, requested_day)
    report = validate_dataset(bars, metadata)
    results = {
        name: resample_bars(bars, Timeframe(timeframe))
        for name, timeframe in (("m5", "5m"), ("m15", "15m"), ("h1", "1h"))
    }
    first, final = bars[0], bars[-1]
    return {
        "provider": PROVIDER,
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_byte_length": len(payload),
        "decoded_byte_length": len(decoded),
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "canonical_schema_version": metadata.schema_version,
        "dataset_id": metadata.dataset_id,
        "instrument": metadata.instrument,
        "price_basis": metadata.price_basis.value,
        "volume_semantics": metadata.volume_semantics.value,
        "source_timezone": metadata.source_timezone,
        "requested_day": requested_day.isoformat(),
        "m1_count": len(bars),
        "first_m1_open_time": first.open_time.isoformat(),
        "final_m1_close_time": final.close_time.isoformat(),
        "first_bar_ohlc": [first.open, first.high, first.low, first.close],
        "final_bar_ohlc": [final.open, final.high, final.low, final.close],
        "gap_count": len(report.gaps),
        "resamples": {
            name: {
                "count": len(result.bars),
                "incomplete_window_count": len(result.incomplete_windows),
            }
            for name, result in results.items()
        },
        "minimum_price": min(bar.low for bar in bars),
        "maximum_price": max(bar.high for bar in bars),
        "volume": {
            "minimum": min(bar.volume for bar in bars if bar.volume is not None),
            "maximum": max(bar.volume for bar in bars if bar.volume is not None),
            "sum": sum(bar.volume for bar in bars if bar.volume is not None),
        },
    }


def main() -> None:
    """Create the deterministic Stage 1A-P audit on a GitHub-hosted runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != args.expected_sha256:
        raise SystemExit(
            "raw SHA-256 mismatch: "
            f"expected {args.expected_sha256}, got {actual_sha256}"
        )
    audit = canonicalization_audit(payload, args.date)
    expected = {"m1_count": 1440, "gap_count": 0}
    for key, value in expected.items():
        if audit[key] != value:
            raise SystemExit(f"frozen expectation failed: {key}={audit[key]!r}")
    for name, count in (("m5", 288), ("m15", 96), ("h1", 24)):
        result = audit["resamples"][name]
        if result != {"count": count, "incomplete_window_count": 0}:
            raise SystemExit(f"frozen expectation failed: {name}={result!r}")
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
