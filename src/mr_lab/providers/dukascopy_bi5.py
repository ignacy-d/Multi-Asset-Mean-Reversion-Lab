"""Parse production-verified or explicitly verification-only BI5 candidates.

The production entry points fail closed for unverified instruments. A separate
bounded audit entry point may exercise candidate scales against real samples.
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
from mr_lab.providers.instruments import (
    ProviderInstrumentSpec,
    get_candidate_instrument_spec,
    get_instrument_spec,
)

PROVIDER = "Dukascopy"
INSTRUMENT = "EURUSD"
PARSER_SCHEMA_VERSION = "dukascopy-bi5-eurusd-m1-bid-v1"
CANONICAL_SCHEMA_VERSION = "bar-v1"
SOURCE_TIMEZONE = "UTC"
VOLUME_SEMANTICS = VolumeSemantics.QUOTE_ACTIVITY
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


def parse_decoded_m1_bid_bars(
    decoded: bytes, requested_day: date, instrument: str = INSTRUMENT
) -> tuple[Bar, ...]:
    """Convert records using a production-verified instrument spec."""
    return _parse_decoded_m1_bid_bars(
        decoded, requested_day, get_instrument_spec(instrument), PriceBasis.BID
    )


def parse_decoded_m1_ask_bars(
    decoded: bytes, requested_day: date, instrument: str = INSTRUMENT
) -> tuple[Bar, ...]:
    """Convert a native Dukascopy ASK candle payload without side inference."""
    return _parse_decoded_m1_bid_bars(
        decoded, requested_day, get_instrument_spec(instrument), PriceBasis.ASK
    )


def _parse_decoded_m1_bid_bars(
    decoded: bytes,
    requested_day: date,
    spec: ProviderInstrumentSpec,
    price_basis: PriceBasis = PriceBasis.BID,
) -> tuple[Bar, ...]:
    if price_basis not in (PriceBasis.BID, PriceBasis.ASK):
        raise Bi5ParseError("Dukascopy candle side must be BID or ASK")
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

        open_price = spec.decode_price(open_int)
        close_price = spec.decode_price(close_int)
        low_price = spec.decode_price(low_int)
        high_price = spec.decode_price(high_int)
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
                instrument=spec.instrument,
                timeframe=M1,
                open_time=open_time,
                close_time=close_time,
                available_at=close_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                price_basis=price_basis,
                volume=float(volume),
                # JForex IBar defines volume as the sum of best-price volumes
                # for each tick, which is quote activity rather than executed
                # centralized FX trade volume.
                volume_semantics=VOLUME_SEMANTICS,
            )
        )
        previous_offset = offset
    return tuple(bars)


def parse_m1_bid_bars(
    payload: bytes, requested_day: date, instrument: str = INSTRUMENT
) -> tuple[Bar, ...]:
    """Decode raw BI5 bytes and return verified M1 BID bars."""
    return parse_decoded_m1_bid_bars(
        decode_m1_bid_payload(payload), requested_day, instrument
    )


def parse_m1_ask_bars(
    payload: bytes, requested_day: date, instrument: str = INSTRUMENT
) -> tuple[Bar, ...]:
    """Decode a raw native ASK BI5 payload into explicitly ASK-based bars."""
    return parse_decoded_m1_ask_bars(
        decode_m1_bid_payload(payload), requested_day, instrument
    )


def build_side_dataset_metadata(
    payload: bytes,
    requested_day: date,
    instrument: str,
    side: PriceBasis,
) -> DatasetMetadata:
    """Build V2 side-aware provenance while preserving the legacy BID identity."""
    if side is PriceBasis.BID:
        return build_dataset_metadata(payload, requested_day, instrument)
    if side is not PriceBasis.ASK:
        raise Bi5ParseError("side metadata supports BID or ASK")
    spec = get_instrument_spec(instrument)
    inputs = {
        "canonical_schema_version": "bar-v2-side-aware",
        "instrument_spec": spec.as_dict(),
        "price_basis": side.value,
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "requested_day": requested_day.isoformat(),
        "source_timezone": SOURCE_TIMEZONE,
        "timeframe": M1.value,
        "volume_semantics": VOLUME_SEMANTICS.value,
    }
    serialized = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return DatasetMetadata(
        source=spec.provider,
        instrument=spec.instrument,
        price_basis=side,
        volume_semantics=VOLUME_SEMANTICS,
        schema_version="bar-v2-side-aware",
        dataset_id=f"sha256:{hashlib.sha256(serialized).hexdigest()}",
        native_timeframe=M1,
        source_timezone=SOURCE_TIMEZONE,
    )


def build_dataset_metadata(
    payload: bytes, requested_day: date, instrument: str = INSTRUMENT
) -> DatasetMetadata:
    """Build stable canonical metadata whose identity excludes retrieval time."""
    return _build_dataset_metadata(
        payload, requested_day, get_instrument_spec(instrument)
    )


def _build_dataset_metadata(
    payload: bytes, requested_day: date, spec: ProviderInstrumentSpec
) -> DatasetMetadata:
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    identity_inputs = {
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "instrument": spec.instrument,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "price_basis": PriceBasis.BID.value,
        "provider": PROVIDER,
        "raw_sha256": raw_sha256,
        "requested_day": requested_day.isoformat(),
        "source_timezone": SOURCE_TIMEZONE,
        "timeframe": M1.value,
        "volume_semantics": VOLUME_SEMANTICS.value,
    }
    if spec.instrument != INSTRUMENT:
        identity_inputs["instrument_spec"] = spec.as_dict()
    serialized = json.dumps(
        identity_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    dataset_id = f"sha256:{hashlib.sha256(serialized).hexdigest()}"
    return DatasetMetadata(
        source=spec.provider,
        instrument=spec.instrument,
        price_basis=spec.price_basis,
        volume_semantics=VOLUME_SEMANTICS,
        schema_version=CANONICAL_SCHEMA_VERSION,
        dataset_id=dataset_id,
        native_timeframe=M1,
        source_timezone=SOURCE_TIMEZONE,
    )


def canonicalization_audit(
    payload: bytes, requested_day: date, instrument: str = INSTRUMENT
) -> dict[str, object]:
    """Validate, resample, and summarize a payload without storing canonical bars."""
    return _canonicalization_audit(
        payload, requested_day, get_instrument_spec(instrument)
    )


def candidate_canonicalization_audit(
    payload: bytes, requested_day: date, instrument: str
) -> dict[str, object]:
    """Audit one bounded sample using declared, possibly unverified facts."""
    return _canonicalization_audit(
        payload, requested_day, get_candidate_instrument_spec(instrument)
    )


def _canonicalization_audit(
    payload: bytes, requested_day: date, spec: ProviderInstrumentSpec
) -> dict[str, object]:
    decoded = decode_m1_bid_payload(payload)
    bars = _parse_decoded_m1_bid_bars(decoded, requested_day, spec)
    metadata = _build_dataset_metadata(payload, requested_day, spec)
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
        "price_scale": spec.price_scale,
        "price_precision": spec.price_precision,
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
    parser.add_argument("--instrument", default="EURUSD")
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
    audit = canonicalization_audit(payload, args.date, args.instrument)
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
