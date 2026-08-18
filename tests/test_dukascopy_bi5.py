import hashlib
import json
import lzma
import math
import struct
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from mr_lab.data import PriceBasis, Timeframe, VolumeSemantics, resample_bars
from mr_lab.providers.dukascopy_bi5 import (
    Bi5ParseError,
    build_dataset_metadata,
    canonicalization_audit,
    decode_m1_bid_payload,
    parse_decoded_m1_bid_bars,
    parse_m1_bid_bars,
)

DAY = date(2024, 1, 2)
RECORD = struct.Struct(">5if")


def compressed(decoded: bytes) -> bytes:
    return lzma.compress(decoded, format=lzma.FORMAT_ALONE)


def record(
    offset: int = 0,
    open_: int = 110_366,
    close: int = 110_374,
    low: int = 110_366,
    high: int = 110_376,
    volume: float = 102.37,
) -> bytes:
    return RECORD.pack(offset, open_, close, low, high, volume)


def fixture_records() -> bytes:
    path = Path(__file__).parent / "fixtures/dukascopy_eurusd_2024-01-02_records.hex"
    lines = (line for line in path.read_text().splitlines() if not line.startswith("#"))
    return bytes.fromhex("".join(lines))


def test_real_record_fixture_freezes_field_order_endianness_and_timestamps() -> None:
    bars = parse_m1_bid_bars(compressed(fixture_records()), DAY)

    assert len(bars) == 4
    assert (bars[0].open, bars[0].close, bars[0].low, bars[0].high) == (
        1.10366,
        1.10374,
        1.10366,
        1.10376,
    )
    assert bars[0].volume == pytest.approx(102.37000274658203)
    assert (bars[1].open, bars[1].close, bars[1].low, bars[1].high) == (
        1.10371,
        1.10375,
        1.10368,
        1.10375,
    )
    assert bars[2].volume == pytest.approx(29.420000076293945)
    assert bars[-1].open_time == datetime(2024, 1, 2, 23, 59, tzinfo=UTC)
    assert bars[-1].close_time == datetime(2024, 1, 3, tzinfo=UTC)
    assert (bars[-1].open, bars[-1].close, bars[-1].low, bars[-1].high) == (
        1.09411,
        1.09416,
        1.09411,
        1.09419,
    )


def test_canonical_semantics_and_zero_volume_are_preserved() -> None:
    bar = parse_m1_bid_bars(compressed(record(volume=0.0)), DAY)[0]

    assert bar.instrument == "EURUSD"
    assert bar.timeframe == Timeframe("1m")
    assert bar.price_basis is PriceBasis.BID
    assert bar.volume_semantics is VolumeSemantics.QUOTE_ACTIVITY
    assert bar.volume == 0.0
    assert bar.open_time == datetime(2024, 1, 2, tzinfo=UTC)
    assert bar.available_at == bar.close_time == datetime(2024, 1, 2, 0, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"corrupt", "not valid LZMA"),
        (compressed(b""), "empty"),
        (compressed(record() + b"x"), "partial 24-byte"),
    ],
)
def test_malformed_payloads_are_rejected(payload: bytes, message: str) -> None:
    with pytest.raises(Bi5ParseError, match=message):
        decode_m1_bid_payload(payload)


@pytest.mark.parametrize(
    ("decoded", "message"),
    [
        (record(offset=1), "not M1-aligned"),
        (record(offset=-60), "outside"),
        (record(offset=86_400), "outside"),
        (record() + record(), "duplicate"),
        (record(60) + record(0), "decreasing"),
        (record(open_=110_400, high=110_399), "OHLC"),
        (record(volume=math.inf), "not finite"),
    ],
)
def test_invalid_records_are_rejected(decoded: bytes, message: str) -> None:
    with pytest.raises(Bi5ParseError, match=message):
        parse_decoded_m1_bid_bars(decoded, DAY)


def test_price_scaling_and_dataset_identity_are_deterministic() -> None:
    payload = compressed(record())
    first = build_dataset_metadata(payload, DAY)
    second = build_dataset_metadata(payload, DAY)
    identity_inputs = {
        "canonical_schema_version": "bar-v1",
        "instrument": "EURUSD",
        "parser_schema_version": "dukascopy-bi5-eurusd-m1-bid-v1",
        "price_basis": "bid",
        "provider": "Dukascopy",
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "requested_day": "2024-01-02",
        "source_timezone": "UTC",
        "timeframe": "1m",
        "volume_semantics": "quote_activity",
    }
    identity_json = json.dumps(
        identity_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    expected_dataset_id = f"sha256:{hashlib.sha256(identity_json).hexdigest()}"

    assert parse_m1_bid_bars(payload, DAY)[0].open == 1.10366
    assert first == second
    assert first.dataset_id == expected_dataset_id
    assert first.dataset_id == (
        "sha256:72d50f6f54bc6e0f015754a4e3918d847f39ad44b88929a69424dd4ea4ed5285"
    )
    assert (
        first.dataset_id
        != build_dataset_metadata(compressed(record(60)), DAY).dataset_id
    )
    assert first.source == "Dukascopy"
    assert first.source_timezone == "UTC"
    assert first.native_timeframe == Timeframe("1m")
    assert first.volume_semantics is VolumeSemantics.QUOTE_ACTIVITY


def test_parsing_is_prefix_invariant() -> None:
    decoded = b"".join(record(offset=index * 60) for index in range(5))

    prefix = parse_decoded_m1_bid_bars(decoded[: 3 * RECORD.size], DAY)
    complete = parse_decoded_m1_bid_bars(decoded, DAY)

    assert prefix == complete[:3]


def test_full_day_validates_and_uses_generic_resampling() -> None:
    decoded = b"".join(
        record(
            offset=index * 60,
            open_=100_000 + index,
            close=100_001 + index,
            low=99_999 + index,
            high=100_002 + index,
            volume=float(index % 7),
        )
        for index in range(1440)
    )
    payload = compressed(decoded)
    bars = parse_m1_bid_bars(payload, DAY)
    audit = canonicalization_audit(payload, DAY)

    assert audit["m1_count"] == 1440
    assert audit["gap_count"] == 0
    assert audit["resamples"] == {
        "m5": {"count": 288, "incomplete_window_count": 0},
        "m15": {"count": 96, "incomplete_window_count": 0},
        "h1": {"count": 24, "incomplete_window_count": 0},
    }
    first_m5 = resample_bars(bars, Timeframe("5m")).bars[0]
    assert first_m5.open == bars[0].open
    assert first_m5.close == bars[4].close
    assert first_m5.high == max(bar.high for bar in bars[:5])
    assert first_m5.low == min(bar.low for bar in bars[:5])
    assert first_m5.volume == sum(bar.volume for bar in bars[:5])
    assert first_m5.available_at == bars[4].close_time
    assert bars[-1].close_time == datetime(2024, 1, 3, tzinfo=UTC)


def test_audit_is_stable_json_and_contains_content_hash() -> None:
    payload = compressed(record())
    audit = canonicalization_audit(payload, DAY)

    assert audit == json.loads(json.dumps(audit, sort_keys=True))
    assert audit["raw_sha256"] == hashlib.sha256(payload).hexdigest()
    assert audit["volume_semantics"] == "quote_activity"
    assert audit["source_timezone"] == "UTC"
    assert "retrieved_at" not in audit
