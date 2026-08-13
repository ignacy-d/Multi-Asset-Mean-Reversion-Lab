import hashlib
import json
import lzma
import struct
from datetime import UTC, date, datetime

import pytest

from mr_lab.data import (
    Timeframe,
    VolumeSemantics,
    available_bars,
    resample_bars,
    validate_dataset,
)
from mr_lab.providers.dukascopy_bi5 import Bi5ParseError
from mr_lab.providers.dukascopy_multiday import (
    DailyPayload,
    MultiDayAssemblyError,
    assemble_daily_payloads,
)

RECORD = struct.Struct(">5if")
DAY_ONE = date(2024, 1, 2)
DAY_TWO = date(2024, 1, 3)


def payload(offsets: range | tuple[int, ...], *, price_shift: int = 0) -> bytes:
    decoded = b"".join(
        RECORD.pack(
            offset,
            110_000 + price_shift + index,
            110_001 + price_shift + index,
            109_999 + price_shift + index,
            110_002 + price_shift + index,
            0.0 if index == 0 else float(index),
        )
        for index, offset in enumerate(offsets)
    )
    return lzma.compress(decoded, format=lzma.FORMAT_ALONE)


def two_days() -> tuple[DailyPayload, DailyPayload]:
    return (
        DailyPayload(DAY_ONE, payload(tuple(range(23 * 3600, 86_400, 60)))),
        DailyPayload(DAY_TWO, payload(tuple(range(0, 3600, 60)), price_shift=100)),
    )


def test_two_days_are_assembled_chronologically_independent_of_input_order() -> None:
    first, second = two_days()

    forward = assemble_daily_payloads([first, second])
    reverse = assemble_daily_payloads([second, first])

    assert forward == reverse
    assert forward.metadata.dataset_id == reverse.metadata.dataset_id
    assert len(forward.bars) == 120
    assert forward.bars == tuple(sorted(forward.bars, key=lambda bar: bar.open_time))
    assert forward.bars[0].open_time == datetime(2024, 1, 2, 23, 0, tzinfo=UTC)
    assert forward.bars[-1].close_time == datetime(2024, 1, 3, 1, 0, tzinfo=UTC)
    assert validate_dataset(forward.bars, forward.metadata).gaps == ()
    assert all(
        bar.volume_semantics is VolumeSemantics.QUOTE_ACTIVITY for bar in forward.bars
    )
    assert forward.bars[0].volume == 0.0


def test_manifest_has_component_provenance_and_stable_canonical_json() -> None:
    inputs = two_days()
    dataset = assemble_daily_payloads(inputs)
    manifest = dataset.manifest.as_dict()

    assert manifest["component_dates"] == ["2024-01-02", "2024-01-03"]
    assert manifest["component_day_count"] == 2
    assert manifest["m1_bar_count"] == 120
    assert manifest["gap_count"] == 0
    assert manifest["resample_counts"] == {"m5": 24, "m15": 8, "h1": 2}
    assert manifest["incomplete_window_counts"] == {"m5": 0, "m15": 0, "h1": 0}
    assert manifest["volume_semantics"] == "quote_activity"
    assert manifest["source_timezone"] == "UTC"
    assert "retrieved_at" not in manifest
    assert json.loads(dataset.manifest.to_json()) == manifest
    assert (
        dataset.manifest.to_json() == assemble_daily_payloads(inputs).manifest.to_json()
    )
    for component, source in zip(manifest["components"], inputs, strict=True):
        assert component["raw_sha256"] == hashlib.sha256(source.payload).hexdigest()
        assert component["daily_dataset_id"].startswith("sha256:")


def test_valid_raw_payload_change_changes_assembled_identity() -> None:
    first, second = two_days()
    changed = DailyPayload(
        second.requested_day,
        payload(tuple(range(0, 3600, 60)), price_shift=101),
    )

    original = assemble_daily_payloads([first, second])
    modified = assemble_daily_payloads([first, changed])

    assert original.metadata.dataset_id != modified.metadata.dataset_id
    assert (
        original.manifest.components[1].raw_sha256
        != modified.manifest.components[1].raw_sha256
    )
    assert (
        original.manifest.components[1].daily_dataset_id
        != modified.manifest.components[1].daily_dataset_id
    )


def test_duplicate_day_and_malformed_component_fail_explicitly() -> None:
    first, _ = two_days()
    with pytest.raises(MultiDayAssemblyError, match="duplicate requested day"):
        assemble_daily_payloads([first, first])
    with pytest.raises(Bi5ParseError, match="not valid LZMA"):
        assemble_daily_payloads([first, DailyPayload(DAY_TWO, b"malformed")])


def test_missing_calendar_periods_are_gaps_and_are_not_filled() -> None:
    friday = DailyPayload(date(2024, 1, 5), payload((23 * 3600 + 59 * 60,)))
    monday = DailyPayload(date(2024, 1, 8), payload((0,)))

    dataset = assemble_daily_payloads([monday, friday])

    assert len(dataset.bars) == 2
    assert dataset.manifest.gap_count == 1
    assert dataset.manifest.as_dict()["component_dates"] == [
        "2024-01-05",
        "2024-01-08",
    ]
    assert dataset.manifest.as_dict()["resample_counts"] == {
        "m5": 0,
        "m15": 0,
        "h1": 0,
    }
    assert dataset.manifest.as_dict()["incomplete_window_counts"] == {
        "m5": 2,
        "m15": 2,
        "h1": 2,
    }


@pytest.mark.parametrize("target", ["5m", "15m", "1h"])
def test_multiday_resampling_is_prefix_invariant_at_day_boundary(target: str) -> None:
    dataset = assemble_daily_payloads(list(two_days()))
    cutoff = datetime(2024, 1, 3, 0, 30, tzinfo=UTC)

    full = resample_bars(dataset.bars, Timeframe(target))
    prefix = resample_bars(available_bars(dataset.bars, cutoff), Timeframe(target))

    assert prefix.bars == available_bars(full.bars, cutoff)
