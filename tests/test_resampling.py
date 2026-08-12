from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.config import normalize_timeframe
from mr_lab.data import (
    Bar,
    DataContractError,
    DatasetMetadata,
    PriceBasis,
    Timeframe,
    VolumeType,
    available_bars,
    resample_bars,
)

ONE_MINUTE = Timeframe.parse("1m")
FIVE_MINUTES = Timeframe.parse("5m")
START = datetime(2025, 1, 1, 10, tzinfo=UTC)


def source_bar(index: int, **changes: object) -> Bar:
    open_time = START + index * ONE_MINUTE.duration
    values = {
        "instrument": "EURUSD",
        "timeframe": ONE_MINUTE,
        "open_time": open_time,
        "close_time": open_time + ONE_MINUTE.duration,
        "available_at": open_time + ONE_MINUTE.duration,
        "open": 100.0 + index,
        "high": 101.0 + index,
        "low": 99.0 + index,
        "close": 100.5 + index,
        "price_basis": PriceBasis.MID,
        "volume": float(index + 1),
        "volume_type": VolumeType.TICK,
    }
    values.update(changes)
    return Bar(**values)


def metadata() -> DatasetMetadata:
    return DatasetMetadata(
        source="synthetic",
        instrument="EURUSD",
        native_timeframe=ONE_MINUTE,
        price_basis=PriceBasis.MID,
        volume_type=VolumeType.TICK,
        schema_version="1",
        dataset_id="synthetic-a",
    )


def test_one_minute_to_five_minute_ohlcv_aggregation() -> None:
    result = resample_bars(
        [source_bar(index) for index in range(5)], FIVE_MINUTES, metadata()
    )

    assert result.incomplete_windows == ()
    assert len(result.bars) == 1
    output = result.bars[0]
    assert (output.open, output.high, output.low, output.close) == (
        100.0,
        105.0,
        99.0,
        104.5,
    )
    assert output.volume == 15.0
    assert output.open_time == START
    assert output.close_time == START + FIVE_MINUTES.duration


def test_incomplete_window_is_skipped_and_reported() -> None:
    result = resample_bars([source_bar(0), source_bar(1)], FIVE_MINUTES, metadata())

    assert result.bars == ()
    assert len(result.incomplete_windows) == 1
    assert result.incomplete_windows[0].observed_bars == 2
    assert result.incomplete_windows[0].expected_bars == 5


def test_latest_component_availability_propagates() -> None:
    source = [source_bar(index) for index in range(5)]
    delayed = START + timedelta(minutes=6)
    source[-1] = replace(source[-1], available_at=delayed)

    output = resample_bars(source, FIVE_MINUTES, metadata()).bars[0]

    assert output.available_at == delayed
    assert not output.is_available_at(output.close_time)


def test_incompatible_volume_semantics_are_rejected() -> None:
    source = [source_bar(index) for index in range(5)]
    source[2] = replace(source[2], volume_type=VolumeType.TRADED)

    with pytest.raises(DataContractError, match="volume semantics"):
        resample_bars(source, FIVE_MINUTES, metadata())


def test_absent_volume_remains_absent() -> None:
    source = [
        source_bar(index, volume=None, volume_type=VolumeType.NONE)
        for index in range(5)
    ]
    no_volume_metadata = replace(metadata(), volume_type=VolumeType.NONE)

    output = resample_bars(source, FIVE_MINUTES, no_volume_metadata).bars[0]

    assert output.volume is None
    assert output.volume_type is VolumeType.NONE


@pytest.mark.parametrize("target", ["30s", "1m", "90s", "250s"])
def test_target_must_be_larger_integer_multiple(target: str) -> None:
    with pytest.raises(DataContractError, match="integer multiple"):
        resample_bars([], Timeframe.parse(target), metadata())


@pytest.mark.parametrize(
    ("value", "canonical"),
    [("M5", "5m"), ("M15", "15m"), ("H1", "1h"), ("5m", "5m"), ("1h", "1h")],
)
def test_experiment_timeframe_normalization(value: str, canonical: str) -> None:
    assert normalize_timeframe(value) == Timeframe.parse(canonical)


@pytest.mark.parametrize("value", ["m15", "15M", "T5", "M0"])
def test_invalid_experiment_timeframe_is_rejected(value: str) -> None:
    with pytest.raises(DataContractError, match="experiment timeframe"):
        normalize_timeframe(value)


def test_prefix_invariance_for_resampling_and_availability() -> None:
    source = [source_bar(index) for index in range(10)]
    research_time = START + timedelta(minutes=5)

    full_outputs = resample_bars(source, FIVE_MINUTES, metadata()).bars
    full_prefix = available_bars(full_outputs, research_time)
    source_prefix = available_bars(source, research_time)
    truncated_outputs = resample_bars(source_prefix, FIVE_MINUTES, metadata()).bars

    assert truncated_outputs == full_prefix
