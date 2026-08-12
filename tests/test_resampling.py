from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import (
    Bar,
    DataContractError,
    DatasetValidationError,
    PriceBasis,
    Timeframe,
    VolumeSemantics,
    available_bars,
    resample_bars,
)

BASE = datetime(2025, 1, 2, 10, 0, tzinfo=UTC)


def bars(
    count: int,
    *,
    start: datetime = BASE,
    timeframe: str = "5m",
    volume: bool = True,
) -> list[Bar]:
    duration = Timeframe(timeframe).duration
    result = []
    for index in range(count):
        open_time = start + index * duration
        result.append(
            Bar(
                instrument="EURUSD",
                timeframe=Timeframe(timeframe),
                open_time=open_time,
                close_time=open_time + duration,
                available_at=open_time + duration,
                open=100.0 + index,
                high=102.0 + index,
                low=99.0 + index,
                close=101.0 + index,
                price_basis=PriceBasis.TRADE,
                volume=float(index + 1) if volume else None,
                volume_semantics=(
                    VolumeSemantics.TRADED if volume else VolumeSemantics.NONE
                ),
            )
        )
    return result


def test_complete_window_aggregates_ohlcv_and_epoch_aligned_utc_times() -> None:
    result = resample_bars(bars(3), Timeframe("15m"))

    assert result.incomplete_windows == ()
    assert len(result.bars) == 1
    output = result.bars[0]
    assert (output.open_time, output.close_time) == (
        BASE,
        BASE + timedelta(minutes=15),
    )
    assert (output.open, output.high, output.low, output.close) == (
        100.0,
        104.0,
        99.0,
        103.0,
    )
    assert output.volume == 6.0
    assert output.volume_semantics is VolumeSemantics.TRADED


@pytest.mark.parametrize(
    ("source", "target", "count"),
    [("1m", "5m", 5), ("1m", "15m", 15), ("5m", "15m", 3), ("15m", "1h", 4)],
)
def test_supported_fixed_duration_ratios_resample(
    source: str, target: str, count: int
) -> None:
    result = resample_bars(bars(count, timeframe=source), Timeframe(target))

    assert len(result.bars) == 1
    assert result.bars[0].timeframe == Timeframe(target)
    assert result.incomplete_windows == ()


def test_absent_volume_is_preserved() -> None:
    output = resample_bars(bars(3, volume=False), Timeframe("15m")).bars[0]

    assert output.volume is None
    assert output.volume_semantics is VolumeSemantics.NONE


def test_mixed_volume_or_price_semantics_are_rejected_not_reinterpreted() -> None:
    source = bars(3)
    source[1] = Bar(
        **{
            **{
                field: getattr(source[1], field)
                for field in source[1].__dataclass_fields__
            },
            "volume_semantics": VolumeSemantics.TICK,
        }
    )
    with pytest.raises(DatasetValidationError, match="homogeneous"):
        resample_bars(source, Timeframe("15m"))


@pytest.mark.parametrize("target", ["1m", "5m", "7m"])
def test_invalid_target_source_relationship_is_rejected(target: str) -> None:
    with pytest.raises(DataContractError, match="larger|integer"):
        resample_bars(bars(3), Timeframe(target))


def test_source_bar_may_not_cross_epoch_aligned_target_boundary() -> None:
    source = bars(1, start=BASE + timedelta(minutes=13), timeframe="5m")
    with pytest.raises(DataContractError, match="crosses"):
        resample_bars(source, Timeframe("15m"))


def test_incomplete_windows_are_skipped_and_reported() -> None:
    result = resample_bars(bars(2), Timeframe("15m"))

    assert result.bars == ()
    assert len(result.incomplete_windows) == 1
    assert result.incomplete_windows[0].expected_components == 3
    assert result.incomplete_windows[0].observed_components == 2


def test_fully_absent_target_windows_across_large_gap_are_not_synthesized() -> None:
    friday = datetime(2025, 1, 3, 21, 0, tzinfo=UTC)
    monday = datetime(2025, 1, 6, 9, 0, tzinfo=UTC)
    source = bars(3, start=friday) + bars(3, start=monday)

    result = resample_bars(source, Timeframe("15m"))

    assert [bar.open_time for bar in result.bars] == [friday, monday]
    assert result.incomplete_windows == ()


def test_delayed_component_propagates_output_availability() -> None:
    source = bars(3)
    last = source[-1]
    source[-1] = Bar(
        **{
            field: (
                BASE + timedelta(minutes=16)
                if field == "available_at"
                else getattr(last, field)
            )
            for field in last.__dataclass_fields__
        }
    )
    output = resample_bars(source, Timeframe("15m")).bars[0]

    assert output.available_at == BASE + timedelta(minutes=16)
    assert available_bars([output], BASE + timedelta(minutes=15)) == ()


def test_resampling_is_prefix_invariant_by_explicit_availability() -> None:
    source = bars(6)
    research_time = BASE + timedelta(minutes=15)
    full = resample_bars(source, Timeframe("15m"))
    full_available = available_bars(full.bars, research_time)

    prefix_source = available_bars(source, research_time)
    prefix = resample_bars(prefix_source, Timeframe("15m"))

    assert prefix.bars == full_available
