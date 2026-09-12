from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.failed_breakout import (
    LOWER,
    UPPER,
    FailedBreakoutSpec,
    StructuralLevel,
    detect_failed_breakouts,
)
from mr_lab.research import Direction


def _bar(minute, *, close, high=None, low=None):
    start = datetime(2024, 1, 2, 8, minute, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return Bar(
        instrument="EURUSD",
        timeframe=Timeframe("1m"),
        open_time=start,
        close_time=end,
        available_at=end,
        open=close,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        price_basis=PriceBasis.BID,
        volume=1.0,
        volume_semantics=VolumeSemantics.TICK,
    )


def test_upper_breakout_reclaim_emits_short_event():
    level = StructuralLevel(
        "EURUSD",
        "pdh-2024-01-02",
        "pdh-pdl",
        UPPER,
        1.1000,
        0.0100,
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
    )
    bars = (
        _bar(0, close=1.0995),
        _bar(1, close=1.1010, high=1.1015, low=1.0998),
        _bar(2, close=1.1008, high=1.1012, low=1.1002),
        _bar(3, close=1.0997, high=1.1009, low=1.0995),
    )
    event = detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.10))[0]
    assert event.direction is Direction.SHORT
    assert event.minutes_to_reclaim == 2
    assert event.outside_close_count == 2
    assert event.max_depth_fraction == pytest.approx(0.15)


def test_lower_breakout_reclaim_emits_long_event():
    level = StructuralLevel(
        "EURUSD",
        "pdl-2024-01-02",
        "pdh-pdl",
        LOWER,
        1.1000,
        0.0100,
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
    )
    bars = (
        _bar(0, close=1.1005),
        _bar(1, close=1.0990, high=1.1002, low=1.0985),
        _bar(2, close=1.1003, high=1.1005, low=1.0989),
    )
    event = detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.10))[0]
    assert event.direction is Direction.LONG
    assert event.minutes_to_reclaim == 1


def test_insufficient_breakout_depth_does_not_emit():
    level = StructuralLevel(
        "EURUSD",
        "pdh",
        "pdh-pdl",
        UPPER,
        1.1000,
        0.0100,
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
    )
    bars = (
        _bar(0, close=1.0995),
        _bar(1, close=1.1002, high=1.1004, low=1.0999),
        _bar(2, close=1.0999),
    )
    assert detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.10)) == ()


def test_reclaim_after_window_is_rejected():
    level = StructuralLevel(
        "EURUSD",
        "pdh",
        "pdh-pdl",
        UPPER,
        1.1000,
        0.0100,
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
    )
    bars = tuple(
        [_bar(0, close=1.0995), _bar(1, close=1.1011, high=1.1015, low=1.1001)]
        + [_bar(m, close=1.1008, high=1.1010, low=1.1002) for m in range(2, 33)]
        + [_bar(33, close=1.0998)]
    )
    assert detect_failed_breakouts(
        bars, (level,), FailedBreakoutSpec(0.10, reclaim_window_minutes=30)
    ) == ()


def test_gap_cancels_episode_fail_closed():
    level = StructuralLevel(
        "EURUSD",
        "pdh",
        "pdh-pdl",
        UPPER,
        1.1000,
        0.0100,
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
    )
    bars = (
        _bar(0, close=1.0995),
        _bar(1, close=1.1011, high=1.1015, low=1.1001),
        _bar(3, close=1.0998),
    )
    assert detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.10)) == ()


def test_future_bars_do_not_change_prefix_events():
    level = StructuralLevel(
        "EURUSD",
        "pdh",
        "pdh-pdl",
        UPPER,
        1.1000,
        0.0100,
        datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
    )
    prefix = (
        _bar(0, close=1.0995),
        _bar(1, close=1.1010, high=1.1015, low=1.0998),
        _bar(2, close=1.0997),
    )
    complete = prefix + (
        _bar(3, close=1.1001, high=1.1003, low=1.0995),
        _bar(4, close=1.0995),
    )
    spec = FailedBreakoutSpec(0.10)
    prefix_events = detect_failed_breakouts(prefix, (level,), spec)
    complete_events = detect_failed_breakouts(complete, (level,), spec)
    assert prefix_events == complete_events[: len(prefix_events)]
