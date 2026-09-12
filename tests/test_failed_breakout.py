from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.failed_breakout import (
    LOWER,
    UPPER,
    FailedBreakoutSpec,
    StructuralLevel,
    detect_failed_breakouts,
    generate_structural_levels,
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


def _history(start, minutes):
    bars = []
    for index in range(minutes):
        opened = start + timedelta(minutes=index)
        base = 1.10 + (index % 37) / 100_000
        bars.append(
            Bar(
                instrument="EURUSD",
                timeframe=Timeframe("1m"),
                open_time=opened,
                close_time=opened + timedelta(minutes=1),
                available_at=opened + timedelta(minutes=1),
                open=base,
                high=base + 0.0002,
                low=base - 0.0002,
                close=base,
                price_basis=PriceBasis.BID,
                volume=1.0,
                volume_semantics=VolumeSemantics.TICK,
            )
        )
    return tuple(bars)


def test_structural_levels_are_causal_complete_and_use_one_pre_event_scale():
    bars = _history(datetime(2024, 1, 2, tzinfo=UTC), 2 * 24 * 60)
    levels = generate_structural_levels(bars)
    day = [level for level in levels if level.level_id.endswith("2024-01-03")]

    assert {level.anchor_family for level in day} == {
        "previous-day",
        "asia-session",
        "london-or60",
    }
    assert len(day) == 6
    assert len({level.scale for level in day}) == 1
    assert all(level.available_at < level.expires_at for level in day)
    assert all(level.available_at >= datetime(2024, 1, 3, tzinfo=UTC) for level in day)


@pytest.mark.parametrize(("month", "day", "expected_hour"), [(1, 2, 9), (7, 2, 8)])
def test_london_or60_uses_historical_dst(month, day, expected_hour):
    start = datetime(2024, month, day - 1, tzinfo=UTC)
    levels = generate_structural_levels(_history(start, 2 * 24 * 60))
    london = next(
        level
        for level in levels
        if level.anchor_family == "london-or60" and level.side == UPPER
    )

    assert london.available_at.hour == expected_hour


def test_missing_anchor_observation_fails_closed_without_mutating_input():
    bars = _history(datetime(2024, 1, 2, tzinfo=UTC), 2 * 24 * 60)
    missing_at = datetime(2024, 1, 3, 0, 30, tzinfo=UTC)
    incomplete = tuple(bar for bar in bars if bar.open_time != missing_at)
    before = tuple(incomplete)

    levels = generate_structural_levels(incomplete)

    assert not any(
        level.anchor_family == "asia-session" and level.level_id.endswith("2024-01-03")
        for level in levels
    )
    assert incomplete == before


def test_structural_level_generation_is_prefix_invariant():
    bars = _history(datetime(2024, 1, 2, tzinfo=UTC), 3 * 24 * 60)
    cutoff = datetime(2024, 1, 3, 10, tzinfo=UTC)
    prefix = tuple(bar for bar in bars if bar.available_at <= cutoff)

    prefix_levels = generate_structural_levels(prefix)
    full_levels = generate_structural_levels(bars)

    assert prefix_levels == tuple(
        level for level in full_levels if level.available_at <= cutoff
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
    event = detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.05))[0]
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
    event = detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.05))[0]
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
    assert detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.05)) == ()


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
    assert (
        detect_failed_breakouts(
            bars, (level,), FailedBreakoutSpec(0.05, reclaim_window_minutes=30)
        )
        == ()
    )


def test_reclaim_exactly_at_30_minute_deadline_is_included():
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
        [_bar(0, close=1.0995), _bar(1, close=1.1010, high=1.1015)]
        + [_bar(minute, close=1.1005) for minute in range(2, 31)]
        + [_bar(31, close=1.0998)]
    )

    event = detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.05))[0]

    assert event.minutes_to_reclaim == 30


def test_level_rearms_after_completed_reclaim_without_duplicate_event():
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
        _bar(1, close=1.1010, high=1.1015),
        _bar(2, close=1.0998),
        _bar(3, close=1.1010, high=1.1015),
        _bar(4, close=1.0998),
    )

    events = detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.05))

    assert len(events) == 2
    assert len({event.candidate_event_id for event in events}) == 2


def test_detector_spec_rejects_non_preregistered_grid_values():
    with pytest.raises(ValueError, match="preregistered grid"):
        FailedBreakoutSpec(0.10)
    with pytest.raises(ValueError, match="frozen at 30"):
        FailedBreakoutSpec(0.05, reclaim_window_minutes=15)


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
    assert detect_failed_breakouts(bars, (level,), FailedBreakoutSpec(0.05)) == ()


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
    complete = (
        *prefix,
        _bar(3, close=1.1001, high=1.1003, low=1.0995),
        _bar(4, close=1.0995),
    )
    spec = FailedBreakoutSpec(0.05)
    prefix_events = detect_failed_breakouts(prefix, (level,), spec)
    complete_events = detect_failed_breakouts(complete, (level,), spec)
    assert prefix_events == complete_events[: len(prefix_events)]
