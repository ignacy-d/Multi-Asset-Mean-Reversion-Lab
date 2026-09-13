from dataclasses import replace
from datetime import UTC, datetime, time, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.research import Direction
from mr_lab.sessions import SessionSpec, TimeWindow
from mr_lab.trend_exhaustion import (
    DISPLACEMENT_THRESHOLDS,
    TrendExhaustionError,
    TrendExhaustionSpec,
    detect_trend_exhaustion,
)

START = datetime(2024, 1, 2, tzinfo=UTC)


def _bars(mirror: bool = False, start: datetime = START) -> tuple[Bar, ...]:
    closes = [100.0] * 13 + [100 + 0.3 * index for index in range(8)]
    closes += [102.22, 102.25, 102.10, 102.0]
    if mirror:
        closes = [200 - value for value in closes]
    output = []
    previous = closes[0]
    for index, close in enumerate(closes):
        opened = previous
        output.append(
            Bar(
                "EURUSD",
                Timeframe("15m"),
                start + timedelta(minutes=15 * index),
                start + timedelta(minutes=15 * (index + 1)),
                start + timedelta(minutes=15 * (index + 1)),
                opened,
                max(opened, close) + 0.02,
                min(opened, close) - 0.02,
                close,
                PriceBasis.BID,
                float(index),
                VolumeSemantics.QUOTE_ACTIVITY,
            )
        )
        previous = close
    return tuple(output)


def test_event_arithmetic_atr_freeze_and_direction_symmetry() -> None:
    source = _bars()
    event = detect_trend_exhaustion(source, TrendExhaustionSpec(1.5))[0]
    mirrored = detect_trend_exhaustion(_bars(True), TrendExhaustionSpec(1.5))[0]
    assert event.direction is Direction.SHORT
    assert mirrored.direction is Direction.LONG
    assert event.trend_displacement == pytest.approx(2.1)
    assert event.efficiency_ratio == pytest.approx(1.0)
    assert event.exhaustion_extension == pytest.approx(0.17)
    assert event.retained_progress == pytest.approx(-0.1)
    # Changing Phase B extremes changes extension but cannot change frozen ATR.
    changed = list(source)
    changed[-2] = replace(changed[-2], high=changed[-2].high + 5)
    changed_event = detect_trend_exhaustion(changed, TrendExhaustionSpec(1.5))[0]
    assert changed_event.frozen_atr20 == event.frozen_atr20
    assert changed_event.exhaustion_extension > event.exhaustion_extension


def test_prefix_invariance_input_immutability_and_fail_closed_gap() -> None:
    source = _bars()
    snapshot = tuple(source)
    future = tuple(
        replace(
            source[-1],
            open_time=source[-1].close_time + timedelta(minutes=15 * index),
            close_time=source[-1].close_time + timedelta(minutes=15 * (index + 1)),
            available_at=source[-1].close_time + timedelta(minutes=15 * (index + 1)),
        )
        for index in range(3)
    )
    full = detect_trend_exhaustion(source + future, TrendExhaustionSpec(1.5))
    prefix = detect_trend_exhaustion(source, TrendExhaustionSpec(1.5))
    assert full[: len(prefix)] == prefix
    assert source == snapshot
    gapped = (*source[:23], replace(source[24], open_time=source[24].open_time))
    assert detect_trend_exhaustion(gapped, TrendExhaustionSpec(1.5)) == ()


def test_threshold_grid_is_exact_and_episode_is_unique() -> None:
    assert DISPLACEMENT_THRESHOLDS == (1.25, 1.5, 1.75)
    with pytest.raises(TrendExhaustionError):
        TrendExhaustionSpec(1.4)
    # The first eligible window emits only once; later history cannot duplicate it.
    source = _bars()
    assert len(detect_trend_exhaustion(source, TrendExhaustionSpec(1.25))) == 1


@pytest.mark.parametrize(
    ("day", "minutes_from_open", "minutes_to_close"),
    [
        (datetime(2024, 1, 15, 3, 45, tzinfo=UTC), 120, 420),
        (datetime(2024, 7, 15, 3, 45, tzinfo=UTC), 180, 360),
    ],
)
def test_session_diagnostics_use_historical_dst_boundaries(
    day: datetime, minutes_from_open: int, minutes_to_close: int
) -> None:
    event = detect_trend_exhaustion(_bars(start=day), TrendExhaustionSpec(1.5))[0]

    assert event.signal_timestamp == day + timedelta(hours=6, minutes=15)
    assert event.session_label == "london"
    assert event.minutes_from_session_open == minutes_from_open
    assert event.minutes_to_session_close == minutes_to_close
    assert event.session_spec_id is not None


def test_session_diagnostics_never_change_eligibility() -> None:
    source = _bars(start=datetime(2024, 1, 15, 3, 45, tzinfo=UTC))
    london = SessionSpec(
        "diagnostic-only-v1",
        (TimeWindow("custom", "Europe/London", time(8), time(17)),),
    )
    no_sessions = SessionSpec("diagnostic-only-empty-v1", ())
    with_session = detect_trend_exhaustion(source, TrendExhaustionSpec(1.5), london)[0]
    without_session = detect_trend_exhaustion(
        source, TrendExhaustionSpec(1.5), no_sessions
    )[0]

    assert with_session.event_id == without_session.event_id
    assert with_session.direction is without_session.direction
    assert (
        with_session.normalized_displacement == without_session.normalized_displacement
    )
    assert with_session.session_label == "custom"
    assert without_session.session_label is None


def test_detector_rejects_non_m15() -> None:
    with pytest.raises(TrendExhaustionError):
        detect_trend_exhaustion(
            (replace(_bars()[0], timeframe=Timeframe("1m")),),
            TrendExhaustionSpec(1.5),
        )
