from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.research import Direction
from mr_lab.trend_exhaustion import (
    DISPLACEMENT_THRESHOLDS,
    TrendExhaustionError,
    TrendExhaustionSpec,
    detect_trend_exhaustion,
)

START = datetime(2024, 1, 2, tzinfo=UTC)


def _bars(mirror: bool = False) -> tuple[Bar, ...]:
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
                START + timedelta(minutes=15 * index),
                START + timedelta(minutes=15 * (index + 1)),
                START + timedelta(minutes=15 * (index + 1)),
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


def test_detector_rejects_non_m15() -> None:
    with pytest.raises(TrendExhaustionError):
        detect_trend_exhaustion(
            (replace(_bars()[0], timeframe=Timeframe("1m")),),
            TrendExhaustionSpec(1.5),
        )
