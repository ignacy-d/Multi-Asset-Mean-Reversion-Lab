from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.pca_stage0_runner import (
    INSTRUMENTS,
    REGISTRY,
    Stage0Error,
    build_panel,
    concentration,
    event_id,
    is_padding,
    load_registry,
    month_block_bootstrap,
)


def bar(name, minute, close, *, open_=None, volume=1.0):
    start = datetime(2024, 1, 2, tzinfo=UTC) + timedelta(minutes=minute)
    open_ = close if open_ is None else open_
    return Bar(
        name,
        Timeframe("1m"),
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=1),
        open_,
        max(open_, close),
        min(open_, close),
        close,
        PriceBasis.BID,
        volume,
        VolumeSemantics.QUOTE_ACTIVITY,
    )


def panel_series(*, gap_name=None, padding_name=None):
    result = {}
    for j, name in enumerate(INSTRUMENTS):
        bars = [
            bar(
                name,
                i,
                100 + j + i * (0.01 + j / 10000),
                open_=100 + j + max(0, i - 1) * (0.01 + j / 10000),
            )
            for i in range(7)
        ]
        if name == gap_name:
            bars.pop(3)
        if name == padding_name:
            bars[3] = bar(name, 3, 101, volume=0)
        result[name] = bars
    return result


def test_only_canonical_registry_path_is_accepted():
    with pytest.raises(Stage0Error, match="explicit 2024 registry"):
        load_registry(Path("anything.json"))
    assert set(load_registry(REGISTRY)["instruments"]) == set(INSTRUMENTS)


def test_padding_requires_both_zero_activity_and_flat_ohlc():
    assert is_padding(bar("EURUSD", 0, 1, volume=0))
    assert not is_padding(bar("EURUSD", 0, 1.1, open_=1, volume=0))
    assert not is_padding(bar("EURUSD", 0, 1, volume=1))


def test_exact_intersection_does_not_bridge_gap_or_padding_or_forward_fill():
    datasets = {n: "id-" + n for n in INSTRUMENTS}
    complete = build_panel(panel_series(), datasets)
    gap = build_panel(panel_series(gap_name="EURUSD"), datasets)
    padded = build_panel(panel_series(padding_name="GBPUSD"), datasets)
    assert len(complete.panel.timestamps) == 7
    # Missing/padded minutes invalidate adjacent timestamps.
    assert len(gap.panel.timestamps) == 5
    assert len(padded.panel.timestamps) == 5
    assert gap.counts["EURUSD"]["invalid_or_gap_returns"] == 1
    assert padded.counts["GBPUSD"]["padding"] == 1


def test_incomplete_universe_fails_closed():
    series = panel_series()
    series.pop("EURGBP")
    with pytest.raises(Stage0Error, match="nine"):
        build_panel(series, {n: "x" for n in INSTRUMENTS})


def test_event_ids_and_bootstrap_are_deterministic():
    stamp = datetime(2024, 1, 1, tzinfo=UTC)
    assert event_id(stamp, "EURUSD", "panel") == event_id(stamp, "EURUSD", "panel")
    events = [
        {
            "timestamp": "2024-01-02T00:00:00+00:00",
            "instrument": "EURUSD",
            "h15_signed_bps_return": 1.0,
        },
        {
            "timestamp": "2024-02-02T00:00:00+00:00",
            "instrument": "GBPUSD",
            "h15_signed_bps_return": -0.5,
        },
    ]
    assert (
        month_block_bootstrap(events, replicates=20, seed=7)
        == month_block_bootstrap(events, replicates=20, seed=7)
    ).all()


def test_concentration_arithmetic():
    events = [
        {"timestamp": "2024-01-01", "instrument": "A", "h15_signed_bps_return": 2},
        {"timestamp": "2024-02-01", "instrument": "A", "h15_signed_bps_return": 0},
        {"timestamp": "2024-04-01", "instrument": "B", "h15_signed_bps_return": -1},
        {"timestamp": "2024-07-01", "instrument": "C", "h15_signed_bps_return": 1},
    ]
    assert concentration(events) == (0.5, 0.5, 2)
