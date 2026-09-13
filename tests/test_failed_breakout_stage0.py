import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.failed_breakout import (
    FAILED_BREAKOUT_FAMILY,
    FailedBreakoutEvent,
    FailedBreakoutSpec,
)
from mr_lab.failed_breakout_stage0 import (
    FailedBreakoutObservation,
    ModuleAReference,
    aggregate_observations,
    calculate_overlap,
    write_outputs,
)
from mr_lab.research import Direction
from mr_lab.stage4a import ForwardOutcome, diagnose_directional_path

T = datetime(2024, 1, 31, 23, tzinfo=UTC)


def _bar(timestamp, close, *, instrument="EURUSD", high=None, low=None):
    return Bar(
        instrument,
        Timeframe("1m"),
        timestamp - timedelta(minutes=1),
        timestamp,
        timestamp,
        close,
        close if high is None else high,
        close if low is None else low,
        close,
        PriceBasis.BID,
        1.0,
        VolumeSemantics.TICK,
    )


def _path(direction=Direction.LONG, *, missing=(), instrument="EURUSD", start=T):
    sign = int(direction)
    return tuple(
        _bar(
            start + timedelta(minutes=minute),
            100 + sign * minute / 100,
            instrument=instrument,
            high=100 + sign * minute / 100 + 0.02,
            low=100 + sign * minute / 100 - 0.02,
        )
        for minute in range(1, 121)
        if minute not in missing
    )


def _event(
    *,
    timestamp=T,
    direction=Direction.LONG,
    instrument="EURUSD",
    anchor="previous-day",
    suffix="a",
):
    spec = FailedBreakoutSpec(0.05)
    return FailedBreakoutEvent(
        f"event-{suffix}",
        FAILED_BREAKOUT_FAMILY,
        spec.spec_id,
        instrument,
        timestamp,
        direction,
        anchor,
        f"level-{suffix}",
        100.0,
        1.0,
        timestamp - timedelta(minutes=2),
        timestamp,
        0.05,
        0.10,
        1,
        2,
        0.02,
    )


def _observation(**kwargs):
    event = _event(**kwargs)
    path = diagnose_directional_path(
        instrument=event.instrument,
        signal_timestamp=event.signal_timestamp,
        direction=event.direction,
        p0=100.0,
        m1_bars=_path(
            event.direction,
            instrument=event.instrument,
            start=event.signal_timestamp,
        ),
    )
    return FailedBreakoutObservation(event, 0.05, path)


@pytest.mark.parametrize("direction", (Direction.LONG, Direction.SHORT))
def test_exact_clock_signed_outcomes_and_mfe_mae(direction):
    result = diagnose_directional_path(
        instrument="EURUSD",
        signal_timestamp=T,
        direction=direction,
        p0=100.0,
        m1_bars=_path(direction),
    )

    assert [item.horizon_minutes for item in result.horizons] == [15, 30, 60, 120]
    assert [item.signed_price_movement for item in result.horizons] == pytest.approx(
        [0.15, 0.30, 0.60, 1.20]
    )
    assert result.mfe_price == pytest.approx(1.22)
    assert result.mae_price == pytest.approx(0.01)
    assert result.time_to_mfe_minutes == 120
    assert result.time_to_mae_minutes == 1


def test_missing_exact_clock_is_not_filled_and_path_metrics_fail_closed():
    result = diagnose_directional_path(
        instrument="EURUSD",
        signal_timestamp=T,
        direction=Direction.LONG,
        p0=100.0,
        m1_bars=_path(missing=(30,)),
    )

    assert [item.horizon_minutes for item in result.horizons] == [15, 60, 120]
    assert result.missing_future_minutes == (30,)
    assert not result.future_path_complete
    assert result.mfe_price is result.mae_price is None


def test_generic_path_uses_verified_jpy_pip_conversion():
    result = diagnose_directional_path(
        instrument="USDJPY",
        signal_timestamp=T,
        direction=Direction.LONG,
        p0=100.0,
        m1_bars=_path(instrument="USDJPY"),
    )

    assert result.horizons[0].signed_return_pips == pytest.approx(15.0)


def test_generic_path_is_invariant_to_bars_after_requested_path():
    prefix = _path()
    future = tuple(
        _bar(T + timedelta(minutes=minute), 500.0) for minute in range(121, 141)
    )

    expected = diagnose_directional_path(
        instrument="EURUSD",
        signal_timestamp=T,
        direction=Direction.LONG,
        p0=100.0,
        m1_bars=prefix,
    )
    actual = diagnose_directional_path(
        instrument="EURUSD",
        signal_timestamp=T,
        direction=Direction.LONG,
        p0=100.0,
        m1_bars=(*prefix, *future),
    )

    assert actual == expected


def test_overlap_windows_deduplicate_timestamps_and_isolate_instruments():
    duplicate = _observation(suffix="duplicate")
    observations = (
        _observation(),
        duplicate,
        _observation(
            timestamp=T + timedelta(minutes=40), suffix="later", anchor="asia-session"
        ),
        _observation(instrument="GBPUSD", suffix="other"),
    )
    references = (
        ModuleAReference("EURUSD", T),
        ModuleAReference("EURUSD", T + timedelta(minutes=16)),
        ModuleAReference("GBPUSD", T + timedelta(minutes=61)),
    )

    rows = calculate_overlap(observations, references)
    aggregate = next(row for row in rows if row["scope"] == "aggregate")
    eurusd = next(
        row
        for row in rows
        if row["scope"] == "instrument" and row["instrument"] == "EURUSD"
    )

    assert aggregate["failed_breakout_family_opportunity_count"] == 3
    assert [aggregate[f"overlap_count_w{window}"] for window in (0, 15, 30, 60)] == [
        1,
        1,
        2,
        2,
    ]
    assert eurusd["failed_breakout_family_opportunity_count"] == 2
    assert eurusd["overlap_count_w30"] == 2


def test_parameter_month_quarter_aggregation_and_all_cells():
    observations = (
        _observation(),
        _observation(timestamp=datetime(2024, 4, 2, tzinfo=UTC), suffix="q2"),
    )
    rows = aggregate_observations(observations, ("EURUSD", "GBPUSD"))
    family = next(row for row in rows if row["scope"] == "family")

    assert len([row for row in rows if row["scope"] == "cell"]) == 12
    assert family["active_months"] == 2
    assert set(family["monthly_expectancy_h60"]) == {"2024-01", "2024-04"}
    assert set(family["quarterly_expectancy_h60"]) == {"2024-Q1", "2024-Q2"}
    assert any(
        row["raw_structural_event_count"] == 0 for row in rows if row["scope"] == "cell"
    )


def test_family_distinguishes_opportunities_from_global_clocks():
    rows = aggregate_observations(
        (
            _observation(instrument="EURUSD", suffix="eur"),
            _observation(instrument="GBPUSD", suffix="gbp"),
        ),
        ("EURUSD", "GBPUSD"),
    )
    family = next(row for row in rows if row["scope"] == "family")

    assert family["unique_family_opportunity_count"] == 2
    assert family["unique_global_signal_clocks"] == 1


def test_anchor_collisions_count_once_at_family_and_depth_but_remain_diagnostic():
    observations = tuple(
        _observation(anchor=anchor, suffix=anchor)
        for anchor in ("previous-day", "asia-session", "london-or60")
    )

    rows = aggregate_observations(observations, ("EURUSD",))
    family = next(row for row in rows if row["scope"] == "family")
    depth = next(
        row
        for row in rows
        if row["scope"] == "depth" and row["minimum_depth_fraction"] == 0.05
    )
    populated_anchor_cells = [
        row
        for row in rows
        if row["scope"] == "cell" and row["raw_structural_event_count"] == 1
    ]

    assert family["raw_structural_event_count"] == 3
    assert family["unique_family_opportunity_count"] == 1
    assert family["n_h60"] == 1
    assert (
        family["forward_pips_h60_mean"]
        == observations[0].path.horizons[2].signed_return_pips
    )
    assert family["contributing_anchor_event_counts"] == {
        "asia-session": 1,
        "london-or60": 1,
        "previous-day": 1,
    }
    assert depth["raw_structural_event_count"] == 3
    assert depth["unique_family_opportunity_count"] == 1
    assert len(populated_anchor_cells) == 3


def test_depth_plateau_uses_all_observations_not_unweighted_cell_means():
    observations = []
    for index in range(10):
        item = _observation(
            timestamp=T + timedelta(minutes=index * 180), suffix=f"large-{index}"
        )
        observations.append(
            replace(
                item,
                path=replace(
                    item.path,
                    horizons=(ForwardOutcome(60, 0.0001, 0.0, 0.0, 1.0),),
                ),
            )
        )
    small = _observation(instrument="GBPUSD", anchor="asia-session", suffix="small")
    observations.append(
        replace(
            small,
            path=replace(
                small.path,
                horizons=(ForwardOutcome(60, -0.0005, 0.0, 0.0, -5.0),),
            ),
        )
    )

    rows = aggregate_observations(observations, ("EURUSD", "GBPUSD"))
    depth = next(
        row
        for row in rows
        if row["scope"] == "depth" and row["minimum_depth_fraction"] == 0.05
    )
    cell_means = [
        row["forward_pips_h60_mean"]
        for row in rows
        if row["scope"] == "cell" and row["raw_structural_event_count"]
    ]

    assert sum(cell_means) / len(cell_means) == pytest.approx(-2.0)
    assert depth["forward_pips_h60_mean"] == pytest.approx(5 / 11)


def test_report_outputs_are_byte_deterministic(tmp_path):
    observations = (_observation(),)
    refs = (ModuleAReference("EURUSD", T),)
    first = write_outputs(observations, refs, ("EURUSD",), tmp_path / "first")
    second = write_outputs(observations, refs, ("EURUSD",), tmp_path / "second")

    assert set(first) == set(second)
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
    summary = json.loads(first["summary.json"].read_text())
    assert summary["raw_structural_event_count"] == 1
    assert summary["unique_family_opportunity_count"] == 1
    assert summary["grid_observation_count"] == 1
    assert summary["classification"]["family_classification"] == "INCONCLUSIVE"
    report = first["report.md"].read_text()
    assert "## OBSERVED RESULT" in report
    assert "## INTERPRETATION" in report
