from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import (
    SLIPPAGES,
    SPREAD_STATISTICS,
    CostProfile,
    Stage4CError,
    break_even_total_slippage,
    scenario_rows,
    transform_trade,
)
from mr_lab.stage4c_runner import run_rows

PROFILE = Path("configs/stage4c-ftmo-cost-profile-v1.json")


def trade(
    event="event-1", instrument="EURUSD", session="london", gross=2.0, favorable=2.0
):
    return {
        "instrument": instrument,
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "session": session,
        "direction": "LONG",
        "lookback": 20,
        "signal_threshold": 2.0,
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "entry_mode": "immediate",
        "tp_target_fraction": 0.5,
        "sl_extension_fraction": 0.5,
        "time_stop_minutes": 30,
        "candidate_event_id": event,
        "complete": True,
        "exit_ordering": "unambiguous",
        "holding_minutes": 4,
        "mae_pips_certain": 1.0,
        "mfe_pips_certain": 3.0,
        "gross_return_pips_adverse_first": gross,
        "gross_return_pips_favorable_first": favorable,
    }


@pytest.fixture
def profile():
    return CostProfile.load(PROFILE)


def test_one_bid_bid_spread_and_usd_commission(profile):
    row = transform_trade(trade(), profile, "mean", 0.0)
    assert row["commission_pips"] == 0.5
    assert row["net_pips_adverse_first"] == pytest.approx(
        2.0 - row["spread_pips"] - 0.5
    )


def test_audusd_commission_is_half_pip(profile):
    assert (
        transform_trade(trade(instrument="AUDUSD"), profile, "p75", 0)[
            "commission_pips"
        ]
        == 0.5
    )


def test_jpy_commission_and_conversion_adjustment(profile):
    positive = transform_trade(trade(instrument="USDJPY", gross=4), profile, "mean", 0)
    execution = 4 - positive["spread_pips"]
    assert positive["commission_pips"] == pytest.approx(0.795265089)
    assert positive["net_pips_adverse_first"] == pytest.approx(
        execution * 0.993 - positive["commission_pips"]
    )
    negative = transform_trade(trade(instrument="AUDJPY", gross=-4), profile, "mean", 0)
    assert negative["net_pips_adverse_first"] == pytest.approx(
        (-4 - negative["spread_pips"]) * 1.007 - negative["commission_pips"]
    )


def test_scenario_monotonicity_and_full_matrix(profile):
    rows = list(scenario_rows(trade(), profile))
    assert len(rows) == 16
    by_spread = {(r["spread_statistic"], r["slippage_pips"]): r for r in rows}
    for statistic in SPREAD_STATISTICS:
        values = [by_spread[statistic, x]["net_pips_adverse_first"] for x in SLIPPAGES]
        assert values == sorted(values, reverse=True)
    for slip in SLIPPAGES:
        # "Increasing spread" means the numeric frozen cost, not the statistic
        # label: an empirical mean can exceed p75 in a right-tailed sample.
        ordered = sorted(
            SPREAD_STATISTICS,
            key=lambda x: by_spread[x, slip]["spread_pips"],
        )
        values = [by_spread[x, slip]["net_pips_adverse_first"] for x in ordered]
        assert values == sorted(values, reverse=True)


def test_gross_and_ambiguity_are_preserved(profile):
    original = trade(gross=-2, favorable=3) | {
        "exit_ordering": "ambiguous_same_minute",
        "exit_reason": "sl",
    }
    snapshot = deepcopy(original)
    result = transform_trade(original, profile, "mean", 0.1)
    assert original == snapshot
    assert result["gross_return_pips_adverse_first"] == -2
    assert result["gross_return_pips_favorable_first"] == 3
    assert result["exit_ordering"] == "ambiguous_same_minute"
    assert result["net_pips_adverse_first"] < result["net_pips_favorable_first"]


def test_null_session_uses_overall(profile):
    row = transform_trade(trade(session=None), profile, "mean", 0)
    assert row["spread_pips"] == pytest.approx(0.160480694)


def test_streaming_matches_reference_and_sharded_equals_unsharded(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    rows = [trade("a", gross=2), trade("b", gross=-1)]
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
    }
    first, second = tmp_path / "one", tmp_path / "two"
    run_rows(
        iter(rows),
        first,
        PROFILE,
        source_mode="regenerated_stage4b",
        source_audit=audit,
        debug_raw=True,
    )
    run_rows(
        iter(rows[:1] + rows[1:]),
        second,
        PROFILE,
        source_mode="existing_stage4b_raw",
        source_audit=audit,
    )
    with (first / "stage4c-debug-trades.jsonl").open() as stream:
        actual = [json.loads(line) for line in stream]
    profile = CostProfile.load(PROFILE)
    expected = [result for row in rows for result in scenario_rows(row, profile)]
    assert actual == expected
    for name in (
        "stage4c-trade-matrix.csv",
        "stage4c-regime-breadth.csv",
        "stage4c-cost-scenarios.csv",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_usd_break_even_fixture(profile):
    spread, commission = profile.costs("EURUSD", "london", "mean")
    assert break_even_total_slippage(
        [2.0, 4.0], "EURUSD", spread, commission
    ) == pytest.approx(3.0 - spread - commission, abs=1e-9)


def test_mixed_sign_jpy_break_even_is_deterministic_and_spread_monotone(profile):
    values = (8.0, -1.0, 3.0)
    results = []
    for statistic in SPREAD_STATISTICS:
        spread, commission = profile.costs("USDJPY", "london", statistic)
        first = break_even_total_slippage(values, "USDJPY", spread, commission)
        assert first == break_even_total_slippage(values, "USDJPY", spread, commission)
        results.append(first)
    assert results == sorted(results, reverse=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1.0", None])
def test_nonfinite_or_malformed_gross_fails_closed(profile, bad):
    with pytest.raises(Stage4CError, match="gross"):
        transform_trade(trade(gross=bad), profile, "mean", 0)


def test_duplicate_and_wrong_methodology_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
    }
    with pytest.raises(Stage4CError, match="duplicate"):
        run_rows(
            iter([trade(), trade()]),
            tmp_path / "duplicate",
            PROFILE,
            source_mode="existing_stage4b_raw",
            source_audit=audit,
        )
    with pytest.raises(Stage4CError, match="methodology"):
        run_rows(
            iter([trade()]),
            tmp_path / "method",
            PROFILE,
            source_mode="existing_stage4b_raw",
            source_audit=audit | {"stage4b_methodology_id": "wrong"},
        )


def test_reports_required_metrics_and_no_2025_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
    }
    run_rows(
        iter([trade()]),
        tmp_path,
        PROFILE,
        source_mode="regenerated_stage4b",
        source_audit=audit,
    )
    with (tmp_path / "stage4c-trade-matrix.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert {
        "mean_net_pips",
        "median_net_pips",
        "net_pips_p10",
        "net_profit_factor",
        "cost_headroom_pips",
        "break_even_total_slippage_pips",
        "break_even_extra_slippage_pips",
    } <= set(row)
    assert "2025" not in Path("src/mr_lab/stage4c.py").read_text()
    assert "2025" not in Path("src/mr_lab/stage4c_runner.py").read_text()
