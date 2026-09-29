from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.ou_longusd_tradable_runner import (
    AUX_DIRECTIONS,
    CORE_DIRECTIONS,
    METHODOLOGY_ID,
    Component,
    EnsembleSignal,
    OuTradableError,
    ensemble_components,
    execute_signal,
    geometry,
    lifecycle_deadline,
    ou_eligible,
    portfolio,
    required_direction,
    safe_london_end,
    scope_for,
)
from mr_lab.pca_stage0_runner import REGISTRY, Stage0Error, load_registry
from mr_lab.research import Direction


def component(
    instrument="EURUSD",
    timestamp=datetime(2024, 1, 2, 10, tzinfo=UTC),
    family="vwap",
    lookback=20,
    e0=1.0,
    p0=1.01,
):
    direction = required_direction(instrument)
    return Component(
        f"{family}-{lookback}",
        instrument,
        timestamp,
        direction,
        family,
        lookback,
        p0,
        e0,
        1.500001,
        120.0,
        "corpus",
        "dataset",
    )


def signal(
    instrument="EURUSD", timestamp=datetime(2024, 1, 2, 10, tzinfo=UTC), p0=1.01, e0=1.0
):
    c = component(instrument, timestamp, p0=p0, e0=e0)
    return ensemble_components((c,))[0]


def bar(instrument, start, o, h=None, low=None, close=None):
    h = o if h is None else h
    low = o if low is None else low
    close = o if close is None else close
    return Bar(
        instrument,
        Timeframe("1m"),
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=1),
        o,
        h,
        low,
        close,
        PriceBasis.BID,
        1.0,
        VolumeSemantics.TICK,
    )


def test_universe_orientation_and_scopes():
    assert CORE_DIRECTIONS == {
        "AUDUSD": Direction.SHORT,
        "EURUSD": Direction.SHORT,
        "GBPUSD": Direction.SHORT,
        "NZDUSD": Direction.SHORT,
        "USDCAD": Direction.LONG,
        "USDCHF": Direction.LONG,
        "USDJPY": Direction.LONG,
    }
    assert AUX_DIRECTIONS == {"EURGBP": Direction.SHORT}
    assert scope_for("EURGBP") == "AUX_EURGBP_SHORT"
    assert all(scope_for(x) == "CORE_LONG_USD" for x in CORE_DIRECTIONS)
    with pytest.raises(OuTradableError):
        required_direction("AUDJPY")


def test_union_median_exact_p0_and_vote_count_diagnostic_only():
    votes = (
        component(e0=1.00),
        component(family="vwap", lookback=40, e0=1.02),
        component(family="vwap-canonical-m1", lookback=20, e0=1.01),
    )
    out = ensemble_components(votes)
    assert len(out) == 1 and out[0].ensemble_e0 == 1.01 and out[0].model_vote_count == 3
    assert len(ensemble_components((votes[0],))) == 1
    with pytest.raises(OuTradableError):
        ensemble_components(
            (votes[0], component(family="vwap", lookback=40, p0=1.0100001))
        )


def test_ou_boundaries_are_strict_score_and_inclusive_half_life():
    assert not ou_eligible(1.5, 120)
    assert ou_eligible(1.5000001, 120)
    assert not ou_eligible(2, 120.00001)
    assert not ou_eligible(None, 10)


def test_geometry_ladder_and_shared_stop():
    targets, stop = geometry(signal())
    assert targets == pytest.approx((1.005, 1.0025, 1.0))
    assert stop == pytest.approx(1.015)


def test_exact_open_missing_bar_and_stale_entry():
    s = signal()
    future = bar("EURUSD", s.timestamp + timedelta(minutes=1), 1.009)
    assert (
        execute_signal(s, (future,), "PRIMARY")["incomplete_reason"]
        == "missing_exact_m1_entry"
    )
    stale = bar("EURUSD", s.timestamp, 1.004)
    assert (
        execute_signal(s, (stale,), "PRIMARY")["incomplete_reason"]
        == "entry_already_past_target_or_stop"
    )


def test_partial_tp_then_stop_and_actual_entry_r():
    s = signal()
    bars = [
        bar("EURUSD", s.timestamp, 1.011, 1.0115, 1.004, 1.006),
        bar("EURUSD", s.timestamp + timedelta(minutes=1), 1.006, 1.016, 1.005, 1.015),
    ]
    trade = execute_signal(s, bars, "PRIMARY")
    assert trade["executed"]
    assert [x["reason"] for x in trade["tranches"]] == ["tp1", "stop", "stop"]
    assert trade["initial_risk_distance"] == pytest.approx(0.004)
    assert trade["tranches"][0]["gross_r"] == pytest.approx(1.5)


def test_adverse_first_same_minute():
    s = signal()
    trade = execute_signal(
        s, (bar("EURUSD", s.timestamp, 1.011, 1.016, 0.999, 1.0),), "PRIMARY"
    )
    assert trade["executed"] and {x["reason"] for x in trade["tranches"]} == {"stop"}


def test_exact_deadline_close_no_forward_substitution():
    ts = datetime(2024, 1, 2, 16, 44, tzinfo=UTC)
    s = signal(timestamp=ts)
    only = bar("EURUSD", ts, 1.011, 1.012, 1.010, 1.0105)
    trade = execute_signal(s, (only,), "PRIMARY")
    assert trade["executed"] and {x["reason"] for x in trade["tranches"]} == {
        "deadline"
    }
    assert all(x["price"] == 1.0105 for x in trade["tranches"])


def test_london_safe_end_dst_and_deadlines():
    assert safe_london_end(datetime(2024, 1, 2, 12, tzinfo=UTC)).hour == 16
    assert safe_london_end(datetime(2024, 7, 2, 12, tzinfo=UTC)).hour == 15
    winter = datetime(2024, 1, 2, 15, 45, tzinfo=UTC)
    assert lifecycle_deadline(winter, "PRIMARY") == datetime(
        2024, 1, 2, 16, 45, tzinfo=UTC
    )
    assert lifecycle_deadline(winter, "ROBUSTNESS") == datetime(
        2024, 1, 2, 16, 25, tzinfo=UTC
    )
    with pytest.raises(OuTradableError):
        lifecycle_deadline(datetime(2024, 1, 2, 16, 45, tzinfo=UTC), "PRIMARY")


def fake_trade(identity, start, end, instrument="EURUSD", scope="CORE_LONG_USD"):
    return {
        "signal_id": identity,
        "instrument": instrument,
        "scope": scope,
        "executed": True,
        "entry_timestamp": start.isoformat(),
        "exit_timestamp": end.isoformat(),
    }


def test_portfolio_overlap_modes_and_core_independence():
    t = datetime(2024, 1, 2, 10, tzinfo=UTC)
    rows = [
        fake_trade("a", t, t + timedelta(minutes=30)),
        fake_trade("b", t + timedelta(minutes=15), t + timedelta(minutes=45)),
        fake_trade("aux", t, t + timedelta(minutes=20), "EURGBP", "AUX_EURGBP_SHORT"),
    ]
    unconstrained, _ = portfolio(rows, "UNCONSTRAINED")
    constrained, audit = portfolio(rows, "ONE_PER_INSTRUMENT")
    assert len(unconstrained) == 3 and len(constrained) == 2
    assert audit["skipped_due_to_open_instrument"] == 1
    assert [x["signal_id"] for x in constrained if x["scope"] == "CORE_LONG_USD"] == [
        "a"
    ]


def test_bollinger_diagnostics_cannot_change_signal_identity_or_trade_set():
    original = signal()
    # Rebuilding with altered diagnostics leaves execution geometry and identity fixed.
    altered = EnsembleSignal(
        original.signal_id,
        original.instrument,
        original.timestamp,
        original.direction,
        original.scope,
        original.p0,
        original.ensemble_e0,
        original.d0,
        original.components,
        {"z20": 99},
    )
    assert altered.signal_id == original.signal_id and geometry(altered) == geometry(
        original
    )


def test_deterministic_identity():
    votes = (component(), component(family="vwap", lookback=40, e0=0.999))
    assert ensemble_components(votes) == ensemble_components(tuple(reversed(votes)))
    assert METHODOLOGY_ID.startswith("sha256:")


def test_registry_is_exact_explicit_authenticated_contract(tmp_path):
    registry = load_registry(REGISTRY)
    assert registry["registry_schema_version"] == "fx-universe-2024-registry-v1"
    wrong = tmp_path / "registry.json"
    wrong.write_text(json.dumps({"registry_schema_version": "wrong"}))
    with pytest.raises(Stage0Error, match="only the explicit 2024 registry"):
        load_registry(wrong)
