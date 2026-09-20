from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.rv_leadlag import LeadLagEvent
from mr_lab.rv_leadlag_runner import (
    HORIZONS,
    INSTRUMENTS,
    LeadLagRunnerError,
    PreparedData,
    cost_profile_session,
    economic_summary,
    execute_event,
    load_registry,
    prepare_data,
    write_outputs,
)


def bar(instrument: str, minute: int, price: float = 1.0, volume: float = 1) -> Bar:
    start = datetime(2024, 1, 2, tzinfo=UTC) + timedelta(minutes=minute)
    return Bar(
        instrument,
        Timeframe("1m"),
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=1),
        price,
        price,
        price,
        price,
        PriceBasis.BID,
        volume,
        VolumeSemantics.TICK,
    )


def event(direction: int = 1) -> LeadLagEvent:
    timestamp = datetime(2024, 1, 2, 0, 5, tzinfo=UTC)
    return LeadLagEvent(
        "RV-LEADLAG-2024-v1",
        "event",
        timestamp,
        "EURUSD_GBPUSD",
        "EURUSD",
        "GBPUSD",
        direction,  # type: ignore[arg-type]
        2.1,
        0.5,
        2.1,
        0.5,
        0.2,
        0.01,
        0.002,
        "2.00 <= |z| < 2.50",
    )


def test_registry_is_exact_and_fail_closed(tmp_path):
    registry = load_registry()
    assert registry["registry_id"].startswith("sha256:")
    with pytest.raises(LeadLagRunnerError, match="explicit frozen registry"):
        load_registry(tmp_path / "copy.json")


def test_exact_six_and_padding_or_missing_invalidates_m5():
    series = {name: [bar(name, i) for i in range(5)] for name in INSTRUMENTS}
    assert len(prepare_data(series).m5_bars) == 6
    series["EURUSD"][2] = bar("EURUSD", 2, volume=0)
    assert not [x for x in prepare_data(series).m5_bars if x.instrument == "EURUSD"]
    del series["GBPUSD"][2]
    assert not [x for x in prepare_data(series).m5_bars if x.instrument == "GBPUSD"]
    with pytest.raises(LeadLagRunnerError, match="six instruments"):
        prepare_data({"EURUSD": []})


@pytest.mark.parametrize("direction", [-1, 1])
def test_exact_entry_exits_and_signed_pnl(direction):
    series = {name: [bar(name, i) for i in range(66)] for name in INSTRUMENTS}
    series["GBPUSD"][5] = bar("GBPUSD", 5, 1.0)
    for horizon in HORIZONS:
        series["GBPUSD"][4 + horizon] = bar("GBPUSD", 4 + horizon, 1.01)
    prepared = prepare_data(series)
    record, outcomes = execute_event(event(direction), prepared)
    assert record["entry_complete"] is True
    assert {row.horizon_minutes for row in outcomes} == set(HORIZONS)
    assert all(row.entry_timestamp == event().timestamp for row in outcomes)
    assert all(
        row.exit_timestamp == event().timestamp + timedelta(minutes=row.horizon_minutes)
        for row in outcomes
    )
    assert all((row.gross_signed_bps_return > 0) == (direction > 0) for row in outcomes)
    assert outcomes[0].gross_pips == pytest.approx(direction * 100)


def test_jpy_pips_and_incomplete_reasons():
    series = {
        name: [bar(name, i, 150 if name == "USDJPY" else 1) for i in range(66)]
        for name in INSTRUMENTS
    }
    prepared = prepare_data(series)
    usd_event = event()
    usd_event = LeadLagEvent(
        *(
            usd_event.study_id,
            usd_event.event_id,
            usd_event.timestamp,
            "USDJPY_USDCHF",
            "USDCHF",
            "USDJPY",
            1,
            usd_event.leader_z,
            usd_event.laggard_z,
            usd_event.leader_abs_z,
            usd_event.laggard_abs_z,
            usd_event.z_ratio,
            usd_event.leader_return,
            usd_event.laggard_return,
            usd_event.leader_z_bin,
        )
    )
    record, outcomes = execute_event(usd_event, prepared)
    assert outcomes[0].gross_pips == 0
    missing = dict(prepared.m1_by_instrument)
    missing["USDJPY"] = {
        k: v for k, v in missing["USDJPY"].items() if k != usd_event.timestamp
    }
    record, outcomes = execute_event(
        usd_event, PreparedData(prepared.m5_bars, missing, {})
    )
    assert outcomes == ()
    assert record["entry_incomplete_reason"] == "MISSING_EXACT_ENTRY_BAR"


def test_cost_session_selection_is_order_independent():
    assert cost_profile_session(()) == "overall"
    assert cost_profile_session(("london",)) == "london"
    assert cost_profile_session(("new_york", "london")) == "overall"
    assert cost_profile_session(("london", "new_york")) == "overall"


def test_shared_analyzer_and_frozen_screen_labels():
    series = {name: [bar(name, i) for i in range(66)] for name in INSTRUMENTS}
    prepared = prepare_data(series)
    _, outcomes = execute_event(event(), prepared)
    result = economic_summary(outcomes, None)
    relationship = result["segments"]["relationship:EURUSD_GBPUSD"]
    assert relationship["n"] == 1
    assert relationship["methodology_id"] == "stage4c-economic-analysis-v2"
    assert relationship["screen_labels"] == [
        "FAILS_GROSS_1PIP_SCREEN",
        "BLOCKED_MISSING_COST_PROFILE",
    ]
    assert result["segments"]["relationship:AUDUSD_NZDUSD"]["n"] == 0
    assert result["family_status"] == "PARK_NO_ECONOMICALLY_LARGE_V1_EFFECT"


def test_deterministic_atomic_outputs_and_refusal(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_outputs(first, [{"x": 1}], {"summary": 1}, {"audit": 1})
    write_outputs(second, [{"x": 1}], {"summary": 1}, {"audit": 1})
    assert (first / "events.jsonl.gz").read_bytes() == (
        second / "events.jsonl.gz"
    ).read_bytes()
    assert json.loads(gzip.decompress((first / "events.jsonl.gz").read_bytes())) == {
        "x": 1
    }
    with pytest.raises(LeadLagRunnerError, match="overwrite"):
        write_outputs(first, [], {}, {})
