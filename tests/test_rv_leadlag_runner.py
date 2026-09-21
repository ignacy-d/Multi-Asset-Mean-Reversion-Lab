from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import mr_lab.rv_leadlag_runner as runner
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
    load_cost_profile,
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
        "INSUFFICIENT_N_FOR_DISCOVERY_SCREEN",
        "FAILS_GROSS_1PIP_SCREEN",
        "BLOCKED_MISSING_COST_PROFILE",
    ]
    assert result["segments"]["relationship:AUDUSD_NZDUSD"]["n"] == 0
    assert result["family_status"] == "PARK_NO_ECONOMICALLY_LARGE_V1_EFFECT"


def test_family_screen_requires_n_mean_and_lower_bootstrap(monkeypatch):
    series = {name: [bar(name, i) for i in range(66)] for name in INSTRUMENTS}
    _, outcomes = execute_event(event(), prepare_data(series))

    def result(rows, _profile, **_kwargs):
        return {
            "methodology_id": "stage4c-economic-analysis-v2",
            "n": len(rows),
            "gross_pips": {"mean": 1.0},
            "gross_bps": {"mean": 1.0},
            "bootstrap_summary": {"p2_5": 0.01},
            "calendar_month_breadth": 1,
            "cost_floor": {"net_mean_pips": None},
        }

    monkeypatch.setattr(runner, "analyze", result)
    insufficient = economic_summary([outcomes[1]] * 199, None)
    assert insufficient["family_status"] == "PARK_NO_ECONOMICALLY_LARGE_V1_EFFECT"
    labels = insufficient["segments"]["relationship:EURUSD_GBPUSD"]["screen_labels"]
    assert "INSUFFICIENT_N_FOR_DISCOVERY_SCREEN" in labels
    sufficient = economic_summary([outcomes[1]] * 200, None)
    assert sufficient["family_status"] == "CONTINUE_RV_LEADLAG_V1_DISCOVERY"
    assert (
        "INSUFFICIENT_N_FOR_DISCOVERY_SCREEN"
        not in sufficient["segments"]["relationship:EURUSD_GBPUSD"]["screen_labels"]
    )


def test_explicit_v2_cost_profile_accepts_non_template_identity(tmp_path):
    source = runner.COST_PROFILE_PATH
    value = json.loads(source.read_text(encoding="utf-8"))
    value["generated_for_test"] = True
    explicit = tmp_path / "populated-profile.json"
    explicit.write_text(json.dumps(value), encoding="utf-8")
    profile, identity = load_cost_profile(explicit)
    assert profile.raw["generated_for_test"] is True
    assert identity != runner._sha(source.read_bytes())
    with pytest.raises(LeadLagRunnerError, match="explicit Stage4C-v2"):
        load_cost_profile(tmp_path / "missing.json")


def test_run_records_explicit_cost_profile_provenance(tmp_path, monkeypatch):
    entries = {
        name: {"instrument": name, "assembled_dataset_id": f"dataset-{name}"}
        for name in INSTRUMENTS
    }
    explicit = tmp_path / "explicit-profile.json"
    captured = {}
    monkeypatch.setattr(
        runner,
        "load_registry",
        lambda: {"registry_id": "registry", "instruments": entries},
    )
    monkeypatch.setattr(
        runner, "load_cost_profile", lambda path: (None, "sha256:explicit")
    )
    monkeypatch.setattr(runner, "authenticate_corpus", lambda entry, name: tmp_path)
    monkeypatch.setattr(
        runner, "load_offline_corpus", lambda path: SimpleNamespace(bars=())
    )
    monkeypatch.setattr(
        runner,
        "prepare_data",
        lambda series: PreparedData((), {name: {} for name in INSTRUMENTS}, {}),
    )
    monkeypatch.setattr(runner, "detect_leadlag_events", lambda bars: ())
    monkeypatch.setattr(
        runner,
        "write_outputs",
        lambda output, records, summary, audit: captured.update(
            output=output, summary=summary, audit=audit
        ),
    )
    runner.run(tmp_path / "output", explicit)
    assert captured["summary"]["provenance"] == {
        "cost_profile_path": str(explicit),
        "cost_profile_sha256": "sha256:explicit",
    }
    assert captured["audit"]["cost_profile_path"] == str(explicit)
    assert captured["audit"]["cost_profile_sha256"] == "sha256:explicit"


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
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".first.")]


def test_failed_atomic_directory_rename_leaves_no_partial_result(tmp_path, monkeypatch):
    output = tmp_path / "result"

    def fail_rename(_source, _target):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(runner.os, "rename", fail_rename)
    with pytest.raises(OSError, match="publication failure"):
        write_outputs(output, [{"x": 1}], {"summary": 1}, {"audit": 1})
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
