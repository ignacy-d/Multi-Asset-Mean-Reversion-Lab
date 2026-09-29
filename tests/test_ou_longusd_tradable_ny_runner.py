from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import mr_lab.ou_longusd_tradable_ny_runner as runner
from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.research import Direction


def _event(session: str) -> SimpleNamespace:
    return SimpleNamespace(
        signal=SimpleNamespace(
            instrument="EURUSD",
            direction=Direction.SHORT,
            signal_timeframe=Timeframe("15m"),
            session=session,
            benchmark_family="vwap",
            lookback=20,
        )
    )


def _component(timestamp: datetime) -> runner.Component:
    return runner.Component(
        "component",
        "EURUSD",
        timestamp,
        Direction.SHORT,
        "vwap",
        20,
        1.01,
        1.0,
        1.500001,
        120.0,
        "corpus",
        "dataset",
    )


def _bar(start: datetime, price: float = 1.011) -> Bar:
    return Bar(
        "EURUSD",
        Timeframe("1m"),
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=1),
        price,
        price + 0.001,
        price - 0.001,
        price,
        PriceBasis.BID,
        1.0,
        VolumeSemantics.TICK,
    )


def test_london_sources_remain_byte_for_byte_unchanged():
    expected = {
        Path("configs/ou-longusd-tradable-2024-v1.json"): (
            "8a53eaf1031fe8d5010dd802a814f3ca433c67c1ffbe72db51c6d7ae7b3a7e8d"
        ),
        Path("src/mr_lab/ou_longusd_tradable_runner.py"): (
            "b2710f3be003e76c97bb7bb41d746425ac7798f6044c7665d149dc3516ebe582"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_preregistration_freezes_ny_parallel_discovery_contract():
    value, identity = runner.load_preregistration()
    assert value["study_id"] == "OU-LONGUSD-TRADABLE-2024-NY-v1"
    assert value["scientific_status"] == "2024_FOLLOW_UP_DISCOVERY_NOT_CONFIRMATION"
    assert value["signal"]["session"] == "new_york"
    assert value["lifecycle"] == {
        "safe_new_york_end": "16:45 America/New_York",
        "primary": "min(180_minutes,safe_new_york_end-entry)",
        "robustness": "min(180_minutes,floor_two_thirds_safe_remaining)",
    }
    assert identity == runner.EXPECTED_PREREGISTRATION_SHA


def test_only_new_york_components_are_admitted():
    assert runner._is_study_event(_event("new_york"))
    assert not runner._is_study_event(_event("london"))


def test_safe_ny_end_is_local_1645_and_dst_aware():
    winter = runner.safe_new_york_end(datetime(2024, 1, 2, 12, tzinfo=UTC))
    summer = runner.safe_new_york_end(datetime(2024, 7, 2, 12, tzinfo=UTC))
    assert winter == datetime(2024, 1, 2, 21, 45, tzinfo=UTC)
    assert summer == datetime(2024, 7, 2, 20, 45, tzinfo=UTC)
    zone = runner.ZoneInfo("America/New_York")
    assert winter.astimezone(zone).strftime("%H:%M") == "16:45"
    assert summer.astimezone(zone).strftime("%H:%M") == "16:45"


def test_primary_and_robustness_deadline_formulas_and_boundary():
    entry = datetime(2024, 1, 2, 19, tzinfo=UTC)  # 165 safe minutes remain.
    assert runner.lifecycle_deadline(entry, "PRIMARY") == datetime(
        2024, 1, 2, 21, 45, tzinfo=UTC
    )
    assert runner.lifecycle_deadline(entry, "ROBUSTNESS") == datetime(
        2024, 1, 2, 20, 50, tzinfo=UTC
    )
    with pytest.raises(runner.OuTradableError, match="at_or_after"):
        runner.lifecycle_deadline(datetime(2024, 1, 2, 21, 45, tzinfo=UTC), "PRIMARY")
    with pytest.raises(runner.OuTradableError, match="at_or_after"):
        runner.lifecycle_deadline(datetime(2024, 1, 2, 21, 46, tzinfo=UTC), "PRIMARY")


def test_exact_m1_open_execution_has_no_forward_substitution():
    timestamp = datetime(2024, 1, 2, 18, tzinfo=UTC)
    signal = runner.ensemble_components((_component(timestamp),))[0]
    future_only = (_bar(timestamp + timedelta(minutes=1)),)
    result = runner.execute_signal(signal, future_only, "PRIMARY")
    assert not result["executed"]
    assert result["incomplete_reason"] == "missing_exact_m1_entry"


def test_cost_session_uses_actual_default_spec_classification():
    # 09:00 New York is also inside London, so overlap must use overall.
    assert runner._cost_session_key(datetime(2024, 1, 2, 14, tzinfo=UTC)) == "overall"
    # 16:00 New York is after London and therefore has exactly one active session.
    assert runner._cost_session_key(datetime(2024, 1, 2, 21, tzinfo=UTC)) == "new_york"


def test_default_output_and_artifacts_are_separate_and_deterministic(
    tmp_path, monkeypatch
):
    assert Path("results/ou-longusd-tradable-2024-ny-v1") == runner.DEFAULT_OUTPUT
    assert Path("results/ou-longusd-tradable-2024-v1") != runner.DEFAULT_OUTPUT

    monkeypatch.setattr(runner, "STUDY_DIRECTIONS", {})
    monkeypatch.setattr(
        runner,
        "load_registry",
        lambda _path: {"registry_id": "unit-test-registry", "instruments": {}},
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="fixed-revision\n"),
    )

    first = tmp_path / "ny-first"
    second = tmp_path / "ny-second"
    missing_costs = tmp_path / "missing-cost-profile.json"
    runner.run(first, missing_costs)
    runner.run(second, missing_costs)

    assert first.is_dir() and second.is_dir()
    assert not (tmp_path / "ou-longusd-tradable-2024-v1").exists()
    first_files = {p.name: p.read_bytes() for p in first.iterdir()}
    second_files = {p.name: p.read_bytes() for p in second.iterdir()}
    assert first_files == second_files
    assert json.loads(first_files["summary.json"])["study_id"] == runner.STUDY_ID
