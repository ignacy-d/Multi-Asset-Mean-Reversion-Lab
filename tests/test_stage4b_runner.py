from pathlib import Path

import pytest

from mr_lab.stage4a_runner import (
    Stage4ARunnerError,
    load_corpus_registry,
    select_verified_registry_entries,
)
from mr_lab.stage4b_runner import (
    GROUP_FIELDS,
    OUTPUTS,
    Aggregate,
    _aggregate_diagnostics,
    _aggregate_row,
    _filter_provenance,
    _json,
)


def test_gbpusd_pending_fails_closed():
    registry = load_corpus_registry(Path("configs/stage4a-2024-corpus-registry.json"))
    with pytest.raises(Stage4ARunnerError, match="not verified"):
        select_verified_registry_entries(registry, "GBPUSD")


def test_workflow_has_only_instrument_input_and_review_diagnostics():
    text = Path(".github/workflows/run-stage-4b-real-2024.yml").read_text()
    head = text.split("permissions:", 1)[0]
    assert head.count("type: choice") == 1
    assert all(
        name not in head
        for name in ("threshold:", "tp:", "sl:", "time_stop:", "entry_mode:")
    )
    compact = text.split("review-source", 1)[1]
    assert "result/entry-diagnostics.csv" in compact


def test_deterministic_json_and_output_contract():
    assert _json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert set(OUTPUTS) == {
        "candidate-events.jsonl",
        "trades.jsonl",
        "entry-diagnostics.csv",
        "trade-matrix.csv",
        "summary.json",
        "report.md",
    }


def test_tp_target_is_not_overwritten_by_observed_hit_fraction():
    key = (
        "EURUSD",
        "vwap",
        "15m",
        "london",
        "LONG",
        20,
        2.0,
        "none",
        "none-v1",
        "immediate",
        0.5,
        None,
        30,
    )
    aggregate = Aggregate(complete=4, tp=1)
    row = _aggregate_row(key, aggregate)
    assert GROUP_FIELDS[10] == "tp_target_fraction"
    assert row["tp_target_fraction"] == 0.5
    assert row["tp_exit_fraction"] == 0.25


def test_diagnostics_preserve_session_direction_and_lookback():
    base = {
        "instrument": "EURUSD",
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "lookback": 20,
        "horizon_minutes": 5,
        "reversion25_hit": True,
        "extension25_hit": False,
        "time_to_reversion25": 2,
        "time_to_extension25": None,
        "ordering_25": "reversion",
        "reversion50_hit": False,
        "extension50_hit": False,
        "time_to_reversion50": None,
        "time_to_extension50": None,
        "ordering_50": "none",
    }
    rows = _aggregate_diagnostics(
        [
            base | {"session": "london", "direction": "LONG"},
            base | {"session": "new_york", "direction": "SHORT"},
        ]
    )
    assert len(rows) == 2
    assert {(r["session"], r["direction"]) for r in rows} == {
        ("london", "LONG"),
        ("new_york", "SHORT"),
    }


def test_audit_filter_provenance_uses_actual_identity():
    assert _filter_provenance({("future-ou", "ou-v1")}) == {
        "filter_family": "future-ou",
        "filter_spec_id": "ou-v1",
    }
