from pathlib import Path
from types import SimpleNamespace

import pytest

from mr_lab.stage4a_runner import (
    load_corpus_registry,
    select_verified_registry_entries,
)
from mr_lab.stage4b_runner import (
    GROUP_FIELDS,
    OUTPUTS,
    Aggregate,
    _aggregate_diagnostics,
    _aggregate_row,
    _entry_wait_distributions,
    _filter_provenance,
    _first_passage_distributions,
    _json,
    _long_distributions,
    _quantile,
    partition_candidate_groups,
    stable_group_key,
    validate_grid_restriction,
)

VALID_RESTRICTION = {
    "signal_timeframes": ("15m",),
    "sessions": ("london",),
    "benchmark_families": ("vwap", "vwap-canonical-m1"),
    "lookbacks": (20, 40),
    "signal_threshold": 2.0,
    "directions": ("LONG", "SHORT"),
    "entry_modes": ("immediate",),
    "tp_fractions": (0.75, 1.0),
    "sl_fractions": (0.25, 0.5),
    "time_stops_minutes": (60, 120),
}


def test_grid_restriction_accepts_only_nonempty_frozen_subsets():
    assert validate_grid_restriction(VALID_RESTRICTION) == VALID_RESTRICTION


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("signal_timeframes", ("2m",)),
        ("sessions", ("weekend",)),
        ("benchmark_families", ("future-alpha",)),
        ("lookbacks", (19,)),
        ("directions", ("BOTH",)),
        ("entry_modes", ("future-entry",)),
        ("tp_fractions", (1.25,)),
        ("sl_fractions", (0.75,)),
        ("time_stops_minutes", (90,)),
        ("signal_threshold", 1.5),
        ("directions", ()),
    ],
)
def test_grid_restriction_rejects_nonfrozen_or_empty_values(key, value):
    with pytest.raises(ValueError, match="grid_restriction"):
        validate_grid_restriction(VALID_RESTRICTION | {key: value})


def test_grid_restriction_rejects_unknown_or_missing_keys():
    with pytest.raises(ValueError, match="exactly"):
        validate_grid_restriction(VALID_RESTRICTION | {"surprise": (1,)})
    incomplete = dict(VALID_RESTRICTION)
    del incomplete["tp_fractions"]
    with pytest.raises(ValueError, match="exactly"):
        validate_grid_restriction(incomplete)


def test_gbpusd_promoted_entry_is_accepted():
    registry = load_corpus_registry(Path("configs/stage4a-2024-corpus-registry.json"))
    entry = select_verified_registry_entries(registry, "GBPUSD")[0]
    assert entry["source_mode"] == "local-checkpointed"
    assert entry["source_workflow_run_id"] is None
    assert entry["source_artifact_id"] is None


def test_workflow_artifact_upload_contract():
    text = Path(".github/workflows/run-stage-4b-real-2024.yml").read_text()
    head = text.split("permissions:", 1)[0]
    assert head.count("type: choice") == 1
    assert "upload_raw:" in head
    upload_raw_input = head.split("upload_raw:", 1)[1]
    assert "required: true" in upload_raw_input
    assert "type: boolean" in upload_raw_input
    assert "default: false" in upload_raw_input
    assert all(
        name not in head
        for name in ("threshold:", "tp:", "sl:", "time_stop:", "entry_mode:")
    )
    assert 'STAGE4B_SHARD_COUNT: "4"' in text
    assert "shard_index: [0, 1, 2, 3]" in text
    assert '--shard-count "$STAGE4B_SHARD_COUNT"' in text
    raw = "raw-shard-${{ matrix.shard_index }}-attempt-${{ github.run_attempt }}"
    compact = (
        "compact-shard-${{ matrix.shard_index }}-attempt-${{ github.run_attempt }}"
    )
    assert raw in text
    assert compact in text
    assert "result/candidate-events.jsonl" in text
    assert "result/trades.jsonl" in text
    compact_upload = text.index(
        "name: stage-4b-${{ env.INSTRUMENT_LOWER }}-2024-compact-shard-"
    )
    raw_upload = text.index(
        "name: stage-4b-${{ env.INSTRUMENT_LOWER }}-2024-raw-shard-"
    )
    assert compact_upload < raw_upload
    compact_block = text[compact_upload:raw_upload]
    raw_block = text[raw_upload : text.index("\n\n  reducer:")]
    assert "retention-days: 30" in compact_block
    assert "if: ${{ inputs.upload_raw == true }}" in text[compact_upload:raw_upload]
    assert "retention-days: 7" in raw_block
    reducer = text.split("  reducer:", 1)[1]
    assert (
        "pattern: stage-4b-${{ env.INSTRUMENT_LOWER }}-2024-compact-shard-*" in reducer
    )
    assert (
        "pattern: stage-4b-${{ env.INSTRUMENT_LOWER }}-2024-raw-shard-*" not in reducer
    )
    assert "candidate-events.jsonl" not in reducer
    assert "trades.jsonl" not in reducer
    assert "stage-4b-${{ env.INSTRUMENT_LOWER }}-2024-combined-review" in text
    assert "retention-days: 90" in reducer


def test_deterministic_json_and_output_contract():
    assert _json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert set(OUTPUTS) == {
        "candidate-events.jsonl",
        "trades.jsonl",
        "entry-diagnostics.csv",
        "trade-matrix.csv",
        "summary.json",
        "report.md",
        "first-passage-distributions.csv",
        "conditional-hit.csv",
        "entry-wait-distributions.csv",
        "stage4b-distributions.csv",
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


def _diagnostic(event_id, hit):
    return {
        "candidate_event_id": event_id,
        "instrument": "EURUSD",
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "LONG",
        "lookback": 20,
        "horizon_minutes": 30,
        "time_to_reversion25": hit,
        "time_to_reversion50": None,
        "time_to_extension25": None,
        "time_to_extension50": None,
    }


def test_first_passage_distributions_preserve_observations_and_no_hits():
    # A secondary arithmetic mean may be 30, but no quantile invents a 30m hit.
    assert _quantile([15, 45], 0.5) == 15
    assert _quantile([15, 45], 0.9) == 45
    rows = [_diagnostic("a", 5), _diagnostic("b", 15), _diagnostic("c", None)]
    distributions, conditional = _first_passage_distributions(rows)
    reversion = next(r for r in distributions if r["barrier"] == "R>=+0.25")
    assert reversion["candidate_event_count"] == 3
    assert reversion["hit_count"] == 2
    assert reversion["hit_fraction"] == pytest.approx(2 / 3)
    assert reversion["time_to_hit_median"] == 5
    assert reversion["mean_time_to_hit"] == 10
    second = next(
        r
        for r in conditional
        if r["barrier"] == "R>=+0.25" and r["interval_start_minutes"] == 10
    )
    assert second["events_at_risk_at_interval_start"] == 2
    assert second["new_hits_in_interval"] == 1
    assert second["conditional_hit_fraction"] == 0.5


def test_entry_wait_distribution_keeps_no_entry_separate():
    common = {
        "instrument": "EURUSD",
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "LONG",
        "lookback": 20,
        "mode": "m1-reclaim-p0",
    }
    rows = [
        common | {"executed": True, "wait_minutes": 1},
        common | {"executed": True, "wait_minutes": 10},
        common | {"executed": False, "wait_minutes": None},
    ]
    result = _entry_wait_distributions(rows)[0]
    assert result["candidate_event_count"] == 3
    assert result["executed_entry_count"] == 2
    assert result["no_entry_count"] == 1
    assert result["wait_median"] == 1


def test_long_distribution_uses_certain_mae_mfe_and_is_deterministic():
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
    aggregate = Aggregate(mae=[3, 15], mfe=[2, 8], pips=[-5, 4], holding=[5, 20])
    first = _long_distributions({key: aggregate})
    second = _long_distributions({key: aggregate})
    assert first == second
    mae = [r for r in first if r["metric"] == "mae_certain_pips"]
    assert {r["value"] for r in mae} <= {3, 15}


def test_diagnostic_aggregation_accepts_none_and_named_sessions():
    common = {
        "instrument": "EURUSD",
        "benchmark_family": "bollinger",
        "signal_timeframe": "15m",
        "direction": "LONG",
        "lookback": 20,
        "horizon_minutes": 30,
        "time_to_reversion25": 5,
        "time_to_extension25": None,
        "ordering_25": "reversion",
        "time_to_reversion50": None,
        "time_to_extension50": None,
        "ordering_50": "none",
    }

    rows = _aggregate_diagnostics(
        [
            common | {"session": None},
            common | {"session": "london"},
        ]
    )

    assert len(rows) == 2
    assert {row["session"] for row in rows} == {None, "london"}


def _event(event_id, family, session, direction="LONG", lookback=20):
    signal = SimpleNamespace(
        instrument="EURUSD",
        benchmark_family=family,
        signal_timeframe="15m",
        session=session,
        direction=SimpleNamespace(name=direction),
        lookback=lookback,
    )
    return SimpleNamespace(candidate_event_id=event_id, signal=signal)


def test_partition_is_deterministic_complete_and_supports_none_session():
    events = tuple(
        _event(f"a-{index}", "bollinger", None) for index in range(3)
    ) + tuple(_event(f"b-{index}", "vwap", "london") for index in range(5))
    first = partition_candidate_groups(events, 2)
    assert first == partition_candidate_groups(events, 2)
    selected = [event for _, shard in first for event in shard]
    assert [event.candidate_event_id for event in selected] == [
        event.candidate_event_id for event in events
    ]
    assert len({event.candidate_event_id for event in selected}) == len(events)
    ownership = {
        stable_group_key(event): shard_index
        for shard_index, (_, shard) in enumerate(first)
        for event in shard
    }
    assert len(ownership) == 2


def test_partition_operates_on_post_dedup_candidate_events():
    candidates = (
        _event("before-rearm", "vwap", "london"),
        _event("after-rearm", "vwap", "london"),
    )
    shards = partition_candidate_groups(candidates, 1)
    assert [event.candidate_event_id for event in shards[0][1]] == [
        "before-rearm",
        "after-rearm",
    ]
