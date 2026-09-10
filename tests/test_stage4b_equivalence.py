import csv
import hashlib
import json
import lzma
import math
import struct
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import mr_lab.stage4b_runner as stage4b_runner
from mr_lab.ornstein_uhlenbeck import residual_observations
from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import build_corpus_manifest
from mr_lab.stage4b import ENTRY_MODES, SL_FRACTIONS, TIME_STOPS_MINUTES, TP_FRACTIONS
from mr_lab.stage4b_reducer import COMPACT_CSVS, reduce_shards
from mr_lab.stage4b_runner import STABLE_GROUP_FIELDS, run

RECORD = struct.Struct(">5if")
RESEARCH_SUMMARY_FIELDS = (
    "reporting_schema_version",
    "stage4b_methodology_id",
    "candidate_event_count",
    "no_entry_count",
    "target_already_passed_count",
    "executed_trade_configuration_count",
    "incomplete_count",
    "ambiguous_count",
    "filter_ineligible_count",
)


def _write_offline_fixture(root: Path) -> tuple[Path, Path]:
    day = date(2024, 1, 2)
    records = []
    for minute in range(360, 960):
        price = 100_000 + int(40 * math.sin(minute / 17) + 20 * math.sin(minute / 5))
        if 85 <= minute % 180 <= 90:
            price += 180
        if 115 <= minute % 180 <= 120:
            price -= 180
        previous = (
            100_000
            + int(40 * math.sin((minute - 1) / 17) + 20 * math.sin((minute - 1) / 5))
            if minute > 360
            else price
        )
        records.append(
            RECORD.pack(
                minute * 60,
                previous,
                price,
                min(previous, price) - 5,
                max(previous, price) + 5,
                1.0,
            )
        )
    raw = lzma.compress(b"".join(records))
    end = date(2024, 12, 31)
    current = date(2024, 1, 1)
    absent = []
    while current <= end:
        if current != day:
            absent.append(current)
        current += timedelta(days=1)
    manifest = build_corpus_manifest(
        date(2024, 1, 1), end, [DailyPayload(day, raw)], absent
    )
    corpus = root / "corpus"
    corpus.mkdir()
    stem = corpus / f"EURUSD-{day.isoformat()}-M1-BID"
    stem.with_suffix(".bi5").write_bytes(raw)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "requested_date": day.isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    (corpus / "corpus-manifest.json").write_text(manifest.to_json())

    registry = json.loads(Path("configs/stage4a-2024-corpus-registry.json").read_text())
    registry["instruments"]["EURUSD"] = {
        "instrument": "EURUSD",
        "source_workflow_run_id": 1,
        "source_artifact_id": 2,
        "source_artifact_name": "dukascopy-EURUSD-m1-bid-2024-full-year",
        "corpus_id": manifest.corpus_id,
        "assembled_dataset_id": manifest.dataset.manifest.dataset_id,
        "requested_start_date": "2024-01-01",
        "requested_end_date": "2024-12-31",
        "verification_status": "verified",
    }
    registry_path = root / "registry.json"
    registry_path.write_text(json.dumps(registry, sort_keys=True))
    return corpus, registry_path


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _csv_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _canonical_rows(rows):
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def test_real_runner_unsharded_equals_two_shards_plus_reducer(tmp_path):
    corpus, registry = _write_offline_fixture(tmp_path)
    unsharded = tmp_path / "unsharded"
    shard_dirs = [tmp_path / f"shard-{index}" for index in range(2)]
    reduced = tmp_path / "reduced"

    run(corpus, unsharded, "EURUSD", registry)
    for index, directory in enumerate(shard_dirs):
        run(
            corpus,
            directory,
            "EURUSD",
            registry,
            shard_index=index,
            shard_count=2,
        )
    reduce_shards(shard_dirs, reduced, 2)

    expected_candidates = _jsonl(unsharded / "candidate-events.jsonl")
    assert all(
        not (
            {"eligibility", "filter_family", "filter_spec_id", "filter_metadata"}
            & row.keys()
        )
        for row in expected_candidates
    )
    actual_candidates = [
        row
        for directory in shard_dirs
        for row in _jsonl(directory / "candidate-events.jsonl")
    ]
    expected_ids = [row["candidate_event_id"] for row in expected_candidates]
    actual_ids = [row["candidate_event_id"] for row in actual_candidates]
    assert len(actual_ids) == len(set(actual_ids)) == len(expected_ids)
    assert set(actual_ids) == set(expected_ids)
    assert _canonical_rows(actual_candidates) == _canonical_rows(expected_candidates)

    signals = [row["signal"] for row in expected_candidates]
    assert len({row["benchmark_family"] for row in signals}) >= 2
    assert {row["session"] for row in signals} >= {None, "london"}
    assert {row["direction"] for row in signals} == {"LONG", "SHORT"}
    assert len({row["lookback"] for row in signals}) >= 2
    stable_counts = Counter(
        tuple(row[field] for field in STABLE_GROUP_FIELDS) for row in signals
    )
    assert max(stable_counts.values()) > 1  # A real group re-arms and emits again.

    expected_trades = _jsonl(unsharded / "trades.jsonl")
    actual_trades = [
        row for directory in shard_dirs for row in _jsonl(directory / "trades.jsonl")
    ]
    assert _canonical_rows(actual_trades) == _canonical_rows(expected_trades)
    assert {row["entry_mode"] for row in expected_trades} >= {
        "immediate",
        "m1-reclaim-p0",
    }
    assert len({row["tp_target_fraction"] for row in expected_trades}) > 1
    assert len({row["sl_extension_fraction"] for row in expected_trades}) > 1
    assert len({row["time_stop_minutes"] for row in expected_trades}) > 1
    baseline_matrix = _csv_rows(unsharded / "trade-matrix.csv")
    assert {row["entry_mode"] for row in baseline_matrix} == set(ENTRY_MODES)
    assert {float(row["tp_target_fraction"]) for row in baseline_matrix} == set(
        TP_FRACTIONS
    )
    assert {
        None
        if row["sl_extension_fraction"] == ""
        else float(row["sl_extension_fraction"])
        for row in baseline_matrix
    } == set(SL_FRACTIONS)
    assert {int(row["time_stop_minutes"]) for row in baseline_matrix} == set(
        TIME_STOPS_MINUTES
    )
    baseline_audit = json.loads((unsharded / "execution-audit.json").read_text())
    assert (baseline_audit["filter_family"], baseline_audit["filter_spec_id"]) == (
        "none",
        "none-v1",
    )
    assert _csv_rows(unsharded / "first-passage-distributions.csv")

    for name in COMPACT_CSVS:
        assert _canonical_rows(_csv_rows(reduced / name)) == _canonical_rows(
            _csv_rows(unsharded / name)
        ), name

    expected_summary = json.loads((unsharded / "summary.json").read_text())
    actual_summary = json.loads((reduced / "summary.json").read_text())
    assert {field: actual_summary[field] for field in RESEARCH_SUMMARY_FIELDS} == {
        field: expected_summary[field] for field in RESEARCH_SUMMARY_FIELDS
    }
    assert expected_summary["executed_trade_configuration_count"] > 0
    assert expected_summary["no_entry_count"] > 0
    assert expected_summary["target_already_passed_count"] > 0

    manifests = [
        json.loads((directory / "shard-manifest.json").read_text())
        for directory in shard_dirs
    ]
    group_owners = {}
    for manifest in manifests:
        for group in manifest["group_keys"]:
            key = tuple(group)
            assert key not in group_owners
            group_owners[key] = manifest["shard_index"]
    assert list(group_owners) == [tuple(key) for key in manifests[0]["full_group_keys"]]
    assert all(len(group) == len(STABLE_GROUP_FIELDS) for group in group_owners)

    # Raw shard order is operationally group-contiguous, while the unsharded file
    # is time ordered. No research fields are excluded: complete parsed semantic
    # dictionaries are compared after deterministic canonical sorting.


def test_filtered_runner_internal_state_slice_and_shard_reducer_parity(
    tmp_path, monkeypatch
):
    corpus, registry = _write_offline_fixture(tmp_path)
    original_builder = stage4b_runner.build_candidate_ou_states
    original_assemble = stage4b_runner.assemble_signal_states
    calls = []

    def with_frozen_scope_candidate(dataset, manifest):
        states = list(original_assemble(dataset, manifest))
        selected = next(
            index
            for index, state in enumerate(states)
            if str(state.signal_timeframe) == "15m"
            and state.session == "london"
            and state.direction.name == "SHORT"
            and state.benchmark_family in {"vwap", "vwap-canonical-m1"}
            and state.lookback in {20, 40}
            and state.p0 > state.e0
        )
        states[selected] = replace(states[selected], z=2.1, qualifying=True)
        return tuple(states)

    def eligible_internal_states(signal_states, process_spec, required):
        # The runner supplies its complete internal SignalState history, rather than
        # a path or externally authenticated OU artifact.
        calls.append((signal_states, process_spec.process_spec_id, required))
        states = original_builder(signal_states, process_spec, required)
        by_process = {
            state.process_id: source
            for source in signal_states
            for state in states
            if state.available_at == source.timestamp
            and state.process_id == residual_observations((source,))[0].process_id
        }
        return tuple(
            replace(
                state,
                status="valid",
                invalid_reason=None,
                half_life_minutes=120.0,
                ornstein_uhlenbeck_score=2.0,
            )
            if (
                (source := by_process.get(state.process_id)) is not None
                and str(source.signal_timeframe) == "15m"
                and source.session == "london"
                and source.direction.name == "SHORT"
                and source.benchmark_family in {"vwap", "vwap-canonical-m1"}
                and source.lookback in {20, 40}
            )
            else state
            for state in states
        )

    monkeypatch.setattr(
        stage4b_runner, "build_candidate_ou_states", eligible_internal_states
    )
    monkeypatch.setattr(
        stage4b_runner, "assemble_signal_states", with_frozen_scope_candidate
    )
    unsharded = tmp_path / "filtered-unsharded"
    shards = [tmp_path / f"filtered-shard-{index}" for index in range(2)]
    reduced = tmp_path / "filtered-reduced"
    selected = "frozen-ou-crossasset-v1"
    run(corpus, unsharded, "EURUSD", registry, eligibility_filter=selected)
    for index, directory in enumerate(shards):
        run(
            corpus,
            directory,
            "EURUSD",
            registry,
            eligibility_filter=selected,
            shard_index=index,
            shard_count=2,
        )
    reduce_shards(shards, reduced, 2)

    assert len(calls) == 3 and all(call[0] and call[2] for call in calls)
    candidates = _jsonl(unsharded / "candidate-events.jsonl")
    assert candidates and all(
        {"eligibility", "filter_family", "filter_spec_id", "filter_metadata"}
        <= row.keys()
        for row in candidates
    )
    eligible_ids = {
        row["candidate_event_id"] for row in candidates if row["eligibility"]
    }
    assert eligible_ids
    assert all(
        not row["eligibility"]
        for row in candidates
        if row["signal"]["session"] != "london"
    )
    trades = _jsonl(unsharded / "trades.jsonl")
    assert {row["candidate_event_id"] for row in trades} == eligible_ids
    assert {row["entry_mode"] for row in trades} == {"immediate"}
    assert {row["tp_target_fraction"] for row in trades} == {0.75, 1.0}
    assert {row["sl_extension_fraction"] for row in trades} == {0.25, 0.5}
    assert {row["time_stop_minutes"] for row in trades} == {60, 120}
    assert len(trades) == 8 * len(eligible_ids)

    audit = json.loads((unsharded / "execution-audit.json").read_text())
    assert audit["filter_spec_id"] and audit["process_spec_id"]
    assert all(row["filter_family"] == audit["filter_family"] for row in trades)
    assert all(row["filter_spec_id"] == audit["filter_spec_id"] for row in trades)
    manifests = [
        json.loads((directory / "shard-manifest.json").read_text())
        for directory in shards
    ]
    assert {item["filter_family"] for item in manifests} == {audit["filter_family"]}
    assert {item["filter_spec_id"] for item in manifests} == {audit["filter_spec_id"]}
    for name in COMPACT_CSVS:
        assert _canonical_rows(_csv_rows(reduced / name)) == _canonical_rows(
            _csv_rows(unsharded / name)
        )
