import csv
import hashlib
import json
import tracemalloc
from datetime import datetime
from pathlib import Path

import pytest

import mr_lab.stage4_discovery_atlas as atlas_module
from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4_discovery_atlas import (
    DiscoveryAtlasError,
    _cell_metrics,
    _create_spool,
    _manifest_identity_key,
    _scenario_values,
    _stream_execution_family_rows,
    _stream_overlap_counts,
    _stream_spooled_cell_metrics,
    build_atlas,
)
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import CostProfile

PROFILE = Path("configs/stage4c-ftmo-cost-profile-v1.json")


def _dump(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _registry(tmp_path, instruments=("EURUSD", "AUDUSD")):
    entries = {}
    for instrument in instruments:
        entries[instrument] = {
            "instrument": instrument,
            "corpus_id": f"corpus-{instrument}",
            "assembled_dataset_id": f"dataset-{instrument}",
            "requested_start_date": "2024-01-01",
            "requested_end_date": "2024-12-31",
            "source_workflow_run_id": f"run-{instrument}",
            "source_artifact_id": f"artifact-{instrument}",
            "source_artifact_name": f"raw-{instrument}",
            "verification_status": "verified",
        }
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "registry_schema_version": "stage-4a-2024-corpus-registry-v1",
                "instruments": entries,
            }
        )
    )
    return path


def _trade(
    event_id,
    *,
    instrument="EURUSD",
    gross=4.0,
    timestamp=None,
    benchmark="vwap",
    lookback=20,
    tp=0.75,
    sl=0.25,
    stop=60,
    session="london",
    timeframe="15m",
    direction="SHORT",
    filter_family="none",
    filter_spec_id="none-v1",
):
    return {
        "instrument": instrument,
        "signal_timeframe": timeframe,
        "session": session,
        "direction": direction,
        "benchmark_family": benchmark,
        "lookback": lookback,
        "signal_threshold": 2.0,
        "filter_family": filter_family,
        "filter_spec_id": filter_spec_id,
        "entry_mode": "immediate",
        "tp_target_fraction": tp,
        "sl_extension_fraction": sl,
        "time_stop_minutes": stop,
        "candidate_event_id": event_id,
        "complete": True,
        "gross_return_pips_adverse_first": gross,
        "gross_return_pips_favorable_first": gross,
    }


def _shard(
    tmp_path,
    registry,
    trades,
    *,
    instrument="EURUSD",
    index=0,
    count=1,
    filter_family="none",
    filter_spec_id="none-v1",
    process_spec_id=None,
    label="evidence",
    full_count=None,
    group_keys=None,
    full_groups=None,
):
    root = tmp_path / f"{label}-{instrument}-{index}"
    root.mkdir()
    unique_ids = list(dict.fromkeys(row["candidate_event_id"] for row in trades))
    events = []
    for position, event_id in enumerate(unique_ids):
        timestamp = trades[
            next(
                i
                for i, row in enumerate(trades)
                if row["candidate_event_id"] == event_id
            )
        ].get("_timestamp", f"2024-02-{position + 1:02d}T10:00:00+00:00")
        events.append(
            {
                "candidate_event_id": event_id,
                "signal": {
                    "signal_timestamp": timestamp,
                    "instrument": instrument,
                    "source_corpus_id": f"corpus-{instrument}",
                    "assembled_dataset_id": f"dataset-{instrument}",
                },
            }
        )
    clean_trades = [
        {k: v for k, v in row.items() if k != "_timestamp"} for row in trades
    ]
    _dump(root / "candidate-events.jsonl", events)
    _dump(root / "trades.jsonl", clean_trades)
    registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "source_commit_sha": "a" * 40,
        "instrument": instrument,
        "corpus_id": f"corpus-{instrument}",
        "assembled_dataset_id": f"dataset-{instrument}",
        "filter_family": filter_family,
        "filter_spec_id": filter_spec_id,
        "process_spec_id": process_spec_id,
    }
    (root / "execution-audit.json").write_text(json.dumps(audit))
    raw_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("candidate-events.jsonl", "trades.jsonl", "execution-audit.json")
    }
    groups = group_keys if group_keys is not None else [[f"g{index}"]]
    universe = full_groups if full_groups is not None else groups
    manifest = {
        **audit,
        "schema_version": "stage4b-runtime-shard-v1",
        "registry_identity": registry_sha,
        "source_workflow_run_id": f"run-{instrument}",
        "source_artifact_id": f"artifact-{instrument}",
        "source_artifact_name": f"raw-{instrument}",
        "shard_count": count,
        "shard_index": index,
        "full_candidate_count": full_count or len(unique_ids),
        "shard_candidate_count": len(unique_ids),
        "full_group_keys": universe,
        "group_keys": groups,
        "file_sha256": raw_hashes,
        "row_counts": {
            "candidate-events.jsonl": len(events),
            "trades.jsonl": len(clean_trades),
        },
    }
    (root / "shard-manifest.json").write_text(json.dumps(manifest))
    return root


def _rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _run(tmp_path, trades, **kwargs):
    registry = _registry(tmp_path)
    source = _shard(tmp_path, registry, trades, **kwargs)
    output = tmp_path / "atlas"
    build_atlas([source], output, PROFILE, registry)
    return output, registry, source


def test_exit_cells_do_not_multiply_frequency_or_pool_performance(tmp_path):
    trades = [
        _trade("event", gross=value, tp=tp, sl=sl, stop=stop)
        for value, tp, sl, stop in [
            (8, 0.75, 0.25, 60),
            (-8, 0.75, 0.5, 60),
            (8, 1, 0.25, 60),
            (-8, 1, 0.5, 60),
            (8, 0.75, 0.25, 120),
            (-8, 0.75, 0.5, 120),
            (8, 1, 0.25, 120),
            (-8, 1, 0.5, 120),
        ]
    ]
    output, _, _ = _run(tmp_path, trades)
    family = _rows(output / "setup-family-matrix.csv")[0]
    assert family["exit_cell_count"] == "8"
    assert float(family["trades_per_month"]) == pytest.approx(1 / 12)
    costs = _rows(output / "cost-robustness.csv")
    assert (
        len({row["expectancy"] for row in costs if row["cost_scenario"] == "mean+0"})
        == 2
    )
    assert family["mean_0_positive_exit_cells"] == "4"
    assert family["mean_0_positive_fraction"] == "0.5"


def test_plateau_uses_distribution_of_per_cell_expectancies(tmp_path):
    output, _, _ = _run(
        tmp_path,
        [
            _trade("same", gross=-8, tp=0.75),
            _trade("same", gross=8, tp=1.0),
        ],
    )
    family = _rows(output / "setup-family-matrix.csv")[0]
    cells = [
        float(row["expectancy"])
        for row in _rows(output / "cost-robustness.csv")
        if row["cost_scenario"] == "mean+0"
    ]
    assert float(family["mean_0_min_expectancy"]) == min(cells)
    assert float(family["mean_0_median_expectancy"]) == pytest.approx(sum(cells) / 2)
    assert float(family["mean_0_max_expectancy"]) == max(cells)


def test_signal_threshold_and_exit_parameters_define_exact_cells(tmp_path):
    first, second = _trade("a"), _trade("a", tp=1.0)
    second["signal_threshold"] = 2.5
    output, _, _ = _run(tmp_path, [first, second])
    rows = _rows(output / "cost-robustness.csv")
    assert {(row["signal_threshold"], row["tp_target_fraction"]) for row in rows} == {
        ("2.0", "0.75"),
        ("2.5", "1.0"),
    }


def test_execution_dedup_is_order_invariant_and_has_no_pl(tmp_path):
    registry = _registry(tmp_path)
    trades = [
        _trade("a", benchmark="vwap"),
        _trade("a", benchmark="vwap-canonical-m1", lookback=40),
    ]
    first = _shard(tmp_path, registry, trades, label="first")
    second = _shard(tmp_path, registry, list(reversed(trades)), label="second")
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    build_atlas([first], out1, PROFILE, registry)
    build_atlas([second], out2, PROFILE, registry)
    assert (out1 / "execution-family-matrix.csv").read_text() == (
        out2 / "execution-family-matrix.csv"
    ).read_text()
    row = _rows(out1 / "execution-family-matrix.csv")[0]
    assert row["unique_execution_opportunities"] == "1"
    assert not any(
        "pips" in key or "expectancy" in key or "profit" in key for key in row
    )


def test_zero_sum_trade_month_is_not_no_trade_month(tmp_path):
    trades = [_trade("a", gross=4), _trade("b", gross=-4)]
    trades[0]["_timestamp"] = "2024-02-01T10:00:00+00:00"
    trades[1]["_timestamp"] = "2024-02-02T10:00:00+00:00"
    rows = [
        row | {"timestamp": datetime.fromisoformat(row["_timestamp"])} for row in trades
    ]
    assert _cell_metrics(rows, [1.0, -1.0])["no_trade_months"] == 11


def test_losing_streak_is_chronological_not_input_order(tmp_path):
    trades = [
        _trade("third", gross=-3),
        _trade("first", gross=-3),
        _trade("second", gross=10),
    ]
    for row, day in zip(trades, (3, 1, 2), strict=True):
        row["_timestamp"] = f"2024-02-{day:02d}T10:00:00+00:00"
    output, _, _ = _run(tmp_path, trades)
    row = next(
        r
        for r in _rows(output / "cost-robustness.csv")
        if r["cost_scenario"] == "mean+0"
    )
    assert row["max_losing_streak"] == "1"


def test_cross_asset_support_contains_actual_instrument_evidence(tmp_path):
    registry = _registry(tmp_path)
    eur = _shard(tmp_path, registry, [_trade("eur", gross=5)], label="eur")
    aud = _shard(
        tmp_path,
        registry,
        [_trade("aud", instrument="AUDUSD", gross=-5)],
        instrument="AUDUSD",
        label="aud",
    )
    output = tmp_path / "atlas"
    build_atlas([eur, aud], output, PROFILE, registry)
    rows = _rows(output / "cross-asset-hypotheses.csv")
    assert {row["instrument"] for row in rows} == {"EURUSD", "AUDUSD"}
    assert all(
        set(json.loads(row["cross_asset_support"])) == {"EURUSD", "AUDUSD"}
        for row in rows
    )


def test_mixed_instrument_partial_shards_fail_per_run(tmp_path):
    registry = _registry(tmp_path)
    universe = [["g0"], ["g1"]]
    eur0 = _shard(
        tmp_path,
        registry,
        [_trade("e")],
        count=2,
        full_count=2,
        group_keys=[["g0"]],
        full_groups=universe,
        label="eur",
    )
    aud1 = _shard(
        tmp_path,
        registry,
        [_trade("a", instrument="AUDUSD")],
        instrument="AUDUSD",
        index=1,
        count=2,
        full_count=2,
        group_keys=[["g1"]],
        full_groups=universe,
        label="aud",
    )
    with pytest.raises(DiscoveryAtlasError, match="per-run"):
        build_atlas([eur0, aud1], tmp_path / "out", PROFILE, registry)


@pytest.mark.parametrize(
    ("field", "message"),
    [("corpus_id", "corpus"), ("assembled_dataset_id", "assembled_dataset")],
)
def test_registry_identity_mismatch_fails(tmp_path, field, message):
    registry = _registry(tmp_path)
    source = _shard(tmp_path, registry, [_trade("a")])
    manifest_path = source / "shard-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DiscoveryAtlasError, match=message):
        build_atlas([source], tmp_path / "out", PROFILE, registry)


def test_local_checkpointed_registry_provenance_authenticates(tmp_path):
    registry = _registry(tmp_path, instruments=("EURUSD",))
    value = json.loads(registry.read_text())
    entry = value["instruments"]["EURUSD"]
    entry.update(
        source_mode="local-checkpointed",
        source_acquisition_commit_sha="a" * 40,
        source_workflow_run_id=None,
        source_artifact_id=None,
    )
    registry.write_text(json.dumps(value))
    source = _shard(tmp_path, registry, [_trade("local")])
    manifest_path = source / "shard-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        source_mode="local-checkpointed",
        source_acquisition_commit_sha="a" * 40,
        source_workflow_run_id=None,
        source_artifact_id=None,
    )
    manifest_path.write_text(json.dumps(manifest))

    build_atlas([source], tmp_path / "out", PROFILE, registry)


@pytest.mark.parametrize("target", ["trades.jsonl", "shard-manifest.json"])
def test_tampered_manifest_or_raw_fails(tmp_path, target):
    registry = _registry(tmp_path)
    source = _shard(tmp_path, registry, [_trade("a")])
    (source / target).write_text("{}\n")
    with pytest.raises((DiscoveryAtlasError, KeyError)):
        build_atlas([source], tmp_path / "out", PROFILE, registry)


def test_baseline_vwap_is_raw_overlap_not_module_a(tmp_path):
    output, _, _ = _run(tmp_path, [_trade("a")])
    row = _rows(output / "overlap-with-module-a.csv")[0]
    assert row["raw_vwap_signal_family_intersection_count"] == "1"
    assert row["actual_frozen_module_a_available"] == "False"


def test_explicit_module_a_role_is_overlap_only(tmp_path):
    registry = _registry(tmp_path)
    baseline = _shard(tmp_path, registry, [_trade("base")], label="base")
    spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
    module_trade = _trade(
        "base",
        filter_family="ornstein-uhlenbeck",
        filter_spec_id=spec.filter_spec_id,
    )
    module = _shard(
        tmp_path,
        registry,
        [module_trade],
        label="module",
        filter_family="ornstein-uhlenbeck",
        filter_spec_id=spec.filter_spec_id,
        process_spec_id=spec.process_spec.process_spec_id,
    )
    output = tmp_path / "atlas"
    build_atlas([baseline], output, PROFILE, registry, module_a_dirs=[module])
    assert len(_rows(output / "cost-robustness.csv")) == 4
    overlap = _rows(output / "overlap-with-module-a.csv")[0]
    assert overlap["actual_frozen_module_a_available"] == "True"
    assert overlap["raw_vwap_signal_family_intersection_count"] == "1"
    assert overlap["raw_vwap_signal_family_overlap_rate"] == "1.0"
    assert overlap["actual_frozen_module_a_intersection_count"] == "1"
    assert overlap["actual_frozen_module_a_overlap_rate"] == "1.0"
    assert overlap["unique_to_setup_vs_module_a"] == "0"
    assert overlap["module_a_only_count"] == "0"


def test_no_automatic_promising_or_fail_thresholds(tmp_path):
    output, _, _ = _run(tmp_path, [_trade("a")])
    row = _rows(output / "candidate-shortlist.csv")[0]
    assert (
        row["evidence_status"] == "descriptive_only_no_preregistered_triage_thresholds"
    )
    assert not any("category" in key or "score" in key or "rank" in key for key in row)


def test_sealed_path_rejected_before_filesystem_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Path, "is_file", lambda self: (_ for _ in ()).throw(AssertionError("inspected"))
    )
    with pytest.raises(DiscoveryAtlasError, match="sealed-period"):
        build_atlas(
            [tmp_path / "sealed-2025"], tmp_path / "out", PROFILE, tmp_path / "registry"
        )


def test_duplicate_trade_identity_is_rejected(tmp_path):
    registry = _registry(tmp_path)
    source = _shard(tmp_path, registry, [_trade("same"), _trade("same")])
    with pytest.raises(DiscoveryAtlasError, match="duplicate trade identity"):
        build_atlas([source], tmp_path / "out", PROFILE, registry)


def test_manifest_identity_keys_are_compact_deterministic_and_distinct():
    common = {"full_group_keys": [[f"group-{index}"] for index in range(10_000)]}
    first = common | {"source_commit_sha": "a" * 40}
    second = common | {"source_commit_sha": "b" * 40}

    assert _manifest_identity_key(first) == _manifest_identity_key(first)
    assert _manifest_identity_key(first) != _manifest_identity_key(second)
    assert len(_manifest_identity_key(first)) == 64


def test_spooled_trade_stores_digest_not_full_manifest_identity(tmp_path):
    registry = _registry(tmp_path)
    full_groups = [[f"large-logical-group-{index}"] for index in range(10_000)]
    source = _shard(
        tmp_path,
        registry,
        [_trade("event")],
        full_groups=full_groups,
        group_keys=full_groups,
    )
    database = _create_spool(tmp_path / "spool.sqlite3")
    profile = CostProfile.load(PROFILE)
    try:
        atlas_module._collect(
            [source],
            json.loads(registry.read_text()),
            hashlib.sha256(registry.read_bytes()).hexdigest(),
            "evidence",
            database,
            profile,
        )
        run_key = database.execute("SELECT run_key FROM trades").fetchone()[0]
    finally:
        database.close()

    assert len(run_key) == 64
    assert run_key == _manifest_identity_key(
        json.loads((source / "shard-manifest.json").read_text())
    )
    assert "large-logical-group" not in run_key


def test_missing_candidate_reference_is_rejected(tmp_path):
    registry = _registry(tmp_path)
    source = _shard(tmp_path, registry, [_trade("known")])
    trades_path = source / "trades.jsonl"
    trade = json.loads(trades_path.read_text())
    trade["candidate_event_id"] = "missing"
    _dump(trades_path, [trade])
    manifest_path = source / "shard-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"]["trades.jsonl"] = hashlib.sha256(
        trades_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DiscoveryAtlasError, match="trade provenance mismatch"):
        build_atlas([source], tmp_path / "out", PROFILE, registry)


def test_spool_is_cleaned_after_aggregation_exception(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    source = _shard(tmp_path, registry, [_trade("a")])
    spool = tmp_path / "spool"

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic aggregation failure")

    monkeypatch.setattr(
        "mr_lab.stage4_discovery_atlas._stream_spooled_cell_metrics", fail
    )
    with pytest.raises(RuntimeError, match="synthetic aggregation failure"):
        build_atlas([source], tmp_path / "out", PROFILE, registry, spool_dir=spool)
    assert list(spool.iterdir()) == []


def test_large_single_cell_decodes_each_trade_payload_only_once(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    trades = [
        _trade(f"event-{index}", gross=(-1) ** index * 4)
        | {"_timestamp": "2024-02-01T10:00:00+00:00"}
        for index in range(2000)
    ]
    source = _shard(tmp_path, registry, trades)

    original_loads = atlas_module.json.loads
    decoded_trades = 0

    def counting_loads(value, *args, **kwargs):
        nonlocal decoded_trades
        if isinstance(value, str) and "gross_return_pips_adverse_first" in value:
            decoded_trades += 1
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(atlas_module.json, "loads", counting_loads)
    output = tmp_path / "out"
    build_atlas([source], output, PROFILE, registry, spool_dir=tmp_path / "spool")
    assert _rows(output / "cost-robustness.csv")[0]["trade_count"] == "2000"
    assert decoded_trades == len(trades)
    assert list((tmp_path / "spool").iterdir()) == []


def test_spool_schema_does_not_retain_serialized_trade_payloads(tmp_path):
    database = _create_spool(tmp_path / "spool.sqlite3")
    try:
        columns = {
            row[1] for row in database.execute("PRAGMA table_info(trades)").fetchall()
        }
    finally:
        database.close()
    assert "payload" not in columns
    assert {
        "cell_key",
        "setup_key",
        "execution_family",
        "execution_key",
        "gross_pips",
        "net_mean",
        "net_p75",
        "net_p90",
        "net_p95",
    } <= columns


def test_fused_cell_metrics_match_previous_implementation(tmp_path):
    trades = [
        _trade("third", gross=-3),
        _trade("first", gross=-2),
        _trade("second", gross=7),
    ]
    for row, day in zip(trades, (3, 1, 2), strict=True):
        row["_timestamp"] = f"2024-02-{day:02d}T10:00:00+00:00"
    output, _, _ = _run(tmp_path, trades)
    rows = [
        row | {"timestamp": datetime.fromisoformat(row["_timestamp"])} for row in trades
    ]
    profile = CostProfile.load(PROFILE)
    actual = {
        row["cost_scenario"]: row for row in _rows(output / "cost-robustness.csv")
    }
    gross = _cell_metrics(
        rows, [float(row["gross_return_pips_adverse_first"]) for row in rows]
    )
    for statistic, slippage in atlas_module.HEADLINE_COSTS:
        scenario = f"{statistic}+{slippage:g}"
        expected = _cell_metrics(
            rows, _scenario_values(rows, profile, statistic, slippage)
        )
        for field, value in expected.items():
            assert actual[scenario][field] == str(value)
        for field, value in gross.items():
            assert actual[scenario][f"gross_{field}"] == str(value)


def _insert_stream_trade(database, seq, cell, candidate, timestamp, values):
    database.execute(
        "INSERT INTO trades VALUES (?, 'evidence', ?, ?, ?, ?, 'execution', "
        "?, ?, ?, 'vwap', 20, ?, ?, ?, ?, 0)",
        (
            seq,
            f"run-{seq}",
            candidate,
            cell,
            cell,
            candidate,
            timestamp,
            *values,
        ),
    )


def test_streamed_cells_match_legacy_metrics_and_distinct_candidates(tmp_path):
    database = _create_spool(tmp_path / "stream.sqlite3")
    rows = [
        {
            "timestamp": datetime.fromisoformat("2024-02-03T10:00:00+00:00"),
            "candidate_event_id": "repeat",
        },
        {
            "timestamp": datetime.fromisoformat("2024-02-01T10:00:00+00:00"),
            "candidate_event_id": "repeat",
        },
        {
            "timestamp": datetime.fromisoformat("2024-02-02T10:00:00+00:00"),
            "candidate_event_id": "other",
        },
    ]
    gross = [-3.0, -2.0, 7.0]
    for seq, (row, value) in enumerate(zip(rows, gross, strict=True), 1):
        _insert_stream_trade(
            database,
            seq,
            "cell",
            row["candidate_event_id"],
            row["timestamp"].isoformat(),
            (value, value - 1, value - 2, value - 3, value - 4),
        )
    actual = dict(_stream_spooled_cell_metrics(database, progress_rows=0))["cell"]
    assert actual["gross"] == _cell_metrics(rows, gross)
    assert actual["gross"]["candidate_count"] == 2
    assert actual["gross"]["max_losing_streak"] == 1
    for offset, scenario in enumerate(("mean", "p75", "p90", "p95"), 1):
        assert actual[scenario] == _cell_metrics(rows, [v - offset for v in gross])
    database.close()


def test_streamed_cells_preserve_seq_order_float_accumulation(tmp_path):
    database = _create_spool(tmp_path / "float-order.sqlite3")
    timestamps = (
        "2024-01-03T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
        "2024-01-04T00:00:00+00:00",
    )
    rows = [
        {
            "timestamp": datetime.fromisoformat(timestamp),
            "candidate_event_id": f"candidate-{seq}",
        }
        for seq, timestamp in enumerate(timestamps, 1)
    ]
    values = [1e16, 1.0, 1.0, -1e16]
    for seq, (row, value) in enumerate(zip(rows, values, strict=True), 1):
        _insert_stream_trade(
            database,
            seq,
            "cell",
            row["candidate_event_id"],
            row["timestamp"].isoformat(),
            (value,) * 5,
        )
    actual = dict(_stream_spooled_cell_metrics(database, progress_rows=0))["cell"]
    chronological = [value for _, value in sorted(zip(timestamps, values, strict=True))]
    seq_total = chronological_total = 0.0
    for value in values:
        seq_total += value
    for value in chronological:
        chronological_total += value
    expected = _cell_metrics(rows, values) | {
        "total_pips": seq_total,
        "profit_factor": 1.0,
    }
    assert seq_total == 0.0
    assert chronological_total == 2.0
    assert actual["gross"] == expected
    assert all(
        actual[scenario] == expected for scenario in ("mean", "p75", "p90", "p95")
    )
    database.close()


def test_streamed_aggregation_uses_one_global_trade_cursor(tmp_path):
    database = _create_spool(tmp_path / "queries.sqlite3")
    for seq in range(20):
        _insert_stream_trade(
            database,
            seq,
            f"cell-{seq:03d}",
            f"candidate-{seq}",
            "2024-01-01T00:00:00+00:00",
            (1.0,) * 5,
        )
    statements = []
    database.set_trace_callback(statements.append)
    assert len(list(_stream_spooled_cell_metrics(database, progress_rows=0))) == 20
    trade_selects = [
        sql for sql in statements if sql.startswith("SELECT") and "FROM trades" in sql
    ]
    assert len(trade_selects) == 1
    assert "cell_key=?" not in trade_selects[0]
    database.close()


def test_streamed_aggregation_memory_does_not_scale_with_cell_count(tmp_path):
    database = _create_spool(tmp_path / "memory.sqlite3")
    for seq in range(5_000):
        _insert_stream_trade(
            database,
            seq,
            f"cell-{seq:05d}",
            f"candidate-{seq}",
            "2024-01-01T00:00:00+00:00",
            (1.0,) * 5,
        )
    tracemalloc.start()
    for _ in _stream_spooled_cell_metrics(database, progress_rows=0):
        pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 2_000_000
    database.close()


def test_downstream_grouped_query_count_does_not_scale_with_families(tmp_path):
    database = _create_spool(tmp_path / "downstream-queries.sqlite3")
    for seq in range(50):
        _insert_stream_trade(
            database,
            seq,
            f"cell-{seq:03d}",
            f"candidate-{seq}",
            "2024-01-01T00:00:00+00:00",
            (1.0,) * 5,
        )
    statements = []
    database.set_trace_callback(statements.append)
    assert len(list(_stream_execution_family_rows(database))) == 1
    assert len(list(_stream_overlap_counts(database))) == 50
    selects = [statement for statement in statements if statement.startswith("SELECT")]
    assert len(selects) == 3
    assert not any("setup_key=?" in statement for statement in selects)
    database.close()
