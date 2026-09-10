import hashlib
import json
from pathlib import Path

import pytest

from mr_lab.bollinger_incremental import BollingerIncrementalError, run
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID

REGISTRY = Path("configs/stage4a-2024-corpus-registry.json")
COSTS = Path("configs/stage4c-ftmo-cost-profile-v1.json")


def _candidate(event_id, family, timestamp, registry, lookback=20):
    identity = registry["instruments"]["EURUSD"]
    return {
        "candidate_event_id": event_id,
        "signal": {
            "instrument": "EURUSD",
            "signal_timestamp": timestamp,
            "benchmark_family": family,
            "signal_timeframe": "15m",
            "session": "london",
            "direction": "SHORT",
            "lookback": lookback,
            "source_corpus_id": identity["corpus_id"],
            "assembled_dataset_id": identity["assembled_dataset_id"],
        },
    }


def _trade(event_id, family, gross, *, lookback=20, tp=0.75, sl=0.25, stop=60):
    return {
        "instrument": "EURUSD",
        "benchmark_family": family,
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "SHORT",
        "lookback": lookback,
        "signal_threshold": 2.0,
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "entry_mode": "immediate",
        "tp_target_fraction": tp,
        "sl_extension_fraction": sl,
        "time_stop_minutes": stop,
        "candidate_event_id": event_id,
        "complete": True,
        "gross_return_pips_adverse_first": gross,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authenticated_shard(directory, candidates, trades, registry, *, index=0, count=1):
    directory.mkdir()
    candidate_path = directory / "candidate-events.jsonl"
    trade_path = directory / "trades.jsonl"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(trade_path, trades)
    identity = registry["instruments"]["EURUSD"]
    groups = [["EURUSD", "bollinger", "15m", "london", "SHORT", 20]]
    manifest = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "source_commit_sha": "a" * 40,
        "instrument": "EURUSD",
        "registry_identity": "registry-fixture",
        "corpus_id": identity["corpus_id"],
        "assembled_dataset_id": identity["assembled_dataset_id"],
        "source_workflow_run_id": 123,
        "source_artifact_id": 456,
        "raw_artifact_name": f"stage4b-raw-shard-{index}",
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "shard_index": index,
        "shard_count": count,
        "full_candidate_count": len(candidates),
        "shard_candidate_count": len(candidates),
        "full_group_keys": groups,
        "group_keys": groups,
        "file_sha256": {
            "candidate-events.jsonl": _hash(candidate_path),
            "trades.jsonl": _hash(trade_path),
        },
        "row_counts": {
            "candidate-events.jsonl": len(candidates),
            "trades.jsonl": len(trades),
        },
    }
    (directory / "shard-manifest.json").write_text(json.dumps(manifest) + "\n")
    return manifest


def _run(tmp_path, candidates, trades):
    registry = json.loads(REGISTRY.read_text())
    source = tmp_path / "source"
    _authenticated_shard(source, candidates, trades, registry)
    output = tmp_path / "output"
    return (*run([source], output, REGISTRY, COSTS), output, source)


def test_full_module_a_classification_and_overlap_levels(tmp_path):
    registry = json.loads(REGISTRY.read_text())
    candidates = [
        _candidate("bb-canonical", "bollinger", "2024-01-02T09:15:00+00:00", registry),
        _candidate(
            "canonical", "vwap-canonical-m1", "2024-01-02T09:15:00+00:00", registry
        ),
        _candidate("bb-l20", "bollinger", "2024-01-02T09:30:00+00:00", registry),
        _candidate("vwap-l40", "vwap", "2024-01-02T09:30:00+00:00", registry, 40),
        _candidate("native", "vwap", "2024-02-02T09:45:00+00:00", registry),
        _candidate(
            "canonical-same", "vwap-canonical-m1", "2024-02-02T09:45:00+00:00", registry
        ),
    ]
    trades = [
        _trade("bb-canonical", "bollinger", 1),
        _trade("canonical", "vwap-canonical-m1", 2),
        _trade("bb-l20", "bollinger", -1),
        _trade("vwap-l40", "vwap", -2, lookback=40),
        _trade("native", "vwap", 1),
        _trade("canonical-same", "vwap-canonical-m1", 1),
    ]
    results, overlap, output, _source = _run(tmp_path, candidates, trades)
    assert not [row for row in results if row["event_class"] == "bollinger-only"]
    strict20 = next(
        row
        for row in overlap
        if row["overlap_definition"] == "strict-specification" and row["lookback"] == 20
    )
    execution = next(
        row for row in overlap if row["overlap_definition"] == "execution-level"
    )
    assert strict20["intersection_count"] == 1
    assert execution["intersection_count"] == 2
    assert (
        execution["vwap_module_a_candidates"] == 3
    )  # native+canonical timestamp is one
    assert (
        "vwap-canonical-m1" in (output / "bollinger-incremental-matrix.csv").read_text()
    )


def test_baseline_authenticated_real_format_fixture_passes(tmp_path):
    registry = json.loads(REGISTRY.read_text())
    candidates = [_candidate("bb", "bollinger", "2024-01-02T09:15:00+00:00", registry)]
    results, _overlap, output, _source = _run(
        tmp_path, candidates, [_trade("bb", "bollinger", 3)]
    )
    assert (
        results
        and json.loads((output / "execution-audit.json").read_text())[
            "selected_trade_rows"
        ]
        == 1
    )


@pytest.mark.parametrize("filename", ["candidate-events.jsonl", "trades.jsonl"])
def test_tampered_raw_file_fails_authentication(tmp_path, filename):
    registry = json.loads(REGISTRY.read_text())
    candidate = _candidate("bb", "bollinger", "2024-01-02T09:15:00+00:00", registry)
    _results, _overlap, _output, source = _run(
        tmp_path, [candidate], [_trade("bb", "bollinger", 3)]
    )
    with (source / filename).open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(BollingerIncrementalError, match="authenticated hash mismatch"):
        run([source], tmp_path / "again", REGISTRY, COSTS)


def test_mismatched_manifest_fails(tmp_path):
    registry = json.loads(REGISTRY.read_text())
    candidate = _candidate("bb", "bollinger", "2024-01-02T09:15:00+00:00", registry)
    source = tmp_path / "source"
    manifest = _authenticated_shard(
        source, [candidate], [_trade("bb", "bollinger", 3)], registry
    )
    manifest["corpus_id"] = "sha256:wrong"
    (source / "shard-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BollingerIncrementalError, match="corpus_id"):
        run([source], tmp_path / "out", REGISTRY, COSTS)


def test_incomplete_shard_set_fails(tmp_path):
    registry = json.loads(REGISTRY.read_text())
    candidate = _candidate("bb", "bollinger", "2024-01-02T09:15:00+00:00", registry)
    source = tmp_path / "source"
    _authenticated_shard(
        source, [candidate], [_trade("bb", "bollinger", 3)], registry, count=2
    )
    with pytest.raises(BollingerIncrementalError, match="incomplete or mixed"):
        run([source], tmp_path / "out", REGISTRY, COSTS)


def test_rejects_any_non_2024_candidate(tmp_path):
    registry = json.loads(REGISTRY.read_text())
    candidate = _candidate("sealed", "bollinger", "2025-01-02T09:15:00+00:00", registry)
    with pytest.raises(BollingerIncrementalError, match="only frozen 2024"):
        _run(tmp_path, [candidate], [])
