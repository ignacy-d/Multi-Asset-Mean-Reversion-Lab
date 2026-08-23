import csv
import hashlib
import json

import pytest

from mr_lab.stage4b_reducer import COMPACT_CSVS, ShardReductionError, reduce_shards
from mr_lab.stage4b_runner import _sha256_file


def _make_shard(root, index, groups, full_groups):
    directory = root / f"shard-{index}"
    directory.mkdir()
    fields = ["instrument", "signal_timeframe", "executed_trade_count", "group"]
    for name in COMPACT_CSVS:
        with (directory / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for group in groups:
                writer.writerow(
                    {
                        "instrument": "EURUSD",
                        "signal_timeframe": "15m",
                        "executed_trade_count": 1,
                        "group": group[1],
                    }
                )
    summary = {
        "reporting_schema_version": "report-v1",
        "stage4b_methodology_id": "method-v1",
        "candidate_event_count": len(groups),
        "no_entry_count": 0,
        "target_already_passed_count": 0,
        "executed_trade_configuration_count": len(groups),
        "incomplete_count": 0,
        "ambiguous_count": 0,
        "filter_ineligible_count": 0,
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "report.md").write_text("fixture report\n")
    (directory / "execution-audit.json").write_text("{}\n")
    (directory / "candidate-events.jsonl").write_text("{}\n" * len(groups))
    (directory / "trades.jsonl").write_text("{}\n" * len(groups))
    files = sorted(directory.iterdir())
    manifest = {
        "source_commit_sha": "a" * 40,
        "stage4b_methodology_id": "method-v1",
        "instrument": "EURUSD",
        "registry_identity": "registry",
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "shard_count": 2,
        "full_candidate_count": len(full_groups),
        "shard_candidate_count": len(groups),
        "shard_index": index,
        "raw_artifact_name": f"raw-shard-{index}",
        "compact_artifact_name": f"compact-shard-{index}",
        "full_group_keys": full_groups,
        "group_keys": groups,
        "file_sha256": {path.name: _sha256_file(path) for path in files},
        "row_counts": {
            path.name: sum(1 for _ in path.open("rb"))
            for path in files
            if path.suffix in {".csv", ".jsonl"}
        },
    }
    (directory / "shard-manifest.json").write_text(json.dumps(manifest))
    (directory / "candidate-events.jsonl").unlink()
    (directory / "trades.jsonl").unlink()
    return directory


def test_streaming_sha256_matches_whole_content_digest(tmp_path):
    content = (bytes(range(256)) * 8193) + b"final"
    path = tmp_path / "large-enough-for-chunks.bin"
    path.write_bytes(content)
    assert _sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_reducer_concatenates_group_complete_quantile_rows(tmp_path):
    groups = [["EURUSD", "a"], ["EURUSD", "b"]]
    shards = [
        _make_shard(tmp_path, 0, groups[:1], groups),
        _make_shard(tmp_path, 1, groups[1:], groups),
    ]
    out = tmp_path / "out"
    reduce_shards(shards, out, 2)
    assert json.loads((out / "summary.json").read_text())["candidate_event_count"] == 2
    with (out / "stage4b-distributions.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 2
    audit = json.loads((out / "execution-audit.json").read_text())
    assert [item["shard_index"] for item in audit["shards"]] == [0, 1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_id", "wrong"),
        ("source_commit_sha", "b" * 40),
        ("stage4b_methodology_id", "wrong"),
    ],
)
def test_reducer_rejects_provenance_mismatch(tmp_path, field, value):
    groups = [["EURUSD", "a"], ["EURUSD", "b"]]
    shards = [
        _make_shard(tmp_path, 0, groups[:1], groups),
        _make_shard(tmp_path, 1, groups[1:], groups),
    ]
    path = shards[1] / "shard-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ShardReductionError, match="provenance mismatch"):
        reduce_shards(shards, tmp_path / "out", 2)


def test_reducer_rejects_missing_duplicate_and_overlapping_groups(tmp_path):
    groups = [["EURUSD", "a"], ["EURUSD", "b"]]
    first = _make_shard(tmp_path, 0, groups[:1], groups)
    with pytest.raises(ShardReductionError, match="missing"):
        reduce_shards([first], tmp_path / "missing", 2)
    second = _make_shard(tmp_path, 1, groups[1:], groups)
    path = second / "shard-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shard_index"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ShardReductionError, match="duplicate"):
        reduce_shards([first, second], tmp_path / "duplicate", 2)
    manifest["shard_index"] = 1
    manifest["group_keys"] = groups[:1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ShardReductionError, match="overlap"):
        reduce_shards([first, second], tmp_path / "overlap", 2)
