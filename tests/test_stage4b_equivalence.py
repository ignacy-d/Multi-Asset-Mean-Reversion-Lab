import csv
import hashlib
import json
import lzma
import math
import struct
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import build_corpus_manifest
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
