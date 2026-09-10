import csv
import hashlib
import json
from pathlib import Path

import pytest

from mr_lab.stage4_discovery_atlas import DiscoveryAtlasError, build_atlas
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID


def _dump(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(
    tmp_path,
    *,
    instrument="EURUSD",
    session="london",
    timeframe="15m",
    direction="SHORT",
    benchmark="vwap",
    lookback=20,
    filter_family="none",
):
    root = tmp_path / (
        f"input-{instrument}-{session}-{timeframe}-{direction}-{benchmark}-{lookback}"
    )
    root.mkdir()
    event_id = f"event-{instrument}-{session}-{timeframe}-{direction}-{lookback}"
    event = {
        "candidate_event_id": event_id,
        "signal": {
            "signal_timestamp": "2024-02-01T10:00:00+00:00",
            "instrument": instrument,
            "source_corpus_id": f"corpus-{instrument}",
            "assembled_dataset_id": f"dataset-{instrument}",
        },
    }
    trade = {
        "instrument": instrument,
        "signal_timeframe": timeframe,
        "session": session,
        "direction": direction,
        "benchmark_family": benchmark,
        "lookback": lookback,
        "signal_threshold": 2.0,
        "filter_family": filter_family,
        "filter_spec_id": f"{filter_family}-v1",
        "entry_mode": "immediate",
        "tp_target_fraction": 0.75,
        "sl_extension_fraction": 0.25,
        "time_stop_minutes": 60,
        "candidate_event_id": event_id,
        "complete": True,
        "gross_return_pips_adverse_first": 4.0,
        "gross_return_pips_favorable_first": 4.0,
    }
    _dump(root / "candidate-events.jsonl", [event])
    _dump(root / "trades.jsonl", [trade])
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("candidate-events.jsonl", "trades.jsonl")
    }
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "source_commit_sha": "a" * 40,
        "requested_start_date": "2024-01-01",
        "requested_end_date": "2024-12-31",
        "instrument": instrument,
        "corpus_id": f"corpus-{instrument}",
        "assembled_dataset_id": f"dataset-{instrument}",
        "filter_family": filter_family,
        "filter_spec_id": f"{filter_family}-v1",
        "output_sha256": hashes,
    }
    (root / "execution-audit.json").write_text(json.dumps(audit))
    return root


def _rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_frozen_outputs_duplicate_adjustment_and_costs(tmp_path):
    inputs = [
        _fixture(tmp_path, benchmark="vwap", lookback=20),
        _fixture(tmp_path, benchmark="vwap-canonical-m1", lookback=40),
    ]
    output = tmp_path / "atlas"
    build_atlas(inputs, output, Path("configs/stage4c-ftmo-cost-profile-v1.json"))
    assert len(_rows(output / "setup-family-matrix.csv")) == 2
    execution = _rows(output / "execution-family-matrix.csv")
    assert len(execution) == 1
    assert execution[0]["unique_execution_opportunities"] == "1"
    scenarios = _rows(output / "cost-robustness.csv")
    assert {(r["spread_statistic"], r["slippage_pips"]) for r in scenarios} == {
        ("mean", "0.0"),
        ("p75", "0.1"),
        ("p90", "0.25"),
        ("p95", "0.5"),
    }


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("session", "london", "new_york"),
        ("timeframe", "15m", "30m"),
        ("direction", "SHORT", "LONG"),
    ],
)
def test_distinct_regimes_remain_distinct(tmp_path, field, first, second):
    kwargs1, kwargs2 = {field: first}, {field: second}
    output = tmp_path / "atlas"
    build_atlas(
        [
            _fixture(tmp_path, lookback=20, **kwargs1),
            _fixture(tmp_path, lookback=40, **kwargs2),
        ],
        output,
        Path("configs/stage4c-ftmo-cost-profile-v1.json"),
    )
    assert len(_rows(output / "execution-family-matrix.csv")) == 2


def test_module_a_overlap_is_descriptive(tmp_path):
    output = tmp_path / "atlas"
    build_atlas(
        [_fixture(tmp_path)], output, Path("configs/stage4c-ftmo-cost-profile-v1.json")
    )
    row = _rows(output / "overlap-with-module-a.csv")[0]
    assert row["intersection_count"] == "1"
    assert row["same_timestamp_overlap_rate"] == "1.0"


def test_shortlist_has_categories_not_scores(tmp_path):
    output = tmp_path / "atlas"
    build_atlas(
        [_fixture(tmp_path)], output, Path("configs/stage4c-ftmo-cost-profile-v1.json")
    )
    row = _rows(output / "candidate-shortlist.csv")[0]
    assert row["triage_category"].startswith("E.")
    assert not any("score" in key or "rank" in key for key in row)


def test_tampered_input_fails_closed(tmp_path):
    source = _fixture(tmp_path)
    (source / "trades.jsonl").write_text("{}\n")
    with pytest.raises(DiscoveryAtlasError, match="commitment mismatch"):
        build_atlas(
            [source],
            tmp_path / "out",
            Path("configs/stage4c-ftmo-cost-profile-v1.json"),
        )


def test_filtered_and_baseline_cannot_mix(tmp_path):
    sources = [
        _fixture(tmp_path, lookback=20),
        _fixture(tmp_path, lookback=40, filter_family="ornstein-uhlenbeck"),
    ]
    with pytest.raises(DiscoveryAtlasError, match="cannot be mixed"):
        build_atlas(
            sources, tmp_path / "out", Path("configs/stage4c-ftmo-cost-profile-v1.json")
        )


def test_incomplete_shards_fail_closed(tmp_path):
    source = _fixture(tmp_path)
    (source / "shard-manifest.json").write_text(
        json.dumps({"shard_count": 2, "shard_index": 0})
    )
    with pytest.raises(DiscoveryAtlasError, match="shard"):
        build_atlas(
            [source],
            tmp_path / "out",
            Path("configs/stage4c-ftmo-cost-profile-v1.json"),
        )


def test_sealed_path_is_rejected_before_inspection(tmp_path, monkeypatch):
    source = tmp_path / "sealed-2025"
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("inspected")),
    )
    with pytest.raises(DiscoveryAtlasError, match="sealed-period"):
        build_atlas(
            [source],
            tmp_path / "out",
            Path("configs/stage4c-ftmo-cost-profile-v1.json"),
        )
    monkeypatch.setattr(Path, "read_text", original)


def test_non_frozen_audit_rejected(tmp_path):
    source = _fixture(tmp_path)
    audit_path = source / "execution-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["requested_end_date"] = "2023-12-31"
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(DiscoveryAtlasError, match="frozen 2024"):
        build_atlas(
            [source],
            tmp_path / "out",
            Path("configs/stage4c-ftmo-cost-profile-v1.json"),
        )
