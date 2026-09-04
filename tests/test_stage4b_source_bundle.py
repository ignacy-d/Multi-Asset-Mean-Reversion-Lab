from __future__ import annotations

import hashlib
import json

import pytest
from test_stage4c import PROFILE, trade

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4b_source_bundle import SourceBundleError, authenticate_source_bundle
from mr_lab.stage4c import Stage4CError
from mr_lab.stage4c_reducer import reduce_shards
from mr_lab.stage4c_runner import run_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    core = {
        "instrument": "EURUSD",
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "source_commit_sha": "b" * 40,
        "corpus_id": "frozen-corpus",
        "assembled_dataset_id": "frozen-dataset",
        "registry_identity": "registry-sha",
    }
    raw, audit_raw = [], []
    for index in range(2):
        filename = f"trades-{index}.jsonl"
        path = root / filename
        path.write_text(json.dumps(trade(f"event-{index}"), sort_keys=True) + "\n")
        artifact = f"stage4b-raw-{index}"
        shard_manifest = core | {
            "shard_index": index,
            "shard_count": 2,
            "raw_artifact_name": artifact,
            "file_sha256": {filename: sha(path)},
            "row_counts": {filename: 1},
        }
        shard_name = f"shard-manifest-{index}.json"
        write_json(root / shard_name, shard_manifest)
        raw.append(
            {
                "shard_index": index,
                "artifact_name": artifact,
                "artifact_id": 100 + index,
                "filename": filename,
                "sha256": sha(path),
                "row_count": 1,
                "shard_manifest_filename": shard_name,
                "shard_manifest_sha256": sha(root / shard_name),
            }
        )
        audit_raw.append(
            {
                "shard_index": index,
                "artifact_name": artifact,
                "file_sha256": {filename: sha(path)},
                "row_counts": {filename: 1},
            }
        )
    audit = core | {"raw_shard_artifacts": audit_raw}
    audit_path = root / "stage4b-reduction-audit.json"
    write_json(audit_path, audit)
    manifest = core | {
        "schema_version": "stage4b-source-bundle-v1",
        "bundle_id": "synthetic-eurusd",
        "source_workflow_run_id": 42,
        "expected_shard_count": 2,
        "raw_shards": raw,
        "combined_audit": {"filename": audit_path.name, "sha256": sha(audit_path)},
    }
    manifest_path = root / "stage4b-source-bundle.json"
    write_json(manifest_path, manifest)
    registry = tmp_path / "approved.json"
    write_json(
        registry,
        {
            "schema_version": "stage4b-approved-source-registry-v1",
            "approved_bundles": {
                "synthetic-eurusd": {
                    "instrument": "EURUSD",
                    "manifest_sha256": sha(manifest_path),
                }
            },
        },
    )
    return root, registry


def test_exact_authenticated_bundle_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    root, registry = bundle(tmp_path)
    paths, audit = authenticate_source_bundle(root, registry)
    assert [path.name for path in paths] == ["trades-0.jsonl", "trades-1.jsonl"]
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        run_rows(
            (
                json.loads(line)
                for path in paths
                for line in path.read_text().splitlines()
            ),
            output,
            PROFILE,
            source_mode="authenticated_stage4b_bundle",
            source_audit=audit,
        )
    assert (first / "stage4c-trade-matrix.csv").read_bytes() == (
        second / "stage4c-trade-matrix.csv"
    ).read_bytes()
    shards = []
    for index in range(2):
        output = tmp_path / f"shard-{index}"
        run_rows(
            (
                json.loads(line)
                for path in paths
                for line in path.read_text().splitlines()
            ),
            output,
            PROFILE,
            source_mode="authenticated_stage4b_bundle",
            source_audit=audit,
            shard_index=index,
            shard_count=2,
        )
        shards.append(output)
    reduced = tmp_path / "reduced"
    reduce_shards(shards, reduced, PROFILE, 2)
    reduced_audit = json.loads((reduced / "execution-audit.json").read_text())
    assert reduced_audit["source_authentication"] == audit["source_authentication"]
    assert (
        reduced_audit["source_bundle_manifest_sha256"]
        == audit["source_bundle_manifest_sha256"]
    )


def test_instrument_mismatch_and_mixed_rows_fail_before_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    root, registry = bundle(tmp_path)
    _, audit = authenticate_source_bundle(root, registry)
    for rows in (
        [trade(instrument="AUDUSD")],
        [trade("e"), trade("a", instrument="AUDUSD")],
    ):
        output = tmp_path / f"bad-{len(list(output for output in tmp_path.iterdir()))}"
        with pytest.raises(Stage4CError, match="instrument disagrees"):
            run_rows(
                iter(rows),
                output,
                PROFILE,
                source_mode="authenticated_stage4b_bundle",
                source_audit=audit,
            )
        assert not (output / "execution-audit.json").exists()


@pytest.mark.parametrize(
    "case",
    [
        "methodology",
        "corpus",
        "dataset",
        "substitute",
        "filename",
        "missing",
        "duplicate",
        "bytes",
        "count",
        "audit",
    ],
)
def test_bundle_tampering_is_rejected(tmp_path, case):
    root, registry = bundle(tmp_path)
    manifest_path = root / "stage4b-source-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    if case in {
        "methodology",
        "corpus",
        "dataset",
        "filename",
        "missing",
        "duplicate",
        "count",
    }:
        if case == "methodology":
            manifest["stage4b_methodology_id"] = "wrong"
        elif case == "corpus":
            manifest["corpus_id"] = "wrong"
        elif case == "dataset":
            manifest["assembled_dataset_id"] = "wrong"
        elif case == "filename":
            manifest["raw_shards"][0]["filename"] = "trades-1.jsonl"
        elif case == "missing":
            manifest["raw_shards"].pop()
        elif case == "duplicate":
            manifest["raw_shards"][1]["shard_index"] = 0
        elif case == "count":
            manifest["raw_shards"][0]["row_count"] = 2
        write_json(manifest_path, manifest)
    elif case in {"substitute", "bytes"}:
        (root / "trades-0.jsonl").write_text(json.dumps(trade("changed")) + "\n")
    else:
        (root / "stage4b-reduction-audit.json").write_text("{}\n")
    with pytest.raises(SourceBundleError):
        authenticate_source_bundle(root, registry)
