from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    raw, audit_raw, audit_shards = [], [], []
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
        audit_shards.append(shard_manifest)
        raw.append(
            {
                "shard_index": index,
                "artifact_name": artifact,
                "artifact_identity": {"github_artifact_id": 100 + index},
                "filename": filename,
                "source_filename": filename,
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
    audit = core | {"raw_shard_artifacts": audit_raw, "shards": audit_shards}
    audit_path = root / "stage4b-reduction-audit.json"
    write_json(audit_path, audit)
    manifest = core | {
        "schema_version": "stage4b-source-bundle-v1",
        "source_provenance": {"kind": "github_workflow", "run_id": 42},
        "expected_shard_count": 2,
        "raw_shards": raw,
        "combined_audit": {"filename": audit_path.name, "sha256": sha(audit_path)},
    }
    body = {key: value for key, value in manifest.items() if key != "bundle_id"}
    manifest["bundle_id"] = (
        "stage4b:"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    manifest_path = root / "stage4b-source-bundle.json"
    write_json(manifest_path, manifest)
    registry = tmp_path / "approved.json"
    write_json(
        registry,
        {
            "schema_version": "stage4b-approved-source-registry-v1",
            "approved_bundles": {
                manifest["bundle_id"]: {
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


def approve_manifest(root, registry):
    manifest_path = root / "stage4b-source-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    body = {key: value for key, value in manifest.items() if key != "bundle_id"}
    manifest["bundle_id"] = (
        "stage4b:"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    write_json(manifest_path, manifest)
    value = json.loads(registry.read_text())
    approval = next(iter(value["approved_bundles"].values()))
    approval["manifest_sha256"] = sha(manifest_path)
    value["approved_bundles"] = {manifest["bundle_id"]: approval}
    write_json(registry, value)


def test_hash_and_row_count_are_exact_and_streaming(tmp_path, monkeypatch):
    from mr_lab.stage4b_source_bundle import inspect_file

    path = tmp_path / "large-style.jsonl"
    payload = b'{"n":1}\n' * 31 + b'{"last":true}'
    path.write_bytes(payload)
    original_open = Path.open
    reads = []

    class Tracking:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.wrapped.close()

        def read(self, size=-1):
            reads.append(size)
            return self.wrapped.read(size)

    def tracked_open(self, *args, **kwargs):
        opened = original_open(self, *args, **kwargs)
        return Tracking(opened) if self == path else opened

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unbounded read"))
    digest, rows = inspect_file(path, count_lines=True, chunk_size=7)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert rows == 32
    assert len(reads) > 2
    assert set(reads) == {7}


@pytest.mark.parametrize(
    "malicious", ["/etc/passwd", "../outside", "a/../../outside", "a//b"]
)
def test_manifest_paths_must_be_safe_bundle_relative(tmp_path, malicious):
    root, registry = bundle(tmp_path)
    manifest_path = root / "stage4b-source-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_shards"][0]["filename"] = malicious
    write_json(manifest_path, manifest)
    approve_manifest(root, registry)
    with pytest.raises(SourceBundleError, match="path"):
        authenticate_source_bundle(root, registry)


def test_manifest_symlink_escape_is_rejected(tmp_path):
    root, registry = bundle(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("secret\n")
    (root / "escape.jsonl").symlink_to(outside)
    manifest_path = root / "stage4b-source-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_shards"][0]["filename"] = "escape.jsonl"
    write_json(manifest_path, manifest)
    approve_manifest(root, registry)
    with pytest.raises(SourceBundleError, match="escapes"):
        authenticate_source_bundle(root, registry)


def builder_fixture(tmp_path):
    from mr_lab.stage4b_source_bundle import build_candidate_source_bundle

    root = tmp_path / "candidate"
    artifact = "raw-shard-0"
    directory = root / artifact
    directory.mkdir(parents=True)
    raw = directory / "trades.jsonl"
    raw.write_text(json.dumps(trade("builder"), sort_keys=True) + "\n")
    core = {
        "instrument": "EURUSD",
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "source_commit_sha": "b" * 40,
        "corpus_id": "frozen-corpus",
        "assembled_dataset_id": "frozen-dataset",
        "registry_identity": "registry-sha",
        "shard_count": 1,
    }
    shard = core | {
        "shard_index": 0,
        "raw_artifact_name": artifact,
        "file_sha256": {"trades.jsonl": sha(raw)},
        "row_counts": {"trades.jsonl": 1},
    }
    write_json(directory / "shard-manifest.json", shard)
    audit = core | {
        "raw_shard_artifacts": [
            {
                "shard_index": 0,
                "artifact_name": artifact,
                "file_sha256": {"trades.jsonl": sha(raw)},
                "row_counts": {"trades.jsonl": 1},
            }
        ],
        "shards": [shard],
    }
    write_json(root / "execution-audit.json", audit)
    provenance = {
        "kind": "local_run",
        "run_id": "deterministic-test-run",
        "raw_artifacts": {artifact: {"local_artifact": artifact}},
    }
    return root, provenance, build_candidate_source_bundle


def test_candidate_builder_is_deterministic_and_cannot_approve(tmp_path):
    root, provenance, build = builder_fixture(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text('{"sentinel":"unchanged"}\n')
    before = approval.read_bytes()
    first, first_hash = build(root, provenance)
    first_bytes = first.read_bytes()
    second, second_hash = build(root, provenance)
    assert second.read_bytes() == first_bytes
    assert second_hash == first_hash == sha(first)
    assert approval.read_bytes() == before
    assert "approved" not in json.loads(first.read_text())
    manifest = json.loads(first.read_text())
    registry = tmp_path / "reviewed-registry.json"
    write_json(
        registry,
        {
            "schema_version": "stage4b-approved-source-registry-v1",
            "approved_bundles": {
                manifest["bundle_id"]: {
                    "instrument": "EURUSD",
                    "manifest_sha256": first_hash,
                }
            },
        },
    )
    paths, authenticated = authenticate_source_bundle(root, registry)
    assert [path.relative_to(root).as_posix() for path in paths] == [
        "raw-shard-0/trades.jsonl"
    ]
    assert authenticated["stage4b_methodology_id"] == STAGE4B_METHODOLOGY_ID


def test_historical_methodology_cannot_build_current_candidate(tmp_path):
    root, provenance, build = builder_fixture(tmp_path)
    audit_path = root / "execution-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["stage4b_methodology_id"] = "sha256:historical-v1"
    write_json(audit_path, audit)
    with pytest.raises(SourceBundleError, match="current Stage 4B methodology"):
        build(root, provenance)


def test_approved_historical_methodology_rejected_for_current_production(tmp_path):
    root, registry = bundle(tmp_path)
    manifest_path = root / "stage4b-source-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stage4b_methodology_id"] = "sha256:historical-vwap-v1"
    write_json(manifest_path, manifest)
    approve_manifest(root, registry)
    with pytest.raises(SourceBundleError, match="methodology"):
        authenticate_source_bundle(root, registry)
