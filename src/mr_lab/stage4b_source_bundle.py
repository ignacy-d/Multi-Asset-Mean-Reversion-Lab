"""Authentication boundary for frozen Stage 4B raw source bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID


class SourceBundleError(ValueError):
    """The bundle is not exactly one of the trusted, frozen source bundles."""


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SourceBundleError(f"cannot read authenticated JSON: {path}") from error


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_text(value, field):
    if not isinstance(value, str) or not value:
        raise SourceBundleError(f"missing or invalid source identity: {field}")


def authenticate_source_bundle(bundle_dir: Path, trust_registry_path: Path):
    """Return ordered trade paths and a source audit after anchored verification.

    The registry is the trust anchor.  A hash supplied beside an arbitrary bundle is
    deliberately insufficient: the bundle ID and manifest hash must already occur
    in that independently selected registry.
    """
    bundle_dir, trust_registry_path = Path(bundle_dir), Path(trust_registry_path)
    manifest_path = bundle_dir / "stage4b-source-bundle.json"
    manifest, registry = _load(manifest_path), _load(trust_registry_path)
    if registry.get("schema_version") != "stage4b-approved-source-registry-v1":
        raise SourceBundleError("invalid approved-source registry schema")
    if manifest.get("schema_version") != "stage4b-source-bundle-v1":
        raise SourceBundleError("invalid source-bundle schema")
    bundle_id = manifest.get("bundle_id")
    approvals = registry.get("approved_bundles")
    if not isinstance(approvals, dict) or bundle_id not in approvals:
        raise SourceBundleError("source bundle is not in the approved registry")
    approval = approvals[bundle_id]
    if not isinstance(approval, dict) or approval.get("manifest_sha256") != _sha256(
        manifest_path
    ):
        raise SourceBundleError("source-bundle manifest is not approved")

    required = (
        "instrument",
        "stage4b_methodology_id",
        "source_commit_sha",
        "corpus_id",
        "assembled_dataset_id",
        "registry_identity",
    )
    for field in required:
        _require_text(manifest.get(field), field)
    if (
        type(manifest.get("source_workflow_run_id")) is not int
        or manifest["source_workflow_run_id"] <= 0
    ):
        raise SourceBundleError(
            "missing or invalid source identity: source_workflow_run_id"
        )
    if manifest["stage4b_methodology_id"] != STAGE4B_METHODOLOGY_ID:
        raise SourceBundleError("wrong Stage 4B methodology")
    if len(manifest["source_commit_sha"]) != 40:
        raise SourceBundleError("invalid historical source commit")
    if approval.get("instrument") != manifest["instrument"]:
        raise SourceBundleError("approved instrument mismatch")

    audit_spec = manifest["combined_audit"]
    if not isinstance(audit_spec, dict):
        raise SourceBundleError("invalid combined audit identity")
    audit_path = bundle_dir / str(audit_spec.get("filename", ""))
    if not audit_path.is_file() or _sha256(audit_path) != audit_spec.get("sha256"):
        raise SourceBundleError("stale or wrong combined audit")
    audit = _load(audit_path)
    for field in (
        "instrument",
        "stage4b_methodology_id",
        "source_commit_sha",
        "corpus_id",
        "assembled_dataset_id",
        "registry_identity",
    ):
        if audit.get(field) != manifest[field]:
            raise SourceBundleError(f"combined audit provenance mismatch: {field}")

    raw = manifest.get("raw_shards")
    expected_count = manifest.get("expected_shard_count")
    if (
        not isinstance(raw, list)
        or type(expected_count) is not int
        or expected_count < 1
    ):
        raise SourceBundleError("invalid shard universe")
    indexes = [item.get("shard_index") for item in raw if isinstance(item, dict)]
    if len(indexes) != len(set(indexes)):
        raise SourceBundleError("duplicate shard")
    if indexes != list(range(expected_count)):
        raise SourceBundleError("missing or unordered shard")
    audit_raw = audit.get("raw_shard_artifacts")
    if not isinstance(audit_raw, list) or len(audit_raw) != expected_count:
        raise SourceBundleError("combined audit shard universe mismatch")

    paths, identities = [], []
    for item, audit_item in zip(raw, audit_raw, strict=True):
        for field in (
            "artifact_name",
            "artifact_id",
            "filename",
            "sha256",
            "row_count",
            "shard_manifest_filename",
            "shard_manifest_sha256",
        ):
            if item.get(field) in (None, ""):
                raise SourceBundleError(f"missing raw shard identity: {field}")
        if item["shard_index"] != audit_item.get("shard_index") or item[
            "artifact_name"
        ] != audit_item.get("artifact_name"):
            raise SourceBundleError(
                "raw artifact identity disagrees with combined audit"
            )
        filename = item["filename"]
        if (
            audit_item.get("file_sha256", {}).get(filename) != item["sha256"]
            or audit_item.get("row_counts", {}).get(filename) != item["row_count"]
        ):
            raise SourceBundleError(
                "raw hash or row count disagrees with combined audit"
            )
        path = bundle_dir / filename
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise SourceBundleError("substituted or changed raw trades file")
        with path.open("rb") as stream:
            if sum(1 for _ in stream) != item["row_count"]:
                raise SourceBundleError("mismatched raw row count")
        shard_manifest_path = bundle_dir / item["shard_manifest_filename"]
        if (
            not shard_manifest_path.is_file()
            or _sha256(shard_manifest_path) != item["shard_manifest_sha256"]
        ):
            raise SourceBundleError("wrong shard manifest")
        shard_manifest = _load(shard_manifest_path)
        for field in (
            "instrument",
            "stage4b_methodology_id",
            "source_commit_sha",
            "corpus_id",
            "assembled_dataset_id",
            "registry_identity",
        ):
            if shard_manifest.get(field) != manifest[field]:
                raise SourceBundleError(f"shard manifest provenance mismatch: {field}")
        if (
            shard_manifest.get("shard_index") != item["shard_index"]
            or shard_manifest.get("shard_count") != expected_count
            or shard_manifest.get("raw_artifact_name") != item["artifact_name"]
            or shard_manifest.get("file_sha256", {}).get(filename) != item["sha256"]
            or shard_manifest.get("row_counts", {}).get(filename) != item["row_count"]
        ):
            raise SourceBundleError("shard manifest identity mismatch")
        paths.append(path)
        identities.append(item)

    source_audit = {
        "stage4b_methodology_id": manifest["stage4b_methodology_id"],
        "instrument": manifest["instrument"],
        "corpus_id": manifest["corpus_id"],
        "assembled_dataset_id": manifest["assembled_dataset_id"],
        "registry_identity": manifest["registry_identity"],
        "source_trade_sha256": {item["filename"]: item["sha256"] for item in raw},
        "source_shard_identities": identities,
        "expected_shard_count": expected_count,
        "source_bundle_id": bundle_id,
        "source_bundle_manifest_sha256": _sha256(manifest_path),
        "approved_source_registry_sha256": _sha256(trust_registry_path),
        "source_workflow_run_id": manifest["source_workflow_run_id"],
        "stage4b_source_commit": manifest["source_commit_sha"],
        "combined_audit_sha256": audit_spec["sha256"],
        "source_authentication": "approved_registry_manifest_and_local_bytes_verified",
    }
    return tuple(paths), source_audit
