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


def inspect_file(path: Path, *, count_lines=False, chunk_size=1024 * 1024):
    """Hash a file, optionally counting lines, with memory bounded by chunk_size."""
    digest = hashlib.sha256()
    line_count = 0
    last_byte = b""
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
                if count_lines:
                    line_count += chunk.count(b"\n")
                    last_byte = chunk[-1:]
    except OSError as error:
        raise SourceBundleError(f"cannot inspect source-bundle file: {path}") from error
    if count_lines and last_byte and last_byte != b"\n":
        line_count += 1
    return digest.hexdigest(), line_count


def _sha256(path: Path):
    return inspect_file(path)[0]


def _safe_bundle_path(bundle_dir: Path, value, field):
    """Resolve one non-empty portable relative path, rejecting every escape."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceBundleError(f"invalid bundle-relative path: {field}")
    if value.startswith("/") or any(
        part in ("", ".", "..") for part in value.split("/")
    ):
        raise SourceBundleError(f"invalid bundle-relative path: {field}")
    root = bundle_dir.resolve()
    candidate = (root / value).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise SourceBundleError(f"bundle path escapes bundle directory: {field}")
    return candidate


def _require_text(value, field):
    if not isinstance(value, str) or not value:
        raise SourceBundleError(f"missing or invalid source identity: {field}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bundle_id(manifest):
    body = {key: value for key, value in manifest.items() if key != "bundle_id"}
    return "stage4b:" + hashlib.sha256(_canonical(body).encode()).hexdigest()


def build_candidate_source_bundle(
    bundle_dir: Path,
    source_provenance: dict,
    *,
    combined_audit_filename="execution-audit.json",
):
    """Independently verify recovered outputs and write an unapproved candidate.

    Expected layout is ``ARTIFACT_NAME/{trades.jsonl,shard-manifest.json}`` plus
    the Stage 4B reducer audit at bundle root. Approval is intentionally outside
    this function and no approval-registry path is accepted.
    """
    root = Path(bundle_dir)
    audit_path = _safe_bundle_path(root, combined_audit_filename, "combined audit")
    audit = _load(audit_path)
    if audit.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise SourceBundleError("candidate is not current Stage 4B methodology")
    core_fields = (
        "instrument",
        "stage4b_methodology_id",
        "source_commit_sha",
        "corpus_id",
        "assembled_dataset_id",
        "registry_identity",
    )
    for field in core_fields:
        _require_text(audit.get(field), field)
    if not isinstance(source_provenance, dict) or source_provenance.get("kind") not in (
        "github_workflow",
        "local_run",
    ):
        raise SourceBundleError("invalid source provenance")
    artifact_identities = source_provenance.get("raw_artifacts")
    if not isinstance(artifact_identities, dict):
        raise SourceBundleError("source provenance omits raw artifact identities")
    audit_raw = audit.get("raw_shard_artifacts")
    audit_shards = audit.get("shards")
    if not isinstance(audit_raw, list) or not isinstance(audit_shards, list):
        raise SourceBundleError("reducer audit omits shard evidence")
    expected = audit.get("shard_count")
    if type(expected) is not int or expected < 1 or len(audit_raw) != expected:
        raise SourceBundleError("invalid reducer shard universe")
    if [item.get("shard_index") for item in audit_raw] != list(range(expected)):
        raise SourceBundleError("missing, duplicate, or unordered reducer shard")
    shards_by_index = {item.get("shard_index"): item for item in audit_shards}
    if len(shards_by_index) != expected or set(shards_by_index) != set(range(expected)):
        raise SourceBundleError("reducer audit shard manifests are incomplete")

    raw_shards = []
    for item in audit_raw:
        index, artifact_name = item["shard_index"], item.get("artifact_name")
        _require_text(artifact_name, "artifact_name")
        artifact_identity = artifact_identities.get(artifact_name)
        if not isinstance(artifact_identity, dict) or not artifact_identity:
            raise SourceBundleError("missing independently supplied artifact identity")
        raw_name = "trades.jsonl"
        filename = f"{artifact_name}/{raw_name}"
        shard_manifest_filename = f"{artifact_name}/shard-manifest.json"
        raw_path = _safe_bundle_path(root, filename, "raw filename")
        shard_path = _safe_bundle_path(
            root, shard_manifest_filename, "shard manifest filename"
        )
        shard = _load(shard_path)
        if shard != shards_by_index[index]:
            raise SourceBundleError("local shard manifest disagrees with reducer audit")
        for field in core_fields:
            if shard.get(field) != audit[field]:
                raise SourceBundleError(f"candidate shard provenance mismatch: {field}")
        if (
            shard.get("shard_index") != index
            or shard.get("shard_count") != expected
            or shard.get("raw_artifact_name") != artifact_name
        ):
            raise SourceBundleError("candidate shard identity mismatch")
        raw_hash, row_count = inspect_file(raw_path, count_lines=True)
        if (
            item.get("file_sha256", {}).get(raw_name) != raw_hash
            or item.get("row_counts", {}).get(raw_name) != row_count
            or shard.get("file_sha256", {}).get(raw_name) != raw_hash
            or shard.get("row_counts", {}).get(raw_name) != row_count
        ):
            raise SourceBundleError("candidate raw commitment mismatch")
        raw_shards.append(
            {
                "shard_index": index,
                "artifact_name": artifact_name,
                "artifact_identity": artifact_identity,
                "filename": filename,
                "source_filename": raw_name,
                "sha256": raw_hash,
                "row_count": row_count,
                "shard_manifest_filename": shard_manifest_filename,
                "shard_manifest_sha256": _sha256(shard_path),
            }
        )
    body = {
        "schema_version": "stage4b-source-bundle-v1",
        **{field: audit[field] for field in core_fields},
        "source_provenance": source_provenance,
        "expected_shard_count": expected,
        "raw_shards": raw_shards,
        "combined_audit": {
            "filename": combined_audit_filename,
            "sha256": _sha256(audit_path),
        },
    }
    body["bundle_id"] = _bundle_id(body)
    destination = root / "stage4b-source-bundle.json"
    destination.write_text(_canonical(body) + "\n")
    return destination, _sha256(destination)


def authenticate_source_bundle(bundle_dir: Path, trust_registry_path: Path):
    """Return ordered trade paths and a source audit after anchored verification.

    The registry is the trust anchor.  A hash supplied beside an arbitrary bundle is
    deliberately insufficient: the bundle ID and manifest hash must already occur
    in that independently selected registry.
    """
    bundle_dir, trust_registry_path = Path(bundle_dir), Path(trust_registry_path)
    manifest_path = _safe_bundle_path(
        bundle_dir, "stage4b-source-bundle.json", "bundle manifest"
    )
    manifest, registry = _load(manifest_path), _load(trust_registry_path)
    if registry.get("schema_version") != "stage4b-approved-source-registry-v1":
        raise SourceBundleError("invalid approved-source registry schema")
    if manifest.get("schema_version") != "stage4b-source-bundle-v1":
        raise SourceBundleError("invalid source-bundle schema")
    bundle_id = manifest.get("bundle_id")
    if bundle_id != _bundle_id(manifest):
        raise SourceBundleError("non-canonical source bundle ID")
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
    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") not in (
        "github_workflow",
        "local_run",
    ):
        raise SourceBundleError("missing or invalid source provenance")
    if manifest["stage4b_methodology_id"] != STAGE4B_METHODOLOGY_ID:
        raise SourceBundleError("wrong Stage 4B methodology")
    if len(manifest["source_commit_sha"]) != 40:
        raise SourceBundleError("invalid historical source commit")
    if approval.get("instrument") != manifest["instrument"]:
        raise SourceBundleError("approved instrument mismatch")

    audit_spec = manifest["combined_audit"]
    if not isinstance(audit_spec, dict):
        raise SourceBundleError("invalid combined audit identity")
    audit_path = _safe_bundle_path(
        bundle_dir, audit_spec.get("filename"), "combined audit filename"
    )
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
            "artifact_identity",
            "filename",
            "source_filename",
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
        source_filename = item["source_filename"]
        path = _safe_bundle_path(bundle_dir, filename, "raw filename")
        shard_manifest_path = _safe_bundle_path(
            bundle_dir, item["shard_manifest_filename"], "shard manifest filename"
        )
        if (
            audit_item.get("file_sha256", {}).get(source_filename) != item["sha256"]
            or audit_item.get("row_counts", {}).get(source_filename)
            != item["row_count"]
        ):
            raise SourceBundleError(
                "raw hash or row count disagrees with combined audit"
            )
        actual_hash, actual_rows = inspect_file(path, count_lines=True)
        if not path.is_file() or actual_hash != item["sha256"]:
            raise SourceBundleError("substituted or changed raw trades file")
        if actual_rows != item["row_count"]:
            raise SourceBundleError("mismatched raw row count")
        if (
            not shard_manifest_path.is_file()
            or _sha256(shard_manifest_path) != item["shard_manifest_sha256"]
        ):
            raise SourceBundleError("wrong shard manifest")
        shard_manifest = _load(shard_manifest_path)
        audit_shards = audit.get("shards")
        if (
            not isinstance(audit_shards, list)
            or item["shard_index"] >= len(audit_shards)
            or shard_manifest != audit_shards[item["shard_index"]]
        ):
            raise SourceBundleError("shard manifest disagrees with combined audit")
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
            or shard_manifest.get("file_sha256", {}).get(source_filename)
            != item["sha256"]
            or shard_manifest.get("row_counts", {}).get(source_filename)
            != item["row_count"]
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
        "source_provenance": provenance,
        "stage4b_source_commit": manifest["source_commit_sha"],
        "combined_audit_sha256": audit_spec["sha256"],
        "source_authentication": "approved_registry_manifest_and_local_bytes_verified",
    }
    return tuple(paths), source_audit
