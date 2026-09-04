"""Fail-closed recovery of the frozen 2024 Stage 4B Actions artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPOSITORY = "ignacy-d/Multi-Asset-Mean-Reversion-Lab"
HEAD_SHA = "3090682b61090de1b4a30efc18a7547e92fa262e"
SHARD_COUNT = 4
RUNS = {
    "USDJPY": 32638668214,
    "AUDUSD": 32641981092,
    "AUDJPY": 32641988237,
    "EURUSD": 32643840766,
}
KINDS = ("raw", "compact", "review")


class RecoveryError(RuntimeError):
    """An immutable identity or integrity check failed."""


@dataclass(frozen=True)
class ExpectedArtifact:
    instrument: str
    run_id: int
    kind: str
    shard_index: int | None
    name: str


def expected_artifacts(
    instrument: str, source_artifact_id: int
) -> list[ExpectedArtifact]:
    """Return the one exact artifact set produced by the frozen workflow."""
    lower = instrument.lower()
    run_id = RUNS[instrument]
    result = []
    for index in range(SHARD_COUNT):
        suffix = f"shard-{index}-attempt-1-{source_artifact_id}"
        result.extend(
            [
                ExpectedArtifact(
                    instrument,
                    run_id,
                    "raw",
                    index,
                    f"stage-4b-{lower}-2024-raw-{suffix}",
                ),
                ExpectedArtifact(
                    instrument,
                    run_id,
                    "compact",
                    index,
                    f"stage-4b-{lower}-2024-compact-{suffix}",
                ),
            ]
        )
    result.append(
        ExpectedArtifact(
            instrument,
            run_id,
            "review",
            None,
            f"stage-4b-{lower}-2024-combined-review-{run_id}",
        )
    )
    return result


class GitHubClient:
    def __init__(self, token: str | None = None):
        self.token = (
            token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        )

    def _request(self, path: str) -> bytes:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"https://api.github.com/repos/{REPOSITORY}/{path}", headers=headers
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise RecoveryError(
                f"GitHub API request failed ({exc.code}): {path}"
            ) from exc

    def json(self, path: str) -> dict:
        try:
            return json.loads(self._request(path))
        except json.JSONDecodeError as exc:
            raise RecoveryError(f"GitHub returned malformed JSON: {path}") from exc

    def run(self, run_id: int) -> dict:
        return self.json(f"actions/runs/{run_id}")

    def artifacts(self, run_id: int) -> list[dict]:
        data = self.json(f"actions/runs/{run_id}/artifacts?per_page=100")
        if data.get("total_count") != len(data.get("artifacts", [])):
            raise RecoveryError(
                "artifact response was truncated; refusing incomplete inventory"
            )
        return data["artifacts"]

    def download(self, artifact_id: int) -> bytes:
        return self._request(f"actions/artifacts/{artifact_id}/zip")


def _load_registry(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    entries = data.get("instruments", {})
    for instrument in RUNS:
        entry = entries.get(instrument)
        if (
            not entry
            or entry.get("instrument") != instrument
            or entry.get("verification_status") != "verified"
        ):
            raise RecoveryError(f"frozen corpus registry mismatch for {instrument}")
        if (
            entry.get("requested_start_date") != "2024-01-01"
            or entry.get("requested_end_date") != "2024-12-31"
        ):
            raise RecoveryError(f"non-2024 corpus provenance for {instrument}")
    return entries


def inventory(
    client: GitHubClient, instrument: str, entry: dict
) -> tuple[dict, list[tuple[ExpectedArtifact, dict]]]:
    run_id = RUNS[instrument]
    run = client.run(run_id)
    if (
        run.get("id") != run_id
        or run.get("conclusion") != "success"
        or run.get("head_sha") != HEAD_SHA
    ):
        raise RecoveryError(f"frozen run provenance mismatch for {instrument}")
    actual = client.artifacts(run_id)
    by_name: dict[str, list[dict]] = {}
    for artifact in actual:
        by_name.setdefault(artifact.get("name", ""), []).append(artifact)
    matched = []
    for expected in expected_artifacts(instrument, entry["source_artifact_id"]):
        candidates = by_name.get(expected.name, [])
        if len(candidates) != 1:
            raise RecoveryError(
                "expected exactly one artifact named "
                f"{expected.name}; found {len(candidates)}"
            )
        artifact = candidates[0]
        if (
            artifact.get("expired") is not False
            or (artifact.get("workflow_run") or {}).get("id") != run_id
        ):
            raise RecoveryError(f"expired or wrong-run artifact: {expected.name}")
        matched.append((expected, artifact))
    indexes = [e.shard_index for e, _ in matched if e.kind == "raw"]
    if indexes != list(range(SHARD_COUNT)):
        raise RecoveryError("raw artifacts are not exact shards 0,1,2,3")
    return run, matched


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_extract(data: bytes, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "artifact.zip"
        archive_path.write_bytes(data)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                if not names or any(
                    PurePosixPath(name).is_absolute()
                    or ".." in PurePosixPath(name).parts
                    for name in names
                ):
                    raise RecoveryError("malformed or unsafe artifact archive")
                destination.mkdir(parents=True, exist_ok=False)
                archive.extractall(destination)
        except (zipfile.BadZipFile, OSError) as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise RecoveryError("malformed artifact archive") from exc


def _json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"missing or malformed JSON: {path}") from exc


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_local(instrument_root: Path, instrument: str, entry: dict) -> None:
    """Verify downloaded content and all shard cross-commitments."""
    manifests = []
    for index in range(SHARD_COUNT):
        compact = instrument_root / "compact-shards" / f"shard-{index}"
        raw = instrument_root / "raw-shards" / f"shard-{index}"
        manifest = _json_file(compact / "shard-manifest.json")
        audit = _json_file(compact / "execution-audit.json")
        required = {
            "instrument": instrument,
            "source_commit_sha": HEAD_SHA,
            "shard_index": index,
            "shard_count": SHARD_COUNT,
            "source_workflow_run_id": entry["source_workflow_run_id"],
            "source_artifact_id": entry["source_artifact_id"],
            "source_artifact_name": entry["source_artifact_name"],
            "corpus_id": entry["corpus_id"],
            "assembled_dataset_id": entry["assembled_dataset_id"],
        }
        for key, value in required.items():
            if manifest.get(key) != value:
                raise RecoveryError(f"shard {index} manifest mismatch: {key}")
        for key in (
            "instrument",
            "source_commit_sha",
            "source_workflow_run_id",
            "source_artifact_id",
            "source_artifact_name",
            "corpus_id",
            "assembled_dataset_id",
        ):
            if audit.get(key) != required[key]:
                raise RecoveryError(f"shard {index} execution audit mismatch: {key}")
        for name, expected_hash in manifest.get("file_sha256", {}).items():
            path = (
                raw / name
                if name in {"candidate-events.jsonl", "trades.jsonl"}
                else compact / name
            )
            if not path.is_file() or _hash_file(path) != expected_hash:
                raise RecoveryError(f"shard {index} file hash mismatch: {name}")
        for name in ("candidate-events.jsonl", "trades.jsonl"):
            if name not in manifest.get("file_sha256", {}):
                raise RecoveryError(f"shard {index} lacks raw hash commitment: {name}")
        manifests.append(manifest)
    indexes = [m["shard_index"] for m in manifests]
    if len(set(indexes)) != SHARD_COUNT or set(indexes) != set(range(SHARD_COUNT)):
        raise RecoveryError("duplicate or missing local shards")
    review_audit = _json_file(
        instrument_root / "combined-review" / "execution-audit.json"
    )
    if (
        review_audit.get("instrument") != instrument
        or review_audit.get("source_commit_sha") != HEAD_SHA
    ):
        raise RecoveryError("combined review provenance mismatch")
    review_root = instrument_root / "combined-review"
    for name, expected_hash in review_audit.get("combined_output_sha256", {}).items():
        path = review_root / name
        if not path.is_file() or _hash_file(path) != expected_hash:
            raise RecoveryError(f"combined review file hash mismatch: {name}")
    if [s.get("shard_index") for s in review_audit.get("shards", [])] != list(
        range(SHARD_COUNT)
    ):
        raise RecoveryError("combined review does not commit exact shard set")
    expected_raw = {
        m["raw_artifact_name"]: {
            name: m["file_sha256"][name]
            for name in ("candidate-events.jsonl", "trades.jsonl")
        }
        for m in manifests
    }
    review_raw = {
        r.get("artifact_name"): r.get("file_sha256")
        for r in review_audit.get("raw_shard_artifacts", [])
    }
    if review_raw != expected_raw:
        raise RecoveryError(
            "combined review raw commitments differ from shard manifests"
        )


def _target(root: Path, expected: ExpectedArtifact) -> Path:
    folder = {
        "raw": "raw-shards",
        "compact": "compact-shards",
        "review": "combined-review",
    }[expected.kind]
    return (
        root
        / expected.instrument
        / folder
        / (f"shard-{expected.shard_index}" if expected.shard_index is not None else "")
    )


def recover(
    client: GitHubClient, root: Path, instrument: str, entry: dict, force: bool
) -> dict:
    run, artifacts = inventory(client, instrument, entry)
    instrument_root = root / instrument
    if instrument_root.exists() and not force:
        verify_local(instrument_root, instrument, entry)
        return {"instrument": instrument, "status": "already-valid"}
    staging = root / f".{instrument}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        for expected, artifact in artifacts:
            data = client.download(artifact["id"])
            digest = artifact.get("digest")
            if digest:
                algorithm, separator, value = digest.partition(":")
                if separator != ":" or algorithm != "sha256" or _sha256(data) != value:
                    raise RecoveryError(
                        f"GitHub artifact digest mismatch: {expected.name}"
                    )
            _safe_extract(data, _target(staging, expected))
        verify_local(staging / instrument, instrument, entry)
        provenance = staging / instrument / "provenance"
        provenance.mkdir()
        (provenance / "github-metadata.json").write_text(
            json.dumps(
                {"run": run, "artifacts": [a for _, a in artifacts]},
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        if instrument_root.exists():
            shutil.rmtree(instrument_root)
        root.mkdir(parents=True, exist_ok=True)
        (staging / instrument).replace(instrument_root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"instrument": instrument, "status": "downloaded-and-verified"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "check", "download", "verify"))
    parser.add_argument("--instrument", choices=(*RUNS, "all"), default="all")
    parser.add_argument("--destination", type=Path, help="explicit local recovery root")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/stage4a-2024-corpus-registry.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing instrument only after a verified download",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    entries = _load_registry(args.registry)
    instruments = list(RUNS) if args.instrument == "all" else [args.instrument]
    if args.command == "list":
        for instrument in instruments:
            for expected in expected_artifacts(
                instrument, entries[instrument]["source_artifact_id"]
            ):
                print(
                    f"{instrument}\t{expected.run_id}\t{expected.kind}\t{expected.name}"
                )
        return 0
    if args.command in {"download", "verify"} and args.destination is None:
        raise RecoveryError("--destination is required for local operations")
    if args.command == "check":
        client = GitHubClient()
        for instrument in instruments:
            _, found = inventory(client, instrument, entries[instrument])
            print(
                f"{instrument}: available and provenance-valid ({len(found)} artifacts)"
            )
    elif args.command == "verify":
        for instrument in instruments:
            verify_local(args.destination / instrument, instrument, entries[instrument])
            print(f"{instrument}: local files valid")
    else:
        client = GitHubClient()
        for instrument in instruments:
            print(
                json.dumps(
                    recover(
                        client,
                        args.destination,
                        instrument,
                        entries[instrument],
                        args.force,
                    ),
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
