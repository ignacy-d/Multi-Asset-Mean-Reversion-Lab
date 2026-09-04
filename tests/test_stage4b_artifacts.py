import hashlib
import io
import json
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mr_lab.stage4b_artifacts import (
    HEAD_SHA,
    RUNS,
    GitHubClient,
    RecoveryError,
    expected_artifacts,
    inventory,
    recover,
    verify_local,
)

INSTRUMENT = "USDJPY"
ENTRY = {
    "instrument": INSTRUMENT,
    "source_workflow_run_id": 7,
    "source_artifact_id": 11,
    "source_artifact_name": "corpus",
    "corpus_id": "sha256:corpus",
    "assembled_dataset_id": "sha256:dataset",
}


@contextmanager
def _http_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def test_authenticated_api_request_receives_authorization():
    received = []

    class ApiHandler(_QuietHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"api response")

    with _http_server(ApiHandler) as api_url:
        assert GitHubClient("secret-token", api_url).download(1) == b"api response"

    assert received == ["Bearer secret-token"]


def test_cross_host_artifact_redirect_drops_authorization_and_follows():
    api_authorization = []
    storage_authorization = []

    class StorageHandler(_QuietHandler):
        def do_GET(self):
            storage_authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"artifact archive")

    with _http_server(StorageHandler) as storage_url:

        class ApiHandler(_QuietHandler):
            def do_GET(self):
                api_authorization.append(self.headers.get("Authorization"))
                self.send_response(302)
                self.send_header("Location", f"{storage_url}/artifact.zip")
                self.end_headers()

        with _http_server(ApiHandler) as api_url:
            result = GitHubClient("secret-token", api_url).download(1)

    assert result == b"artifact archive"
    assert api_authorization == ["Bearer secret-token"]
    assert storage_authorization == [None]


def _metadata(expired=False, sha=HEAD_SHA):
    expected = expected_artifacts(INSTRUMENT, ENTRY["source_artifact_id"])
    artifacts = [
        {
            "id": number,
            "name": item.name,
            "expired": expired,
            "workflow_run": {"id": RUNS[INSTRUMENT]},
            "size_in_bytes": 10,
        }
        for number, item in enumerate(expected, 1)
    ]
    return {"id": RUNS[INSTRUMENT], "conclusion": "success", "head_sha": sha}, artifacts


class Client(GitHubClient):
    def __init__(self, run, artifacts, archives=None):
        self._run, self._artifacts, self._archives = run, artifacts, archives or {}

    def run(self, run_id):
        return self._run

    def artifacts(self, run_id):
        return self._artifacts

    def download(self, artifact_id):
        return self._archives[artifact_id]


def test_exact_run_selection_and_deterministic_names():
    run, artifacts = _metadata()
    _, matched = inventory(Client(run, artifacts), INSTRUMENT, ENTRY)
    assert [item.name for item, _ in matched] == [
        a.name for a in expected_artifacts(INSTRUMENT, 11)
    ]
    assert matched[0][0].run_id == 32638668214


@pytest.mark.parametrize("change", ["wrong_sha", "expired", "missing", "duplicate"])
def test_metadata_rejections(change):
    run, artifacts = _metadata()
    if change == "wrong_sha":
        run["head_sha"] = "0" * 40
    elif change == "expired":
        artifacts[0]["expired"] = True
    elif change == "missing":
        artifacts.pop(0)
    else:
        artifacts.append(dict(artifacts[0], id=999))
    with pytest.raises(RecoveryError):
        inventory(Client(run, artifacts), INSTRUMENT, ENTRY)


def _zip(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return output.getvalue()


def _local(root: Path, instrument=INSTRUMENT):
    manifests = []
    for index in range(4):
        raw = root / instrument / "raw-shards" / f"shard-{index}"
        compact = root / instrument / "compact-shards" / f"shard-{index}"
        raw.mkdir(parents=True)
        compact.mkdir(parents=True)
        raw_hashes = {}
        for name in ("candidate-events.jsonl", "trades.jsonl"):
            path = raw / name
            path.write_text(f"{index}-{name}\n")
            raw_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit = ENTRY | {"source_commit_sha": HEAD_SHA}
        (compact / "execution-audit.json").write_text(json.dumps(audit))
        audit_hash = hashlib.sha256(
            (compact / "execution-audit.json").read_bytes()
        ).hexdigest()
        manifest = ENTRY | {
            "source_commit_sha": HEAD_SHA,
            "shard_index": index,
            "shard_count": 4,
            "raw_artifact_name": f"raw-{index}",
            "file_sha256": raw_hashes | {"execution-audit.json": audit_hash},
        }
        (compact / "shard-manifest.json").write_text(json.dumps(manifest))
        manifests.append(manifest)
    review = root / instrument / "combined-review"
    review.mkdir(parents=True)
    review_audit = ENTRY | {
        "source_commit_sha": HEAD_SHA,
        "shards": manifests,
        "raw_shard_artifacts": [
            {
                "shard_index": i,
                "artifact_name": m["raw_artifact_name"],
                "file_sha256": {
                    k: v for k, v in m["file_sha256"].items() if k.endswith("jsonl")
                },
            }
            for i, m in enumerate(manifests)
        ],
    }
    (review / "execution-audit.json").write_text(json.dumps(review_audit))


def test_local_verification_is_idempotent_and_layout_is_deterministic(tmp_path):
    _local(tmp_path)
    verify_local(tmp_path / INSTRUMENT, INSTRUMENT, ENTRY)
    verify_local(tmp_path / INSTRUMENT, INSTRUMENT, ENTRY)
    assert (tmp_path / INSTRUMENT / "raw-shards" / "shard-3" / "trades.jsonl").is_file()


@pytest.mark.parametrize("problem", ["hash", "instrument", "duplicate", "missing"])
def test_local_provenance_rejections(tmp_path, problem):
    _local(tmp_path)
    if problem == "hash":
        (tmp_path / INSTRUMENT / "raw-shards/shard-0/trades.jsonl").write_text(
            "tampered"
        )
    elif problem == "instrument":
        path = tmp_path / INSTRUMENT / "compact-shards/shard-0/shard-manifest.json"
        value = json.loads(path.read_text())
        value["instrument"] = "EURUSD"
        path.write_text(json.dumps(value))
    else:
        path = tmp_path / INSTRUMENT / "combined-review/execution-audit.json"
        value = json.loads(path.read_text())
        value["shards"][3]["shard_index"] = 2 if problem == "duplicate" else 4
        path.write_text(json.dumps(value))
    with pytest.raises(RecoveryError):
        verify_local(tmp_path / INSTRUMENT, INSTRUMENT, ENTRY)


def test_digest_mismatch_and_malformed_archive_are_rejected(tmp_path):
    run, artifacts = _metadata()
    artifacts[0]["digest"] = "sha256:" + "0" * 64
    archives = {artifact["id"]: _zip({"file": "value"}) for artifact in artifacts}
    with pytest.raises(RecoveryError, match="digest mismatch"):
        recover(Client(run, artifacts, archives), tmp_path, INSTRUMENT, ENTRY, False)
    artifacts[0].pop("digest")
    archives[artifacts[0]["id"]] = b"not zip"
    with pytest.raises(RecoveryError, match="malformed"):
        recover(Client(run, artifacts, archives), tmp_path, INSTRUMENT, ENTRY, False)
