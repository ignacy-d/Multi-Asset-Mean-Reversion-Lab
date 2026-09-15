import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mr_lab.run_manifest import (
    AuthenticatedDataIdentity,
    GeneratedArtifact,
    RunManifest,
    RunManifestError,
)
from mr_lab.spec_lock import SpecLockError, freeze_study, hash_spec_bytes
from mr_lab.spec_lock import main as freeze_main


def test_spec_hash_is_deterministic_and_uses_exact_bytes() -> None:
    payload = b"# Study\n\n- **status:** `FROZEN`\n"

    assert hash_spec_bytes(payload) == hashlib.sha256(payload).hexdigest()
    assert hash_spec_bytes(payload) == hash_spec_bytes(payload)
    assert hash_spec_bytes(payload + b"\n") != hash_spec_bytes(payload)


def test_freeze_writes_canonical_lock_for_explicit_spec(tmp_path: Path) -> None:
    spec = tmp_path / "study.md"
    spec.write_bytes(b"study_id: TEST-1\nstatus: FROZEN\n")
    output = tmp_path / "lock.json"

    freeze_study("TEST-1", spec, output)

    assert json.loads(output.read_text()) == {
        "lock_schema_version": "mr-lab-study-lock-v1",
        "spec_path": spec.as_posix(),
        "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "study_id": "TEST-1",
    }
    assert output.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    "status_line",
    (
        "status: DRAFT",
        "- **status:** `DRAFT` (allowed values: `DRAFT`, `FROZEN`)",
        'Status: "DRAFT"',
    ),
)
def test_freeze_refuses_obvious_draft_status(tmp_path: Path, status_line: str) -> None:
    spec = tmp_path / "draft.md"
    spec.write_text(f"# Study\n{status_line}\n")

    with pytest.raises(SpecLockError, match="DRAFT"):
        freeze_study("TEST-1", spec, tmp_path / "lock.json")


def test_freeze_refuses_any_draft_status_and_mismatched_declared_id(
    tmp_path: Path,
) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text("status: FROZEN\nstatus: DRAFT\n")
    with pytest.raises(SpecLockError, match="DRAFT"):
        freeze_study("TEST-1", draft, tmp_path / "draft-lock.json")

    mismatch = tmp_path / "mismatch.md"
    mismatch.write_text("- **study_id:** `OTHER-1`\n- **status:** `FROZEN`\n")
    with pytest.raises(SpecLockError, match="does not match"):
        freeze_study("TEST-1", mismatch, tmp_path / "mismatch-lock.json")


def test_freeze_refuses_missing_spec_and_silent_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "lock.json"
    with pytest.raises(SpecLockError, match="missing or unreadable"):
        freeze_study("TEST-1", tmp_path / "missing.md", output)

    spec = tmp_path / "frozen.md"
    spec.write_text("status: FROZEN\n")
    output.write_text("preserve me")
    with pytest.raises(SpecLockError, match="overwrite"):
        freeze_study("TEST-1", spec, output)
    assert output.read_text() == "preserve me"


def test_freeze_cli_reports_created_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    spec = tmp_path / "frozen.md"
    spec.write_text("status: FROZEN\n")
    output = tmp_path / "lock.json"

    assert (
        freeze_main(
            ["--study-id", "TEST-1", "--spec", str(spec), "--output", str(output)]
        )
        == 0
    )
    assert capsys.readouterr().out == f"study_lock={output}\n"


@pytest.mark.parametrize("study_id", ("", "test-1", "../TEST", "TEST_1", "-TEST"))
def test_freeze_rejects_invalid_study_id(tmp_path: Path, study_id: str) -> None:
    spec = tmp_path / "frozen.md"
    spec.write_text("status: FROZEN\n")

    with pytest.raises(SpecLockError, match="study_id"):
        freeze_study(study_id, spec, tmp_path / "lock.json")


def valid_manifest() -> RunManifest:
    return RunManifest(
        study_id="TEST-1",
        frozen_spec_sha256="a" * 64,
        git_revision="b" * 40,
        registry_identity="sha256:" + "c" * 64,
        registry_path="configs/registry.json",
        authenticated_data=(
            AuthenticatedDataIdentity(
                name="development-corpus",
                identity="sha256:" + "d" * 64,
                path="explicit/corpus",
            ),
        ),
        runner_command=("uv", "run", "example-runner", "--frozen-lock", "lock.json"),
        parameters={"horizon": 60, "thresholds": [1.0, 2.0]},
        random_seed=7,
        execution_status="SUCCEEDED",
        result_classification="KILL",
        generated_artifacts=(GeneratedArtifact("results/report.json", "e" * 64),),
    )


def test_manifest_schema_round_trip_is_canonical() -> None:
    manifest = valid_manifest()

    restored = RunManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.to_json() == manifest.to_json()
    assert restored.to_json().endswith("\n")


def test_manifest_detaches_and_freezes_mutable_parameters() -> None:
    parameters = {"thresholds": [1.0, 2.0]}
    manifest = replace(valid_manifest(), parameters=parameters)
    initial_json = manifest.to_json()

    parameters["thresholds"].append(3.0)

    assert manifest.to_json() == initial_json
    with pytest.raises(TypeError):
        manifest.parameters["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("frozen_spec_sha256", "unknown", "frozen_spec_sha256"),
        ("git_revision", "main", "git_revision"),
        ("registry_identity", "", "registry_identity"),
        ("registry_path", "", "registry_path"),
        ("authenticated_data", (), "authenticated_data"),
        ("authenticated_data", [], "authenticated_data"),
        ("runner_command", (), "runner_command"),
        ("runner_command", ["runner"], "runner_command"),
        ("generated_artifacts", (), "generated artifacts"),
        ("generated_artifacts", [], "generated_artifacts"),
    ),
)
def test_manifest_rejects_missing_required_provenance(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(RunManifestError, match=message):
        replace(valid_manifest(), **{field: value})


def test_manifest_rejects_fabricated_or_inconsistent_completion() -> None:
    with pytest.raises(RunManifestError, match="successful run requires"):
        replace(valid_manifest(), result_classification=None)
    with pytest.raises(RunManifestError, match="only a successful"):
        replace(valid_manifest(), execution_status="FAILED")
    with pytest.raises(RunManifestError, match="non-finite"):
        replace(valid_manifest(), parameters={"result": float("nan")})


def test_manifest_rejects_ambiguous_or_untyped_provenance() -> None:
    data = valid_manifest().authenticated_data[0]
    artifact = valid_manifest().generated_artifacts[0]
    with pytest.raises(RunManifestError, match="identity objects"):
        replace(valid_manifest(), authenticated_data=("invented",))
    with pytest.raises(RunManifestError, match="duplicate name"):
        replace(valid_manifest(), authenticated_data=(data, data))
    with pytest.raises(RunManifestError, match="artifact objects"):
        replace(valid_manifest(), generated_artifacts=("report.json",))
    with pytest.raises(RunManifestError, match="duplicate paths"):
        replace(valid_manifest(), generated_artifacts=(artifact, artifact))


def test_manifest_rejects_missing_and_unknown_schema_fields() -> None:
    value = valid_manifest().to_dict()
    del value["registry_identity"]
    value["invented"] = "fallback"

    with pytest.raises(RunManifestError, match="fields mismatch"):
        RunManifest.from_dict(value)
