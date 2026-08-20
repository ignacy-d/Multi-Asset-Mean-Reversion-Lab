import json
from pathlib import Path

import pytest

from mr_lab.stage4a_runner import Stage4ARunnerError, read_and_validate_manifest


def manifest(tmp_path: Path, **changes) -> Path:
    value = {
        "instrument": "EURUSD",
        "requested_start_date": "2024-01-01",
        "requested_end_date": "2024-12-31",
        "successful_component_dates": ["2024-01-02"],
        "confirmed_absent_dates": ["2024-01-01"],
        "components": [{"requested_day": "2024-01-02"}],
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
    }
    value.update(changes)
    (tmp_path / "corpus-manifest.json").write_text(json.dumps(value))
    return tmp_path


def test_manifest_identity_is_preserved(tmp_path):
    result = read_and_validate_manifest(manifest(tmp_path), "EURUSD")
    assert (result["corpus_id"], result["assembled_dataset_id"]) == (
        "corpus",
        "dataset",
    )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("requested_end_date", "2025-01-01"),
        ("successful_component_dates", ["2025-01-01"]),
        ("confirmed_absent_dates", ["2025-01-01"]),
        ("components", [{"requested_day": "2025-01-01"}]),
        ("instrument", "GBPUSD"),
    ],
)
def test_manifest_guard_rejects_before_load(tmp_path, change, value):
    with pytest.raises(Stage4ARunnerError):
        read_and_validate_manifest(manifest(tmp_path, **{change: value}), "EURUSD")


def test_registry_is_explicitly_blocked_until_github_pins_can_be_verified():
    registry = json.loads(Path("configs/stage4a-2024-corpus-registry.json").read_text())
    assert registry["registry_schema_version"] == "stage-4a-2024-corpus-registry-v1"
    assert set(registry["instruments"]) == {
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "AUDJPY",
    }
    for instrument, entry in registry["instruments"].items():
        assert entry["instrument"] == instrument
        assert entry["source_artifact_name"] == (
            f"dukascopy-{instrument}-m1-bid-2024-full-year"
        )
        assert entry["verification_status"].startswith("unverified-")


def test_workflow_has_fixed_selector_and_safety_controls():
    workflow = Path(".github/workflows/run-stage-4a-real-2024.yml").read_text()
    for choice in ("ALL", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDJPY"):
        assert choice in workflow
    assert "source_workflow_run_id:" not in workflow
    assert "fail-fast: false" in workflow
    assert "refs/heads/main" in workflow
    assert "configs/stage4a-2024-corpus-registry.json" in workflow
    assert "uv run mr-lab-stage4a" in workflow
    for name in (
        *("events.jsonl", "matrix.csv", "summary.json", "report.md"),
        "execution-audit.json",
    ):
        assert f"result/{name}" in workflow
