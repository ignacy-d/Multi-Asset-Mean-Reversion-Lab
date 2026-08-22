from pathlib import Path

import pytest

from mr_lab.stage4a_runner import (
    Stage4ARunnerError,
    load_corpus_registry,
    select_verified_registry_entries,
)
from mr_lab.stage4b_runner import OUTPUTS, _json


def test_gbpusd_pending_fails_closed():
    registry = load_corpus_registry(Path("configs/stage4a-2024-corpus-registry.json"))
    with pytest.raises(Stage4ARunnerError, match="not verified"):
        select_verified_registry_entries(registry, "GBPUSD")


def test_workflow_has_only_instrument_input_and_review_diagnostics():
    text = Path(".github/workflows/run-stage-4b-real-2024.yml").read_text()
    head = text.split("permissions:", 1)[0]
    assert head.count("type: choice") == 1
    assert all(
        name not in head
        for name in ("threshold:", "tp:", "sl:", "time_stop:", "entry_mode:")
    )
    compact = text.split("review-source", 1)[1]
    assert "result/entry-diagnostics.csv" in compact


def test_deterministic_json_and_output_contract():
    assert _json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert set(OUTPUTS) == {
        "candidate-events.jsonl",
        "trades.jsonl",
        "entry-diagnostics.csv",
        "trade-matrix.csv",
        "summary.json",
        "report.md",
    }
