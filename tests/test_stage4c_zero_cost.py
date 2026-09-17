from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy

import pytest

from mr_lab.stage4b import SEMANTICS, STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import INSTRUMENTS, adjusted_execution_pips
from mr_lab.stage4c_zero_cost import (
    METHODOLOGY_ID,
    MODE,
    ZeroCostDiagnosticError,
    transform_trade,
    validate_instrument,
)
from mr_lab.stage4c_zero_cost_runner import run_manifest, run_rows, validate_2024_path


def trade(instrument="EURCAD", adverse=2.0, favorable=3.0, event="one"):
    return {
        "instrument": instrument,
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "LONG",
        "lookback": 20,
        "signal_threshold": 2.0,
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "entry_mode": "immediate",
        "tp_target_fraction": 0.5,
        "sl_extension_fraction": 0.5,
        "time_stop_minutes": 30,
        "candidate_event_id": event,
        "complete": True,
        "gross_return_pips_adverse_first": adverse,
        "gross_return_pips_favorable_first": favorable,
    }


def audit(instrument="EURCAD"):
    return {
        "instrument": instrument,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus-identity",
        "assembled_dataset_id": "dataset-identity",
        "requested_start_date": "2024-01-01",
        "requested_end_date": "2024-12-31",
    }


def test_identity_transform_and_jpy_has_no_adjustment():
    original = trade("CADJPY", adverse=-4.0, favorable=5.0)
    frozen = deepcopy(original)
    result = transform_trade(original, "CADJPY")
    assert result["zero_cost_pips_adverse_first"] == -4.0
    assert result["zero_cost_pips_favorable_first"] == 5.0
    assert result["zero_cost_pips_favorable_first"] != adjusted_execution_pips(
        5.0, "USDJPY"
    )
    assert original == frozen


def test_new_pair_runs_and_artifacts_are_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_ZERO_COST_SOURCE_COMMIT", "a" * 40)
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        run_rows(
            [trade(event="b"), trade(adverse=-1.0, event="a")],
            output,
            "EURCAD",
            audit(),
            "/explicit/EURCAD_2024_CORPUS",
        )
        outputs.append(output)
    names = (
        "stage4c-zero-cost-trade-matrix.csv",
        "stage4c-zero-cost-regime-breadth.csv",
        "stage4c-zero-cost-summary.json",
        "stage4c-zero-cost-report.md",
        "execution-audit.json",
    )
    assert all(
        (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
        for name in names
    )
    recorded = json.loads((outputs[0] / "execution-audit.json").read_text())
    assert recorded["mode"] == MODE
    assert recorded["zero_cost_diagnostic_methodology_id"] == METHODOLOGY_ID
    assert recorded["currency_conversion_adjustment"] is False


def test_mixed_rows_and_malformed_ticker_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_ZERO_COST_SOURCE_COMMIT", "a" * 40)
    with pytest.raises(ZeroCostDiagnosticError, match="mixed"):
        run_rows(
            [trade(), trade("GBPCHF", event="two")],
            tmp_path,
            "EURCAD",
            audit(),
            "/explicit/EURCAD_2024_CORPUS",
        )
    for malformed in ("eurcad", "EUR/CAD", "EUROCAD", "EUR"):
        with pytest.raises(ZeroCostDiagnosticError):
            validate_instrument(malformed)


def test_year_is_rejected_before_file_inspection(tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(type(tmp_path), "open", lambda *_a, **_k: touched.append(True))
    with pytest.raises(ZeroCostDiagnosticError, match="discovery year"):
        validate_2024_path(tmp_path / "sealed", 2025)
    assert touched == []
    with pytest.raises(ZeroCostDiagnosticError, match="ambiguous year"):
        validate_2024_path(tmp_path / "corpus", 2024)


def _make_corpus(path, instrument):
    path.mkdir()
    row = _json_line(trade(instrument))
    trades = path / "trades.jsonl"
    trades.write_text(row)
    source = audit(instrument)
    source["output_sha256"] = {
        "trades.jsonl": hashlib.sha256(trades.read_bytes()).hexdigest()
    }
    (path / "execution-audit.json").write_text(json.dumps(source))


def _json_line(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def test_manifest_processes_only_explicit_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_ZERO_COST_SOURCE_COMMIT", "a" * 40)
    corpus = tmp_path / "EURCAD_2024"
    _make_corpus(corpus, "EURCAD")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("instrument", "corpus_path", "year")
        )
        writer.writeheader()
        writer.writerow({"instrument": "EURCAD", "corpus_path": corpus, "year": 2024})
    output = tmp_path / "output"
    run_manifest(manifest, output)
    assert (output / "EURCAD" / "execution-audit.json").is_file()
    assert not (output / "GBPCHF").exists()
    assert (output / "stage4c-zero-cost-cross-asset.csv").is_file()


def test_frozen_contracts_are_not_changed():
    assert INSTRUMENTS == ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDJPY")
    before = json.dumps(SEMANTICS, sort_keys=True)
    expected = "sha256:" + hashlib.sha256(before.encode()).hexdigest()
    assert expected == STAGE4B_METHODOLOGY_ID
