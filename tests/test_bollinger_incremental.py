import json
from pathlib import Path

import pytest

from mr_lab.bollinger_incremental import BollingerIncrementalError, run


def _candidate(event_id, family, timestamp, registry, lookback=20):
    identity = registry["instruments"]["EURUSD"]
    return {
        "candidate_event_id": event_id,
        "signal": {
            "instrument": "EURUSD",
            "signal_timestamp": timestamp,
            "benchmark_family": family,
            "signal_timeframe": "15m",
            "session": "london",
            "direction": "SHORT",
            "lookback": lookback,
            "source_corpus_id": identity["corpus_id"],
            "assembled_dataset_id": identity["assembled_dataset_id"],
        },
    }


def _trade(event_id, family, gross, *, tp=0.75, sl=0.25, stop=60):
    return {
        "instrument": "EURUSD",
        "benchmark_family": family,
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "SHORT",
        "lookback": 20,
        "signal_threshold": 2.0,
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "entry_mode": "immediate",
        "tp_target_fraction": tp,
        "sl_extension_fraction": sl,
        "time_stop_minutes": stop,
        "candidate_event_id": event_id,
        "complete": True,
        "gross_return_pips_adverse_first": gross,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_frozen_study_categories_costs_and_correlation(tmp_path):
    registry_path = Path("configs/stage4a-2024-corpus-registry.json")
    registry = json.loads(registry_path.read_text())
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    candidates = [
        _candidate("bb-only", "bollinger", "2024-01-02T09:15:00+00:00", registry),
        _candidate("vwap-only", "vwap", "2024-01-02T09:30:00+00:00", registry),
        _candidate("bb-both-1", "bollinger", "2024-01-02T09:45:00+00:00", registry),
        _candidate("vwap-both-1", "vwap", "2024-01-02T09:45:00+00:00", registry),
        _candidate("bb-both-2", "bollinger", "2024-02-02T09:45:00+00:00", registry),
        _candidate("vwap-both-2", "vwap", "2024-02-02T09:45:00+00:00", registry),
    ]
    _write_jsonl(source / "candidate-events.jsonl", candidates)
    _write_jsonl(
        source / "trades.jsonl",
        [
            _trade("bb-only", "bollinger", 3),
            _trade("vwap-only", "vwap", -2),
            _trade("bb-both-1", "bollinger", 1),
            _trade("vwap-both-1", "vwap", 2),
            _trade("bb-both-2", "bollinger", -1),
            _trade("vwap-both-2", "vwap", -2),
        ],
    )
    results, overlap = run(
        [source],
        output,
        registry_path,
        Path("configs/stage4c-ftmo-cost-profile-v1.json"),
    )
    gross = [row for row in results if row["spread_statistic"] == "gross"]
    standalone = next(
        row for row in gross if row["event_class"] == "bollinger-standalone"
    )
    assert standalone["unique_candidate_count"] == 3
    assert standalone["positive_months"] == 1
    assert len([row for row in results if row["event_class"] == "bollinger-only"]) == 17
    assert overlap[0]["intersection_count"] == 2
    correlation = (output / "intersection-correlation.csv").read_text()
    assert "1.0" in correlation
    audit = json.loads((output / "execution-audit.json").read_text())
    assert audit["research_year"] == 2024 and audit["selected_trade_rows"] == 6


def test_rejects_any_non_2024_candidate_before_reading_results(tmp_path):
    registry_path = Path("configs/stage4a-2024-corpus-registry.json")
    registry = json.loads(registry_path.read_text())
    source = tmp_path / "source"
    source.mkdir()
    _write_jsonl(
        source / "candidate-events.jsonl",
        [_candidate("sealed", "bollinger", "2025-01-02T09:15:00+00:00", registry)],
    )
    _write_jsonl(source / "trades.jsonl", [])
    with pytest.raises(BollingerIncrementalError, match="only frozen 2024"):
        run(
            [source],
            tmp_path / "out",
            registry_path,
            Path("configs/stage4c-ftmo-cost-profile-v1.json"),
        )
