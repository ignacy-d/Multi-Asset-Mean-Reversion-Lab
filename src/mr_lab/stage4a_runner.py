"""Operational Stage 4A wiring for the frozen 2024 discovery corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from mr_lab.bollinger_benchmark import (
    CONTEXT_SESSIONS,
    DEFAULT_LOOKBACKS,
    DEFAULT_THRESHOLDS,
    BollingerStrategySpec,
    build_bollinger_features,
)
from mr_lab.data import Timeframe, resample_bars
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.providers.instruments import get_instrument_spec
from mr_lab.research import build_research_observations
from mr_lab.sessions import DEFAULT_SESSION_SPEC
from mr_lab.stage4a import (
    diagnose_events,
    frozen_signal_from_bollinger,
    frozen_signal_from_vwap,
)
from mr_lab.stage4a_reporting import (
    STAGE4A_REPORT_SCHEMA_VERSION,
    write_stage4a_outputs,
)
from mr_lab.vwap_benchmark import VwapStrategySpec, build_vwap_features
from mr_lab.vwap_m1_robustness import (
    CANONICAL_M1,
    VwapRobustnessStrategySpec,
    build_canonical_m1_vwap_features,
)

FROZEN_INSTRUMENTS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDJPY")
FROZEN_TIMEFRAMES = (Timeframe("5m"), Timeframe("15m"), Timeframe("1h"))
REGISTRY_SCHEMA_VERSION = "stage-4a-2024-corpus-registry-v1"
RESEARCH_FILES = ("events.jsonl", "matrix.csv", "summary.json", "report.md")


class Stage4ARunnerError(ValueError):
    """Raised before market data is loaded when operational identity is unsafe."""


def read_and_validate_manifest(corpus_dir: Path, instrument: str) -> dict[str, object]:
    """Validate every manifest-declared date before corpus bytes are inspected."""
    selected = get_instrument_spec(instrument).instrument
    try:
        manifest = json.loads(
            (corpus_dir / "corpus-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise Stage4ARunnerError("invalid or missing corpus manifest") from error
    if manifest.get("instrument") != selected:
        raise Stage4ARunnerError("selected instrument does not match corpus manifest")
    if (manifest.get("requested_start_date"), manifest.get("requested_end_date")) != (
        "2024-01-01",
        "2024-12-31",
    ):
        raise Stage4ARunnerError("corpus must declare the exact frozen 2024 range")
    declared = list(manifest.get("successful_component_dates", ()))
    declared += list(manifest.get("confirmed_absent_dates", ()))
    components = manifest.get("components", ())
    if not isinstance(components, list):
        raise Stage4ARunnerError("manifest components must be a list")
    declared += [
        item.get("requested_day") for item in components if isinstance(item, dict)
    ]
    try:
        dates = tuple(date.fromisoformat(value) for value in declared)
    except (TypeError, ValueError) as error:
        raise Stage4ARunnerError(
            "manifest contains an invalid declared date"
        ) from error
    if any(day.year != 2024 for day in dates):
        raise Stage4ARunnerError("manifest declares a date outside frozen 2024")
    for field in ("corpus_id", "assembled_dataset_id"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise Stage4ARunnerError(f"manifest {field} must be non-empty")
    return manifest


def assemble_frozen_signals(dataset, manifest: Mapping[str, object]):
    """Build the complete frozen grid while reusing each Stage 3B feature builder."""
    m1_observations = build_research_observations(dataset.bars, DEFAULT_SESSION_SPEC)
    signals = []
    for timeframe in FROZEN_TIMEFRAMES:
        observations = build_research_observations(
            resample_bars(dataset.bars, timeframe).bars, DEFAULT_SESSION_SPEC
        )
        for lookback in DEFAULT_LOOKBACKS:
            native = build_vwap_features(observations, DEFAULT_SESSION_SPEC, lookback)
            canonical = build_canonical_m1_vwap_features(
                m1_observations, observations, DEFAULT_SESSION_SPEC, lookback
            )
            bollinger = build_bollinger_features(observations, lookback)
            for threshold in DEFAULT_THRESHOLDS:
                native_id = VwapStrategySpec(lookback, threshold).strategy_spec_id
                robust_id = VwapRobustnessStrategySpec(
                    lookback, threshold, CANONICAL_M1
                ).strategy_spec_id
                bollinger_id = BollingerStrategySpec(
                    lookback, threshold
                ).strategy_spec_id
                common = {
                    "threshold": threshold,
                    "source_corpus_id": manifest["corpus_id"],
                    "assembled_dataset_id": manifest["assembled_dataset_id"],
                }
                signals.extend(
                    signal
                    for feature in native
                    if (
                        signal := frozen_signal_from_vwap(
                            feature,
                            benchmark_family="vwap",
                            strategy_spec_id=native_id,
                            **common,
                        )
                    )
                )
                signals.extend(
                    signal
                    for feature in canonical
                    if (
                        signal := frozen_signal_from_vwap(
                            feature,
                            benchmark_family="vwap-canonical-m1",
                            strategy_spec_id=robust_id,
                            **common,
                        )
                    )
                )
                for feature in bollinger:
                    contexts = (
                        None,
                        *(
                            context
                            for context in CONTEXT_SESSIONS
                            if context is not None
                            and context in feature.observation.sessions.active_sessions
                        ),
                    )
                    for context in contexts:
                        signal = frozen_signal_from_bollinger(
                            feature,
                            session=context,
                            strategy_spec_id=bollinger_id,
                            **common,
                        )
                        if signal is not None:
                            signals.append(signal)
    return tuple(signals)


def run_stage4a_2024(corpus_dir: Path, output_dir: Path, instrument: str):
    """Load once, diagnose one instrument batch, and reuse frozen reporting."""
    manifest = read_and_validate_manifest(corpus_dir, instrument)
    dataset = load_offline_corpus(corpus_dir)
    if dataset.metadata.instrument != instrument:
        raise Stage4ARunnerError("loaded dataset instrument disagrees with selection")
    if dataset.metadata.dataset_id != manifest["assembled_dataset_id"]:
        raise Stage4ARunnerError("loaded dataset identity disagrees with manifest")
    signals = assemble_frozen_signals(dataset, manifest)
    events = diagnose_events(signals, dataset.bars)
    paths = write_stage4a_outputs(events, output_dir)
    return paths, events, manifest


def write_execution_audit(
    output_dir: Path,
    entry: Mapping[str, object],
    source_commit_sha: str,
) -> Path:
    """Add deterministic operational provenance after the four research outputs."""
    if not source_commit_sha:
        raise Stage4ARunnerError("source commit SHA is required")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    hashes = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in RESEARCH_FILES
    }
    if any(hashes[name] != digest for name, digest in summary["hashes"].items()):
        raise Stage4ARunnerError("reporting hashes disagree with generated files")
    audit = {
        "assembled_dataset_id": entry["assembled_dataset_id"],
        "complete_path_event_count": summary["complete_path_total"],
        "corpus_id": entry["corpus_id"],
        "incomplete_path_event_count": summary["incomplete_path_total"],
        "instrument": entry["instrument"],
        "matrix_row_count": summary["matrix_row_count"],
        "registry_entry": dict(entry),
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "reporting_schema_version": STAGE4A_REPORT_SCHEMA_VERSION,
        "requested_end_date": entry["requested_end_date"],
        "requested_start_date": entry["requested_start_date"],
        "research_output_sha256": hashes,
        "source_artifact_id": entry["source_artifact_id"],
        "source_artifact_name": entry["source_artifact_name"],
        "source_commit_sha": source_commit_sha,
        "source_workflow_run_id": entry["source_workflow_run_id"],
        "stage4a_methodology_ids": summary["stage4a_methodology_ids"],
        "total_event_count": summary["event_total"],
    }
    path = output_dir / "execution-audit.json"
    path.write_text(json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Stage 4A 2024 diagnostics")
    parser.add_argument("--instrument", required=True, choices=FROZEN_INSTRUMENTS)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry-entry", type=Path)
    args = parser.parse_args(argv)
    paths, _events, _manifest = run_stage4a_2024(
        args.corpus_dir, args.output_dir, args.instrument
    )
    if args.registry_entry:
        entry = json.loads(args.registry_entry.read_text(encoding="utf-8"))
        paths["execution-audit.json"] = write_execution_audit(
            args.output_dir, entry, os.environ.get("GITHUB_SHA", "")
        )
    for path in paths.values():
        print(f"result={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
