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
    BollingerStrategySpec,
    build_bollinger_features,
)
from mr_lab.bollinger_benchmark import (
    DEFAULT_LOOKBACKS as BOLLINGER_LOOKBACKS,
)
from mr_lab.bollinger_benchmark import (
    DEFAULT_THRESHOLDS as BOLLINGER_THRESHOLDS,
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
from mr_lab.vwap_benchmark import (
    DEFAULT_THRESHOLDS as VWAP_THRESHOLDS,
)
from mr_lab.vwap_benchmark import (
    DEFAULT_VOLATILITY_LOOKBACKS as VWAP_LOOKBACKS,
)
from mr_lab.vwap_benchmark import (
    VwapStrategySpec,
    build_vwap_features,
)
from mr_lab.vwap_m1_robustness import (
    CANONICAL_M1,
    VwapRobustnessStrategySpec,
    build_canonical_m1_vwap_features,
)

FROZEN_INSTRUMENTS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDJPY")
FROZEN_TIMEFRAMES = (Timeframe("5m"), Timeframe("15m"), Timeframe("1h"))
REGISTRY_SCHEMA_VERSION = "stage-4a-2024-corpus-registry-v1"
RESEARCH_FILES = ("events.jsonl", "matrix.csv", "summary.json", "report.md")
REGISTRY_STATUSES = frozenset(
    ("verified", "pending-reviewer-verification", "pending-acquisition")
)


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
    declared: list[str] = []
    for field in ("successful_component_dates", "confirmed_absent_dates"):
        values = manifest.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise Stage4ARunnerError(f"manifest {field} must be a list of ISO dates")
        declared.extend(values)
    components = manifest.get("components", ())
    if not isinstance(components, list):
        raise Stage4ARunnerError("manifest components must be a list")
    for component in components:
        if not isinstance(component, dict):
            raise Stage4ARunnerError("every manifest component must be an object")
        requested_day = component.get("requested_day")
        if not isinstance(requested_day, str):
            raise Stage4ARunnerError(
                "every component requested_day must be an ISO date"
            )
        declared.append(requested_day)
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


def validate_registry_entry(
    instrument: str,
    entry: object,
    *,
    manifest: Mapping[str, object] | None = None,
    require_verified: bool = False,
) -> dict[str, object]:
    """Validate one source-selection entry and optionally bind it to a manifest."""
    if not isinstance(entry, dict):
        raise Stage4ARunnerError("registry entry must be an object")
    if entry.get("instrument") != instrument:
        raise Stage4ARunnerError("registry key and entry instrument disagree")
    status = entry.get("verification_status")
    if status not in REGISTRY_STATUSES:
        raise Stage4ARunnerError("registry entry has an unsupported status")
    expected_name = f"dukascopy-{instrument}-m1-bid-2024-full-year"
    if entry.get("source_artifact_name") != expected_name:
        raise Stage4ARunnerError("registry entry has an unexpected artifact name")
    if (entry.get("requested_start_date"), entry.get("requested_end_date")) != (
        "2024-01-01",
        "2024-12-31",
    ):
        raise Stage4ARunnerError("registry entry must declare the exact 2024 range")
    if require_verified and status != "verified":
        raise Stage4ARunnerError(f"{instrument} registry entry is not verified")
    if status == "verified":
        for field in ("source_workflow_run_id", "source_artifact_id"):
            value = entry.get(field)
            if type(value) is not int or value <= 0:
                raise Stage4ARunnerError(f"verified registry {field} must be positive")
        for field in ("corpus_id", "assembled_dataset_id"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise Stage4ARunnerError(f"verified registry {field} must be non-empty")
    elif any(
        entry.get(field) is not None
        for field in (
            "source_workflow_run_id",
            "source_artifact_id",
            "corpus_id",
            "assembled_dataset_id",
        )
    ):
        raise Stage4ARunnerError(
            "pending registry entries must not contain partial pins"
        )
    if manifest is not None:
        for field in (
            "instrument",
            "corpus_id",
            "assembled_dataset_id",
            "requested_start_date",
            "requested_end_date",
        ):
            if entry.get(field) != manifest.get(field):
                raise Stage4ARunnerError(f"registry and manifest disagree on {field}")
    return dict(entry)


def load_corpus_registry(path: Path) -> dict[str, object]:
    """Load the one supported registry schema and exact frozen universe."""
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage4ARunnerError("invalid or missing corpus registry") from error
    if (
        not isinstance(registry, dict)
        or registry.get("registry_schema_version") != REGISTRY_SCHEMA_VERSION
    ):
        raise Stage4ARunnerError("unsupported or missing registry schema version")
    entries = registry.get("instruments")
    if not isinstance(entries, dict) or set(entries) != set(FROZEN_INSTRUMENTS):
        raise Stage4ARunnerError("registry must contain exactly the frozen universe")
    for instrument in FROZEN_INSTRUMENTS:
        validate_registry_entry(instrument, entries[instrument])
    return registry


def select_verified_registry_entries(
    registry: Mapping[str, object], selection: str
) -> tuple[dict[str, object], ...]:
    """Fail ALL centrally unless every frozen source is ready."""
    if selection not in (*FROZEN_INSTRUMENTS, "ALL"):
        raise Stage4ARunnerError("unsupported instrument selection")
    instruments = FROZEN_INSTRUMENTS if selection == "ALL" else (selection,)
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    return tuple(
        validate_registry_entry(name, entries[name], require_verified=True)
        for name in instruments
    )


def assemble_frozen_signals(dataset, manifest: Mapping[str, object]):
    """Build the complete frozen grid while reusing each Stage 3B feature builder."""
    m1_observations = build_research_observations(dataset.bars, DEFAULT_SESSION_SPEC)
    signals = []
    for timeframe in FROZEN_TIMEFRAMES:
        observations = build_research_observations(
            resample_bars(dataset.bars, timeframe).bars, DEFAULT_SESSION_SPEC
        )
        for lookback in VWAP_LOOKBACKS:
            native = build_vwap_features(observations, DEFAULT_SESSION_SPEC, lookback)
            canonical = build_canonical_m1_vwap_features(
                m1_observations, observations, DEFAULT_SESSION_SPEC, lookback
            )
            for threshold in VWAP_THRESHOLDS:
                native_id = VwapStrategySpec(lookback, threshold).strategy_spec_id
                robust_id = VwapRobustnessStrategySpec(
                    lookback, threshold, CANONICAL_M1
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
        for lookback in BOLLINGER_LOOKBACKS:
            bollinger = build_bollinger_features(observations, lookback)
            for threshold in BOLLINGER_THRESHOLDS:
                bollinger_id = BollingerStrategySpec(
                    lookback, threshold
                ).strategy_spec_id
                common = {
                    "threshold": threshold,
                    "source_corpus_id": manifest["corpus_id"],
                    "assembled_dataset_id": manifest["assembled_dataset_id"],
                }
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
    *,
    registry_schema_version: str,
    manifest: Mapping[str, object],
) -> Path:
    """Add deterministic operational provenance after the four research outputs."""
    if not source_commit_sha:
        raise Stage4ARunnerError("source commit SHA is required")
    if registry_schema_version != REGISTRY_SCHEMA_VERSION:
        raise Stage4ARunnerError("unsupported registry schema version")
    entry = validate_registry_entry(
        str(entry.get("instrument")), entry, manifest=manifest, require_verified=True
    )
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
        "registry_schema_version": registry_schema_version,
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
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args(argv)
    paths, _events, _manifest = run_stage4a_2024(
        args.corpus_dir, args.output_dir, args.instrument
    )
    if args.registry:
        registry = load_corpus_registry(args.registry)
        entries = registry["instruments"]
        assert isinstance(entries, dict)
        entry = validate_registry_entry(
            args.instrument,
            entries[args.instrument],
            manifest=_manifest,
            require_verified=True,
        )
        paths["execution-audit.json"] = write_execution_audit(
            args.output_dir,
            entry,
            os.environ.get("GITHUB_SHA", ""),
            registry_schema_version=str(registry["registry_schema_version"]),
            manifest=_manifest,
        )
    for path in paths.values():
        print(f"result={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
