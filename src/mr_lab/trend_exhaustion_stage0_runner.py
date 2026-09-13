"""CLI for the preregistered 2024-only Trend Exhaustion Stage-0 study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mr_lab.data import Timeframe, resample_bars
from mr_lab.failed_breakout_stage0_runner import _module_a_references
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.stage4a import DirectionalPathRequest, diagnose_directional_paths
from mr_lab.stage4a_runner import (
    FROZEN_INSTRUMENTS,
    load_corpus_registry,
    read_and_validate_manifest,
    validate_registry_entry,
)
from mr_lab.trend_exhaustion import (
    DISPLACEMENT_THRESHOLDS,
    TrendExhaustionSpec,
    detect_trend_exhaustion,
)
from mr_lab.trend_exhaustion_stage0 import (
    ModuleAReference,
    Observation,
    aggregate,
    calculate_overlap,
    classify,
)


def run(corpus_root: Path, output_dir: Path, registry_path: Path) -> tuple[Path, ...]:
    registry = load_corpus_registry(registry_path)
    entries = registry["instruments"]
    if not isinstance(entries, dict):
        raise ValueError("registry instruments must be a mapping")
    observations: list[Observation] = []
    references: list[ModuleAReference] = []
    for instrument in FROZEN_INSTRUMENTS:
        entry = validate_registry_entry(
            instrument, entries[instrument], require_verified=True
        )
        corpus_dir = corpus_root / instrument
        manifest = read_and_validate_manifest(corpus_dir, instrument)
        validate_registry_entry(
            instrument, entry, manifest=manifest, require_verified=True
        )
        dataset = load_offline_corpus(corpus_dir)
        if dataset.metadata.dataset_id != manifest["assembled_dataset_id"]:
            raise ValueError("loaded dataset identity disagrees with manifest")
        m15 = resample_bars(dataset.bars, Timeframe("15m")).bars
        exact = {bar.available_at: bar for bar in m15}
        pairs = []
        for threshold in DISPLACEMENT_THRESHOLDS:
            for event in detect_trend_exhaustion(m15, TrendExhaustionSpec(threshold)):
                signal = exact[event.signal_timestamp]
                pairs.append(
                    (
                        event,
                        DirectionalPathRequest(
                            instrument,
                            event.signal_timestamp,
                            event.direction,
                            signal.close,
                        ),
                    )
                )
        diagnostics = diagnose_directional_paths(
            (request for _, request in pairs), dataset.bars
        )
        observations.extend(
            Observation(event, path)
            for (event, _), path in zip(pairs, diagnostics, strict=True)
        )
        references.extend(
            ModuleAReference(x.instrument, x.signal_timestamp)
            for x in _module_a_references(dataset, manifest)
        )
    rows = aggregate(observations, FROZEN_INSTRUMENTS)
    overlap = calculate_overlap(observations, references)
    result = {
        "schema": "trend-exhaustion-stage0-v1",
        "execution_status": "complete",
        "classification": classify(rows),
        "metrics": rows,
        "module_a_overlap": overlap,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = (output_dir / "events.jsonl", output_dir / "summary.json")
    output_paths[0].write_text(
        "".join(
            json.dumps(
                {"event": item.event.as_dict(), "path": item.path},
                default=str,
                sort_keys=True,
            )
            + "\n"
            for item in observations
        ),
        encoding="utf-8",
    )
    output_paths[1].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered 2024 Trend Exhaustion Stage-0"
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/stage4a-2024-corpus-registry.json"),
    )
    args = parser.parse_args(argv)
    for path in run(args.corpus_root, args.output_dir, args.registry):
        print(f"result={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
