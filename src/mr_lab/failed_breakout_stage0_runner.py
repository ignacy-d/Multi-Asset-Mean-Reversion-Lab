"""CLI for the frozen-2024 standalone Failed Breakout Stage-0 study."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mr_lab.failed_breakout import (
    PREREGISTERED_MINIMUM_DEPTHS,
    FailedBreakoutSpec,
    detect_failed_breakouts,
    generate_structural_levels,
)
from mr_lab.failed_breakout_stage0 import (
    FailedBreakoutObservation,
    ModuleAReference,
    write_outputs,
)
from mr_lab.ornstein_uhlenbeck import (
    FrozenOuEligibilityFilter,
    build_candidate_ou_states,
    candidate_process_keys,
    frozen_ou_eligibility_spec,
)
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.stage4a import DirectionalPathRequest, diagnose_directional_paths
from mr_lab.stage4a_runner import (
    FROZEN_INSTRUMENTS,
    load_corpus_registry,
    read_and_validate_manifest,
    validate_registry_entry,
)
from mr_lab.stage4b import deduplicate_states
from mr_lab.stage4b_runner import assemble_signal_states


def _module_a_references(dataset, manifest):
    """Generate references with the existing frozen Module A + OU definition."""
    states = assemble_signal_states(dataset, manifest)
    candidates = deduplicate_states(states)
    spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
    ou_states = build_candidate_ou_states(
        states, spec.process_spec, candidate_process_keys(candidates)
    )
    eligibility = FrozenOuEligibilityFilter(spec, ou_states)
    return tuple(
        ModuleAReference(event.signal.instrument, event.signal.signal_timestamp)
        for event in candidates
        if eligibility.evaluate(event).eligible
    )


def _instrument_observations(dataset):
    levels = generate_structural_levels(dataset.bars)
    exact = {bar.available_at: bar for bar in dataset.bars}
    pairs = []
    for depth in PREREGISTERED_MINIMUM_DEPTHS:
        for event in detect_failed_breakouts(
            dataset.bars, levels, FailedBreakoutSpec(depth)
        ):
            signal_bar = exact.get(event.signal_timestamp)
            if signal_bar is None:
                continue
            pairs.append(
                (
                    event,
                    depth,
                    DirectionalPathRequest(
                        event.instrument,
                        event.signal_timestamp,
                        event.direction,
                        signal_bar.close,
                    ),
                )
            )
    paths = diagnose_directional_paths(
        (request for _, _, request in pairs), dataset.bars
    )
    return tuple(
        FailedBreakoutObservation(event, depth, path)
        for (event, depth, _), path in zip(pairs, paths, strict=True)
    )


def run(corpus_root: Path, output_dir: Path, registry_path: Path):
    registry = load_corpus_registry(registry_path)
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    observations, module_a = [], []
    for instrument in FROZEN_INSTRUMENTS:
        entry = validate_registry_entry(
            instrument,
            entries[instrument],
            require_verified=True,
        )
        corpus_dir = corpus_root / instrument
        manifest = read_and_validate_manifest(corpus_dir, instrument)
        validate_registry_entry(
            instrument, entry, manifest=manifest, require_verified=True
        )
        dataset = load_offline_corpus(corpus_dir)
        if dataset.metadata.dataset_id != manifest["assembled_dataset_id"]:
            raise ValueError("loaded dataset identity disagrees with manifest")
        observations.extend(_instrument_observations(dataset))
        module_a.extend(_module_a_references(dataset, manifest))
    return write_outputs(observations, module_a, FROZEN_INSTRUMENTS, output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen-2024 standalone Failed Breakout Stage-0 study"
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/stage4a-2024-corpus-registry.json"),
    )
    args = parser.parse_args(argv)
    paths = run(args.corpus_root, args.output_dir, args.registry)
    for path in paths.values():
        print(f"result={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
