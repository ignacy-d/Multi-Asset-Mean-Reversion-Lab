"""Local-only candidate-aligned Ornstein--Uhlenbeck diagnostic runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from mr_lab.ornstein_uhlenbeck import (
    OrnsteinUhlenbeckProcessSpec,
    align_candidate_states,
    build_candidate_ou_states,
    candidate_process_keys,
)
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.stage4a_runner import (
    FROZEN_INSTRUMENTS,
    load_corpus_registry,
    read_and_validate_manifest,
    validate_registry_entry,
)
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID, deduplicate_states
from mr_lab.stage4b_runner import assemble_signal_states

OUTPUTS = (
    "ornstein-uhlenbeck-candidate-states.csv",
    "summary.json",
    "execution-audit.json",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _commit_sha():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(corpus_dir, output_dir, instrument, registry_path, windows):
    """Run diagnostics against one explicitly selected, verified 2024 corpus."""
    specs = tuple(OrnsteinUhlenbeckProcessSpec(value) for value in windows)
    if len({spec.process_spec_id for spec in specs}) != len(specs):
        raise ValueError("window transition specifications must be unique")
    registry = load_corpus_registry(registry_path)
    entry = validate_registry_entry(
        instrument, registry["instruments"][instrument], require_verified=True
    )
    manifest = read_and_validate_manifest(corpus_dir, instrument)
    validate_registry_entry(instrument, entry, manifest=manifest, require_verified=True)
    dataset = load_offline_corpus(corpus_dir)
    signal_states = assemble_signal_states(dataset, manifest)
    events = deduplicate_states(signal_states)
    required_keys = candidate_process_keys(events)
    process_states = tuple(
        state
        for spec in specs
        for state in build_candidate_ou_states(signal_states, spec, required_keys)
    )
    rows = align_candidate_states(events, process_states)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUTS[0]
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    statuses = Counter(row["status"] for row in rows)
    summary = {
        "candidate_event_count": len(events),
        "candidate_state_row_count": len(rows),
        "process_spec_ids": [spec.process_spec_id for spec in specs],
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "status_counts": dict(sorted(statuses.items())),
        "window_transitions": [spec.window_transitions for spec in specs],
    }
    (output_dir / "summary.json").write_text(_json(summary) + "\n")
    audit = {
        **summary,
        "assembled_dataset_id": entry["assembled_dataset_id"],
        "corpus_id": entry["corpus_id"],
        "instrument": instrument,
        "requested_end_date": entry["requested_end_date"],
        "requested_start_date": entry["requested_start_date"],
        "source_artifact_id": entry["source_artifact_id"],
        "source_artifact_name": entry["source_artifact_name"],
        "source_mode": entry.get("source_mode", "github-artifact"),
        "source_acquisition_commit_sha": entry.get("source_acquisition_commit_sha"),
        "source_commit_sha": _commit_sha(),
        "source_workflow_run_id": entry["source_workflow_run_id"],
        "output_sha256": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in OUTPUTS[:2]
        },
    }
    (output_dir / "execution-audit.json").write_text(_json(audit) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=FROZEN_INSTRUMENTS, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--window-transitions", nargs="+", type=int, required=True, metavar="N"
    )
    args = parser.parse_args(argv)
    run(
        args.corpus_dir,
        args.output_dir,
        args.instrument,
        args.registry,
        args.window_transitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
