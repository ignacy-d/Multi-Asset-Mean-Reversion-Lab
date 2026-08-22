"""CLI and deterministic outputs for frozen Stage 4B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.stage4a_runner import (
    FROZEN_INSTRUMENTS,
    assemble_frozen_signals,
    load_corpus_registry,
    read_and_validate_manifest,
    validate_registry_entry,
)
from mr_lab.stage4b import (
    ENTRY_MODES,
    SIGNAL_THRESHOLD,
    SL_FRACTIONS,
    STAGE4B_METHODOLOGY_ID,
    STAGE4B_REPORT_SCHEMA_VERSION,
    TIME_STOPS_MINUTES,
    TP_FRACTIONS,
    construct_entry,
    deduplicate_signals,
    path_diagnostic,
    simulate_exit,
    target_already_passed,
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def run(
    corpus_dir: Path, output_dir: Path, instrument: str, registry_path: Path
) -> dict[str, Path]:
    registry = load_corpus_registry(registry_path)
    entry = validate_registry_entry(
        instrument, registry["instruments"][instrument], require_verified=True
    )
    manifest = read_and_validate_manifest(corpus_dir, instrument)
    validate_registry_entry(instrument, entry, manifest=manifest, require_verified=True)
    dataset = load_offline_corpus(corpus_dir)
    print("STAGE4B_PROGRESS corpus_loaded", flush=True)
    signals = [
        s
        for s in assemble_frozen_signals(dataset, manifest)
        if s.threshold == SIGNAL_THRESHOLD
    ]
    print("STAGE4B_PROGRESS signal_generation_complete", flush=True)
    events = deduplicate_signals(signals)
    print(
        f"STAGE4B_PROGRESS candidate_dedup_complete candidate_count={len(events)}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate-events.jsonl"
    candidate_path.write_text("".join(_json(e.as_dict()) + "\n" for e in events))
    diagnostic_rows = []
    trade_rows = []
    audit_counts = defaultdict(int)
    with (output_dir / "trades.jsonl").open("w", encoding="utf-8") as trades:
        for number, event in enumerate(events, 1):
            for diagnostic in path_diagnostic(event, dataset.bars):
                diagnostic_rows.append(
                    {"candidate_event_id": event.candidate_event_id, **diagnostic}
                )
            for mode in ENTRY_MODES:
                constructed = construct_entry(event, dataset.bars, mode)
                if not constructed.executed:
                    audit_counts[(mode, "no_entry")] += 1
                    continue
                for tp in TP_FRACTIONS:
                    if target_already_passed(constructed, tp):
                        audit_counts[(mode, f"passed_{tp}")] += len(SL_FRACTIONS) * len(
                            TIME_STOPS_MINUTES
                        )
                        continue
                    for sl in SL_FRACTIONS:
                        for stop in TIME_STOPS_MINUTES:
                            result = simulate_exit(
                                event,
                                constructed,
                                dataset.bars,
                                tp_fraction=tp,
                                sl_fraction=sl,
                                time_stop_minutes=stop,
                            )
                            row = {
                                "candidate_event_id": event.candidate_event_id,
                                "instrument": instrument,
                                "benchmark_family": event.signal.benchmark_family,
                                "signal_timeframe": str(event.signal.signal_timeframe),
                                "session": event.signal.session,
                                "direction": event.signal.direction.name,
                                "lookback": event.signal.lookback,
                                "signal_threshold": SIGNAL_THRESHOLD,
                                "entry_mode": mode,
                                "tp_fraction": tp,
                                "sl_extension_fraction": sl,
                                "time_stop_minutes": stop,
                                "entry": asdict(constructed),
                                "result": asdict(result),
                            }
                            trades.write(_json(row) + "\n")
                            trade_rows.append(row)
            if number % 1000 == 0:
                print(
                    f"STAGE4B_PROGRESS trade_simulation candidates={number}", flush=True
                )
    fields = sorted({key for row in diagnostic_rows for key in row})
    with (output_dir / "entry-diagnostics.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    groups = defaultdict(list)
    group_fields = (
        "instrument",
        "benchmark_family",
        "signal_timeframe",
        "session",
        "direction",
        "lookback",
        "signal_threshold",
        "entry_mode",
        "tp_fraction",
        "sl_extension_fraction",
        "time_stop_minutes",
    )
    for row in trade_rows:
        groups[tuple(row[k] for k in group_fields)].append(row)
    matrix_fields = (
        *group_fields,
        "executed_trade_count",
        "complete_trade_count",
        "incomplete_trade_count",
        "ambiguous_same_minute_count",
        "mean_gross_pips_adverse_first",
    )
    with (output_dir / "trade-matrix.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=matrix_fields)
        writer.writeheader()
        for key in sorted(groups, key=str):
            rows = groups[key]
            complete = [r for r in rows if r["result"]["complete"]]
            pips = [r["result"]["gross_pips_adverse_first"] for r in complete]
            writer.writerow(
                dict(zip(group_fields, key, strict=True))
                | {
                    "executed_trade_count": len(rows),
                    "complete_trade_count": len(complete),
                    "incomplete_trade_count": len(rows) - len(complete),
                    "ambiguous_same_minute_count": sum(
                        r["result"]["exit_ordering"] == "ambiguous_same_minute"
                        for r in rows
                    ),
                    "mean_gross_pips_adverse_first": sum(pips) / len(pips)
                    if pips
                    else "",
                }
            )
    hashes = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in (
            "candidate-events.jsonl",
            "trades.jsonl",
            "entry-diagnostics.csv",
            "trade-matrix.csv",
        )
    }
    summary = {
        "reporting_schema_version": STAGE4B_REPORT_SCHEMA_VERSION,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "candidate_event_count": len(events),
        "executed_trade_configuration_count": len(trade_rows),
        "incomplete_count": sum(not r["result"]["complete"] for r in trade_rows),
        "hashes": hashes,
    }
    (output_dir / "summary.json").write_text(_json(summary) + "\n")
    (output_dir / "report.md").write_text(
        "# Stage 4B — gross trade construction\n\n"
        "**2024 in-sample research. No 2025 data was used.** Configurations, "
        "benchmark and lookback rows are dependent, not independent trials. "
        "Gross BID expectancy is not executable net expectancy; Stage 4C costs "
        "are required. M15 is the lead interpretation; M5 and H1 are robustness "
        "evidence. No winner is selected automatically.\n"
    )
    audit = {
        "source_commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "reporting_schema_version": STAGE4B_REPORT_SCHEMA_VERSION,
        "registry_schema": registry["registry_schema_version"],
        "instrument": instrument,
        "source_workflow_run_id": entry["source_workflow_run_id"],
        "source_artifact_id": entry["source_artifact_id"],
        "source_artifact_name": entry["source_artifact_name"],
        "corpus_id": entry["corpus_id"],
        "assembled_dataset_id": entry["assembled_dataset_id"],
        "requested_start_date": entry["requested_start_date"],
        "requested_end_date": entry["requested_end_date"],
        **summary,
    }
    (output_dir / "execution-audit.json").write_text(_json(audit) + "\n")
    print("STAGE4B_PROGRESS reporting audit_complete", flush=True)
    return {p.name: p for p in output_dir.iterdir() if p.is_file()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", choices=FROZEN_INSTRUMENTS, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.corpus_dir, args.output_dir, args.instrument, args.registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
