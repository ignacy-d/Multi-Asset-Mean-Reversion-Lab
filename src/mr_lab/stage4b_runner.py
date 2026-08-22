"""Operational runner and incremental reporting for frozen Stage 4B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mr_lab.bollinger_benchmark import (
    CONTEXT_SESSIONS,
    BollingerStrategySpec,
    build_bollinger_features,
)
from mr_lab.bollinger_benchmark import (
    signal_direction as bollinger_direction,
)
from mr_lab.data import resample_bars
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.research import Direction, build_research_observations
from mr_lab.sessions import DEFAULT_SESSION_SPEC
from mr_lab.stage4a_runner import (
    FROZEN_INSTRUMENTS,
    FROZEN_TIMEFRAMES,
    load_corpus_registry,
    read_and_validate_manifest,
    validate_registry_entry,
)
from mr_lab.stage4b import (
    SIGNAL_THRESHOLD,
    SL_FRACTIONS,
    STAGE4B_METHODOLOGY_ID,
    STAGE4B_REPORT_SCHEMA_VERSION,
    TIME_STOPS_MINUTES,
    TP_FRACTIONS,
    M1Index,
    NoEntryEligibilityFilter,
    SignalState,
    construct_eligible_entries,
    deduplicate_states,
    path_diagnostic,
    simulate_exit,
    target_already_passed,
)
from mr_lab.vwap_benchmark import (
    VwapStrategySpec,
    build_vwap_features,
)
from mr_lab.vwap_benchmark import (
    signal_direction as vwap_direction,
)
from mr_lab.vwap_m1_robustness import (
    CANONICAL_M1,
    VwapRobustnessStrategySpec,
    build_canonical_m1_vwap_features,
)

LOOKBACKS = (20, 40)
OUTPUTS = (
    "candidate-events.jsonl",
    "trades.jsonl",
    "entry-diagnostics.csv",
    "trade-matrix.csv",
    "summary.json",
    "report.md",
)
GROUP_FIELDS = (
    "instrument",
    "benchmark_family",
    "signal_timeframe",
    "session",
    "direction",
    "lookback",
    "signal_threshold",
    "filter_family",
    "filter_spec_id",
    "entry_mode",
    "tp_fraction",
    "sl_extension_fraction",
    "time_stop_minutes",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _state(
    feature, family, session, lookback, direction, z, e0, manifest, spec_id, qualifying
):
    observation = feature.observation
    return SignalState(
        observation.bar.instrument,
        observation.available_at,
        family,
        observation.bar.timeframe,
        session,
        direction,
        lookback,
        observation.bar.close,
        e0,
        z,
        manifest["corpus_id"],
        manifest["assembled_dataset_id"],
        spec_id,
        qualifying,
    )


def _two_sides(
    feature, family, session, lookback, z, e0, manifest, spec_id, direction_fn
):
    if z is None or not feature.observation.is_research_active:
        return ()
    selected = direction_fn(feature, SIGNAL_THRESHOLD)
    return tuple(
        _state(
            feature,
            family,
            session,
            lookback,
            d,
            z,
            e0,
            manifest,
            spec_id,
            selected is d,
        )
        for d in (Direction.LONG, Direction.SHORT)
    )


def assemble_signal_states(dataset, manifest):
    """Reuse Stage 3B builders while retaining every valid below-threshold state."""
    m1 = build_research_observations(dataset.bars, DEFAULT_SESSION_SPEC)
    states = []
    for timeframe in FROZEN_TIMEFRAMES:
        observations = build_research_observations(
            resample_bars(dataset.bars, timeframe).bars, DEFAULT_SESSION_SPEC
        )
        for lookback in LOOKBACKS:
            native = build_vwap_features(observations, DEFAULT_SESSION_SPEC, lookback)
            canonical = build_canonical_m1_vwap_features(
                m1, observations, DEFAULT_SESSION_SPEC, lookback
            )
            for family, features, spec in (
                (
                    "vwap",
                    native,
                    VwapStrategySpec(lookback, SIGNAL_THRESHOLD).strategy_spec_id,
                ),
                (
                    "vwap-canonical-m1",
                    canonical,
                    VwapRobustnessStrategySpec(
                        lookback, SIGNAL_THRESHOLD, CANONICAL_M1
                    ).strategy_spec_id,
                ),
            ):
                for feature in features:
                    states.extend(
                        _two_sides(
                            feature,
                            family,
                            feature.anchor_session,
                            lookback,
                            feature.vwap_deviation_z,
                            feature.vwap,
                            manifest,
                            spec,
                            vwap_direction,
                        )
                    )
        for lookback in LOOKBACKS:
            spec = BollingerStrategySpec(lookback, SIGNAL_THRESHOLD).strategy_spec_id
            for feature in build_bollinger_features(observations, lookback):
                sessions = (
                    None,
                    *(
                        s
                        for s in CONTEXT_SESSIONS
                        if s is not None
                        and s in feature.observation.sessions.active_sessions
                    ),
                )
                for session in sessions:
                    states.extend(
                        _two_sides(
                            feature,
                            "bollinger",
                            session,
                            lookback,
                            feature.bollinger_z,
                            feature.middle,
                            manifest,
                            spec,
                            bollinger_direction,
                        )
                    )
    return tuple(states)


@dataclass
class Aggregate:
    candidate_ids: set = field(default_factory=set)
    executed: int = 0
    no_entry: int = 0
    passed: int = 0
    complete: int = 0
    incomplete: int = 0
    ambiguous: int = 0
    tp: int = 0
    sl: int = 0
    time_stop: int = 0
    pips: list = field(default_factory=list)
    optimistic_pips: list = field(default_factory=list)
    bps: list = field(default_factory=list)
    mae: list = field(default_factory=list)
    mfe: list = field(default_factory=list)
    holding: list = field(default_factory=list)
    waits: list = field(default_factory=list)
    entry_r: list = field(default_factory=list)
    pre_adverse: list = field(default_factory=list)

    def add(self, event_id, entry, result):
        self.candidate_ids.add(event_id)
        self.executed += 1
        self.complete += result.complete
        self.incomplete += not result.complete
        if not result.complete:
            return
        self.ambiguous += result.exit_ordering == "ambiguous_same_minute"
        setattr(
            self,
            result.exit_reason
            if result.exit_reason in ("tp", "sl", "time_stop")
            else "sl",
            getattr(
                self,
                result.exit_reason
                if result.exit_reason in ("tp", "sl", "time_stop")
                else "sl",
            )
            + 1,
        )
        self.pips.append(result.gross_return_pips_adverse_first)
        self.optimistic_pips.append(result.gross_return_pips_favorable_first)
        self.bps.append(result.gross_return_bp_adverse_first)
        self.mae.append(result.mae_pips_certain)
        self.mfe.append(result.mfe_pips_certain)
        self.holding.append(result.holding_minutes)
        self.waits.append(entry.wait_minutes)
        self.entry_r.append(entry.r_at_entry)
        self.pre_adverse.append(entry.pre_entry_max_adverse)


def _mean(x):
    return statistics.fmean(x) if x else ""


def _median(x):
    return statistics.median(x) if x else ""


def _fraction(n, d):
    return n / d if d else 0.0


def _p90(x):
    return sorted(x)[max(0, math.ceil(0.9 * len(x)) - 1)] if x else ""


def _aggregate_row(key, a):
    n = a.complete
    wins = sum(x > 0 for x in a.pips)
    losses = sum(x < 0 for x in a.pips)
    return dict(zip(GROUP_FIELDS, key, strict=True)) | {
        "candidate_event_count": len(a.candidate_ids),
        "executed_trade_count": a.executed,
        "no_entry_count": a.no_entry,
        "target_already_passed_count": a.passed,
        "complete_trade_count": a.complete,
        "incomplete_trade_count": a.incomplete,
        "ambiguous_same_minute_count": a.ambiguous,
        "ambiguous_fraction": _fraction(a.ambiguous, n),
        "tp_count": a.tp,
        "tp_fraction": _fraction(a.tp, n),
        "sl_count": a.sl,
        "sl_fraction": _fraction(a.sl, n),
        "time_stop_count": a.time_stop,
        "time_stop_fraction": _fraction(a.time_stop, n),
        "mean_gross_pips": _mean(a.pips),
        "median_gross_pips": _median(a.pips),
        "mean_gross_bp": _mean(a.bps),
        "median_gross_bp": _median(a.bps),
        "win_fraction": _fraction(wins, n),
        "loss_fraction": _fraction(losses, n),
        "zero_fraction": _fraction(n - wins - losses, n),
        "mean_gross_pips_adverse_first": _mean(a.pips),
        "median_gross_pips_adverse_first": _median(a.pips),
        "mean_gross_pips_favorable_first": _mean(a.optimistic_pips),
        "mean_mae_pips": _mean(a.mae),
        "median_mae_pips": _median(a.mae),
        "mean_mfe_pips": _mean(a.mfe),
        "median_mfe_pips": _median(a.mfe),
        "median_holding_minutes": _median(a.holding),
        "p90_holding_minutes": _p90(a.holding),
        "median_entry_wait": _median(a.waits),
        "mean_entry_wait": _mean(a.waits),
        "median_r_at_entry": _median(a.entry_r),
        "median_pre_entry_max_adverse_extension": _median(a.pre_adverse),
    }


def _commit_sha():
    value = os.environ.get("GITHUB_SHA")
    if not value:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip()
    if len(value) != 40:
        return (_ for _ in ()).throw(ValueError("valid source commit SHA required"))
    return value


def run(corpus_dir, output_dir, instrument, registry_path, eligibility_filter=None):
    registry = load_corpus_registry(registry_path)
    registry_entry = validate_registry_entry(
        instrument, registry["instruments"][instrument], require_verified=True
    )
    manifest = read_and_validate_manifest(corpus_dir, instrument)
    validate_registry_entry(
        instrument, registry_entry, manifest=manifest, require_verified=True
    )
    dataset = load_offline_corpus(corpus_dir)
    index = M1Index(dataset.bars)
    print("STAGE4B_PROGRESS corpus_loaded m1_index_complete", flush=True)
    events = deduplicate_states(assemble_signal_states(dataset, manifest))
    print(
        "STAGE4B_PROGRESS signal_generation_complete "
        f"candidate_dedup_complete candidate_count={len(events)}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate-events.jsonl").write_text(
        "".join(_json(e.as_dict()) + "\n" for e in events)
    )
    groups = defaultdict(Aggregate)
    diagnostics = []
    filter_ = eligibility_filter or NoEntryEligibilityFilter()
    totals = defaultdict(int)
    with (output_dir / "trades.jsonl").open("w") as trades:
        for event in events:
            diagnostics.extend(
                {
                    "candidate_event_id": event.candidate_event_id,
                    "instrument": instrument,
                    "signal_timeframe": str(event.signal.signal_timeframe),
                    "benchmark_family": event.signal.benchmark_family,
                    **d,
                }
                for d in path_diagnostic(event, index)
            )
            decision, entries = construct_eligible_entries(event, index, filter_)
            if not decision.eligible:
                totals["filter_ineligible"] += 1
                continue
            paths = {
                e.timestamp: index.path(e.timestamp, 120)
                for e in entries.values()
                if e.executed and e.timestamp is not None
            }
            for mode, constructed in entries.items():
                base = (
                    instrument,
                    event.signal.benchmark_family,
                    str(event.signal.signal_timeframe),
                    event.signal.session,
                    event.signal.direction.name,
                    event.signal.lookback,
                    SIGNAL_THRESHOLD,
                    decision.filter_family,
                    decision.filter_spec_id,
                    mode,
                )
                for tp in TP_FRACTIONS:
                    for sl in SL_FRACTIONS:
                        for stop in TIME_STOPS_MINUTES:
                            key = (*base, tp, sl, stop)
                            agg = groups[key]
                            agg.candidate_ids.add(event.candidate_event_id)
                            if not constructed.executed:
                                agg.no_entry += 1
                                totals["no_entry"] += 1
                                continue
                            if target_already_passed(constructed, tp):
                                agg.passed += 1
                                totals["passed"] += 1
                                continue
                            result = simulate_exit(
                                event,
                                constructed,
                                paths[constructed.timestamp],
                                tp_fraction=tp,
                                sl_fraction=sl,
                                time_stop_minutes=stop,
                            )
                            agg.add(event.candidate_event_id, constructed, result)
                            totals["executed"] += 1
                            totals["incomplete"] += not result.complete
                            totals["ambiguous"] += (
                                result.exit_ordering == "ambiguous_same_minute"
                            )
                            row = {
                                **dict(zip(GROUP_FIELDS, key, strict=True)),
                                "candidate_event_id": event.candidate_event_id,
                                "filter_metadata": dict(decision.metadata),
                                "entry_wait_minutes": constructed.wait_minutes,
                                "r_at_entry": constructed.r_at_entry,
                                "pre_entry_max_favorable": (
                                    constructed.pre_entry_max_favorable
                                ),
                                "pre_entry_max_adverse": (
                                    constructed.pre_entry_max_adverse
                                ),
                                **asdict(result),
                            }
                            trades.write(_json(row) + "\n")
    _write_outputs(
        output_dir,
        events,
        groups,
        diagnostics,
        totals,
        registry,
        registry_entry,
        filter_,
    )
    return {p.name: p for p in output_dir.iterdir() if p.is_file()}


def _write_outputs(out, events, groups, diagnostics, totals, registry, entry, filter_):
    # Aggregate descriptive diagnostics by stable dimensions and horizon.
    dg = defaultdict(list)
    for row in diagnostics:
        dg[
            (
                row["instrument"],
                row["benchmark_family"],
                row["signal_timeframe"],
                row["horizon_minutes"],
            )
        ].append(row)
    drows = []
    for key, rows in sorted(dg.items()):
        result = dict(
            zip(
                (
                    "instrument",
                    "benchmark_family",
                    "signal_timeframe",
                    "horizon_minutes",
                ),
                key,
                strict=True,
            )
        ) | {"candidate_event_count": len(rows)}
        for level in ("25", "50"):
            for side in ("reversion", "extension"):
                times = [
                    r[f"time_to_{side}{level}"]
                    for r in rows
                    if r[f"time_to_{side}{level}"] is not None
                ]
                result[f"{side}{level}_hit_fraction"] = _fraction(len(times), len(rows))
                result[f"median_time_to_{side}{level}"] = _median(times)
            for ordering in ("reversion", "extension", "ambiguous_same_minute", "none"):
                result[f"ordering_{level}_{ordering}_fraction"] = _fraction(
                    sum(r[f"ordering_{level}"] == ordering for r in rows), len(rows)
                )
        drows.append(result)
    _csv(out / "entry-diagnostics.csv", drows)
    matrix = [
        _aggregate_row(k, a) for k, a in sorted(groups.items(), key=lambda x: str(x[0]))
    ]
    _csv(out / "trade-matrix.csv", matrix)
    summary = {
        "reporting_schema_version": STAGE4B_REPORT_SCHEMA_VERSION,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "candidate_event_count": len(events),
        "no_entry_count": totals["no_entry"],
        "target_already_passed_count": totals["passed"],
        "executed_trade_configuration_count": totals["executed"],
        "incomplete_count": totals["incomplete"],
        "ambiguous_count": totals["ambiguous"],
    }
    (out / "summary.json").write_text(_json(summary) + "\n")
    (out / "report.md").write_text(_report(matrix, summary))
    hashes = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in OUTPUTS
    }
    audit = {
        **summary,
        "source_commit_sha": _commit_sha(),
        "registry_schema": registry["registry_schema_version"],
        "instrument": entry["instrument"],
        "source_workflow_run_id": entry["source_workflow_run_id"],
        "source_artifact_id": entry["source_artifact_id"],
        "source_artifact_name": entry["source_artifact_name"],
        "corpus_id": entry["corpus_id"],
        "assembled_dataset_id": entry["assembled_dataset_id"],
        "requested_start_date": entry["requested_start_date"],
        "requested_end_date": entry["requested_end_date"],
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "output_sha256": hashes,
    }
    (out / "execution-audit.json").write_text(_json(audit) + "\n")
    print("STAGE4B_PROGRESS reporting audit_complete", flush=True)


def _csv(path, rows):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _report(matrix, summary):
    lead = sum(
        r["executed_trade_count"] for r in matrix if r["signal_timeframe"] == "15m"
    )
    robust = {
        tf: sum(
            r["executed_trade_count"] for r in matrix if r["signal_timeframe"] == tf
        )
        for tf in ("5m", "1h")
    }
    return (
        "# Stage 4B — deterministic gross trade construction\n\n"
        "## M15 lead interpretation\n"
        f"M15 executed configurations: {lead}. Entry modes and the frozen "
        "TP/SL/time-stop matrix are shown without winner ranking in "
        f"`trade-matrix.csv`. No-entry: {summary['no_entry_count']}; "
        f"ambiguous: {summary['ambiguous_count']}.\n\n"
        "## Entry-mode and exit behavior\n"
        "The matrix reports no-entry, ambiguity, TP, SL, and time-stop rates for "
        "each frozen entry mode and exit combination.\n\n"
        "## M5/H1 robustness\n"
        f"M5: {robust['5m']}; H1: {robust['1h']}. These are robustness/regime "
        "evidence.\n\n## Research caveats\n"
        "2024 is in-sample. Configurations and benchmark/lookback rows are "
        "dependent, not independent trials. Results are gross BID only, not "
        "executable net expectancy; Stage 4C costs are required. No 2025 data "
        "was used. No automatic winner is ranked.\n"
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", choices=FROZEN_INSTRUMENTS, required=True)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    a = p.parse_args(argv)
    run(a.corpus_dir, a.output_dir, a.instrument, a.registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
