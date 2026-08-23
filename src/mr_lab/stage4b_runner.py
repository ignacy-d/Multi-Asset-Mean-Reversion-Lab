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
import time
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
    "first-passage-distributions.csv",
    "conditional-hit.csv",
    "entry-wait-distributions.csv",
    "stage4b-distributions.csv",
)
SHARD_SCHEMA_VERSION = "stage4b-runtime-shard-v1"
STABLE_GROUP_FIELDS = (
    "instrument",
    "benchmark_family",
    "signal_timeframe",
    "session",
    "direction",
    "lookback",
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
    "tp_target_fraction",
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
    return _quantile(x, 0.5)


def _fraction(n, d):
    return n / d if d else 0.0


def _p90(x):
    return _quantile(x, 0.9)


def _quantile(values, probability):
    """Return an observed empirical nearest-rank quantile without interpolation."""
    if not values:
        return ""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


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
        "tp_exit_fraction": _fraction(a.tp, n),
        "sl_count": a.sl,
        "sl_fraction": _fraction(a.sl, n),
        "time_stop_count": a.time_stop,
        "time_stop_fraction": _fraction(a.time_stop, n),
        "mean_gross_pips": _mean(a.pips),
        "median_gross_pips": _median(a.pips),
        "gross_pips_p10": _quantile(a.pips, 0.10),
        "gross_pips_p25": _quantile(a.pips, 0.25),
        "gross_pips_p75": _quantile(a.pips, 0.75),
        "gross_pips_p90": _quantile(a.pips, 0.90),
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
        "mae_pips_p10": _quantile(a.mae, 0.10),
        "mae_pips_p25": _quantile(a.mae, 0.25),
        "mae_pips_p75": _quantile(a.mae, 0.75),
        "mae_pips_p90": _quantile(a.mae, 0.90),
        "mae_pips_p95": _quantile(a.mae, 0.95),
        "mean_mfe_pips": _mean(a.mfe),
        "median_mfe_pips": _median(a.mfe),
        "mfe_pips_p10": _quantile(a.mfe, 0.10),
        "mfe_pips_p25": _quantile(a.mfe, 0.25),
        "mfe_pips_p75": _quantile(a.mfe, 0.75),
        "mfe_pips_p90": _quantile(a.mfe, 0.90),
        "mfe_pips_p95": _quantile(a.mfe, 0.95),
        "holding_minutes_p10": _quantile(a.holding, 0.10),
        "holding_minutes_p25": _quantile(a.holding, 0.25),
        "holding_minutes_p75": _quantile(a.holding, 0.75),
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


def stable_group_key(event):
    """Return the frozen reporting group used only for runtime partitioning."""
    return (
        event.signal.instrument,
        event.signal.benchmark_family,
        str(event.signal.signal_timeframe),
        event.signal.session,
        event.signal.direction.name,
        event.signal.lookback,
    )


def _group_sort_key(key):
    # JSON gives None and named sessions one explicit, portable ordering.
    return _json(key)


def partition_candidate_groups(events, shard_count):
    """Deterministically balance contiguous complete groups by event count."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    grouped = defaultdict(list)
    for event in events:
        grouped[stable_group_key(event)].append(event)
    ordered = sorted(grouped, key=_group_sort_key)
    if shard_count > len(ordered):
        raise ValueError("shard_count cannot exceed stable group count")
    shards = []
    cursor = 0
    remaining_events = len(events)
    for shard_index in range(shard_count):
        remaining_shards = shard_count - shard_index
        if remaining_shards == 1:
            end = len(ordered)
        else:
            target = remaining_events / remaining_shards
            end = cursor
            count = 0
            max_end = len(ordered) - (remaining_shards - 1)
            while end < max_end:
                next_count = len(grouped[ordered[end]])
                if end > cursor and abs(count - target) <= abs(
                    count + next_count - target
                ):
                    break
                count += next_count
                end += 1
        keys = tuple(ordered[cursor:end])
        selected = tuple(event for key in keys for event in grouped[key])
        shards.append((keys, selected))
        remaining_events -= len(selected)
        cursor = end
    return tuple(shards)


def run(
    corpus_dir,
    output_dir,
    instrument,
    registry_path,
    eligibility_filter=None,
    *,
    shard_index=None,
    shard_count=None,
):
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
    # This ordering is methodology-critical: partition only the complete output
    # of the existing global state assembly and global re-arm/deduplication.
    full_events = deduplicate_states(assemble_signal_states(dataset, manifest))
    print(
        "STAGE4B_PROGRESS signal_generation_complete "
        f"candidate_dedup_complete candidate_count={len(full_events)}",
        flush=True,
    )
    all_groups = partition_candidate_groups(full_events, 1)[0][0]
    selected_group_keys = all_groups
    events = full_events
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be supplied together")
    if shard_count is not None:
        if not 0 <= shard_index < shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        partitions = partition_candidate_groups(full_events, shard_count)
        selected_group_keys, events = partitions[shard_index]
        print(
            "STAGE4B_PROGRESS "
            f"full_candidate_count={len(full_events)} "
            f"stable_group_count={len(all_groups)} "
            f"shard={shard_index + 1}/{shard_count} "
            f"shard_candidate_count={len(events)} group_range="
            f"{_json([selected_group_keys[0], selected_group_keys[-1]])}",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate-events.jsonl").open("w") as candidates:
        for event in events:
            candidates.write(_json(event.as_dict()) + "\n")
    groups = defaultdict(Aggregate)
    diagnostics = []
    entry_observations = []
    filter_ = eligibility_filter or NoEntryEligibilityFilter()
    filter_identities = set()
    totals = defaultdict(int)
    started = time.monotonic()
    label = (
        f"{shard_index + 1}/{shard_count}" if shard_count is not None else "unsharded"
    )
    print(f"STAGE4B_PROGRESS shard={label} simulation_start", flush=True)
    trade_rows_written = 0
    with (output_dir / "trades.jsonl").open("w") as trades:
        for processed, event in enumerate(events, 1):
            diagnostics.extend(
                {
                    "candidate_event_id": event.candidate_event_id,
                    "instrument": instrument,
                    "signal_timeframe": str(event.signal.signal_timeframe),
                    "benchmark_family": event.signal.benchmark_family,
                    "session": event.signal.session,
                    "direction": event.signal.direction.name,
                    "lookback": event.signal.lookback,
                    **d,
                }
                for d in path_diagnostic(event, index)
            )
            decision, entries = construct_eligible_entries(event, index, filter_)
            filter_identities.add((decision.filter_family, decision.filter_spec_id))
            if not decision.eligible:
                totals["filter_ineligible"] += 1
                continue
            paths = {
                e.timestamp: index.path(e.timestamp, 120)
                for e in entries.values()
                if e.executed and e.timestamp is not None
            }
            entry_observations.extend(
                {
                    "instrument": instrument,
                    "benchmark_family": event.signal.benchmark_family,
                    "signal_timeframe": str(event.signal.signal_timeframe),
                    "session": event.signal.session,
                    "direction": event.signal.direction.name,
                    "lookback": event.signal.lookback,
                    "candidate_event_id": event.candidate_event_id,
                    **asdict(constructed),
                }
                for constructed in entries.values()
                if constructed.mode != "immediate"
            )
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
                            trade_rows_written += 1
            if processed % 1000 == 0 or processed == len(events):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    "STAGE4B_PROGRESS "
                    f"shard={label} processed={processed}/{len(events)} "
                    f"pct={100 * processed / len(events):.1f} "
                    f"elapsed_seconds={elapsed:.1f} "
                    f"candidate_rate={processed / elapsed:.2f} "
                    f"trade_rows_written={trade_rows_written} "
                    f"trades_bytes_written={trades.tell()}",
                    flush=True,
                )
    print(f"STAGE4B_PROGRESS shard={label} simulation_complete", flush=True)
    _write_outputs(
        output_dir,
        events,
        groups,
        diagnostics,
        totals,
        registry,
        registry_entry,
        filter_identities,
        entry_observations,
    )
    if shard_count is not None:
        _write_shard_manifest(
            output_dir,
            registry_path,
            registry,
            registry_entry,
            manifest,
            full_events,
            all_groups,
            events,
            selected_group_keys,
            shard_index,
            shard_count,
        )
        print(f"STAGE4B_PROGRESS shard={label} reporting_complete", flush=True)
    return {p.name: p for p in output_dir.iterdir() if p.is_file()}


def _line_count(path):
    with path.open("rb") as file:
        return sum(1 for _ in file)


def _write_shard_manifest(
    out,
    registry_path,
    registry,
    entry,
    corpus_manifest,
    full_events,
    all_groups,
    events,
    selected_groups,
    shard_index,
    shard_count,
):
    files = sorted(path for path in out.iterdir() if path.is_file())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    row_counts = {
        p.name: _line_count(p) for p in files if p.suffix in {".csv", ".jsonl"}
    }
    audit = json.loads((out / "execution-audit.json").read_text())
    value = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "source_commit_sha": _commit_sha(),
        "instrument": entry["instrument"],
        "registry_identity": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "registry_schema": registry["registry_schema_version"],
        "corpus_id": corpus_manifest["corpus_id"],
        "assembled_dataset_id": corpus_manifest["assembled_dataset_id"],
        "source_workflow_run_id": entry["source_workflow_run_id"],
        "source_artifact_id": entry["source_artifact_id"],
        "source_artifact_name": entry["source_artifact_name"],
        "shard_count": shard_count,
        "shard_index": shard_index,
        "full_candidate_count": len(full_events),
        "shard_candidate_count": len(events),
        "full_group_keys": [list(key) for key in all_groups],
        "group_keys": [list(key) for key in selected_groups],
        "first_group_key": list(selected_groups[0]),
        "last_group_key": list(selected_groups[-1]),
        "first_candidate_event_id": events[0].candidate_event_id,
        "last_candidate_event_id": events[-1].candidate_event_id,
        "row_counts": row_counts,
        "file_sha256": hashes,
        "filter_family": audit["filter_family"],
        "filter_spec_id": audit["filter_spec_id"],
    }
    (out / "shard-manifest.json").write_text(_json(value) + "\n")


def _aggregate_diagnostics(diagnostics):
    # Aggregate descriptive diagnostics by stable dimensions and horizon.
    dg = defaultdict(list)
    for row in diagnostics:
        dg[
            (
                row["instrument"],
                row["benchmark_family"],
                row["signal_timeframe"],
                row["session"],
                row["direction"],
                row["lookback"],
                row["horizon_minutes"],
            )
        ].append(row)
    drows = []
    for key, rows in sorted(dg.items(), key=lambda item: str(item[0])):
        result = dict(
            zip(
                (
                    "instrument",
                    "benchmark_family",
                    "signal_timeframe",
                    "session",
                    "direction",
                    "lookback",
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
    return drows


def _filter_provenance(identities):
    ordered = sorted(identities)
    if not ordered:
        raise ValueError("at least one actual filter identity is required")
    families = sorted({family for family, _spec in ordered})
    specs = sorted({spec for _family, spec in ordered})
    return {
        "filter_family": families[0] if len(families) == 1 else families,
        "filter_spec_id": specs[0] if len(specs) == 1 else specs,
    }


DIAGNOSTIC_GROUP_FIELDS = (
    "instrument",
    "benchmark_family",
    "signal_timeframe",
    "session",
    "direction",
    "lookback",
)
_BARRIERS = (
    ("reversion25", "R>=+0.25"),
    ("reversion50", "R>=+0.50"),
    ("extension25", "R<=-0.25"),
    ("extension50", "R<=-0.50"),
)
_INTERVALS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30))


def _first_passage_distributions(diagnostics):
    """Summarize candidate first passage without interpolating M1 hit times."""
    groups = defaultdict(list)
    for row in diagnostics:
        if row["horizon_minutes"] == 30:
            groups[tuple(row[field] for field in DIAGNOSTIC_GROUP_FIELDS)].append(row)
    distributions = []
    conditional = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        identity = dict(zip(DIAGNOSTIC_GROUP_FIELDS, key, strict=True))
        for barrier_field, barrier in _BARRIERS:
            times = [
                row[f"time_to_{barrier_field}"]
                for row in rows
                if row[f"time_to_{barrier_field}"] is not None
            ]
            distributions.append(
                identity
                | {
                    "barrier": barrier,
                    "candidate_event_count": len(rows),
                    "hit_count": len(times),
                    "hit_fraction": _fraction(len(times), len(rows)),
                    "time_to_hit_p10": _quantile(times, 0.10),
                    "time_to_hit_p25": _quantile(times, 0.25),
                    "time_to_hit_median": _median(times),
                    "time_to_hit_p75": _quantile(times, 0.75),
                    "time_to_hit_p90": _quantile(times, 0.90),
                    "mean_time_to_hit": _mean(times),
                    **{
                        f"hit_by_{minute}m_fraction": _fraction(
                            sum(time <= minute for time in times), len(rows)
                        )
                        for minute in (5, 10, 15, 20, 30)
                    },
                }
            )
            for start, end in _INTERVALS:
                at_risk = sum(
                    time is None or time > start
                    for time in (row[f"time_to_{barrier_field}"] for row in rows)
                )
                new_hits = sum(start < time <= end for time in times)
                conditional.append(
                    identity
                    | {
                        "barrier": barrier,
                        "interval_start_minutes": start,
                        "interval_end_minutes": end,
                        "events_at_risk_at_interval_start": at_risk,
                        "new_hits_in_interval": new_hits,
                        "conditional_hit_fraction": _fraction(new_hits, at_risk),
                    }
                )
    return distributions, conditional


def _entry_wait_distributions(observations):
    fields = (*DIAGNOSTIC_GROUP_FIELDS, "mode")
    groups = defaultdict(list)
    for row in observations:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        executed = [row for row in rows if row["executed"]]
        waits = [row["wait_minutes"] for row in executed]
        result = dict(zip(fields, key, strict=True)) | {
            "candidate_event_count": len(rows),
            "executed_entry_count": len(executed),
            "no_entry_count": len(rows) - len(executed),
            "no_entry_fraction": _fraction(len(rows) - len(executed), len(rows)),
            "wait_p10": _quantile(waits, 0.10),
            "wait_p25": _quantile(waits, 0.25),
            "wait_median": _median(waits),
            "wait_p75": _quantile(waits, 0.75),
            "wait_p90": _quantile(waits, 0.90),
            "mean_wait": _mean(waits),
        }
        for minute in (1, 5, 10, 15):
            result[f"entry_by_{minute}m_fraction"] = _fraction(
                sum(wait <= minute for wait in waits), len(rows)
            )
        if key[-1] == "extension25-then-reclaim-p0":
            for name, source in (
                ("time_to_extension25", "time_to_extension25"),
                ("time_extension25_to_reclaim", "time_from_extension25_to_entry"),
            ):
                values = [row[source] for row in executed if row[source] is not None]
                for label, probability in (
                    ("p10", 0.10),
                    ("p25", 0.25),
                    ("median", 0.50),
                    ("p75", 0.75),
                    ("p90", 0.90),
                ):
                    result[f"{name}_{label}"] = _quantile(values, probability)
                result[f"{name}_mean"] = _mean(values)
            result["same_m1_extension_reclaim_fraction"] = _fraction(
                sum(row["time_from_extension25_to_entry"] == 0 for row in executed),
                len(executed),
            )
        output.append(result)
    return output


def _long_distributions(groups):
    rows = []
    metrics = (
        ("gross_pips_adverse_first", "pips"),
        ("mae_certain_pips", "mae"),
        ("mfe_certain_pips", "mfe"),
        ("holding_minutes", "holding"),
    )
    for key, aggregate in sorted(groups.items(), key=lambda item: str(item[0])):
        identity = dict(zip(GROUP_FIELDS, key, strict=True))
        for metric, attribute in metrics:
            values = getattr(aggregate, attribute)
            for label, probability in (
                ("p10", 0.10),
                ("p25", 0.25),
                ("median", 0.50),
                ("p75", 0.75),
                ("p90", 0.90),
                ("p95", 0.95),
            ):
                rows.append(
                    identity
                    | {
                        "metric": metric,
                        "quantile": label,
                        "value": _quantile(values, probability),
                        "sample_count": len(values),
                    }
                )
    return rows


def _write_outputs(
    out,
    events,
    groups,
    diagnostics,
    totals,
    registry,
    entry,
    filter_identities,
    entry_observations,
):
    drows = _aggregate_diagnostics(diagnostics)
    _csv(out / "entry-diagnostics.csv", drows)
    first_passage, conditional = _first_passage_distributions(diagnostics)
    _csv(out / "first-passage-distributions.csv", first_passage)
    _csv(out / "conditional-hit.csv", conditional)
    _csv(
        out / "entry-wait-distributions.csv",
        _entry_wait_distributions(entry_observations),
    )
    matrix = [
        _aggregate_row(k, a) for k, a in sorted(groups.items(), key=lambda x: str(x[0]))
    ]
    _csv(out / "trade-matrix.csv", matrix)
    _csv(out / "stage4b-distributions.csv", _long_distributions(groups))
    summary = {
        "reporting_schema_version": STAGE4B_REPORT_SCHEMA_VERSION,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "candidate_event_count": len(events),
        "no_entry_count": totals["no_entry"],
        "target_already_passed_count": totals["passed"],
        "executed_trade_configuration_count": totals["executed"],
        "incomplete_count": totals["incomplete"],
        "ambiguous_count": totals["ambiguous"],
        "filter_ineligible_count": totals["filter_ineligible"],
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
        **_filter_provenance(filter_identities),
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
        "## Distributional path shape\n"
        "Mean timing and mean excursion values must not be interpreted as typical "
        "entry/exit parameters. Timing and excursion distributions may be "
        "multimodal or strongly skewed. Stage 4B interpretation should prioritize "
        "first-passage distributions, quantiles, and cumulative/conditional hit "
        "behavior. The interpretation hierarchy remains M15, frozen z=2, session, "
        "direction, benchmark and lookback robustness, entry mode, frozen exits, "
        "then distributional path shape.\n\n"
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
    p.add_argument("--shard-index", type=int)
    p.add_argument("--shard-count", type=int)
    a = p.parse_args(argv)
    run(
        a.corpus_dir,
        a.output_dir,
        a.instrument,
        a.registry,
        shard_index=a.shard_index,
        shard_count=a.shard_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
