"""Authenticated, frozen-period discovery atlas for existing Stage 4B evidence.

This module reports families and deliberately has no strategy-generation or
parameter-selection capability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import CostProfile, net_pips

SCHEMA = "stage4-discovery-atlas-v1"
FROZEN_START = "2024-01-01"
FROZEN_END = "2024-12-31"
HEADLINE_COSTS = (("mean", 0.0), ("p75", 0.1), ("p90", 0.25), ("p95", 0.5))
SETUP_FIELDS = (
    "instrument",
    "signal_timeframe",
    "session",
    "direction",
    "benchmark_family",
    "lookback",
    "entry_mode",
)
EXIT_FIELDS = ("tp_target_fraction", "sl_extension_fraction", "time_stop_minutes")
HYPOTHESIS_FIELDS = (
    "signal_timeframe",
    "session",
    "direction",
    "benchmark_family",
    "entry_mode",
)
OUTPUTS = (
    "setup-family-matrix.csv",
    "execution-family-matrix.csv",
    "cross-asset-hypotheses.csv",
    "cost-robustness.csv",
    "overlap-with-module-a.csv",
    "candidate-shortlist.csv",
    "report.md",
)


class DiscoveryAtlasError(ValueError):
    """Input is not authenticated frozen Stage 4B research evidence."""


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_jsonl(path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise DiscoveryAtlasError(
                    f"malformed authenticated input row {number}"
                ) from error


def _reject_sealed_path(path):
    # This check intentionally occurs before any filesystem operation.
    if "2025" in str(path):
        raise DiscoveryAtlasError("sealed-period paths are forbidden")


def _authenticate(directory):
    _reject_sealed_path(directory)
    audit_path = directory / "execution-audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise DiscoveryAtlasError("Stage 4B methodology mismatch")
    if (audit.get("requested_start_date"), audit.get("requested_end_date")) != (
        FROZEN_START,
        FROZEN_END,
    ):
        raise DiscoveryAtlasError("only the frozen 2024 research period is accepted")
    for field in ("instrument", "corpus_id", "assembled_dataset_id"):
        if not audit.get(field):
            raise DiscoveryAtlasError(f"missing provenance: {field}")
    if len(audit.get("source_commit_sha", "")) != 40:
        raise DiscoveryAtlasError("missing Stage 4B source commitment")
    hashes = audit.get("output_sha256")
    if not isinstance(hashes, dict):
        raise DiscoveryAtlasError("missing output commitments")
    required = ("trades.jsonl", "candidate-events.jsonl")
    for name in required:
        path = directory / name
        if hashes.get(name) != _sha(path):
            raise DiscoveryAtlasError(f"source commitment mismatch: {name}")
    return audit, *(directory / name for name in required)


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DiscoveryAtlasError("timestamps must be timezone-aware")
    parsed = parsed.astimezone(UTC)
    if parsed.year != 2024:
        raise DiscoveryAtlasError("row outside frozen 2024 period")
    return parsed


def _family(row):
    return tuple(row.get(field) for field in SETUP_FIELDS)


def _write(path, rows, fields=None):
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _profit_factor(values):
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else (math.inf if gains else 0.0)


def _losing_streak(values):
    best = current = 0
    for value in values:
        current = current + 1 if value < 0 else 0
        best = max(best, current)
    return best


def _metrics(records, value_key):
    values = [row[value_key] for row in records]
    months = defaultdict(float)
    for row in records:
        months[row["timestamp"].month] += row[value_key]
    month_values = [months.get(month, 0.0) for month in range(1, 13)]
    return {
        "candidate_count": len({row["candidate_event_id"] for row in records}),
        "unique_execution_opportunities": len(
            {row["execution_key"] for row in records}
        ),
        "trades_per_month": len(records) / 12,
        "expectancy": statistics.fmean(values),
        "total_pips": sum(values),
        "profit_factor": _profit_factor(values),
        "win_rate": sum(value > 0 for value in values) / len(values),
        "positive_months": sum(value > 0 for value in month_values),
        "negative_months": sum(value < 0 for value in month_values),
        "no_trade_months": sum(not value for value in month_values),
        "worst_month": min(month_values),
        "best_month": max(month_values),
        "monthly_mean": statistics.fmean(month_values),
        "monthly_standard_deviation": statistics.pstdev(month_values),
        "max_losing_streak": _losing_streak(values),
    }


def _is_module_a(row):
    return (
        row["signal_timeframe"] == "15m"
        and row["session"] == "london"
        and row["direction"] == "SHORT"
        and "vwap" in row["benchmark_family"].lower()
        and row["lookback"] in (20, 40)
        and row["entry_mode"] == "immediate"
    )


def build_atlas(stage4b_dirs, output_dir, profile_path):
    stage4b_dirs = tuple(map(Path, stage4b_dirs))
    for directory in stage4b_dirs:
        _reject_sealed_path(directory)
    profile = CostProfile.load(profile_path)
    records, provenances = [], set()
    shard_manifests = []
    authenticated = []
    for directory in stage4b_dirs:
        audit, trades_path, candidates_path = _authenticate(directory)
        manifest_path = directory / "shard-manifest.json"
        if manifest_path.is_file():
            shard_manifests.append(json.loads(manifest_path.read_text()))
        provenance = (audit["filter_family"], audit["filter_spec_id"])
        provenances.add(provenance)
        timestamps = {}
        for event in _read_jsonl(candidates_path):
            signal = event["signal"]
            if signal.get("instrument") != audit["instrument"]:
                raise DiscoveryAtlasError("candidate instrument provenance mismatch")
            if signal.get("source_corpus_id") != audit["corpus_id"]:
                raise DiscoveryAtlasError("candidate corpus provenance mismatch")
            if signal.get("assembled_dataset_id") != audit["assembled_dataset_id"]:
                raise DiscoveryAtlasError("candidate dataset provenance mismatch")
            timestamps[event["candidate_event_id"]] = _time(signal["signal_timestamp"])
        authenticated.append((audit, trades_path, timestamps, provenance))
    if shard_manifests:
        counts = {item.get("shard_count") for item in shard_manifests}
        indexes = {item.get("shard_index") for item in shard_manifests}
        if len(counts) != 1 or indexes != set(range(counts.pop())):
            raise DiscoveryAtlasError("missing or inconsistent raw shard input")
    for audit, trades_path, timestamps, provenance in authenticated:
        for row in _read_jsonl(trades_path):
            if not row.get("complete"):
                continue
            if row.get("instrument") != audit["instrument"]:
                raise DiscoveryAtlasError("instrument provenance mismatch")
            timestamp = timestamps.get(row["candidate_event_id"])
            if timestamp is None:
                raise DiscoveryAtlasError(
                    "trade absent from authenticated candidate stream"
                )
            spread, commission = profile.costs(
                row["instrument"], row["session"], "mean"
            )
            records.append(
                row
                | {
                    "timestamp": timestamp,
                    "provenance": provenance,
                    "execution_key": (
                        row["instrument"],
                        timestamp.isoformat(),
                        row["session"],
                        row["direction"],
                        row["entry_mode"],
                    ),
                    "gross": float(row["gross_return_pips_adverse_first"]),
                    "net": net_pips(
                        float(row["gross_return_pips_adverse_first"]),
                        row["instrument"],
                        spread,
                        0.0,
                        commission,
                    ),
                }
            )
    if len(provenances) != 1:
        raise DiscoveryAtlasError("baseline and filtered provenance cannot be mixed")
    if not records:
        raise DiscoveryAtlasError("no complete authenticated trades")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in records:
        grouped[_family(row)].append(row)
    setup_rows = []
    for key, group in sorted(grouped.items()):
        gross, net = _metrics(group, "gross"), _metrics(group, "net")
        exits = {tuple(row[field] for field in EXIT_FIELDS) for row in group}
        setup_rows.append(
            dict(zip(SETUP_FIELDS, key, strict=True))
            | {
                **{f"gross_{k}": v for k, v in gross.items()},
                **{f"net_{k}": v for k, v in net.items()},
                "exit_plateau_breadth": len(exits),
            }
        )
    _write(output_dir / OUTPUTS[0], setup_rows)
    execution_groups = defaultdict(list)
    for row in records:
        execution_groups[
            (
                row["instrument"],
                row["signal_timeframe"],
                row["session"],
                row["direction"],
                row["entry_mode"],
            )
        ].append(row)
    execution_rows = []
    for key, group in sorted(execution_groups.items()):
        representatives = {}
        for row in group:
            representatives.setdefault(row["execution_key"], row)
        metrics = _metrics(list(representatives.values()), "net")
        execution_rows.append(
            dict(
                zip(
                    (
                        "instrument",
                        "signal_timeframe",
                        "session",
                        "direction",
                        "entry_mode",
                    ),
                    key,
                    strict=True,
                )
            )
            | metrics
        )
    _write(output_dir / OUTPUTS[1], execution_rows)
    hypotheses = defaultdict(list)
    for row in records:
        hypotheses[tuple(row[field] for field in HYPOTHESIS_FIELDS)].append(row)
    hypothesis_rows, shortlist = [], []
    for index, (key, group) in enumerate(sorted(hypotheses.items()), 1):
        instruments = sorted({row["instrument"] for row in group})
        representatives = {row["execution_key"]: row for row in group}.values()
        m = _metrics(list(representatives), "net")
        category = (
            "E. insufficient frequency"
            if m["trades_per_month"] < 1
            else "A. promising existing hypothesis"
            if m["expectancy"] > 0
            and m["positive_months"] >= 7
            and len(instruments) >= 2
            else "C. instrument-specific only"
            if m["expectancy"] > 0 and len(instruments) == 1
            else "F. clear fail"
            if m["expectancy"] <= 0
            else "B. fragile / cost-sensitive"
        )
        base = dict(zip(HYPOTHESIS_FIELDS, key, strict=True))
        hypothesis_rows.append(
            base
            | m
            | {
                "instrument_count": len(instruments),
                "instruments": "|".join(instruments),
            }
        )
        shortlist.append(
            {
                "hypothesis_id": f"H{index:04d}",
                **base,
                "cross_asset_support": len(instruments),
                "frequency_band": "low" if m["trades_per_month"] < 1 else "adequate",
                "cost_robustness": "see cost-robustness.csv",
                "plateau_robustness": "see setup-family-matrix.csv",
                "module_a_overlap": "see overlap-with-module-a.csv",
                "triage_category": category,
                "notes": "descriptive family triage; not a ranking",
            }
        )
    _write(output_dir / OUTPUTS[2], hypothesis_rows)
    _write(output_dir / OUTPUTS[5], shortlist)
    cost_rows = []
    for key, group in sorted(grouped.items()):
        for statistic, slippage in HEADLINE_COSTS:
            spread, commission = profile.costs(
                group[0]["instrument"], group[0]["session"], statistic
            )
            values = [
                net_pips(row["gross"], row["instrument"], spread, slippage, commission)
                for row in group
            ]
            cost_rows.append(
                dict(zip(SETUP_FIELDS, key, strict=True))
                | {
                    "spread_statistic": statistic,
                    "slippage_pips": slippage,
                    "net_expectancy": statistics.fmean(values),
                    "total_net_pips": sum(values),
                }
            )
    _write(output_dir / OUTPUTS[3], cost_rows)
    module = {row["execution_key"] for row in records if _is_module_a(row)}
    overlap = []
    for key, group in sorted(grouped.items()):
        opportunities = {row["execution_key"] for row in group}
        intersection = opportunities & module
        overlap.append(
            dict(zip(SETUP_FIELDS, key, strict=True))
            | {
                "same_timestamp_overlap_rate": len(intersection) / len(opportunities),
                "unique_to_new_setup_opportunity_count": len(opportunities - module),
                "module_a_only_count": len(module - opportunities),
                "intersection_count": len(intersection),
            }
        )
    _write(output_dir / OUTPUTS[4], overlap)
    (output_dir / OUTPUTS[6]).write_text(
        "# Frozen 2024 discovery atlas\n\n"
        "Family-level triage of authenticated existing evidence. It does not "
        "rank parameter cells or create strategies.\n"
    )
    audit = {
        "schema_version": SCHEMA,
        "research_period": {"start": FROZEN_START, "end": FROZEN_END},
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "inputs": [
            {
                "directory": str(Path(d)),
                "execution_audit_sha256": _sha(Path(d) / "execution-audit.json"),
            }
            for d in stage4b_dirs
        ],
        "filter_provenance": list(next(iter(provenances))),
        "output_sha256": {name: _sha(output_dir / name) for name in OUTPUTS},
    }
    (output_dir / "execution-audit.json").write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Report existing frozen-2024 Stage 4 research families"
    )
    parser.add_argument("--stage4b-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cost-profile",
        type=Path,
        default=Path("configs/stage4c-ftmo-cost-profile-v1.json"),
    )
    args = parser.parse_args(argv)
    build_atlas(args.stage4b_dir, args.output_dir, args.cost_profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
