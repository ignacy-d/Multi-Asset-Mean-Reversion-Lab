"""Authenticated frozen-2024 triage of already-computed Stage 4B evidence.

Performance is computed only for exact strategy/exit cells.  Family and
execution views contain descriptive robustness and opportunity evidence; they
never manufacture a portfolio or choose a representative trade outcome.
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

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import CostProfile, net_pips

SCHEMA = "stage4-discovery-atlas-v2"
METHODOLOGY = "exact-cell-performance-family-plateau-v1"
FROZEN_START, FROZEN_END = "2024-01-01", "2024-12-31"
HEADLINE_COSTS = (("mean", 0.0), ("p75", 0.1), ("p90", 0.25), ("p95", 0.5))
SETUP_FIELDS = (
    "instrument",
    "signal_timeframe",
    "session",
    "direction",
    "benchmark_family",
    "lookback",
    "signal_threshold",
    "entry_mode",
    "filter_family",
    "filter_spec_id",
    "process_spec_id",
)
EXIT_FIELDS = ("tp_target_fraction", "sl_extension_fraction", "time_stop_minutes")
CELL_FIELDS = (*SETUP_FIELDS, *EXIT_FIELDS)
EXECUTION_FIELDS = (
    "instrument",
    "signal_timeframe",
    "session",
    "direction",
    "entry_mode",
)
HYPOTHESIS_FIELDS = (
    "signal_timeframe",
    "session",
    "direction",
    "benchmark_family",
    "lookback",
    "signal_threshold",
    "entry_mode",
    "filter_family",
    "filter_spec_id",
    "process_spec_id",
    *EXIT_FIELDS,
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
RAW_FILES = ("candidate-events.jsonl", "trades.jsonl", "execution-audit.json")


class DiscoveryAtlasError(ValueError):
    """Evidence violates the frozen atlas input contract."""


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _reject_sealed(path):
    # Must precede exists/stat/open/rglob for every user-supplied evidence path.
    if "2025" in str(path):
        raise DiscoveryAtlasError("sealed-period paths are forbidden")


def _jsonl(path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise DiscoveryAtlasError(
                    f"malformed authenticated row {number}"
                ) from error


def _line_count(path):
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def _load_registry(path):
    registry = json.loads(path.read_text())
    if registry.get("registry_schema_version") != "stage-4a-2024-corpus-registry-v1":
        raise DiscoveryAtlasError("unexpected frozen registry schema")
    return registry


def _manifest_identity(manifest):
    fields = (
        "source_commit_sha",
        "stage4b_methodology_id",
        "instrument",
        "registry_identity",
        "corpus_id",
        "assembled_dataset_id",
        "source_workflow_run_id",
        "source_artifact_id",
        "source_artifact_name",
        "source_mode",
        "source_acquisition_commit_sha",
        "filter_family",
        "filter_spec_id",
        "process_spec_id",
        "shard_count",
        "full_candidate_count",
        "full_group_keys",
    )
    return tuple(_json(manifest.get(field)) for field in fields)


def _authenticate_run(records, registry, registry_sha, role):
    """Authenticate one complete logical run, never indexes across runs."""
    records.sort(key=lambda item: item[1].get("shard_index", -1))
    reference = records[0][1]
    count = reference.get("shard_count")
    if not isinstance(count, int) or count < 1:
        raise DiscoveryAtlasError("invalid shard count")
    if {manifest.get("shard_index") for _, manifest in records} != set(range(count)):
        raise DiscoveryAtlasError("incomplete per-run shard indexes")
    identity = _manifest_identity(reference)
    if any(_manifest_identity(manifest) != identity for _, manifest in records):
        raise DiscoveryAtlasError("logical-run shard provenance mismatch")
    if reference.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise DiscoveryAtlasError("Stage 4B methodology mismatch")
    if len(reference.get("source_commit_sha", "")) != 40:
        raise DiscoveryAtlasError("missing Stage 4B source commitment")
    if reference.get("registry_identity") != registry_sha:
        raise DiscoveryAtlasError("registry commitment mismatch")
    instrument = reference.get("instrument")
    expected = registry.get("instruments", {}).get(instrument)
    if not expected or expected.get("verification_status") != "verified":
        raise DiscoveryAtlasError("instrument lacks verified frozen registry evidence")
    source_mode = expected.get("source_mode", "github-artifact")
    if reference.get("source_mode", "github-artifact") != source_mode:
        raise DiscoveryAtlasError("registry source_mode mismatch")
    for field in (
        "corpus_id",
        "assembled_dataset_id",
        "source_workflow_run_id",
        "source_artifact_id",
        "source_artifact_name",
        "source_acquisition_commit_sha",
    ):
        if reference.get(field) != expected.get(field):
            raise DiscoveryAtlasError(f"registry {field} mismatch")
    if (expected.get("requested_start_date"), expected.get("requested_end_date")) != (
        FROZEN_START,
        FROZEN_END,
    ):
        raise DiscoveryAtlasError("registry is not frozen 2024")
    full_groups = reference.get("full_group_keys")
    combined_groups = [
        group for _, manifest in records for group in manifest.get("group_keys", [])
    ]
    if not isinstance(full_groups, list) or combined_groups != full_groups:
        raise DiscoveryAtlasError("incomplete or overlapping full group universe")
    if sum(
        manifest.get("shard_candidate_count", -1) for _, manifest in records
    ) != reference.get("full_candidate_count"):
        raise DiscoveryAtlasError("incomplete candidate coverage")

    events, trades, commitments, process_ids = {}, [], [], set()
    for directory, manifest in records:
        hashes, counts = manifest.get("file_sha256", {}), manifest.get("row_counts", {})
        for name in RAW_FILES:
            path = directory / name
            if hashes.get(name) != _sha(path):
                raise DiscoveryAtlasError(f"shard raw hash mismatch: {name}")
            if name.endswith("jsonl") and counts.get(name) != _line_count(path):
                raise DiscoveryAtlasError(f"shard raw row-count mismatch: {name}")
        audit = json.loads((directory / "execution-audit.json").read_text())
        for field in (
            "stage4b_methodology_id",
            "source_commit_sha",
            "instrument",
            "corpus_id",
            "assembled_dataset_id",
            "filter_family",
            "filter_spec_id",
        ):
            if audit.get(field) != reference.get(field):
                raise DiscoveryAtlasError(f"execution-audit identity mismatch: {field}")
        process_ids.add(audit.get("process_spec_id"))
        if reference.get("process_spec_id") is not None and audit.get(
            "process_spec_id"
        ) != reference.get("process_spec_id"):
            raise DiscoveryAtlasError("execution-audit process provenance mismatch")
        for event in _jsonl(directory / "candidate-events.jsonl"):
            signal = event["signal"]
            if (
                signal.get("instrument") != instrument
                or signal.get("source_corpus_id") != reference["corpus_id"]
                or signal.get("assembled_dataset_id")
                != reference["assembled_dataset_id"]
            ):
                raise DiscoveryAtlasError("candidate provenance mismatch")
            event_id = event["candidate_event_id"]
            if event_id in events:
                raise DiscoveryAtlasError("duplicate candidate across shards")
            events[event_id] = _time(signal["signal_timestamp"])
        trades.extend(_jsonl(directory / "trades.jsonl"))
        commitments.append(
            {
                "role": role,
                "manifest_sha256": _sha(directory / "shard-manifest.json"),
                "raw_sha256": {name: hashes[name] for name in RAW_FILES},
            }
        )
    if len(events) != reference["full_candidate_count"]:
        raise DiscoveryAtlasError("candidate rows do not cover declared universe")
    if len(process_ids) != 1:
        raise DiscoveryAtlasError("logical-run process provenance mismatch")
    reference = reference | {"process_spec_id": next(iter(process_ids))}
    return reference, events, trades, commitments


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DiscoveryAtlasError("timestamps must be timezone-aware")
    parsed = parsed.astimezone(UTC)
    if parsed.year != 2024:
        raise DiscoveryAtlasError("row outside frozen 2024 period")
    return parsed


def _collect(directories, registry, registry_sha, role):
    directories = tuple(map(Path, directories))
    for directory in directories:
        _reject_sealed(directory)
    manifests = []
    for directory in directories:
        path = directory / "shard-manifest.json"
        if not path.is_file():
            raise DiscoveryAtlasError("every raw input requires shard-manifest.json")
        manifests.append((directory, json.loads(path.read_text())))
    logical = defaultdict(list)
    for directory, manifest in manifests:
        logical[_manifest_identity(manifest)].append((directory, manifest))
    output, commitments = [], []
    for records in logical.values():
        manifest, events, trades, run_commitments = _authenticate_run(
            records, registry, registry_sha, role
        )
        provenance = tuple(
            manifest.get(field)
            for field in ("filter_family", "filter_spec_id", "process_spec_id")
        )
        for row in trades:
            if not row.get("complete"):
                continue
            timestamp = events.get(row.get("candidate_event_id"))
            if timestamp is None or row.get("instrument") != manifest["instrument"]:
                raise DiscoveryAtlasError("trade provenance mismatch")
            output.append(
                row
                | {
                    "timestamp": timestamp,
                    "role": role,
                    "process_spec_id": provenance[2],
                    "execution_key": (
                        row["instrument"],
                        timestamp.isoformat(),
                        row["session"],
                        row["direction"],
                        row["entry_mode"],
                    ),
                }
            )
        commitments.extend(run_commitments)
    return output, commitments


def _write(path, rows):
    fields = (
        sorted({field for row in rows for field in row}) if rows else ["schema_version"]
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pf(values):
    gains, losses = sum(v for v in values if v > 0), -sum(v for v in values if v < 0)
    return gains / losses if losses else (math.inf if gains else 0.0)


def _cell_metrics(rows, values):
    ordered = sorted(
        zip(rows, values, strict=True),
        key=lambda item: (item[0]["timestamp"], item[0]["candidate_event_id"]),
    )
    monthly, observations = defaultdict(float), defaultdict(int)
    streak = longest = 0
    for row, value in ordered:
        monthly[row["timestamp"].month] += value
        observations[row["timestamp"].month] += 1
        streak = streak + 1 if value < 0 else 0
        longest = max(longest, streak)
    month_values = [monthly.get(month, 0.0) for month in range(1, 13)]
    return {
        "trade_count": len(rows),
        "trades_per_month": len(rows) / 12,
        "candidate_count": len({row["candidate_event_id"] for row in rows}),
        "expectancy": statistics.fmean(values),
        "total_pips": sum(values),
        "profit_factor": _pf(values),
        "win_rate": sum(v > 0 for v in values) / len(values),
        "positive_months": sum(monthly[m] > 0 for m in observations),
        "negative_months": sum(monthly[m] < 0 for m in observations),
        "no_trade_months": 12 - len(observations),
        "worst_month": min(month_values),
        "best_month": max(month_values),
        "monthly_mean": statistics.fmean(month_values),
        "monthly_standard_deviation": statistics.pstdev(month_values),
        "max_losing_streak": longest,
    }


def _scenario_values(rows, profile, statistic, slippage):
    result = []
    for row in rows:
        spread, commission = profile.costs(row["instrument"], row["session"], statistic)
        result.append(
            net_pips(
                float(row["gross_return_pips_adverse_first"]),
                row["instrument"],
                spread,
                slippage,
                commission,
            )
        )
    return result


def _module_candidate(row):
    return (
        row["signal_timeframe"] == "15m"
        and row["session"] == "london"
        and row["direction"] == "SHORT"
        and row["benchmark_family"] in ("vwap", "vwap-canonical-m1")
        and row["lookback"] in (20, 40)
        and row["entry_mode"] == "immediate"
    )


def build_atlas(
    stage4b_dirs,
    output_dir,
    profile_path,
    registry_path=Path("configs/stage4a-2024-corpus-registry.json"),
    *,
    module_a_dirs=(),
):
    stage4b_dirs, module_a_dirs = tuple(stage4b_dirs), tuple(module_a_dirs)
    for path in (*stage4b_dirs, *module_a_dirs):
        _reject_sealed(path)
    registry_path, profile_path = Path(registry_path), Path(profile_path)
    registry_sha, profile_sha = _sha(registry_path), _sha(profile_path)
    registry, profile = _load_registry(registry_path), CostProfile.load(profile_path)
    evidence, commitments = _collect(stage4b_dirs, registry, registry_sha, "evidence")
    module, module_commitments = (
        _collect(module_a_dirs, registry, registry_sha, "module_a")
        if module_a_dirs
        else ([], [])
    )
    commitments.extend(module_commitments)
    if not evidence:
        raise DiscoveryAtlasError("no complete authenticated evidence trades")
    if module:
        spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
        expected_module = (
            "ornstein-uhlenbeck",
            spec.filter_spec_id,
            spec.process_spec.process_spec_id,
        )
        if any(
            (
                row.get("filter_family"),
                row.get("filter_spec_id"),
                row.get("process_spec_id"),
            )
            != expected_module
            for row in module
        ):
            raise DiscoveryAtlasError(
                "Module A role requires exact frozen-ou-crossasset-v1 provenance"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = defaultdict(list)
    for row in evidence:
        cells[tuple(row.get(field) for field in CELL_FIELDS)].append(row)
    cost_rows, cell_results = [], {}
    for key, rows in sorted(cells.items(), key=lambda item: _json(item[0])):
        gross_metrics = _cell_metrics(
            rows, [float(row["gross_return_pips_adverse_first"]) for row in rows]
        )
        for statistic, slippage in HEADLINE_COSTS:
            values = _scenario_values(rows, profile, statistic, slippage)
            metrics = _cell_metrics(rows, values)
            scenario = f"{statistic}+{slippage:g}"
            cell_results[(key, scenario)] = metrics
            cost_rows.append(
                dict(zip(CELL_FIELDS, key, strict=True))
                | {
                    "cost_scenario": scenario,
                    "spread_statistic": statistic,
                    "slippage_pips": slippage,
                    **{f"gross_{name}": value for name, value in gross_metrics.items()},
                    **{f"net_{name}": value for name, value in metrics.items()},
                    **metrics,
                }
            )
    _write(output_dir / OUTPUTS[3], cost_rows)

    families = defaultdict(list)
    for key in cells:
        families[key[: len(SETUP_FIELDS)]].append(key)
    setup_rows = []
    for family, exit_cells in sorted(families.items(), key=lambda item: _json(item[0])):
        opportunities = {
            row["execution_key"] for key in exit_cells for row in cells[key]
        }
        candidate_ids = {
            row["candidate_event_id"] for key in exit_cells for row in cells[key]
        }
        base = dict(zip(SETUP_FIELDS, family, strict=True)) | {
            "exit_cell_count": len(exit_cells),
            "candidate_count": len(candidate_ids),
            "unique_execution_opportunities": len(opportunities),
            "trades_per_month": len(opportunities) / 12,
        }
        for statistic, slippage in HEADLINE_COSTS:
            scenario = f"{statistic}+{slippage:g}"
            expectancies = [
                cell_results[(key, scenario)]["expectancy"] for key in exit_cells
            ]
            positive = sum(value > 0 for value in expectancies)
            prefix = scenario.replace("+", "_").replace(".", "p")
            base.update(
                {
                    f"{prefix}_positive_exit_cells": positive,
                    f"{prefix}_positive_fraction": positive / len(expectancies),
                    f"{prefix}_min_expectancy": min(expectancies),
                    f"{prefix}_median_expectancy": statistics.median(expectancies),
                    f"{prefix}_max_expectancy": max(expectancies),
                }
            )
        setup_rows.append(base)
    _write(output_dir / OUTPUTS[0], setup_rows)

    executions = defaultdict(list)
    for row in evidence:
        executions[tuple(row[field] for field in EXECUTION_FIELDS)].append(row)
    execution_rows = []
    for key, rows in sorted(executions.items(), key=lambda item: _json(item[0])):
        by_opportunity = defaultdict(list)
        for row in rows:
            by_opportunity[row["execution_key"]].append(row)
        execution_rows.append(
            dict(zip(EXECUTION_FIELDS, key, strict=True))
            | {
                "unique_execution_opportunities": len(by_opportunity),
                "family_membership_count": len(
                    {tuple(row.get(f) for f in SETUP_FIELDS) for row in rows}
                ),
                "benchmark_variants_firing": "|".join(
                    sorted({row["benchmark_family"] for row in rows})
                ),
                "lookbacks_firing": "|".join(
                    map(str, sorted({row["lookback"] for row in rows}))
                ),
            }
        )
    _write(output_dir / OUTPUTS[1], execution_rows)

    hypothesis_instruments = defaultdict(lambda: defaultdict(list))
    for key, rows in cells.items():
        hypothesis = tuple(key[CELL_FIELDS.index(field)] for field in HYPOTHESIS_FIELDS)
        hypothesis_instruments[hypothesis][key[0]].extend(rows)
    cross_rows, shortlist = [], []
    for index, (hypothesis, instruments) in enumerate(
        sorted(hypothesis_instruments.items(), key=lambda item: _json(item[0])), 1
    ):
        scenario_positive = {}
        per_instrument = {}
        for instrument, rows in sorted(instruments.items()):
            per_instrument[instrument] = {}
            for statistic, slippage in HEADLINE_COSTS:
                scenario = f"{statistic}+{slippage:g}"
                expectancy = statistics.fmean(
                    _scenario_values(rows, profile, statistic, slippage)
                )
                per_instrument[instrument][scenario] = expectancy
                scenario_positive.setdefault(scenario, 0)
                scenario_positive[scenario] += expectancy > 0
        tested = len(instruments)
        base = dict(zip(HYPOTHESIS_FIELDS, hypothesis, strict=True))
        for instrument, evidence_by_cost in per_instrument.items():
            row = base | {
                "instrument": instrument,
                "instrument_observation_count": len(instruments[instrument]),
                "instruments_tested": tested,
                "instruments_with_sufficient_observations": (
                    "not_assessed_no_preregistered_threshold"
                ),
                "cross_asset_support": _json(per_instrument),
            }
            for scenario, positive in scenario_positive.items():
                prefix = scenario.replace("+", "_").replace(".", "p")
                row[f"{prefix}_instruments_positive"] = positive
                row[f"{prefix}_instruments_negative_or_zero"] = tested - positive
                row[f"{prefix}_instrument_expectancy"] = evidence_by_cost[scenario]
            cross_rows.append(row)
        shortlist.append(
            {
                "hypothesis_id": f"H{index:04d}",
                **base,
                "evidence_status": (
                    "descriptive_only_no_preregistered_triage_thresholds"
                ),
                "cross_asset_support": _json(per_instrument),
                "frequency": "see cross-asset-hypotheses.csv",
                "positive_months": "see cost-robustness.csv",
                "plateau_breadth": "see setup-family-matrix.csv",
                "cost_survival": _json(scenario_positive),
                "module_a_overlap": "see overlap-with-module-a.csv",
            }
        )
    _write(output_dir / OUTPUTS[2], cross_rows)
    _write(output_dir / OUTPUTS[5], shortlist)

    raw_vwap = {row["execution_key"] for row in evidence if _module_candidate(row)}
    actual_module = {row["execution_key"] for row in module if _module_candidate(row)}
    overlap_rows = []
    for family, exit_cells in sorted(families.items(), key=lambda item: _json(item[0])):
        opportunities = {
            row["execution_key"] for key in exit_cells for row in cells[key]
        }
        raw_intersection, module_intersection = (
            opportunities & raw_vwap,
            opportunities & actual_module,
        )
        overlap_rows.append(
            dict(zip(SETUP_FIELDS, family, strict=True))
            | {
                "raw_vwap_signal_family_intersection_count": len(raw_intersection),
                "raw_vwap_signal_family_overlap_rate": len(raw_intersection)
                / len(opportunities),
                "actual_frozen_module_a_available": bool(module),
                "actual_frozen_module_a_intersection_count": len(module_intersection),
                "actual_frozen_module_a_overlap_rate": len(module_intersection)
                / len(opportunities),
                "unique_to_setup_vs_module_a": len(opportunities - actual_module),
                "module_a_only_count": len(actual_module - opportunities),
            }
        )
    _write(output_dir / OUTPUTS[4], overlap_rows)
    (output_dir / OUTPUTS[6]).write_text(
        "# Frozen 2024 discovery atlas\n\nExact cells retain performance. Families "
        "report exit-plateau evidence; execution deduplication reports "
        "opportunities only. No automatic triage threshold or ranking is applied.\n"
    )
    audit = {
        "schema_version": SCHEMA,
        "atlas_methodology_version": METHODOLOGY,
        "research_period": {"start": FROZEN_START, "end": FROZEN_END},
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "registry_sha256": registry_sha,
        "cost_profile_sha256": profile_sha,
        "input_shards": commitments,
        "filter_process_provenance": sorted(
            {
                (
                    row.get("filter_family"),
                    row.get("filter_spec_id"),
                    row.get("process_spec_id"),
                    row["role"],
                )
                for row in (*evidence, *module)
            }
        ),
        "output_sha256": {name: _sha(output_dir / name) for name in OUTPUTS},
    }
    (output_dir / "execution-audit.json").write_text(_json(audit) + "\n")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Report authenticated frozen-2024 Stage 4 families"
    )
    parser.add_argument(
        "--stage4b-dir",
        action="append",
        type=Path,
        required=True,
        help="complete baseline/evidence shard set (repeat per shard)",
    )
    parser.add_argument(
        "--module-a-dir",
        action="append",
        type=Path,
        default=[],
        help="separate frozen-OU Module A shard set used only for overlap",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/stage4a-2024-corpus-registry.json"),
    )
    parser.add_argument(
        "--cost-profile",
        type=Path,
        default=Path("configs/stage4c-ftmo-cost-profile-v1.json"),
    )
    args = parser.parse_args(argv)
    build_atlas(
        args.stage4b_dir,
        args.output_dir,
        args.cost_profile,
        args.registry,
        module_a_dirs=args.module_a_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
