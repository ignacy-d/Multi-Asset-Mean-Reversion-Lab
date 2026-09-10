"""Frozen-2024 Bollinger standalone and incremental-to-VWAP analysis.

This is a reporting transform over authenticated Stage 4B raw outputs.  It does
not construct signals, inspect a market-data corpus, or apply an OU filter.
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
from mr_lab.stage4c import SLIPPAGES, SPREAD_STATISTICS, CostProfile, net_pips

SCHEMA_VERSION = "bollinger-incremental-2024-v1"
VWAP_FAMILIES = ("vwap", "vwap-canonical-m1")
FAMILIES = ("bollinger", *VWAP_FAMILIES)
LOOKBACKS = (20, 40)
TP_VALUES = (0.75, 1.0)
SL_VALUES = (0.25, 0.5)
TIME_STOPS = (60, 120)
EVENT_CLASSES = ("bollinger-standalone", "bollinger-only", "vwap-only", "intersection")


class BollingerIncrementalError(ValueError):
    """An input violates the preregistered study contract."""


def _read_jsonl(path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise BollingerIncrementalError(
                    f"malformed JSONL: {path}:{line_number}"
                ) from error


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise BollingerIncrementalError("malformed signal timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise BollingerIncrementalError("signal timestamps must be UTC")
    if parsed.year != 2024:
        raise BollingerIncrementalError("only frozen 2024 inputs are permitted")
    return parsed


def _authenticate_inputs(input_dirs, registry):
    """Authenticate a complete set of native Stage 4B raw shard directories."""
    records = []
    for directory in sorted(map(Path, input_dirs)):
        manifest_path = directory / "shard-manifest.json"
        if not manifest_path.is_file():
            raise BollingerIncrementalError(
                f"missing authenticated Stage 4B shard manifest: {directory}"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as error:
            raise BollingerIncrementalError("malformed shard manifest") from error
        required = (
            "source_commit_sha",
            "registry_identity",
            "raw_artifact_name",
            "full_candidate_count",
            "shard_candidate_count",
            "full_group_keys",
            "group_keys",
        )
        if any(manifest.get(field) is None for field in required):
            raise BollingerIncrementalError("incomplete shard provenance manifest")
        if not (
            isinstance(manifest["source_commit_sha"], str)
            and len(manifest["source_commit_sha"]) == 40
            and isinstance(manifest["full_candidate_count"], int)
            and manifest["full_candidate_count"] >= 0
            and isinstance(manifest["shard_candidate_count"], int)
            and manifest["shard_candidate_count"] >= 0
            and isinstance(manifest["full_group_keys"], list)
            and isinstance(manifest["group_keys"], list)
        ):
            raise BollingerIncrementalError("malformed shard provenance fields")
        if manifest.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
            raise BollingerIncrementalError("unexpected Stage 4B methodology")
        if (
            manifest.get("filter_family") != "none"
            or manifest.get("filter_spec_id") != "none-v1"
        ):
            raise BollingerIncrementalError("only baseline Stage 4B shards are valid")
        instrument = manifest.get("instrument")
        entry = registry.get("instruments", {}).get(instrument, {})
        if entry.get("verification_status") != "verified":
            raise BollingerIncrementalError(f"unverified instrument: {instrument}")
        for field in ("corpus_id", "assembled_dataset_id"):
            if manifest.get(field) != entry.get(field):
                raise BollingerIncrementalError(
                    f"manifest {field} differs from registry"
                )
        source_mode = entry.get("source_mode", "github-artifact")
        if manifest.get("source_mode", "github-artifact") != source_mode:
            raise BollingerIncrementalError(
                "manifest source mode differs from registry"
            )
        provenance_fields = (
            ("source_workflow_run_id", "source_artifact_id")
            if source_mode == "github-artifact"
            else (
                "source_workflow_run_id",
                "source_artifact_id",
                "source_acquisition_commit_sha",
            )
        )
        for field in provenance_fields:
            if source_mode == "github-artifact" and manifest.get(field) is None:
                raise BollingerIncrementalError(
                    f"manifest {field} must identify the GitHub artifact"
                )
            if manifest.get(field) != entry.get(field):
                if source_mode == "github-artifact":
                    continue
                raise BollingerIncrementalError(
                    f"manifest {field} differs from registry"
                )
        for name in ("candidate-events.jsonl", "trades.jsonl"):
            path = directory / name
            expected_hash = manifest.get("file_sha256", {}).get(name)
            expected_rows = manifest.get("row_counts", {}).get(name)
            if (
                not path.is_file()
                or not expected_hash
                or _sha256(path) != expected_hash
            ):
                raise BollingerIncrementalError(f"authenticated hash mismatch: {name}")
            if not isinstance(expected_rows, int) or expected_rows != sum(
                1 for _ in path.open("rb")
            ):
                raise BollingerIncrementalError(
                    f"authenticated row count mismatch: {name}"
                )
        if (
            type(manifest.get("shard_index")) is not int
            or type(manifest.get("shard_count")) is not int
            or not 0 <= manifest["shard_index"] < manifest["shard_count"]
        ):
            raise BollingerIncrementalError("malformed shard identity/count")
        records.append((directory, manifest))

    # A logical run is instrument + immutable source identity. Every logical run
    # must be complete; mixing partial runs cannot manufacture an event universe.
    run_fields = (
        "instrument",
        "source_commit_sha",
        "registry_identity",
        "corpus_id",
        "assembled_dataset_id",
        "source_workflow_run_id",
        "source_artifact_id",
        "source_mode",
        "source_acquisition_commit_sha",
        "shard_count",
        "full_candidate_count",
    )
    runs = defaultdict(list)
    for record in records:
        manifest = record[1]
        runs[tuple(manifest.get(field) for field in run_fields)].append(record)
    seen_instruments = set()
    for run_key, run_records in runs.items():
        instrument = run_key[0]
        if instrument in seen_instruments:
            raise BollingerIncrementalError(
                "mixed logical Stage 4B runs for instrument"
            )
        seen_instruments.add(instrument)
        expected_count = run_key[-2]
        indexes = [manifest["shard_index"] for _, manifest in run_records]
        if len(indexes) != len(set(indexes)):
            raise BollingerIncrementalError("duplicate or overlapping shard identity")
        if set(indexes) != set(range(expected_count)):
            raise BollingerIncrementalError("incomplete or mixed Stage 4B shard set")
        reference = run_records[0][1]
        identical = (*run_fields, "full_group_keys", "filter_family", "filter_spec_id")
        if any(
            any(manifest.get(field) != reference.get(field) for field in identical)
            for _, manifest in run_records
        ):
            raise BollingerIncrementalError("mismatched shard manifest")
        groups = [
            group
            for _, manifest in sorted(
                run_records, key=lambda item: item[1]["shard_index"]
            )
            for group in manifest.get("group_keys", [])
        ]
        if groups != reference.get("full_group_keys") or len(
            {json.dumps(group, sort_keys=True) for group in groups}
        ) != len(groups):
            raise BollingerIncrementalError("incomplete or overlapping shard groups")
        if sum(
            m.get("shard_candidate_count", -1) for _, m in run_records
        ) != reference.get("full_candidate_count"):
            raise BollingerIncrementalError("incomplete candidate coverage")
    return sorted(
        records, key=lambda item: (item[1]["instrument"], item[1]["shard_index"])
    )


def _candidate_index(records, registry):
    candidates = {}
    commitments = []
    for directory, manifest in records:
        path = directory / "candidate-events.jsonl"
        trades = directory / "trades.jsonl"
        if not path.is_file() or not trades.is_file():
            raise BollingerIncrementalError(
                f"missing Stage 4B raw outputs: {directory}"
            )
        commitments.append(
            {
                "directory": str(directory),
                "manifest_sha256": _sha256(directory / "shard-manifest.json"),
                "instrument": manifest["instrument"],
                "shard_index": manifest["shard_index"],
                "shard_count": manifest["shard_count"],
                "candidate_sha256": _sha256(path),
                "trade_sha256": _sha256(trades),
            }
        )
        for row in _read_jsonl(path):
            signal = row.get("signal", {})
            event_id = row.get("candidate_event_id")
            instrument = signal.get("instrument")
            if instrument != manifest["instrument"]:
                raise BollingerIncrementalError(
                    "candidate instrument differs from shard manifest"
                )
            entry = registry.get("instruments", {}).get(instrument, {})
            if entry.get("verification_status") != "verified":
                raise BollingerIncrementalError(f"unverified instrument: {instrument}")
            if signal.get("source_corpus_id") != entry.get("corpus_id"):
                raise BollingerIncrementalError(
                    "candidate corpus identity differs from registry"
                )
            if signal.get("assembled_dataset_id") != entry.get("assembled_dataset_id"):
                raise BollingerIncrementalError(
                    "candidate dataset identity differs from registry"
                )
            if event_id in candidates:
                raise BollingerIncrementalError(
                    f"duplicate candidate event: {event_id}"
                )
            candidates[event_id] = row | {
                "parsed_timestamp": _timestamp(signal.get("signal_timestamp"))
            }
    return candidates, commitments


def _eligible_trade(row, candidate):
    signal = candidate["signal"]
    expected = {
        "instrument": signal["instrument"],
        "benchmark_family": signal["benchmark_family"],
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "SHORT",
        "lookback": signal["lookback"],
    }
    if any(row.get(key) != value for key, value in expected.items()):
        return False
    return (
        row.get("benchmark_family") in FAMILIES
        and row.get("lookback") in LOOKBACKS
        and row.get("signal_threshold") == 2.0
        and row.get("filter_family") == "none"
        and row.get("filter_spec_id") == "none-v1"
        and row.get("entry_mode") == "immediate"
        and row.get("tp_target_fraction") in TP_VALUES
        and row.get("sl_extension_fraction") in SL_VALUES
        and row.get("time_stop_minutes") in TIME_STOPS
        and row.get("complete") is True
    )


def _load_trades(records, candidates):
    rows = []
    identities = set()
    for directory, _manifest in records:
        for row in _read_jsonl(directory / "trades.jsonl"):
            event_id = row.get("candidate_event_id")
            candidate = candidates.get(event_id)
            if candidate is None:
                raise BollingerIncrementalError(
                    f"trade has no candidate row: {event_id}"
                )
            if not _eligible_trade(row, candidate):
                continue
            identity = (
                event_id,
                row["entry_mode"],
                row["tp_target_fraction"],
                row["sl_extension_fraction"],
                row["time_stop_minutes"],
            )
            if identity in identities:
                raise BollingerIncrementalError("duplicate selected trade identity")
            identities.add(identity)
            gross = row.get("gross_return_pips_adverse_first")
            if (
                isinstance(gross, bool)
                or not isinstance(gross, int | float)
                or not math.isfinite(gross)
            ):
                raise BollingerIncrementalError(
                    "selected trade has malformed gross pips"
                )
            rows.append(row | {"signal_timestamp": candidate["parsed_timestamp"]})
    return rows


def _event_membership(candidates):
    strict = defaultdict(set)
    execution = defaultdict(set)
    for _event_id, row in candidates.items():
        signal = row["signal"]
        if (
            signal.get("signal_timeframe") == "15m"
            and signal.get("session") == "london"
            and signal.get("direction") == "SHORT"
            and signal.get("lookback") in LOOKBACKS
            and signal.get("benchmark_family") in FAMILIES
        ):
            key = (signal["instrument"], row["parsed_timestamp"], signal["lookback"])
            strict[key].add(signal["benchmark_family"])
            execution[(signal["instrument"], row["parsed_timestamp"])].add(
                signal["benchmark_family"]
            )
    membership = {}
    for event_id, row in candidates.items():
        signal = row["signal"]
        key = (
            signal.get("instrument"),
            row["parsed_timestamp"],
            signal.get("lookback"),
        )
        families = execution.get(
            (signal.get("instrument"), row["parsed_timestamp"]), set()
        )
        family = signal.get("benchmark_family")
        if family == "bollinger":
            membership[event_id] = (
                "bollinger-standalone",
                "intersection"
                if any(item in families for item in VWAP_FAMILIES)
                else "bollinger-only",
            )
        elif family in VWAP_FAMILIES:
            membership[event_id] = (
                "intersection" if "bollinger" in families else "vwap-only",
            )
    return membership, {"strict": strict, "execution": execution}


def _profit_factor(values):
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else None


def _losing_streak(rows, value_key):
    maximum = current = 0
    for row in sorted(
        rows, key=lambda item: (item["signal_timestamp"], item["candidate_event_id"])
    ):
        if row[value_key] < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _metrics(rows, value_key):
    values = [row[value_key] for row in rows]
    monthly = defaultdict(list)
    for row in rows:
        monthly[row["signal_timestamp"].strftime("%Y-%m")].append(row[value_key])
    month_means = {
        month: statistics.fmean(items) for month, items in sorted(monthly.items())
    }
    worst = min(month_means, key=month_means.get) if month_means else None
    return {
        "trade_count": len(rows),
        "unique_candidate_count": len({row["candidate_event_id"] for row in rows}),
        "duplicate_adjusted_unique_trade_count": len(
            {row["signal_timestamp"] for row in rows}
        ),
        "trades_per_month": len(rows) / 12,
        "expectancy_pips": statistics.fmean(values) if values else None,
        "profit_factor": _profit_factor(values),
        "win_rate": sum(value > 0 for value in values) / len(values)
        if values
        else None,
        "monthly_stability_stddev": statistics.stdev(month_means.values())
        if len(month_means) > 1
        else None,
        "positive_months": sum(value > 0 for value in month_means.values()),
        "total_months": len(month_means),
        "worst_month": worst,
        "worst_month_expectancy_pips": month_means.get(worst),
        "max_losing_streak": _losing_streak(rows, value_key),
    }


def _correlation(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def run(input_dirs, output_dir, registry_path, cost_profile_path):
    registry_path, cost_profile_path = Path(registry_path), Path(cost_profile_path)
    registry = json.loads(registry_path.read_text())
    authenticated = _authenticate_inputs(input_dirs, registry)
    candidates, commitments = _candidate_index(authenticated, registry)
    trades = _load_trades(authenticated, candidates)
    membership, event_sets = _event_membership(candidates)
    profile = CostProfile.load(cost_profile_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    grouped = defaultdict(list)
    for row in trades:
        for event_class in membership.get(row["candidate_event_id"], ()):
            grouped[
                (
                    row["instrument"],
                    event_class,
                    row["benchmark_family"],
                    row["lookback"],
                    row["tp_target_fraction"],
                    row["sl_extension_fraction"],
                    row["time_stop_minutes"],
                )
            ].append(row)
    for key, selected in sorted(grouped.items()):
        instrument, event_class, family, lookback, tp, sl, stop = key
        gross_rows = [
            row | {"gross": row["gross_return_pips_adverse_first"]} for row in selected
        ]
        base = dict(
            zip(
                (
                    "instrument",
                    "event_class",
                    "outcome_family",
                    "lookback",
                    "tp",
                    "sl",
                    "time_stop_minutes",
                ),
                key,
                strict=True,
            )
        )
        results.append(
            base
            | {
                "spread_statistic": "gross",
                "slippage_pips": None,
                **_metrics(gross_rows, "gross"),
            }
        )
        spread, commission = profile.costs(instrument, "london", "mean")
        for statistic in SPREAD_STATISTICS:
            spread, commission = profile.costs(instrument, "london", statistic)
            for slippage in SLIPPAGES:
                net_rows = [
                    row
                    | {
                        "net": net_pips(
                            row["gross_return_pips_adverse_first"],
                            instrument,
                            spread,
                            slippage,
                            commission,
                        )
                    }
                    for row in selected
                ]
                results.append(
                    base
                    | {
                        "spread_statistic": statistic,
                        "slippage_pips": slippage,
                        **_metrics(net_rows, "net"),
                    }
                )
    fieldnames = list(results[0]) if results else []
    with (output_dir / "bollinger-incremental-matrix.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(results)
    overlap = []
    instruments = sorted({key[0] for key in event_sets["execution"]})
    for instrument in instruments:
        for definition, lookbacks in (
            ("strict-specification", LOOKBACKS),
            ("execution-level", ("all",)),
        ):
            for lookback in lookbacks:
                source = event_sets[
                    "strict" if definition == "strict-specification" else "execution"
                ]
                keys = {
                    key: families
                    for key, families in source.items()
                    if key[0] == instrument
                    and (definition == "execution-level" or key[2] == lookback)
                }
                bb = {
                    key[1] for key, families in keys.items() if "bollinger" in families
                }
                vwap = {
                    key[1]
                    for key, families in keys.items()
                    if any(item in families for item in VWAP_FAMILIES)
                }
                both = bb & vwap
                overlap.append(
                    {
                        "instrument": instrument,
                        "overlap_definition": definition,
                        "lookback": lookback,
                        "bollinger_candidates": len(bb),
                        "vwap_module_a_candidates": len(vwap),
                        "intersection_count": len(both),
                        "bollinger_only_incremental_count": len(bb - vwap),
                        "bollinger_only_incremental_rate": len(bb - vwap) / len(bb)
                        if bb
                        else None,
                        "bollinger_overlap_rate": len(both) / len(bb) if bb else None,
                        "vwap_overlap_rate": len(both) / len(vwap) if vwap else None,
                        "union_duplicate_adjusted_unique_trade_count": len(bb | vwap),
                    }
                )
    with (output_dir / "event-overlap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(overlap[0]) if overlap else [])
        if overlap:
            writer.writeheader()
            writer.writerows(overlap)
    correlations = []
    intersection = [
        row
        for row in trades
        if "intersection" in membership.get(row["candidate_event_id"], ())
    ]
    paired = defaultdict(dict)
    for row in intersection:
        key = (
            row["instrument"],
            row["lookback"],
            row["tp_target_fraction"],
            row["sl_extension_fraction"],
            row["time_stop_minutes"],
            row["signal_timestamp"],
        )
        paired[key][row["benchmark_family"]] = row["gross_return_pips_adverse_first"]
    correlation_groups = defaultdict(list)
    for key, values in paired.items():
        for vwap_family in VWAP_FAMILIES:
            if "bollinger" in values and vwap_family in values:
                correlation_groups[(*key[:-1], vwap_family)].append(
                    (values["bollinger"], values[vwap_family])
                )
    for key, pairs in sorted(correlation_groups.items()):
        correlations.append(
            dict(
                zip(
                    (
                        "instrument",
                        "lookback",
                        "tp",
                        "sl",
                        "time_stop_minutes",
                        "vwap_outcome_family",
                    ),
                    key,
                    strict=True,
                )
            )
            | {
                "paired_intersection_count": len(pairs),
                "gross_outcome_pearson_correlation": _correlation(
                    [x for x, _ in pairs], [y for _, y in pairs]
                ),
            }
        )
    with (output_dir / "intersection-correlation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(correlations[0]) if correlations else []
        )
        if correlations:
            writer.writeheader()
            writer.writerows(correlations)
    breadth_groups = defaultdict(list)
    for row in results:
        breadth_groups[
            (
                row["instrument"],
                row["event_class"],
                row["outcome_family"],
                row["spread_statistic"],
                row["slippage_pips"],
            )
        ].append(row)
    breadth = []
    for key, cells in sorted(breadth_groups.items(), key=lambda item: str(item[0])):
        breadth.append(
            dict(
                zip(
                    (
                        "instrument",
                        "event_class",
                        "outcome_family",
                        "spread_statistic",
                        "slippage_pips",
                    ),
                    key,
                    strict=True,
                )
            )
            | {
                "observed_plateau_cells": len(cells),
                "positive_expectancy_cells": sum(
                    row["expectancy_pips"] > 0 for row in cells
                ),
                "positive_expectancy_fraction": sum(
                    row["expectancy_pips"] > 0 for row in cells
                )
                / len(cells),
            }
        )
    with (output_dir / "plateau-breadth.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breadth[0]) if breadth else [])
        if breadth:
            writer.writeheader()
            writer.writerows(breadth)
    report_lines = [
        "# Frozen-2024 Bollinger incremental study",
        "",
        "This report deliberately does not rank or select a parameter cell. Review ",
        "`plateau-breadth.csv` across both lookbacks and all eight frozen exit cells.",
        "The cost floor is `mean` spread with `0.0` slippage; it is optimistic.",
        "",
        f"Selected complete trade rows: **{len(trades)}**.",
        "",
        "See `event-overlap.csv` for incremental BB-only counts, "
        "`intersection-correlation.csv` for paired outcome correlation, and "
        "`bollinger-incremental-matrix.csv` for all requested trade metrics.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "research_year": 2024,
        "dimensions": {
            "timeframe": "15m",
            "session": "london",
            "direction": "SHORT",
            "lookbacks": list(LOOKBACKS),
            "entry_mode": "immediate",
            "tp": list(TP_VALUES),
            "sl": list(SL_VALUES),
            "time_stop_minutes": list(TIME_STOPS),
            "threshold": 2.0,
            "filter": "none-v1",
        },
        "input_commitments": commitments,
        "registry_sha256": _sha256(registry_path),
        "cost_profile_sha256": profile.sha256,
        "selected_trade_rows": len(trades),
        "candidate_rows": len(candidates),
        "outputs": {},
    }
    for name in (
        "bollinger-incremental-matrix.csv",
        "event-overlap.csv",
        "intersection-correlation.csv",
        "plateau-breadth.csv",
        "report.md",
    ):
        audit["outputs"][name] = _sha256(output_dir / name)
    (output_dir / "execution-audit.json").write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return results, overlap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    run(args.input_dir, args.output_dir, args.registry, args.cost_profile)


if __name__ == "__main__":
    main()
