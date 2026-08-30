"""Streaming runner and reporting for frozen Stage 4C-A."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import (
    CONFIG_FIELDS,
    REGIME_FIELDS,
    SCHEMA_VERSION,
    SLIPPAGES,
    SPREAD_STATISTICS,
    STAGE4B_SOURCE_COMMIT,
    CostProfile,
    Stage4CError,
    break_even_total_slippage,
    scenario_rows,
    sha256_file,
    trade_identity,
    transform_trade,
    validate_trade,
)

OUTPUTS = (
    "stage4c-trade-matrix.csv",
    "stage4c-regime-breadth.csv",
    "stage4c-cost-scenarios.csv",
    "stage4c-summary.json",
    "stage4c-report.md",
    "execution-audit.json",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _commit_sha():
    value = os.environ.get("STAGE4C_SOURCE_COMMIT")
    if not value:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip()
    if len(value) != 40:
        raise Stage4CError("valid Stage 4C source commit required")
    return value


def _quantile(values, probability):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _mean(values):
    return statistics.fmean(values)


def _profit_factor(values):
    gains = sum(x for x in values if x > 0)
    losses = -sum(x for x in values if x < 0)
    return gains / losses if losses else (math.inf if gains else 0.0)


def _key(row, fields):
    return tuple(row.get(field) for field in fields)


def _cell_row(key, rows, profile, statistic, slippage):
    transformed = [transform_trade(row, profile, statistic, slippage) for row in rows]
    first = transformed[0]
    rows = transformed
    net = [row["net_pips_adverse_first"] for row in rows]
    favorable = [row["net_pips_favorable_first"] for row in rows]
    gross = [row["gross_return_pips_adverse_first"] for row in rows]
    spread, commission = first["spread_pips"], first["commission_pips"]
    total_be = break_even_total_slippage(gross, first["instrument"], spread, commission)
    n = len(rows)
    wins, losses = sum(x > 0 for x in net), sum(x < 0 for x in net)
    scenario_key = (*key, statistic, slippage)
    return dict(
        zip(
            (*CONFIG_FIELDS, "spread_statistic", "slippage_pips"),
            scenario_key,
            strict=True,
        )
    ) | {
        "complete_trade_count": n,
        "mean_net_pips": _mean(net),
        "median_net_pips": _quantile(net, 0.5),
        "net_pips_p10": _quantile(net, 0.1),
        "net_pips_p25": _quantile(net, 0.25),
        "net_pips_p75": _quantile(net, 0.75),
        "net_pips_p90": _quantile(net, 0.9),
        "net_win_fraction": wins / n,
        "net_loss_fraction": losses / n,
        "net_zero_fraction": (n - wins - losses) / n,
        "mean_net_pips_adverse_first": _mean(net),
        "mean_net_pips_favorable_first": _mean(favorable),
        "net_profit_factor": _profit_factor(net),
        "mean_gross_pips": _mean(gross),
        "modeled_transaction_cost_pips": _mean(
            [r["modeled_cost_before_conversion_adjustment"] for r in rows]
        ),
        "cost_headroom_pips": _mean(gross)
        - (spread + first["slippage_pips"] + commission),
        "break_even_total_slippage_pips": total_be,
        "break_even_extra_slippage_pips": max(0.0, total_be - first["slippage_pips"]),
    }


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else ["schema_version"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _regime_rows(matrix):
    grouped = defaultdict(list)
    for row in matrix:
        grouped[
            _key(row, (*REGIME_FIELDS, "spread_statistic", "slippage_pips"))
        ].append(row)
    output = []
    for key, cells in sorted(grouped.items(), key=lambda item: _json(item[0])):
        gross_positive = [r for r in cells if r["mean_gross_pips"] > 0]
        net_positive = [r for r in cells if r["mean_net_pips"] > 0]
        surviving = [r for r in gross_positive if r["mean_net_pips"] > 0]
        output.append(
            dict(
                zip(
                    (*REGIME_FIELDS, "spread_statistic", "slippage_pips"),
                    key,
                    strict=True,
                )
            )
            | {
                "dependent_configuration_cell_count": len(cells),
                "net_positive_cell_count": len(net_positive),
                "net_positive_cell_fraction": len(net_positive) / len(cells),
                "gross_positive_cell_count": len(gross_positive),
                "gross_positive_cells_remaining_net_positive_fraction": (
                    len(surviving) / len(gross_positive) if gross_positive else 0.0
                ),
            }
        )
    return output


def run_rows(
    rows,
    output_dir: Path,
    profile_path: Path,
    *,
    source_mode: str,
    source_audit: dict,
    debug_raw=False,
):
    """Consume Stage 4B rows once. Only small per-cell samples needed for exact
    frozen quantiles are retained; no 16x production raw artifact is created."""
    if source_mode not in ("regenerated_stage4b", "existing_stage4b_raw"):
        raise Stage4CError("invalid source_mode")
    if source_audit.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise Stage4CError("inconsistent Stage 4B methodology")
    for required in ("corpus_id", "assembled_dataset_id"):
        if not source_audit.get(required):
            raise Stage4CError(f"missing expected input identity: {required}")
    profile = CostProfile.load(profile_path)
    supplied_profile_hash = source_audit.get("cost_profile_sha256")
    if supplied_profile_hash and supplied_profile_hash != profile.sha256:
        raise Stage4CError("inconsistent cost-profile SHA")
    supplied_matrix = source_audit.get("scenario_matrix")
    expected_matrix = {
        "spread_statistics": list(SPREAD_STATISTICS),
        "slippage_round_turn_pips": list(SLIPPAGES),
    }
    if supplied_matrix and supplied_matrix != expected_matrix:
        raise Stage4CError("inconsistent scenario matrix")
    shard_ids = source_audit.get("source_shard_identities", [])
    if len(shard_ids) != len({_json(identity) for identity in shard_ids}):
        raise Stage4CError("duplicate input identity")
    expected_shards = source_audit.get("expected_shard_count")
    if expected_shards is not None and len(shard_ids) != expected_shards:
        raise Stage4CError("missing expected shard/input identity")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exact empirical quantiles need one pair of scalar gross observations per
    # Stage 4B row. Scenario rows are ephemeral, so normal memory is O(N), not
    # O(16N), and no expanded production artifact is written.
    groups, seen = defaultdict(list), set()
    before = complete = expanded = 0
    raw = (output_dir / "stage4c-debug-trades.jsonl").open("w") if debug_raw else None
    try:
        for row in rows:
            before += 1
            if not validate_trade(row):
                continue
            complete += 1
            original = trade_identity(row)
            if original in seen:
                raise Stage4CError("duplicate original trade identity")
            seen.add(original)
            gross_snapshot = (
                row["gross_return_pips_adverse_first"],
                row["gross_return_pips_favorable_first"],
            )
            for transformed in scenario_rows(row, profile):
                expanded += 1
                if raw:
                    raw.write(_json(transformed) + "\n")
            groups[_key(row, CONFIG_FIELDS)].append(row.copy())
            if gross_snapshot != (
                row["gross_return_pips_adverse_first"],
                row["gross_return_pips_favorable_first"],
            ):
                raise Stage4CError("original gross fields mutated")
    finally:
        if raw:
            raw.close()
    matrix = [
        _cell_row(key, values, profile, statistic, slippage)
        for key, values in sorted(groups.items(), key=lambda item: _json(item[0]))
        for statistic in SPREAD_STATISTICS
        for slippage in SLIPPAGES
    ]
    regimes = _regime_rows(matrix)
    scenarios = [
        {
            "spread_statistic": spread,
            "slippage_pips": slip,
            "headline_cost_floor": spread == "mean" and slip == 0.0,
        }
        for spread in SPREAD_STATISTICS
        for slip in SLIPPAGES
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_mode": source_mode,
        "stage4b_rows_read": before,
        "complete_stage4b_trade_count": complete,
        "scenario_evaluation_count": expanded,
        "configuration_scenario_cell_count": len(matrix),
        "interpretation": (
            "Report broad net-positive plateaus; do not select a historical "
            "maximum-mean cell."
        ),
        "accounting_approximation": "Frozen Stage 4C-A accounting approximation",
    }
    _write_csv(output_dir / OUTPUTS[0], matrix)
    _write_csv(output_dir / OUTPUTS[1], regimes)
    _write_csv(output_dir / OUTPUTS[2], scenarios)
    (output_dir / OUTPUTS[3]).write_text(_json(summary) + "\n")
    (output_dir / OUTPUTS[4]).write_text(
        "# Stage 4C-A transaction-cost viability\n\n"
        "This is the frozen accounting approximation. The mean-spread, zero-slippage "
        "case is an optimistic cost floor, not expected live P/L. Review broad "
        "net-positive plateaus; no maximum-mean cell is selected.\n"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "source_mode": source_mode,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "stage4b_source_commit": STAGE4B_SOURCE_COMMIT,
        "stage4c_source_commit": _commit_sha(),
        "cost_profile_sha256": profile.sha256,
        "scenario_matrix": {
            "spread_statistics": list(SPREAD_STATISTICS),
            "slippage_round_turn_pips": list(SLIPPAGES),
        },
        "row_counts": {
            "stage4b_input": before,
            "complete": complete,
            "scenario_evaluations": expanded,
        },
        "account_currency": "USD",
        "volume_bands_modeled": False,
        "swaps_modeled": False,
        **source_audit,
    }
    audit["output_sha256"] = {
        name: sha256_file(output_dir / name) for name in OUTPUTS[:5]
    }
    (output_dir / OUTPUTS[5]).write_text(_json(audit) + "\n")
    return summary


def iter_jsonl(paths):
    for path in paths:
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise Stage4CError(f"malformed JSONL {path}:{number}") from error


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4b-trades", type=Path, action="append", required=True)
    parser.add_argument("--stage4b-audit", type=Path, required=True)
    parser.add_argument(
        "--cost-profile",
        type=Path,
        default=Path("configs/stage4c-ftmo-cost-profile-v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug-raw", action="store_true")
    args = parser.parse_args(argv)
    source = json.loads(args.stage4b_audit.read_text())
    source["source_trade_sha256"] = {
        str(p): sha256_file(p) for p in args.stage4b_trades
    }
    run_rows(
        iter_jsonl(args.stage4b_trades),
        args.output_dir,
        args.cost_profile,
        source_mode="existing_stage4b_raw",
        source_audit=source,
        debug_raw=args.debug_raw,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
