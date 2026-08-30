"""Streaming runner and reporting for frozen Stage 4C-A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import statistics
import subprocess
import tempfile
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
SHARD_STATE = "stage4c-shard-state.jsonl"
SHARD_MANIFEST = "stage4c-shard-manifest.json"
ALLOWED_SOURCE_AUDIT_FIELDS = (
    "instrument",
    "corpus_id",
    "assembled_dataset_id",
    "registry_identity",
    "source_trade_sha256",
    "source_shard_identities",
    "expected_shard_count",
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


def _group_owner(key, shard_count):
    digest = hashlib.sha256(_json(key).encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def _spool_rows(rows, database, *, shard_index=None, shard_count=None):
    """Externally group arbitrary input with uniqueness enforced on disk."""
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE trades (group_key TEXT NOT NULL, identity TEXT PRIMARY KEY, "
        "row_json TEXT NOT NULL) WITHOUT ROWID"
    )
    before = complete = 0
    groups = set()
    try:
        for row in rows:
            before += 1
            if not validate_trade(row):
                continue
            complete += 1
            group = _key(row, CONFIG_FIELDS)
            if (
                shard_count is not None
                and _group_owner(group, shard_count) != shard_index
            ):
                continue
            group_json = _json(group)
            groups.add(group_json)
            try:
                connection.execute(
                    "INSERT INTO trades VALUES (?, ?, ?)",
                    (group_json, _json(trade_identity(row)), _json(row)),
                )
            except sqlite3.IntegrityError as error:
                raise Stage4CError("duplicate original trade identity") from error
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection, before, complete, tuple(sorted(groups))


def _iter_spooled_groups(connection):
    current_key = None
    rows = []
    cursor = connection.execute(
        "SELECT group_key, row_json FROM trades ORDER BY group_key, identity"
    )
    for group_key, row_json in cursor:
        if current_key is not None and group_key != current_key:
            yield tuple(json.loads(current_key)), rows
            rows = []
        current_key = group_key
        rows.append(json.loads(row_json))
    if current_key is not None:
        yield tuple(json.loads(current_key)), rows


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
    shard_index=None,
    shard_count=None,
):
    """Externally group rows, retaining only the currently processed group in RAM."""
    if source_mode not in ("regenerated_stage4b", "existing_stage4b_raw"):
        raise Stage4CError("invalid source_mode")
    if source_audit.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise Stage4CError("inconsistent Stage 4B methodology")
    for required in ("corpus_id", "assembled_dataset_id"):
        if not source_audit.get(required):
            raise Stage4CError(f"missing expected input identity: {required}")
    profile = CostProfile.load(profile_path)
    expected_matrix = {
        "spread_statistics": list(SPREAD_STATISTICS),
        "slippage_round_turn_pips": list(SLIPPAGES),
    }
    if (shard_index is None) != (shard_count is None):
        raise Stage4CError("shard index and count must be supplied together")
    if shard_count is not None and not 0 <= shard_index < shard_count:
        raise Stage4CError("invalid shard index")
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / ".stage4c-spool.sqlite3"
    connection, before, complete, owned_groups = _spool_rows(
        rows, database, shard_index=shard_index, shard_count=shard_count
    )
    expanded = 0
    matrix = []
    raw = (output_dir / "stage4c-debug-trades.jsonl").open("w") if debug_raw else None
    state = (output_dir / SHARD_STATE).open("w") if shard_count is not None else None
    try:
        for group_key, group_rows in _iter_spooled_groups(connection):
            for statistic in SPREAD_STATISTICS:
                for slippage in SLIPPAGES:
                    matrix.append(
                        _cell_row(group_key, group_rows, profile, statistic, slippage)
                    )
                    expanded += len(group_rows)
            if raw:
                for row in group_rows:
                    for transformed in scenario_rows(row, profile):
                        raw.write(_json(transformed) + "\n")
            if state:
                for row in group_rows:
                    compact = {field: row.get(field) for field in CONFIG_FIELDS}
                    compact.update(
                        candidate_event_id=row["candidate_event_id"],
                        complete=True,
                        gross_return_pips_adverse_first=row[
                            "gross_return_pips_adverse_first"
                        ],
                        gross_return_pips_favorable_first=row[
                            "gross_return_pips_favorable_first"
                        ],
                    )
                    state.write(_json(compact) + "\n")
    finally:
        connection.close()
        database.unlink(missing_ok=True)
        if raw:
            raw.close()
        if state:
            state.close()
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
    permitted_source = {
        field: source_audit[field]
        for field in ALLOWED_SOURCE_AUDIT_FIELDS
        if field in source_audit
    }
    audit = permitted_source | {
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
        "currency_conversion_adjustment": {
            "USDJPY/AUDJPY": (
                "after spread and slippage: positive x 0.993, negative x 1.007, "
                "zero unchanged; then subtract commission"
            ),
            "EURUSD/AUDUSD": "no quote-currency adjustment",
        },
        "volume_bands_modeled": False,
        "swaps_modeled": False,
    }
    audit["output_sha256"] = {
        name: sha256_file(output_dir / name) for name in OUTPUTS[:5]
    }
    (output_dir / OUTPUTS[5]).write_text(_json(audit) + "\n")
    if shard_count is not None:
        manifest = {
            "schema_version": "stage4c-compact-shard-v1",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "group_ownership": [json.loads(key) for key in owned_groups],
            "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
            "stage4b_source_commit": STAGE4B_SOURCE_COMMIT,
            "cost_profile_sha256": profile.sha256,
            "scenario_matrix": expected_matrix,
            "instrument": source_audit.get("instrument"),
            "corpus_id": source_audit["corpus_id"],
            "assembled_dataset_id": source_audit["assembled_dataset_id"],
            "state_row_count": sum(1 for _ in (output_dir / SHARD_STATE).open()),
            "state_sha256": sha256_file(output_dir / SHARD_STATE),
        }
        (output_dir / SHARD_MANIFEST).write_text(_json(manifest) + "\n")
    return summary


def iter_jsonl(paths):
    for path in paths:
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise Stage4CError(f"malformed JSONL {path}:{number}") from error


def run_regenerated(
    corpus_dir,
    output_dir,
    instrument,
    registry_path,
    profile_path,
    *,
    debug_raw=False,
    shard_index=None,
    shard_count=None,
):
    """Run frozen Stage 4B and feed its immutable callback rows into Stage 4C."""
    from mr_lab.stage4b_runner import run as run_stage4b

    with tempfile.TemporaryDirectory(prefix="stage4c-regenerated-") as temporary:
        temporary = Path(temporary)
        callback_rows = temporary / "callback-rows.jsonl"
        with callback_rows.open("w") as stream:
            run_stage4b(
                corpus_dir,
                temporary / "stage4b",
                instrument,
                registry_path,
                trade_row_consumer=lambda row: stream.write(_json(row) + "\n"),
            )
        stage4b_audit = json.loads(
            (temporary / "stage4b" / "execution-audit.json").read_text()
        )
        source_audit = {
            "stage4b_methodology_id": stage4b_audit["stage4b_methodology_id"],
            "instrument": stage4b_audit["instrument"],
            "corpus_id": stage4b_audit["corpus_id"],
            "assembled_dataset_id": stage4b_audit["assembled_dataset_id"],
            "registry_identity": sha256_file(Path(registry_path)),
            "source_trade_sha256": {"callback_stream": sha256_file(callback_rows)},
        }
        return run_rows(
            iter_jsonl((callback_rows,)),
            Path(output_dir),
            Path(profile_path),
            source_mode="regenerated_stage4b",
            source_audit=source_audit,
            debug_raw=debug_raw,
            shard_index=shard_index,
            shard_count=shard_count,
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-mode",
        choices=("regenerated_stage4b", "existing_stage4b_raw"),
        default="regenerated_stage4b",
    )
    parser.add_argument("--stage4b-trades", type=Path, action="append")
    parser.add_argument("--stage4b-audit", type=Path)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--instrument")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug-raw", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args(argv)
    if args.source_mode == "regenerated_stage4b":
        if not args.corpus_dir or not args.instrument:
            parser.error("regenerated route requires --corpus-dir and --instrument")
        run_regenerated(
            args.corpus_dir,
            args.output_dir,
            args.instrument,
            args.registry,
            args.cost_profile,
            debug_raw=args.debug_raw,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        return 0
    if not args.stage4b_trades or not args.stage4b_audit:
        parser.error("existing raw route requires --stage4b-trades and --stage4b-audit")
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
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
