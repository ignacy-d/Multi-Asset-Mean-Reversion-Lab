"""Local runner for the frozen-2024 zero-cost Stage 4C diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import CONFIG_FIELDS, REGIME_FIELDS
from mr_lab.stage4c_zero_cost import (
    METHODOLOGY_ID,
    MODE,
    SCHEMA_VERSION,
    ZeroCostDiagnosticError,
    transform_trade,
    validate_instrument,
)

OUTPUTS = (
    "stage4c-zero-cost-trade-matrix.csv",
    "stage4c-zero-cost-regime-breadth.csv",
    "stage4c-zero-cost-summary.json",
    "stage4c-zero-cost-report.md",
)
CROSS_ASSET_OUTPUTS = (
    "stage4c-zero-cost-cross-asset.csv",
    "stage4c-zero-cost-cross-asset.md",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_commit():
    value = os.environ.get("STAGE4C_ZERO_COST_SOURCE_COMMIT")
    if value is None:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ZeroCostDiagnosticError("valid source commit required")
    return value


def validate_2024_path(path: Path, year: int):
    """Validate only lexical provenance; callers must invoke this before I/O."""
    if year != 2024:
        raise ZeroCostDiagnosticError("only the explicit discovery year is permitted")
    parts = path.expanduser().parts
    if not any(part == "2024" or "2024" in part.split("_") for part in parts):
        raise ZeroCostDiagnosticError("corpus path has ambiguous year provenance")


def _write_csv(path: Path, rows: list[dict]):
    fields = list(rows[0]) if rows else ["schema_version"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _median(values):
    return statistics.median(values)


def run_rows(rows, output_dir: Path, instrument: str, source_audit: dict, corpus_path):
    instrument = validate_instrument(instrument)
    if source_audit.get("instrument") != instrument:
        raise ZeroCostDiagnosticError("source audit instrument mismatch")
    if source_audit.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise ZeroCostDiagnosticError("inconsistent Stage 4B methodology")
    for field in ("corpus_id", "assembled_dataset_id"):
        if not source_audit.get(field):
            raise ZeroCostDiagnosticError(f"missing source identity: {field}")
    requested_start = str(source_audit.get("requested_start_date", ""))
    requested_end = str(source_audit.get("requested_end_date", ""))
    if not requested_start.startswith("2024-") or not requested_end.startswith("2024-"):
        raise ZeroCostDiagnosticError("source audit does not prove a 2024-only range")

    groups = defaultdict(list)
    source_count = complete_count = 0
    for row in rows:
        source_count += 1
        if not row.get("complete"):
            if row.get("instrument") != instrument:
                raise ZeroCostDiagnosticError("mixed or unexpected instrument")
            continue
        evaluated = transform_trade(row, instrument)
        complete_count += 1
        groups[tuple(evaluated.get(field) for field in CONFIG_FIELDS)].append(evaluated)

    matrix = []
    for key, trades in sorted(groups.items(), key=lambda item: _json(item[0])):
        adverse = [trade["zero_cost_pips_adverse_first"] for trade in trades]
        favorable = [trade["zero_cost_pips_favorable_first"] for trade in trades]
        wins = sum(value > 0 for value in adverse)
        matrix.append(
            dict(zip(CONFIG_FIELDS, key, strict=True))
            | {
                "complete_trade_count": len(trades),
                "mean_zero_cost_pips_adverse_first": statistics.fmean(adverse),
                "mean_zero_cost_pips_favorable_first": statistics.fmean(favorable),
                "median_zero_cost_pips_adverse_first": _median(adverse),
                "median_zero_cost_pips_favorable_first": _median(favorable),
                "zero_cost_win_fraction_adverse_first": wins / len(trades),
            }
        )
    regimes = []
    regime_groups = defaultdict(list)
    for row in matrix:
        regime_groups[tuple(row.get(field) for field in REGIME_FIELDS)].append(row)
    for key, cells in sorted(regime_groups.items(), key=lambda item: _json(item[0])):
        positive = [
            cell for cell in cells if cell["mean_zero_cost_pips_adverse_first"] > 0
        ]
        regimes.append(
            dict(zip(REGIME_FIELDS, key, strict=True))
            | {
                "dependent_configuration_cell_count": len(cells),
                "zero_cost_positive_cell_count": len(positive),
                "zero_cost_positive_cell_fraction": len(positive) / len(cells),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / OUTPUTS[0], matrix)
    _write_csv(output_dir / OUTPUTS[1], regimes)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "instrument": instrument,
        "stage4b_rows_read": source_count,
        "complete_stage4b_trade_count": complete_count,
        "configuration_cell_count": len(matrix),
        "interpretation": "Descriptive raw-edge diagnostic; no ranking or selection.",
    }
    (output_dir / OUTPUTS[2]).write_text(_json(summary) + "\n")
    (output_dir / OUTPUTS[3]).write_text(
        "# Stage 4C zero-cost diagnostic\n\n"
        "This frozen-2024 diagnostic reports Stage 4B gross pips unchanged. "
        "It is not production net return or Stage 4C-A cost validation. No setup "
        "or instrument is ranked.\n"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "instrument": instrument,
        "corpus_id": source_audit["corpus_id"],
        "assembled_dataset_id": source_audit["assembled_dataset_id"],
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "zero_cost_diagnostic_methodology_id": METHODOLOGY_ID,
        "source_commit": _source_commit(),
        "row_counts": {
            "source_rows": source_count,
            "complete_rows": complete_count,
        },
        "spread_pips": 0,
        "commission_pips": 0,
        "slippage_pips": 0,
        "currency_conversion_adjustment": False,
        "discovery_year": 2024,
        "assertion_2024_only": True,
        "explicit_corpus_path": str(Path(corpus_path).expanduser().absolute()),
        "path_discovery_performed": False,
        "output_sha256": {name: _sha256(output_dir / name) for name in OUTPUTS},
    }
    (output_dir / "execution-audit.json").write_text(_json(audit) + "\n")
    return summary


def _iter_jsonl(path: Path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ZeroCostDiagnosticError(
                    f"malformed Stage 4B trade row at line {number}"
                ) from error


def run_corpus(corpus: Path, output: Path, instrument: str, *, year: int):
    validate_instrument(instrument)
    validate_2024_path(corpus, year)  # Deliberately before any filesystem access.
    audit_path = corpus / "execution-audit.json"
    trades_path = corpus / "trades.jsonl"
    audit = json.loads(audit_path.read_text())
    declared_hash = audit.get("output_sha256", {}).get("trades.jsonl")
    if not declared_hash or declared_hash != _sha256(trades_path):
        raise ZeroCostDiagnosticError("Stage 4B trade hash is missing or mismatched")
    return run_rows(_iter_jsonl(trades_path), output, instrument, audit, corpus)


def summarize_cross_asset(instrument_outputs, output_dir: Path):
    rows = []
    for instrument, directory in sorted(instrument_outputs):
        audit = json.loads((directory / "execution-audit.json").read_text())
        if audit.get("instrument") != instrument or audit.get("mode") != MODE:
            raise ZeroCostDiagnosticError("cross-asset input identity mismatch")
        with (directory / OUTPUTS[0]).open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append(
                    {
                        "instrument": instrument,
                        **{field: row[field] for field in CONFIG_FIELDS[1:]},
                        "complete_trade_count": row["complete_trade_count"],
                        "mean_zero_cost_pips_adverse_first": row[
                            "mean_zero_cost_pips_adverse_first"
                        ],
                        "mean_zero_cost_pips_favorable_first": row[
                            "mean_zero_cost_pips_favorable_first"
                        ],
                        "median_zero_cost_pips_adverse_first": row[
                            "median_zero_cost_pips_adverse_first"
                        ],
                        "zero_cost_win_fraction_adverse_first": row[
                            "zero_cost_win_fraction_adverse_first"
                        ],
                        "corpus_id": audit["corpus_id"],
                        "assembled_dataset_id": audit["assembled_dataset_id"],
                        "stage4b_methodology_id": audit["stage4b_methodology_id"],
                        "zero_cost_diagnostic_methodology_id": audit[
                            "zero_cost_diagnostic_methodology_id"
                        ],
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / CROSS_ASSET_OUTPUTS[0], rows)
    (output_dir / CROSS_ASSET_OUTPUTS[1]).write_text(
        "# Stage 4C zero-cost cross-asset diagnostic\n\n"
        "Deterministically ordered descriptive configuration evidence. No composite "
        "score, best pair, or best setup is produced.\n"
    )


def run_manifest(manifest: Path, output: Path):
    with manifest.open(newline="") as stream:
        entries = list(csv.DictReader(stream))
    if not entries or set(entries[0]) != {"instrument", "corpus_path", "year"}:
        raise ZeroCostDiagnosticError(
            "manifest columns must be instrument,corpus_path,year"
        )
    outputs = []
    seen = set()
    for entry in entries:
        instrument = validate_instrument(entry["instrument"])
        if instrument in seen:
            raise ZeroCostDiagnosticError("duplicate manifest instrument")
        seen.add(instrument)
        try:
            year = int(entry["year"])
        except ValueError as error:
            raise ZeroCostDiagnosticError("manifest year must be 2024") from error
        corpus = Path(entry["corpus_path"])
        validate_2024_path(corpus, year)
        destination = output / instrument
        run_corpus(corpus, destination, instrument, year=year)
        outputs.append((instrument, destination))
    summarize_cross_asset(outputs, output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--manifest", type=Path)
    route.add_argument("--corpus", type=Path)
    parser.add_argument("--instrument")
    parser.add_argument("--year", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.manifest:
        if args.instrument is not None or args.year is not None:
            parser.error("manifest route does not accept --instrument or --year")
        run_manifest(args.manifest, args.output)
    else:
        if args.instrument is None or args.year is None:
            parser.error("single route requires --instrument and --year")
        run_corpus(args.corpus, args.output, args.instrument, year=args.year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
