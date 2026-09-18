"""Zero-cost reporting over the frozen Stage 4B engine and bidirectional OU gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4b_runner import _load_stage4b_registry, run

MODE = "STAGE4C_ZERO_COST_DIAGNOSTIC_V1"
INSTRUMENTS = ("EURGBP", "NZDUSD", "USDCAD", "USDCHF")
OU_FILTER = "frozen-ou-bidirectional-v1"
VARIANTS = ("MR_BASELINE_GROSS", "MR_OU_BIDIRECTIONAL_V1_GROSS")
OUTPUTS = (
    "stage4c-zero-cost-cross-asset.csv",
    "stage4c-zero-cost-cross-asset.json",
    "stage4c-zero-cost-report.md",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _revision():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()


def _orientation(instrument):
    if instrument.endswith("USD"):
        return "XXXUSD"
    if instrument.startswith("USD"):
        return "USDXXX"
    return "non-USD cross"


def _usd_exposure(instrument, direction):
    orientation = _orientation(instrument)
    if orientation == "non-USD cross":
        return "NONE_DIRECT"
    long_usd = (orientation == "USDXXX") == (direction == "LONG")
    return "LONG_USD" if long_usd else "SHORT_USD"


def zero_cost_trade(row):
    """Copy gross Stage 4B returns exactly; deliberately bypass Stage 4C-A."""
    return row | {
        "zero_cost_return_pips_adverse_first": row["gross_return_pips_adverse_first"],
        "zero_cost_return_pips_favorable_first": row[
            "gross_return_pips_favorable_first"
        ],
    }


class Metrics:
    def __init__(self):
        self.rows = []
        self.incomplete = 0

    def add(self, row):
        if not row.get("complete"):
            self.incomplete += 1
            return
        copied = zero_cost_trade(row)
        fields = (
            "zero_cost_return_pips_adverse_first",
            "zero_cost_return_pips_favorable_first",
            "mfe_pips_certain",
            "mae_pips_certain",
        )
        if not all(math.isfinite(float(copied[field])) for field in fields):
            raise ValueError("complete zero-cost rows require finite metrics")
        self.rows.append(copied)

    def result(self):
        adverse = [float(r["zero_cost_return_pips_adverse_first"]) for r in self.rows]
        favorable = [
            float(r["zero_cost_return_pips_favorable_first"]) for r in self.rows
        ]
        gains = sum(max(value, 0.0) for value in adverse)
        losses = sum(max(-value, 0.0) for value in adverse)
        count = len(adverse)
        return {
            "complete_trade_count": count,
            "incomplete_trade_count": self.incomplete,
            "mean_adverse_first_pips": statistics.fmean(adverse) if count else None,
            "median_adverse_first_pips": statistics.median(adverse) if count else None,
            "mean_favorable_first_pips": statistics.fmean(favorable) if count else None,
            "median_favorable_first_pips": statistics.median(favorable)
            if count
            else None,
            "win_rate": sum(value > 0 for value in adverse) / count if count else 0.0,
            "profit_factor": gains / losses if losses else None,
            "mean_mfe_pips": statistics.fmean(
                float(r["mfe_pips_certain"]) for r in self.rows
            )
            if count
            else None,
            "mean_mae_pips": statistics.fmean(
                float(r["mae_pips_certain"]) for r in self.rows
            )
            if count
            else None,
        }


def _comparable(row, spec):
    return (
        row.get("signal_timeframe") == spec.timeframe
        and row.get("session") == spec.session
        and row.get("direction") in ("LONG", "SHORT")
        and row.get("benchmark_family") in spec.benchmark_families
        and row.get("lookback") in spec.lookbacks
        and row.get("signal_threshold") == 2.0
        and row.get("filter_family") == "none"
        and row.get("filter_spec_id") == "none-v1"
        and row.get("entry_mode") == "immediate"
        and row.get("tp_target_fraction") in (0.75, 1.0)
        and row.get("sl_extension_fraction") in (0.25, 0.5)
        and row.get("time_stop_minutes") in (60, 120)
    )


def _validate(registry_path, instruments):
    if not instruments:
        raise ValueError("--instruments is required and must be explicit")
    if tuple(instruments) != INSTRUMENTS:
        raise ValueError(f"instruments must be exactly {','.join(INSTRUMENTS)}")
    # Reject the sealed year before generic identity validation and, critically,
    # before a corpus path is constructed, inspected, or passed to Stage 4B.
    try:
        provenance = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid or missing corpus registry") from error
    if (
        provenance.get("registry_schema_version") != "fx-universe-2024-registry-v1"
        or provenance.get("requested_start_date") != "2024-01-01"
        or provenance.get("requested_end_date") != "2024-12-31"
    ):
        raise ValueError("only the explicit authenticated 2024 registry is accepted")
    for entry in provenance.get("instruments", {}).values():
        if (entry.get("requested_start_date"), entry.get("requested_end_date")) != (
            "2024-01-01",
            "2024-12-31",
        ):
            raise ValueError("non-2024 provenance rejected before corpus inspection")
    registry = _load_stage4b_registry(registry_path)
    if (
        registry.get("registry_schema_version") != "fx-universe-2024-registry-v1"
        or registry.get("requested_start_date") != "2024-01-01"
        or registry.get("requested_end_date") != "2024-12-31"
    ):
        raise ValueError("only the explicit authenticated 2024 registry is accepted")
    for instrument in instruments:
        entry = registry["instruments"].get(instrument)
        if entry is None or entry.get("instrument") != instrument:
            raise ValueError(f"missing exact authenticated instrument: {instrument}")
        if (entry.get("requested_start_date"), entry.get("requested_end_date")) != (
            "2024-01-01",
            "2024-12-31",
        ):
            raise ValueError("non-2024 provenance rejected before corpus inspection")
    return registry


def _flatten(instrument, variant, direction, metrics, baseline_count, ou_count):
    result = metrics.result()
    retention = ou_count / baseline_count if baseline_count else 0.0
    return {
        "instrument": instrument,
        "variant": variant,
        "direction": direction,
        "pair_orientation": _orientation(instrument),
        "usd_exposure": (
            "MIXED" if direction == "COMBINED" else _usd_exposure(instrument, direction)
        ),
        **result,
        "ou_retention_ratio": retention,
    }


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(registry_path: Path, instruments, output_dir: Path, *, runner=run):
    registry = _validate(registry_path, instruments)
    spec = frozen_ou_eligibility_spec(OU_FILTER)
    output_dir.mkdir(parents=True, exist_ok=True)
    gathered = {}
    counts = {}
    for instrument in instruments:
        entry = registry["instruments"][instrument]
        for variant in VARIANTS:
            by_direction = {name: Metrics() for name in ("LONG", "SHORT", "COMBINED")}

            def consume(
                row, *, selected=variant, expected=instrument, metrics=by_direction
            ):
                if row.get("instrument") != expected:
                    raise ValueError("mixed-instrument trade rows are forbidden")
                if selected == VARIANTS[0] and not _comparable(row, spec):
                    return
                direction = row.get("direction")
                if direction not in ("LONG", "SHORT"):
                    raise ValueError("unexpected trade direction")
                metrics[direction].add(row)
                metrics["COMBINED"].add(row)

            runner(
                Path(entry["corpus_path"]),
                output_dir / ".stage4b" / instrument / variant,
                instrument,
                registry_path,
                eligibility_filter=None if variant == VARIANTS[0] else OU_FILTER,
                trade_row_consumer=consume,
                persist_trade_rows=False,
            )
            gathered[instrument, variant] = by_direction
            counts[instrument, variant] = len(by_direction["COMBINED"].rows)

    rows = []
    for instrument in instruments:
        baseline = counts[instrument, VARIANTS[0]]
        ou = counts[instrument, VARIANTS[1]]
        for variant in VARIANTS:
            for direction in ("LONG", "SHORT", "COMBINED"):
                row = _flatten(
                    instrument,
                    variant,
                    direction,
                    gathered[instrument, variant][direction],
                    baseline,
                    ou,
                )
                base_result = gathered[instrument, VARIANTS[0]][direction].result()
                value = row["mean_adverse_first_pips"]
                base_value = base_result["mean_adverse_first_pips"]
                row["delta_mean_adverse_first_pips_baseline_to_ou"] = (
                    value - base_value
                    if variant == VARIANTS[1]
                    and value is not None
                    and base_value is not None
                    else None
                )
                rows.append(row)

    aggregate = []
    for variant in VARIANTS:
        for direction in ("LONG", "SHORT", "COMBINED"):
            metric = Metrics()
            for instrument in instruments:
                source = gathered[instrument, variant][direction]
                metric.rows.extend(source.rows)
                metric.incomplete += source.incomplete
            aggregate.append(
                {"variant": variant, "direction": direction, **metric.result()}
            )
    payload = {
        "mode": MODE,
        "instruments": list(instruments),
        "variants": list(VARIANTS),
        "rows": rows,
        "aggregate": aggregate,
    }
    _write_csv(output_dir / OUTPUTS[0], rows)
    (output_dir / OUTPUTS[1]).write_text(_json(payload) + "\n")
    lines = [
        "# Stage 4C zero-cost diagnostic",
        "",
        "Discovery-only descriptive report; no ranking or production validation.",
        "",
        "| Instrument | Variant | Direction | Complete | "
        "Mean adverse pips | OU retention |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instrument']} | {row['variant']} | {row['direction']} | "
            f"{row['complete_trade_count']} | {row['mean_adverse_first_pips']} | "
            f"{row['ou_retention_ratio']} |"
        )
    (output_dir / OUTPUTS[2]).write_text("\n".join(lines) + "\n")
    audit = {
        "mode": MODE,
        "instruments": list(instruments),
        "spread_pips": 0,
        "commission_pips": 0,
        "slippage_pips": 0,
        "currency_conversion_adjustment": False,
        "registry_id": registry["registry_id"],
        "registry_sha256": "sha256:" + _sha(registry_path),
        "requested_start_date": "2024-01-01",
        "requested_end_date": "2024-12-31",
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "baseline_eligibility": "none",
        "ou_filter_name": OU_FILTER,
        "ou_filter_id": spec.filter_spec_id,
        "ou_process_id": spec.process_spec.process_spec_id,
        "source_git_commit": _revision(),
        "corpora": {
            name: {
                key: registry["instruments"][name][key]
                for key in ("corpus_id", "assembled_dataset_id", "manifest_sha256")
            }
            for name in instruments
        },
        "row_counts": {
            f"{name}/{variant}": gathered[name, variant]["COMBINED"].result()
            for name in instruments
            for variant in VARIANTS
        },
        "output_sha256": {name: _sha(output_dir / name) for name in OUTPUTS},
    }
    (output_dir / "execution-audit.json").write_text(_json(audit) + "\n")
    return tuple(output_dir / name for name in (*OUTPUTS, "execution-audit.json"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--instruments", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    instruments = tuple(
        part.strip() for part in args.instruments.split(",") if part.strip()
    )
    report(args.registry, instruments, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
