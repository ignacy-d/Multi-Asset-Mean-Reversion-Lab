"""Narrow, streaming runner for OU-BIDIRECTIONAL-2024-v1.

This is a follow-up discovery runner.  It deliberately consumes only the explicit
authenticated registry supplied by the caller and has no discovery fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from array import array
from functools import partial
from pathlib import Path

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4b_runner import _load_stage4b_registry, run
from mr_lab.stage4c import CostProfile, Stage4CError, transform_trade

STUDY = "OU-BIDIRECTIONAL-2024-v1"
STATUS = "2024_FOLLOW_UP_DISCOVERY_NOT_CONFIRMATION"
OLD_FILTER = "frozen-ou-crossasset-v1"
NEW_FILTER = "frozen-ou-bidirectional-v1"
BLOCKED = "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
GRID = {
    "signal_timeframes": ("15m",),
    "sessions": ("london",),
    "benchmark_families": ("vwap", "vwap-canonical-m1"),
    "lookbacks": (20, 40),
    "signal_threshold": 2.0,
    "directions": ("LONG", "SHORT"),
    "entry_modes": ("immediate",),
    "tp_fractions": (0.75, 1.0),
    "sl_fractions": (0.25, 0.5),
    "time_stops_minutes": (60, 120),
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


class StreamingMetrics:
    """Exact median plus constant-size accumulators for all other metrics."""

    def __init__(self):
        self.values = array("d")
        self.wins = 0
        self.gains = self.losses = self.mfe = self.mae = 0.0
        self.incomplete = 0

    def add(self, row, field="gross_return_pips_adverse_first"):
        if not row.get("complete"):
            self.incomplete += 1
            return
        value = float(row[field])
        mfe = float(row["mfe_pips_certain"])
        mae = float(row["mae_pips_certain"])
        if not all(math.isfinite(x) for x in (value, mfe, mae)):
            raise ValueError("metrics require finite trade values")
        self.values.append(value)
        self.wins += value > 0
        self.gains += max(value, 0.0)
        self.losses += max(-value, 0.0)
        self.mfe += mfe
        self.mae += mae

    def merge(self, other):
        self.values.extend(other.values)
        self.wins += other.wins
        self.gains += other.gains
        self.losses += other.losses
        self.mfe += other.mfe
        self.mae += other.mae
        self.incomplete += other.incomplete

    def result(self):
        count = len(self.values)
        return {
            "complete_trade_row_count": count,
            "incomplete_trade_row_count": self.incomplete,
            "mean_pips": statistics.fmean(self.values) if count else None,
            "median_pips": statistics.median(self.values) if count else None,
            "win_rate": self.wins / count if count else 0.0,
            "profit_factor": self.gains / self.losses
            if self.losses
            else (math.inf if self.gains else 0.0),
            "mean_mfe": self.mfe / count if count else None,
            "mean_mae": self.mae / count if count else None,
        }


def pair_orientation(instrument):
    base, quote = instrument[:3], instrument[3:]
    position = "BASE" if base == "USD" else "QUOTE" if quote == "USD" else "NONE"
    family = (
        "USDXXX"
        if position == "BASE"
        else "XXXUSD"
        if position == "QUOTE"
        else "non-USD cross"
    )
    return base, quote, position, family


def usd_exposure(instrument, direction):
    _, _, position, _ = pair_orientation(instrument)
    if position == "NONE":
        return "NO_USD"
    long_usd = (position == "BASE") == (direction == "LONG")
    return "LONG_USD" if long_usd else "SHORT_USD"


def _cost_status(profile, instrument):
    try:
        profile.costs(instrument, "london", "mean")
    except Stage4CError:
        return BLOCKED
    return "AUTHENTICATED"


def _empty_bundle(authenticated):
    result = {
        f"mr_{direction}_{variant}_gross": StreamingMetrics()
        for direction in ("long", "short")
        for variant in ("baseline", "ou")
    }
    if authenticated:
        result |= {
            name.replace("_gross", "_net"): StreamingMetrics() for name in tuple(result)
        }
    return result


def _consume(bundle, variant, row, profile, authenticated):
    direction = row["direction"].lower()
    gross = bundle[f"mr_{direction}_{variant}_gross"]
    gross.add(row)
    if authenticated and row.get("complete"):
        net = transform_trade(row, profile, "mean", 0.0)
        bundle[f"mr_{direction}_{variant}_net"].add(net, "net_pips_adverse_first")


def _with_combined(bundle):
    result = dict(bundle)
    for suffix in ("baseline_gross", "ou_gross", "baseline_net", "ou_net"):
        long = bundle.get(f"mr_long_{suffix}")
        short = bundle.get(f"mr_short_{suffix}")
        if long is not None and short is not None:
            combined = StreamingMetrics()
            combined.merge(long)
            combined.merge(short)
            result[f"mr_combined_{suffix}"] = combined
    return result


def _deltas(metrics, scope, net="gross"):
    baseline = metrics[f"mr_{scope}_baseline_{net}"].result()
    ou = metrics[f"mr_{scope}_ou_{net}"].result()
    return {
        "delta_mean_pips": _difference(ou["mean_pips"], baseline["mean_pips"]),
        "delta_median_pips": _difference(ou["median_pips"], baseline["median_pips"]),
        "delta_win_rate": ou["win_rate"] - baseline["win_rate"],
        "delta_profit_factor": _difference(
            ou["profit_factor"], baseline["profit_factor"]
        ),
    }


def _difference(left, right):
    return left - right if left is not None and right is not None else None


def replay(registry_path: Path, cost_path: Path, output_dir: Path):
    registry = _load_stage4b_registry(registry_path)
    if registry.get("registry_schema_version") != "fx-universe-2024-registry-v1":
        raise ValueError("bidirectional replay requires the explicit 2024 FX registry")
    profile = CostProfile.load(cost_path)
    old = frozen_ou_eligibility_spec(OLD_FILTER)
    new = frozen_ou_eligibility_spec(NEW_FILTER)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    identity = {
        "study": STUDY,
        "status": STATUS,
        "source_commit_sha": revision,
        "registry_id": registry["registry_id"],
        "registry_sha256": _sha(registry_path),
        "cost_profile_identity": "sha256:" + profile.sha256,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "old_ou_process_spec_id": old.process_spec.process_spec_id,
        "old_ou_filter_spec_id": old.filter_spec_id,
        "new_ou_process_spec_id": new.process_spec.process_spec_id,
        "new_ou_filter_spec_id": new.filter_spec_id,
        "grid": GRID,
    }
    if (output_dir / "work").exists() or (output_dir / "output").exists():
        raise RuntimeError("refusing to overwrite or delete prior empirical artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    instruments = []
    aggregate: dict[str, StreamingMetrics] = {}
    diagnostic: dict[tuple[str, str], StreamingMetrics] = {}
    for instrument, entry in registry["instruments"].items():
        print(f"BIDIR_PROGRESS instrument={instrument} start", flush=True)
        status = _cost_status(profile, instrument)
        bundle = _empty_bundle(status == "AUTHENTICATED")
        for variant, filter_name in (("baseline", None), ("ou", NEW_FILTER)):
            for direction in ("LONG", "SHORT"):
                print(
                    "BIDIR_PROGRESS "
                    f"instrument={instrument} direction={direction} {variant}_start",
                    flush=True,
                )
            run(
                Path(entry["corpus_path"]),
                output_dir / "work" / instrument / variant,
                instrument,
                registry_path,
                eligibility_filter=filter_name,
                trade_row_consumer=partial(
                    _consume,
                    bundle,
                    variant,
                    profile=profile,
                    authenticated=status == "AUTHENTICATED",
                ),
                persist_trade_rows=False,
                grid_restriction=GRID,
            )
            for direction in ("LONG", "SHORT"):
                print(
                    "BIDIR_PROGRESS "
                    f"instrument={instrument} direction={direction} {variant}_complete",
                    flush=True,
                )
        metrics = _with_combined(bundle)
        base, quote, position, family = pair_orientation(instrument)
        rendered = {name: value.result() for name, value in metrics.items()}
        for scope in ("long", "short", "combined"):
            rendered[f"mr_{scope}_ou_gross"]["ou_retention_ratio"] = (
                len(metrics[f"mr_{scope}_ou_gross"].values)
                / len(metrics[f"mr_{scope}_baseline_gross"].values)
                if metrics[f"mr_{scope}_baseline_gross"].values
                else 0.0
            )
            rendered[f"{scope}_gross_deltas"] = _deltas(metrics, scope)
            if status == "AUTHENTICATED":
                rendered[f"{scope}_net_deltas"] = _deltas(metrics, scope, "net")
            else:
                for variant in ("baseline", "ou"):
                    rendered[f"mr_{scope}_{variant}_net"] = {"status": BLOCKED}
                rendered[f"{scope}_net_deltas"] = {"status": BLOCKED}
        instruments.append(
            {
                "instrument": instrument,
                "base_currency": base,
                "quote_currency": quote,
                "usd_position": position,
                "pair_orientation": family,
                "cost_profile_status": status,
                "metrics": rendered,
            }
        )
        for name, value in metrics.items():
            aggregate.setdefault(name, StreamingMetrics()).merge(value)
        for direction in ("LONG", "SHORT"):
            for group in (usd_exposure(instrument, direction), family):
                for variant in ("baseline_gross", "ou_gross"):
                    source = metrics[f"mr_{direction.lower()}_{variant}"]
                    diagnostic.setdefault((group, variant), StreamingMetrics()).merge(
                        source
                    )
        print(f"BIDIR_PROGRESS instrument={instrument} complete", flush=True)
    aggregate = _with_combined(aggregate)
    result = {
        "study_identity": identity,
        "follow_up_discovery_status": STATUS,
        "instruments": instruments,
        "aggregates": {name: value.result() for name, value in aggregate.items()},
        "usd_orientation_diagnostics": {
            group: {
                variant: diagnostic[(group, variant)].result()
                for variant in ("baseline_gross", "ou_gross")
                if (group, variant) in diagnostic
            }
            for group in (
                "LONG_USD",
                "SHORT_USD",
                "NO_USD",
                "XXXUSD",
                "USDXXX",
                "non-USD cross",
            )
        },
    }
    final_dir = output_dir / "output"
    final_dir.mkdir()
    target = final_dir / "comparison.json"
    target.write_text(_canonical(result) + "\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cost-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    replay(args.registry, args.cost_profile, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
