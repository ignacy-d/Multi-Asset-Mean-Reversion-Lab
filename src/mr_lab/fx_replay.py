"""Deterministic, streaming nine-pair replay of frozen Stage 4B MR and causal OU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from array import array
from pathlib import Path

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4b_runner import OUTPUTS, _load_stage4b_registry, run
from mr_lab.stage4c import CostProfile, Stage4CError, transform_trade

SCHEMA = "frozen-mr-ou-nine-pair-replay-v3"
AGGREGATE_SCHEMA = "frozen-mr-ou-streaming-aggregate-v1"
OU_FILTER = "frozen-ou-crossasset-v1"
HISTORICAL_FIVE = frozenset(("EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "AUDJPY"))
PROGRESS_ROWS = 250_000


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _revision():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()


class StreamingMetrics:
    """Exact metrics retaining only compact return scalars needed for the median."""

    def __init__(self):
        self.values = array("d")
        self.total = 0.0
        self.wins = 0
        self.gains = 0.0
        self.losses = 0.0
        self.mfe_total = 0.0
        self.mae_total = 0.0
        self.incomplete = 0

    def add(self, row, value_field="gross_return_pips_adverse_first"):
        if not row.get("complete"):
            self.incomplete += 1
            return
        value = float(row[value_field])
        mfe = float(row["mfe_pips_certain"])
        mae = float(row["mae_pips_certain"])
        if not all(math.isfinite(item) for item in (value, mfe, mae)):
            raise ValueError("replay metrics require finite values")
        self.values.append(value)
        self.total += value
        self.wins += value > 0
        self.gains += max(value, 0.0)
        self.losses += max(-value, 0.0)
        self.mfe_total += mfe
        self.mae_total += mae

    def merge(self, other):
        self.values.extend(other.values)
        self.total += other.total
        self.wins += other.wins
        self.gains += other.gains
        self.losses += other.losses
        self.mfe_total += other.mfe_total
        self.mae_total += other.mae_total
        self.incomplete += other.incomplete

    def result(self):
        count = len(self.values)
        return {
            "n_trades": count,
            # Preserve the prior list implementation's fmean semantics while
            # operating on the compact float64 buffer retained for the median.
            "mean_pips": statistics.fmean(self.values) if count else None,
            "median_pips": statistics.median(self.values) if count else None,
            "win_rate": self.wins / count if count else 0.0,
            "profit_factor": (
                self.gains / self.losses
                if self.losses
                else (math.inf if self.gains else 0.0)
            ),
            "mean_mfe_pips": self.mfe_total / count if count else None,
            "mean_mae_pips": self.mae_total / count if count else None,
        }

    def write(self, directory, name):
        values_path = directory / f"{name}-values.f64"
        values = array("d", self.values)
        if sys.byteorder != "little":
            values.byteswap()
        values_path.write_bytes(values.tobytes())
        metadata = {
            "schema_version": AGGREGATE_SCHEMA,
            "value_encoding": "little-endian-float64",
            "value_count": len(self.values),
            "values_sha256": _sha(values_path),
            "total": self.total,
            "wins": self.wins,
            "gains": self.gains,
            "losses": self.losses,
            "mfe_total": self.mfe_total,
            "mae_total": self.mae_total,
            "incomplete": self.incomplete,
        }
        (directory / f"{name}.json").write_text(_canonical(metadata) + "\n")

    @classmethod
    def load(cls, directory, name):
        metadata = json.loads((directory / f"{name}.json").read_text())
        if metadata.get("schema_version") != AGGREGATE_SCHEMA:
            raise RuntimeError("incompatible replay aggregate schema")
        values_path = directory / f"{name}-values.f64"
        if _sha(values_path) != metadata.get("values_sha256"):
            raise RuntimeError("replay aggregate values hash mismatch")
        values = array("d")
        values.frombytes(values_path.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) != metadata.get("value_count"):
            raise RuntimeError("replay aggregate value count mismatch")
        result = cls()
        result.values = values
        for field in (
            "total",
            "wins",
            "gains",
            "losses",
            "mfe_total",
            "mae_total",
            "incomplete",
        ):
            setattr(result, field, metadata[field])
        return result


def _cost_status(profile, instrument):
    try:
        for session in (None, "asia", "london", "new_york"):
            profile.costs(instrument, session, "mean")
    except Stage4CError:
        return "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
    return "AUTHENTICATED"


def _is_mr_comparable_row(row, ou_spec):
    """Match the frozen OU signal scope and downstream grid, without its gate."""
    return (
        row.get("signal_timeframe") == ou_spec.timeframe
        and row.get("session") == ou_spec.session
        and row.get("direction") == ou_spec.direction
        and row.get("benchmark_family") in ou_spec.benchmark_families
        and row.get("lookback") in ou_spec.lookbacks
        and row.get("signal_threshold") == 2.0
        and row.get("filter_family") == "none"
        and row.get("filter_spec_id") == "none-v1"
        and row.get("entry_mode") == "immediate"
        and row.get("tp_target_fraction") in (0.75, 1.0)
        and row.get("sl_extension_fraction") in (0.25, 0.5)
        and row.get("time_stop_minutes") in (60, 120)
    )


def _new_bundle(variant, cost_status):
    names = (
        ("mr_full_grid_gross", "mr_comparable_baseline_gross")
        if variant == "mr"
        else ("mr_ou_gross",)
    )
    bundle = {name: StreamingMetrics() for name in names}
    if cost_status == "AUTHENTICATED":
        net_name = "mr_comparable_baseline_net" if variant == "mr" else "mr_ou_net"
        bundle[net_name] = StreamingMetrics()
    return bundle


def _consume(bundle, variant, row, ou_spec, profile, cost_status):
    if variant == "mr":
        bundle["mr_full_grid_gross"].add(row)
        if not _is_mr_comparable_row(row, ou_spec):
            return
        bundle["mr_comparable_baseline_gross"].add(row)
        net_name = "mr_comparable_baseline_net"
    else:
        bundle["mr_ou_gross"].add(row)
        net_name = "mr_ou_net"
    if cost_status == "AUTHENTICATED" and row.get("complete"):
        net = transform_trade(row, profile, "mean", 0.0)
        bundle[net_name].add(net, "net_pips_adverse_first")


def _write_bundle(directory, bundle):
    for name, metrics in bundle.items():
        metrics.write(directory, name)
    manifest = {
        "schema_version": AGGREGATE_SCHEMA,
        "metrics": sorted(bundle),
    }
    (directory / "replay-aggregate.json").write_text(_canonical(manifest) + "\n")


def _load_bundle(directory):
    manifest = json.loads((directory / "replay-aggregate.json").read_text())
    if manifest.get("schema_version") != AGGREGATE_SCHEMA:
        raise RuntimeError("incompatible replay aggregate manifest")
    return {
        name: StreamingMetrics.load(directory, name) for name in manifest["metrics"]
    }


def _compatible_completed_variant(directory, expected, entry, ou_spec):
    marker = json.loads((directory / "replay-identity.json").read_text())
    for field in (
        "registry_id",
        "registry_sha256",
        "stage4b_methodology_id",
        "ou_filter_spec_id",
        "cost_profile_sha256",
        "instrument",
        "variant",
    ):
        if marker.get(field) != expected.get(field):
            raise RuntimeError(f"completed replay identity mismatch: {field}")
    audit = json.loads((directory / "execution-audit.json").read_text())
    required = {
        "instrument": expected["instrument"],
        "corpus_id": entry["corpus_id"],
        "assembled_dataset_id": entry["assembled_dataset_id"],
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
    }
    for field, value in required.items():
        if audit.get(field) != value:
            raise RuntimeError(f"completed Stage4B audit mismatch: {field}")
    expected_filter = (
        ("none", "none-v1")
        if expected["variant"] == "mr"
        else ("ornstein-uhlenbeck", ou_spec.filter_spec_id)
    )
    if (audit.get("filter_family"), audit.get("filter_spec_id")) != expected_filter:
        raise RuntimeError("completed Stage4B filter identity mismatch")
    hashes = audit.get("output_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(OUTPUTS):
        raise RuntimeError("incomplete Stage4B output audit contract")
    for name, digest in hashes.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"completed Stage4B output missing: {name}")
        if name != "trades.jsonl" and _sha(path).removeprefix("sha256:") != digest:
            raise RuntimeError(f"completed Stage4B output hash mismatch: {name}")
    return audit


def _adopt_legacy_rows(
    directory, bundle, variant, ou_spec, profile, cost_status, audit
):
    path = directory / "trades.jsonl"
    digest = hashlib.sha256()
    rows = 0
    print(f"REPLAY_PROGRESS variant={variant} adoption_start path={path}", flush=True)
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            _consume(bundle, variant, json.loads(line), ou_spec, profile, cost_status)
            rows += 1
            if rows % PROGRESS_ROWS == 0:
                print(
                    f"REPLAY_PROGRESS variant={variant} adoption_rows={rows}",
                    flush=True,
                )
    if digest.hexdigest() != audit["output_sha256"]["trades.jsonl"]:
        raise RuntimeError("completed Stage4B trades hash mismatch")
    _write_bundle(directory, bundle)
    print(
        f"REPLAY_PROGRESS variant={variant} adoption_complete rows={rows}", flush=True
    )


def _run_or_reuse_variant(
    corpus,
    directory,
    instrument,
    registry_path,
    identity,
    variant,
    ou_spec,
    profile,
    cost_status,
    entry,
):
    bundle = _new_bundle(variant, cost_status)
    if directory.exists() and (directory / "replay-identity.json").is_file():
        audit = _compatible_completed_variant(directory, identity, entry, ou_spec)
        if (directory / "replay-aggregate.json").is_file():
            print(
                f"REPLAY_PROGRESS instrument={instrument} variant={variant} reuse",
                flush=True,
            )
            return _load_bundle(directory)
        _adopt_legacy_rows(
            directory, bundle, variant, ou_spec, profile, cost_status, audit
        )
        return bundle
    if directory.exists():
        raise RuntimeError(
            f"refusing to modify incomplete replay artifact: {directory}"
        )
    print(
        f"REPLAY_PROGRESS instrument={instrument} variant={variant} start", flush=True
    )
    filter_name = None if variant == "mr" else OU_FILTER
    run(
        corpus,
        directory,
        instrument,
        registry_path,
        eligibility_filter=filter_name,
        trade_row_consumer=lambda row: _consume(
            bundle, variant, row, ou_spec, profile, cost_status
        ),
        persist_trade_rows=False,
    )
    _write_bundle(directory, bundle)
    (directory / "replay-identity.json").write_text(_canonical(identity) + "\n")
    print(
        f"REPLAY_PROGRESS instrument={instrument} variant={variant} complete",
        flush=True,
    )
    return bundle


def _retention(ou, comparable):
    denominator = len(comparable.values)
    return len(ou.values) / denominator if denominator else 0.0


def replay(registry_path: Path, cost_path: Path, output_dir: Path):
    registry = _load_stage4b_registry(registry_path)
    if registry.get("registry_schema_version") != "fx-universe-2024-registry-v1":
        raise ValueError("the nine-pair replay requires the authenticated FX registry")
    profile = CostProfile.load(cost_path)
    revision = _revision()
    ou_spec = frozen_ou_eligibility_spec(OU_FILTER)
    scientific_identity = {
        "registry_id": registry["registry_id"],
        "registry_sha256": _sha(registry_path),
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "ou_filter_spec_id": ou_spec.filter_spec_id,
        "cost_profile_sha256": profile.sha256,
    }
    run_identity = scientific_identity | {
        "schema_version": SCHEMA,
        "orchestrator_source_commit_sha": revision,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_identity_path = output_dir / "run-identity.json"
    identity_path = output_dir / "run-identity-v3.json"
    previous_path = identity_path if identity_path.exists() else legacy_identity_path
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for field, value in scientific_identity.items():
            if previous.get(field) != value:
                raise RuntimeError(f"replay scientific identity mismatch: {field}")
    if not identity_path.exists():
        identity_path.write_text(_canonical(run_identity) + "\n")

    comparisons = []
    aggregate: dict[str, StreamingMetrics] = {}
    for instrument, entry in registry["instruments"].items():
        print(f"REPLAY_PROGRESS instrument={instrument} start", flush=True)
        status = _cost_status(profile, instrument)
        bundles = {}
        for variant in ("mr", "mr_ou"):
            identity = run_identity | {
                "instrument": instrument,
                "variant": variant,
            }
            bundles[variant] = _run_or_reuse_variant(
                Path(entry["corpus_path"]),
                output_dir / instrument / variant,
                instrument,
                registry_path,
                identity,
                variant,
                ou_spec,
                profile,
                status,
                entry,
            )
            if variant == "mr":
                print(
                    f"REPLAY_PROGRESS instrument={instrument} "
                    "mr_comparable_aggregation_complete",
                    flush=True,
                )
        metrics = bundles["mr"] | bundles["mr_ou"]
        for name, value in metrics.items():
            aggregate.setdefault(name, StreamingMetrics()).merge(value)
        full = metrics["mr_full_grid_gross"]
        comparable = metrics["mr_comparable_baseline_gross"]
        ou = metrics["mr_ou_gross"]
        comparisons.append(
            {
                "instrument": instrument,
                **{name: value.result() for name, value in metrics.items()},
                "mr_full_grid_trade_count": len(full.values),
                "mr_comparable_baseline_trade_count": len(comparable.values),
                "mr_ou_trade_count": len(ou.values),
                "ou_retention_ratio": _retention(ou, comparable),
                "mr_incomplete_count": full.incomplete,
                "mr_comparable_baseline_incomplete_count": comparable.incomplete,
                "mr_ou_incomplete_count": ou.incomplete,
                "cost_profile_status": status,
                "corpus_id": entry["corpus_id"],
                "dataset_id": entry["assembled_dataset_id"],
                "historical_regression_reference": instrument in HISTORICAL_FIVE,
                "historical_numeric_regression_status": (
                    "BLOCKED_MISSING_FROZEN_NUMERIC_REFERENCE"
                    if instrument in HISTORICAL_FIVE
                    else "NOT_APPLICABLE_EXTENSION_INSTRUMENT"
                ),
                "config_run_identity": run_identity,
            }
        )
        print(
            f"REPLAY_PROGRESS instrument={instrument} cost_overlay_complete", flush=True
        )
        print(f"REPLAY_PROGRESS instrument={instrument} complete", flush=True)
    result = {
        "identity": run_identity,
        "instruments": comparisons,
        "aggregate": {name: value.result() for name, value in aggregate.items()},
        "aggregate_ou_retention_ratio": _retention(
            aggregate["mr_ou_gross"], aggregate["mr_comparable_baseline_gross"]
        ),
        "aggregate_net_excludes_blocked_instruments": True,
    }
    target = output_dir / "comparison.json"
    if target.exists() and json.loads(target.read_text()) != result:
        raise RuntimeError("refusing to overwrite a different comparison artifact")
    target.write_text(_canonical(result) + "\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/fx-universe-2024-registry-v1.json"),
    )
    parser.add_argument(
        "--cost-profile",
        type=Path,
        default=Path("configs/stage4c-ftmo-cost-profile-v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    replay(args.registry, args.cost_profile, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
