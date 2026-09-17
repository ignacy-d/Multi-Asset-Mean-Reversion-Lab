"""Deterministic nine-pair replay of frozen Stage 4B MR and causal OU.

This is orchestration and reporting only.  Signal, path, OU, and cost semantics
remain owned by their frozen implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4b_runner import _load_stage4b_registry, run
from mr_lab.stage4c import CostProfile, Stage4CError, transform_trade

SCHEMA = "frozen-mr-ou-nine-pair-replay-v1"
OU_FILTER = "frozen-ou-crossasset-v1"
HISTORICAL_FIVE = frozenset(("EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "AUDJPY"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _revision():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()


def _profit_factor(values):
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else (math.inf if gains else 0.0)


def _metrics(rows, value_field="gross_return_pips_adverse_first"):
    complete = [row for row in rows if row.get("complete")]
    values = [float(row[value_field]) for row in complete]
    return {
        "n_trades": len(values),
        "mean_pips": statistics.fmean(values) if values else None,
        "median_pips": statistics.median(values) if values else None,
        "win_rate": sum(value > 0 for value in values) / len(values) if values else 0.0,
        "profit_factor": _profit_factor(values),
        "mean_mfe_pips": statistics.fmean(
            float(row["mfe_pips_certain"]) for row in complete
        )
        if complete
        else None,
        "mean_mae_pips": statistics.fmean(
            float(row["mae_pips_certain"]) for row in complete
        )
        if complete
        else None,
    }


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _cost_status(profile, instrument, rows):
    try:
        for session in {row.get("session") for row in rows}:
            profile.costs(instrument, session, "mean")
    except Stage4CError:
        return "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
    return "AUTHENTICATED"


def _net_metrics(profile, rows):
    transformed = [
        transform_trade(row, profile, "mean", 0.0)
        for row in rows
        if row.get("complete")
    ]
    return _metrics(transformed, "net_pips_adverse_first")


def _finished(directory, identity):
    marker = directory / "replay-identity.json"
    return marker.is_file() and json.loads(marker.read_text()) == identity


def _run_variant(corpus, directory, instrument, registry_path, identity, filter_name):
    if directory.exists():
        if _finished(directory, identity):
            return
        # A marker is written only after Stage 4B has atomically completed its
        # output contract. A marker-free directory is an interrupted attempt
        # under the already-validated parent run identity and is safe to retry.
        if (directory / "replay-identity.json").exists():
            raise RuntimeError(
                f"refusing to overwrite non-matching artifact: {directory}"
            )
        shutil.rmtree(directory)
    run(corpus, directory, instrument, registry_path, eligibility_filter=filter_name)
    (directory / "replay-identity.json").write_text(_canonical(identity) + "\n")


def replay(registry_path: Path, cost_path: Path, output_dir: Path):
    registry = _load_stage4b_registry(registry_path)
    if registry.get("registry_schema_version") != "fx-universe-2024-registry-v1":
        raise ValueError("the nine-pair replay requires the authenticated FX registry")
    profile = CostProfile.load(cost_path)
    revision = _revision()
    ou = frozen_ou_eligibility_spec(OU_FILTER)
    identity = {
        "schema_version": SCHEMA,
        "registry_id": registry["registry_id"],
        "registry_sha256": _sha(registry_path),
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "ou_filter_spec_id": ou.filter_spec_id,
        "cost_profile_sha256": profile.sha256,
        "source_commit_sha": revision,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        marker = output_dir / "run-identity.json"
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise RuntimeError("refusing to mix a different replay identity")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run-identity.json").write_text(_canonical(identity) + "\n")

    comparisons = []
    all_rows: dict[str, list[dict]] = {"mr": [], "mr_ou": []}
    all_net_rows: dict[str, list[dict]] = {"mr": [], "mr_ou": []}
    for instrument, entry in registry["instruments"].items():
        corpus = Path(entry["corpus_path"])
        variants = {}
        for name, filter_name in (("mr", None), ("mr_ou", OU_FILTER)):
            destination = output_dir / instrument / name
            variant_identity = identity | {"instrument": instrument, "variant": name}
            _run_variant(
                corpus,
                destination,
                instrument,
                registry_path,
                variant_identity,
                filter_name,
            )
            rows = _read_jsonl(destination / "trades.jsonl")
            summary = json.loads((destination / "summary.json").read_text())
            variants[name] = (rows, summary)
            all_rows[name].extend(rows)
        mr_rows, mr_summary = variants["mr"]
        ou_rows, ou_summary = variants["mr_ou"]
        status = _cost_status(profile, instrument, mr_rows + ou_rows)
        mr_gross, ou_gross = _metrics(mr_rows), _metrics(ou_rows)
        if status == "AUTHENTICATED":
            for name, rows in (("mr", mr_rows), ("mr_ou", ou_rows)):
                all_net_rows[name].extend(
                    transform_trade(row, profile, "mean", 0.0)
                    for row in rows
                    if row.get("complete")
                )
        comparisons.append(
            {
                "instrument": instrument,
                "mr_gross": mr_gross,
                "mr_net": _net_metrics(profile, mr_rows)
                if status == "AUTHENTICATED"
                else None,
                "mr_ou_gross": ou_gross,
                "mr_ou_net": _net_metrics(profile, ou_rows)
                if status == "AUTHENTICATED"
                else None,
                "mr_trade_count": mr_gross["n_trades"],
                "mr_ou_trade_count": ou_gross["n_trades"],
                "ou_retention_ratio": ou_gross["n_trades"] / mr_gross["n_trades"]
                if mr_gross["n_trades"]
                else 0.0,
                "mr_incomplete_count": mr_summary["incomplete_count"],
                "mr_ou_incomplete_count": ou_summary["incomplete_count"],
                "cost_profile_status": status,
                "corpus_id": entry["corpus_id"],
                "dataset_id": entry["assembled_dataset_id"],
                "historical_regression_reference": instrument in HISTORICAL_FIVE,
                "historical_numeric_regression_status": (
                    "BLOCKED_MISSING_FROZEN_NUMERIC_REFERENCE"
                    if instrument in HISTORICAL_FIVE
                    else "NOT_APPLICABLE_EXTENSION_INSTRUMENT"
                ),
                "config_run_identity": identity,
            }
        )
    aggregate = {name: _metrics(rows) for name, rows in all_rows.items()}
    aggregate_net = {
        name: _metrics(rows, "net_pips_adverse_first")
        for name, rows in all_net_rows.items()
    }
    result = {
        "identity": identity,
        "instruments": comparisons,
        "aggregate_gross": aggregate,
        "aggregate_net_authenticated_profiles_only": aggregate_net,
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
