"""Fail-closed reducer for independently produced compact Stage 4C shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import (
    CONFIG_FIELDS,
    INSTRUMENTS,
    SLIPPAGES,
    SPREAD_STATISTICS,
    STAGE4B_SOURCE_COMMIT,
    CostProfile,
    Stage4CError,
    sha256_file,
)
from mr_lab.stage4c_runner import (
    SHARD_MANIFEST,
    SHARD_STATE,
    _group_owner,
    _source_input_identity,
    iter_jsonl,
    run_rows,
)

IDENTICAL_FIELDS = (
    "shard_count",
    "stage4b_methodology_id",
    "stage4b_source_commit",
    "cost_profile_sha256",
    "scenario_matrix",
    "instrument",
    "corpus_id",
    "assembled_dataset_id",
    "source_mode",
    "source_input_components",
    "source_input_commitment",
    "source_rows_read",
    "source_complete_rows",
)


class Stage4CReductionError(Stage4CError):
    pass


def reduce_shards(shard_dirs, output_dir, profile_path, expected_shard_count):
    records = []
    for directory in map(Path, shard_dirs):
        manifest_path = directory / SHARD_MANIFEST
        state_path = directory / SHARD_STATE
        if not manifest_path.is_file() or not state_path.is_file():
            raise Stage4CReductionError("missing expected shard")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("state_sha256") != sha256_file(state_path):
            raise Stage4CReductionError("malformed shard state hash")
        if manifest.get("state_row_count") != sum(1 for _ in state_path.open()):
            raise Stage4CReductionError("malformed shard state count")
        records.append((directory, manifest))
    indexes = [manifest.get("shard_index") for _, manifest in records]
    if len(indexes) != len(set(indexes)):
        raise Stage4CReductionError("duplicate shard identity")
    if set(indexes) != set(range(expected_shard_count)):
        raise Stage4CReductionError("missing expected shard")
    records.sort(key=lambda item: item[1]["shard_index"])
    reference = records[0][1]
    for _directory, manifest in records:
        if manifest.get("shard_count") != expected_shard_count:
            raise Stage4CReductionError("manifest shard_count mismatch")
        for field in IDENTICAL_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise Stage4CReductionError(f"inconsistent shard field: {field}")
        components = manifest.get("source_input_components")
        if not isinstance(components, dict):
            raise Stage4CReductionError("malformed source input components")
        for component, manifest_field in (
            ("source_mode", "source_mode"),
            ("instrument", "instrument"),
            ("corpus_id", "corpus_id"),
            ("assembled_dataset_id", "assembled_dataset_id"),
        ):
            if components.get(component) != manifest.get(manifest_field):
                raise Stage4CReductionError(f"source component mismatch: {component}")
        try:
            canonical_components, commitment = _source_input_identity(
                manifest["source_mode"], components
            )
        except (KeyError, Stage4CError) as error:
            raise Stage4CReductionError("malformed source input components") from error
        if canonical_components != components:
            raise Stage4CReductionError("non-canonical source input components")
        if commitment != manifest.get("source_input_commitment"):
            raise Stage4CReductionError("source input commitment mismatch")
    if reference["stage4b_methodology_id"] != STAGE4B_METHODOLOGY_ID:
        raise Stage4CReductionError("inconsistent Stage 4B methodology")
    if reference["stage4b_source_commit"] != STAGE4B_SOURCE_COMMIT:
        raise Stage4CReductionError("inconsistent Stage 4B source commit")
    if reference["cost_profile_sha256"] != CostProfile.load(Path(profile_path)).sha256:
        raise Stage4CReductionError("inconsistent cost-profile SHA")
    expected_matrix = {
        "spread_statistics": list(SPREAD_STATISTICS),
        "slippage_round_turn_pips": list(SLIPPAGES),
    }
    if reference["scenario_matrix"] != expected_matrix:
        raise Stage4CReductionError("inconsistent scenario matrix")
    if reference["instrument"] not in INSTRUMENTS:
        raise Stage4CReductionError("unexpected instrument")
    ownership = [
        json.dumps(group, sort_keys=True, separators=(",", ":"))
        for _, manifest in records
        for group in manifest.get("group_ownership", [])
    ]
    if len(ownership) != len(set(ownership)):
        raise Stage4CReductionError("overlapping group ownership")

    for _directory, manifest in records:
        if manifest.get("owned_complete_rows") != manifest.get("state_row_count"):
            raise Stage4CReductionError("inconsistent owned row count")
        if manifest.get("scenario_evaluations") != 16 * manifest.get(
            "owned_complete_rows", -1
        ):
            raise Stage4CReductionError("inconsistent scenario evaluation count")
    owned_total = sum(manifest["owned_complete_rows"] for _, manifest in records)
    if owned_total != reference["source_complete_rows"]:
        raise Stage4CReductionError(
            "incomplete global shard coverage: owned rows do not equal source rows"
        )
    scenario_total = sum(manifest["scenario_evaluations"] for _, manifest in records)
    if scenario_total != 16 * reference["source_complete_rows"]:
        raise Stage4CReductionError(
            "incomplete global scenario coverage for complete source rows"
        )

    def states():
        for directory, manifest in records:
            declared = {
                json.dumps(group, sort_keys=True, separators=(",", ":"))
                for group in manifest.get("group_ownership", [])
            }
            observed = set()
            for row in iter_jsonl((directory / SHARD_STATE,)):
                if row.get("instrument") != manifest["instrument"]:
                    raise Stage4CReductionError("state row instrument mismatch")
                for field in (
                    "gross_return_pips_adverse_first",
                    "gross_return_pips_favorable_first",
                ):
                    value = row.get(field)
                    if not isinstance(value, int | float) or not math.isfinite(value):
                        raise Stage4CReductionError(
                            "malformed/non-finite aggregate state"
                        )
                group = tuple(row.get(field) for field in CONFIG_FIELDS)
                encoded = json.dumps(group, sort_keys=True, separators=(",", ":"))
                if encoded not in declared:
                    raise Stage4CReductionError(
                        "state group absent from declared shard ownership"
                    )
                if (
                    _group_owner(group, manifest["shard_count"])
                    != manifest["shard_index"]
                ):
                    raise Stage4CReductionError(
                        "state group assigned to another deterministic shard"
                    )
                observed.add(encoded)
                yield row
            if observed != declared:
                raise Stage4CReductionError(
                    "declared ownership group has no corresponding state rows"
                )

    components = reference["source_input_components"]
    shard_commitments = [
        {
            "shard_index": manifest["shard_index"],
            "state_row_count": manifest["state_row_count"],
            "state_sha256": manifest["state_sha256"],
        }
        for _, manifest in records
    ]
    source_audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "instrument": reference["instrument"],
        "corpus_id": reference["corpus_id"],
        "assembled_dataset_id": reference["assembled_dataset_id"],
        "source_trade_sha256": components["source_trade_sha256"],
        "registry_identity": components.get("registry_identity"),
        "source_shard_identities": shard_commitments,
        "expected_shard_count": expected_shard_count,
    }
    summary = run_rows(
        states(),
        Path(output_dir),
        Path(profile_path),
        source_mode=reference["source_mode"],
        source_audit=source_audit,
    )
    summary.update(
        stage4b_rows_read=reference["source_rows_read"],
        complete_stage4b_trade_count=reference["source_complete_rows"],
        scenario_evaluation_count=scenario_total,
    )
    output_dir = Path(output_dir)
    summary_path = output_dir / "stage4c-summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    )
    audit_path = output_dir / "execution-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["row_counts"] = {
        "source_rows_read": reference["source_rows_read"],
        "source_complete_rows": reference["source_complete_rows"],
        "owned_complete_rows": owned_total,
        "scenario_evaluations": scenario_total,
    }
    audit["reduction"] = {
        "produced_by_reducer": True,
        "shard_count": expected_shard_count,
        "shard_state_commitments": shard_commitments,
    }
    audit["output_sha256"]["stage4c-summary.json"] = sha256_file(summary_path)
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--cost-profile", type=Path, required=True)
    args = parser.parse_args(argv)
    directories = sorted(path.parent for path in args.shard_root.rglob(SHARD_MANIFEST))
    reduce_shards(directories, args.output_dir, args.cost_profile, args.shard_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
