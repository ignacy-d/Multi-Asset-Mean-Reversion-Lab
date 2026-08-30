"""Fail-closed reducer for independently produced compact Stage 4C shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import (
    INSTRUMENTS,
    SLIPPAGES,
    SPREAD_STATISTICS,
    STAGE4B_SOURCE_COMMIT,
    CostProfile,
    Stage4CError,
    sha256_file,
)
from mr_lab.stage4c_runner import SHARD_MANIFEST, SHARD_STATE, iter_jsonl, run_rows

IDENTICAL_FIELDS = (
    "shard_count",
    "stage4b_methodology_id",
    "stage4b_source_commit",
    "cost_profile_sha256",
    "scenario_matrix",
    "instrument",
    "corpus_id",
    "assembled_dataset_id",
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
        for field in IDENTICAL_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise Stage4CReductionError(f"inconsistent shard field: {field}")
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

    def states():
        for directory, _manifest in records:
            for row in iter_jsonl((directory / SHARD_STATE,)):
                for field in (
                    "gross_return_pips_adverse_first",
                    "gross_return_pips_favorable_first",
                ):
                    value = row.get(field)
                    if not isinstance(value, int | float) or not math.isfinite(value):
                        raise Stage4CReductionError(
                            "malformed/non-finite aggregate state"
                        )
                yield row

    source_audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "instrument": reference["instrument"],
        "corpus_id": reference["corpus_id"],
        "assembled_dataset_id": reference["assembled_dataset_id"],
        "source_shard_identities": indexes,
        "expected_shard_count": expected_shard_count,
    }
    return run_rows(
        states(),
        Path(output_dir),
        Path(profile_path),
        source_mode="existing_stage4b_raw",
        source_audit=source_audit,
    )


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
