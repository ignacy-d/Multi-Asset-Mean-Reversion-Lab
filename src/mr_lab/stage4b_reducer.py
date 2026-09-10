"""Fail-closed deterministic reducer for complete Stage 4B runtime shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mr_lab.stage4b_runner import (
    COMPACT_SHARD_OUTPUTS,
    RAW_SHARD_OUTPUTS,
    _json,
    _report,
    _sha256_file,
)

COMPACT_CSVS = (
    "entry-diagnostics.csv",
    "trade-matrix.csv",
    "first-passage-distributions.csv",
    "conditional-hit.csv",
    "entry-wait-distributions.csv",
    "stage4b-distributions.csv",
)
IDENTICAL_FIELDS = (
    "source_commit_sha",
    "stage4b_methodology_id",
    "instrument",
    "registry_identity",
    "corpus_id",
    "assembled_dataset_id",
    "source_mode",
    "source_acquisition_commit_sha",
    "filter_family",
    "filter_spec_id",
    "shard_count",
    "full_candidate_count",
)
ADDITIVE_SUMMARY_FIELDS = (
    "candidate_event_count",
    "no_entry_count",
    "target_already_passed_count",
    "executed_trade_configuration_count",
    "incomplete_count",
    "ambiguous_count",
    "filter_ineligible_count",
)


class ShardReductionError(ValueError):
    """A shard set is incomplete, corrupt, or provenance-inconsistent."""


def _read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_rows(path, rows):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reduce_shards(shard_dirs, output_dir, expected_shard_count):
    """Validate and concatenate group-complete reports; never merge quantiles."""
    records = []
    for directory in map(Path, shard_dirs):
        manifest_path = directory / "shard-manifest.json"
        if not manifest_path.is_file():
            raise ShardReductionError(f"missing shard manifest: {directory}")
        records.append((directory, json.loads(manifest_path.read_text())))
    indexes = [manifest["shard_index"] for _, manifest in records]
    if len(indexes) != len(set(indexes)):
        raise ShardReductionError("duplicate shard index")
    if set(indexes) != set(range(expected_shard_count)):
        raise ShardReductionError("missing or unexpected shard index")
    records.sort(key=lambda item: item[1]["shard_index"])
    reference = records[0][1]
    if reference["shard_count"] != expected_shard_count:
        raise ShardReductionError("unexpected shard_count")
    for directory, manifest in records:
        for field in IDENTICAL_FIELDS:
            if manifest.get(field) != reference.get(field):
                raise ShardReductionError(f"provenance mismatch: {field}")
        for name in COMPACT_SHARD_OUTPUTS:
            expected_hash = manifest["file_sha256"].get(name)
            path = directory / name
            if (
                not expected_hash
                or not path.is_file()
                or _sha256_file(path) != expected_hash
            ):
                raise ShardReductionError(f"shard file hash mismatch: {name}")
        if not all(
            manifest["file_sha256"].get(name) and name in manifest["row_counts"]
            for name in RAW_SHARD_OUTPUTS
        ):
            raise ShardReductionError(
                "shard manifest omits required raw hash or row-count commitment"
            )
        if not manifest.get("raw_artifact_name") or not manifest.get(
            "compact_artifact_name"
        ):
            raise ShardReductionError("shard artifact identity is required")
        for name in COMPACT_CSVS:
            expected_rows = manifest["row_counts"][name]
            with (directory / name).open("rb") as file:
                actual_rows = sum(1 for _ in file)
            if actual_rows != expected_rows:
                raise ShardReductionError(f"shard row count mismatch: {name}")
    full_groups = reference["full_group_keys"]
    if any(manifest["full_group_keys"] != full_groups for _, manifest in records):
        raise ShardReductionError("full group universe mismatch")
    combined_groups = [
        group for _, manifest in records for group in manifest["group_keys"]
    ]
    if combined_groups != full_groups or len(
        {_json(g) for g in combined_groups}
    ) != len(combined_groups):
        raise ShardReductionError("group ranges overlap, are incomplete, or unordered")
    if (
        sum(m["shard_candidate_count"] for _, m in records)
        != reference["full_candidate_count"]
    ):
        raise ShardReductionError("shard candidate counts do not cover full universe")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {}
    for name in COMPACT_CSVS:
        rows = [row for directory, _ in records for row in _read_rows(directory / name)]
        rows.sort(key=lambda row: _json(row))
        _write_rows(output_dir / name, rows)
        combined[name] = rows
    summaries = [
        json.loads((directory / "summary.json").read_text()) for directory, _ in records
    ]
    summary = {
        "reporting_schema_version": summaries[0]["reporting_schema_version"],
        "stage4b_methodology_id": summaries[0]["stage4b_methodology_id"],
        **{
            field: sum(value[field] for value in summaries)
            for field in ADDITIVE_SUMMARY_FIELDS
        },
    }
    (output_dir / "summary.json").write_text(_json(summary) + "\n")
    report_matrix = [
        row | {"executed_trade_count": int(row["executed_trade_count"])}
        for row in combined["trade-matrix.csv"]
    ]
    (output_dir / "report.md").write_text(_report(report_matrix, summary))
    audit = {
        "schema_version": "stage4b-shard-reduction-v1",
        "canonical_raw_result": (
            "ordered immutable raw shard artifacts committed by shard-generated "
            "streaming SHA-256 and row counts; reducer did not download or rehash "
            "raw files"
        ),
        "raw_shard_artifacts": [
            {
                "shard_index": manifest["shard_index"],
                "artifact_name": manifest["raw_artifact_name"],
                "file_sha256": {
                    name: manifest["file_sha256"][name] for name in RAW_SHARD_OUTPUTS
                },
                "row_counts": {
                    name: manifest["row_counts"][name] for name in RAW_SHARD_OUTPUTS
                },
            }
            for _, manifest in records
        ],
        **{field: reference.get(field) for field in IDENTICAL_FIELDS},
        "shards": [manifest for _, manifest in records],
        "combined_output_sha256": {
            path.name: _sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    (output_dir / "execution-audit.json").write_text(_json(audit) + "\n")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args(argv)
    directories = sorted(
        path.parent for path in args.shard_root.rglob("shard-manifest.json")
    )
    reduce_shards(directories, args.output_dir, args.shard_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
