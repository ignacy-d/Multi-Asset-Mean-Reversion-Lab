"""One-pass, disk-bounded PCA K=1/2/3 follow-up discovery replay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from mr_lab.pca_residual import MultiKPCAResidualEngine, MultiKQualityObservation
from mr_lab.pca_rv_quality_2024 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_PROCESS_ID,
    SOURCE_EVENTS_SHA256,
    STUDY_STATUS,
    _identity,
    _source_events,
    estimate_ar1,
    idio_ratio,
    verify_regenerated_event,
    verify_source_hash,
)
from mr_lab.pca_stage0_postprocess import EventAccumulator, build_aggregated_summary
from mr_lab.pca_stage0_runner import (
    ACTIVITY_POLICY,
    BLOCKER,
    FROZEN_PCA_CONFIG,
    INSTRUMENTS,
    PROCESS_ID,
    REGISTRY,
    PanelBuild,
    authenticate_corpus,
    build_panel,
    event_outcome,
    load_registry,
)
from mr_lab.providers.dukascopy_range import load_offline_corpus

STUDY_NAME = "PCA-MULTIK-QUALITY-2024-v1"
K_VALUES = (1, 2, 3)
SHARD_EVENT_LIMIT = 25_000
DEFAULT_MAX_OUTPUT_BYTES = 2_147_483_648
PROGRESS_INTERVAL = 25_000
HALF_LIFE_BINS = (
    "<=5",
    ">5_to_15",
    ">15_to_30",
    ">30_to_60",
    ">60",
    "NON_MEAN_REVERTING_OR_INVALID",
)
GROUPS = (
    "K1_RAW",
    "K1_OU_HL_LE_60",
    "K2_RAW",
    "K2_OU_HL_LE_60",
    "K3_RAW",
    "K3_OU_HL_LE_60",
)


class MultiKStudyError(ValueError):
    """Raised when a frozen contract or durable-output invariant fails."""


COMMON_SPEC: dict[str, object] = {
    "study": STUDY_NAME,
    "scientific_status": STUDY_STATUS,
    "primary_promotion_status": BLOCKER,
    "activity_policy": ACTIVITY_POLICY,
    "pca_training_window": 1024,
    "residual_normalization": "256-strictly-prior-rows-ddof-1",
    "shock_threshold_abs_z_gte": 2.0,
    "rearm_threshold_abs_z_lt": 1.0,
    "variance_epsilon": 1e-12,
    "standardization": "1024-strictly-prior-synchronized-returns-ddof-1",
    "ar1": "ols-with-intercept:y=r[1:];x=r[:-1];256-prior-same-K-instrument",
    "ou_kappa": "-log(phi)-iff-0-lt-phi-lt-1",
    "ou_half_life": "log(2)/kappa",
    "ou_gate": "0-lt-phi-lt-1-and-finite-half-life-rows-le-60-inclusive",
    "half_life_bins": HALF_LIFE_BINS,
    "idio_ratio": "abs(residual)/(abs(residual)+abs(factor_reconstruction))",
    "explained_variance": "sum(lambda_1..lambda_K)/sum(positive-finite-lambda)",
    "eigengap": "(lambda_K-lambda_K_plus_1)/sum(positive-finite-lambda)",
    "subspace": "norm_fro(V_K.T@V_K-P_t_minus_60)/sqrt(2*K)",
    "subspace_lag": 60,
    "dispersion": "cross-sectional-standardized-return-std-ddof-1",
    "outcomes": "existing-Open[t+1]-H5-H15-H30-H60-fixed-diagnostics",
    "primary_diagnostic": "H15",
    "bootstrap": {
        "unit": "calendar-month",
        "aggregation": "event-weighted-month-sums-and-counts",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    },
}


def _k_spec(k: int, role: str) -> dict[str, object]:
    return {**COMMON_SPEC, "pca_k": k, "role": role}


K1_SPEC = _k_spec(1, "ROBUSTNESS_ONLY_CANNOT_RESCUE_OR_REPLACE_K2")
K2_SPEC = {
    **_k_spec(2, "FROZEN_PRIMARY"),
    "frozen_source_sha256": SOURCE_EVENTS_SHA256,
    "frozen_process_identity": EXPECTED_PROCESS_ID,
}
K3_SPEC = _k_spec(3, "ROBUSTNESS_ONLY_CANNOT_RESCUE_OR_REPLACE_K2")
OU_SPEC = {
    key: COMMON_SPEC[key]
    for key in ("ar1", "ou_kappa", "ou_half_life", "ou_gate", "half_life_bins")
}
STUDY_IDENTITIES = {
    "overall": _identity(COMMON_SPEC),
    "k1_robustness": _identity(K1_SPEC),
    "k2_frozen_primary": _identity(K2_SPEC),
    "k3_robustness": _identity(K3_SPEC),
    "ou_quality": _identity(OU_SPEC),
}
K_PROCESS_IDENTITIES = {
    k: _identity(
        {
            "identity_type": "pca-multik-k-specific-process",
            "pca_spec_identity": STUDY_IDENTITIES[
                {1: "k1_robustness", 2: "k2_frozen_primary", 3: "k3_robustness"}[k]
            ],
        }
    )
    for k in K_VALUES
}
# K2 event provenance remains the frozen Stage0 process identity rather than
# the wrapper identity above.  The mapping is useful only for new K1/K3 streams.
K_PROCESS_IDENTITIES[2] = PROCESS_ID


def _atomic_json(path: Path, value: object) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiKStudyError(f"{field} must be an integer")
    return value


class AtomicGzipShardWriter:
    """Buffered event writer; completed shards are atomic and self-contained."""

    def __init__(
        self,
        output_dir: Path,
        manifest: dict[str, object],
        manifest_path: Path,
        *,
        event_limit: int = SHARD_EVENT_LIMIT,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ):
        self.output_dir = output_dir
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.event_limit = event_limit
        self.max_output_bytes = max_output_bytes
        self._events: list[bytes] = []
        self._first: str | None = None
        self._last: str | None = None

    def write(self, event: Mapping[str, object]) -> None:
        timestamp = str(event["timestamp"])
        self._first = self._first or timestamp
        self._last = timestamp
        self._events.append(
            (
                json.dumps(
                    event, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            ).encode()
        )
        if len(self._events) >= self.event_limit:
            self.finalize()

    def finalize(self) -> None:
        if not self._events:
            return
        shards = self.manifest["shards"]
        assert isinstance(shards, list)
        filename = f"part-{len(shards):05d}.jsonl.gz"
        final = self.output_dir / filename
        temporary = self.output_dir / f".{filename}.tmp"
        try:
            with temporary.open("xb") as raw:
                with gzip.GzipFile(
                    fileobj=raw, mode="wb", filename="", mtime=0
                ) as zipped:
                    for line in self._events:
                        zipped.write(line)
                raw.flush()
                os.fsync(raw.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        size = temporary.stat().st_size
        current_total = _as_int(
            self.manifest["finalized_output_bytes"], "finalized_output_bytes"
        )
        prospective_total = current_total + size
        if prospective_total > self.max_output_bytes:
            temporary.unlink()
            raise MultiKStudyError(
                "finalizing shard would exceed maximum compressed output bytes"
            )
        os.replace(temporary, final)
        metadata = {
            "filename": f"events/{filename}",
            "event_count": len(self._events),
            "compressed_bytes": size,
            "sha256": _sha_file(final),
            "first_event_timestamp": self._first,
            "last_event_timestamp": self._last,
        }
        shards.append(metadata)
        self.manifest["finalized_output_bytes"] = prospective_total
        _atomic_json(self.manifest_path, self.manifest)
        self._events.clear()
        self._first = self._last = None

    def close(self) -> None:
        self.finalize()


class _PrefixIdentities:
    def __init__(self, source_identity: str):
        prefix = json.dumps(
            {"activity_policy": ACTIVITY_POLICY, "source_identity": source_identity},
            sort_keys=True,
            separators=(",", ":"),
        )[:-1]
        self._digest = hashlib.sha256(
            (prefix + ',"synchronized_timestamps":[').encode()
        )
        self._first = True

    def add(self, timestamp: str) -> str:
        if not self._first:
            self._digest.update(b",")
        self._digest.update(json.dumps(timestamp).encode())
        self._first = False
        snapshot = self._digest.copy()
        snapshot.update(b"]}")
        return "sha256:" + snapshot.hexdigest()


def iter_multik_with_panel_identities(
    built: PanelBuild, engine: MultiKPCAResidualEngine
) -> Iterator[tuple[MultiKQualityObservation, str]]:
    prefix = _PrefixIdentities(built.identity)
    warmup = (
        FROZEN_PCA_CONFIG.pca_training_window
        + FROZEN_PCA_CONFIG.residual_normalization_window
    )
    for timestamp in built.panel.timestamps[1 : warmup + 1]:
        prefix.add(timestamp.isoformat())
    previous = None
    identity = ""
    for row in engine.iter_with_quality(built.panel):
        if row.observation.timestamp != previous:
            identity = prefix.add(row.observation.timestamp.isoformat())
            previous = row.observation.timestamp
        yield row, identity


def enrich_event(
    base: dict[str, object], row: MultiKQualityObservation
) -> dict[str, object]:
    if not row.observation.event_emitted or row.prior_residuals is None:
        raise MultiKStudyError("AR/OU may be computed only for emitted events")
    ar = estimate_ar1(row.prior_residuals, FROZEN_PCA_CONFIG.variance_epsilon)
    base.update(
        {
            "pca_k": row.pca_k,
            "multik_study_identity": STUDY_IDENTITIES["overall"],
            "pca_spec_identity": STUDY_IDENTITIES[
                {1: "k1_robustness", 2: "k2_frozen_primary", 3: "k3_robustness"}[
                    row.pca_k
                ]
            ],
            "ou_quality_identity": STUDY_IDENTITIES["ou_quality"],
            "scientific_status": STUDY_STATUS,
            "standardized_return": row.observation.standardized_return,
            "factor_reconstruction": row.observation.factor_reconstruction,
            "idio_ratio": idio_ratio(
                row.observation.residual,
                row.observation.factor_reconstruction,
                FROZEN_PCA_CONFIG.variance_epsilon,
            ),
            "explained_variance_ratio": row.explained_variance_ratio,
            "eigengap_ratio": row.eigengap_ratio,
            "subspace_distance_60": row.subspace_distance_60,
            "cross_sectional_standardized_return_dispersion": (
                row.cross_sectional_standardized_return_dispersion
            ),
            **asdict(ar),
        }
    )
    return base


def apply_k_specific_provenance(
    event: dict[str, object], pca_k: int
) -> dict[str, object]:
    """Replace K2 helper provenance only for the new robustness streams."""
    if pca_k == 2:
        return event
    if pca_k not in (1, 3):
        raise MultiKStudyError("unsupported PCA K provenance")
    event["frozen_k2_reference_process_identity"] = PROCESS_ID
    event["process_identity"] = K_PROCESS_IDENTITIES[pca_k]
    config = asdict(FROZEN_PCA_CONFIG)
    config["components"] = pca_k
    event["pca_config"] = config
    return event


def _normalize_multik_summary_identity(
    summary: dict[str, object], manifest_identity: str
) -> dict[str, object]:
    """Rename the legacy single-file hash field for a multi-shard artifact."""
    summary.pop("source_events_sha256", None)
    summary["source_event_manifest_identity"] = manifest_identity
    summary["frozen_k2_source_events_sha256"] = SOURCE_EVENTS_SHA256
    return summary


def _iter_shards(
    output_dir: Path, manifest: Mapping[str, object]
) -> Iterator[dict[str, object]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise MultiKStudyError("manifest shards must be a list")
    for shard in shards:
        if not isinstance(shard, dict):
            raise MultiKStudyError("invalid shard metadata")
        path = output_dir / str(shard["filename"])
        if _sha_file(path) != shard["sha256"]:
            raise MultiKStudyError("event shard SHA256 mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MultiKStudyError("event shard line is not an object")
                yield value


def postprocess(output_dir: Path) -> Path:
    manifest = cast(
        dict[str, object], json.loads((output_dir / "manifest.json").read_text())
    )
    if not manifest.get("run_complete"):
        raise MultiKStudyError("cannot summarize an incomplete replay")
    accumulators = {group: EventAccumulator() for group in GROUPS}
    bins: dict[str, Counter[str]] = {group: Counter() for group in GROUPS}
    for event in _iter_shards(output_dir, manifest):
        k = _as_int(event["pca_k"], "pca_k")
        raw_name = f"K{k}_RAW"
        accumulators[raw_name].add(event)
        bins[raw_name][str(event["residual_half_life_bin"])] += 1
        if event["residual_ou_eligible"]:
            ou_name = f"K{k}_OU_HL_LE_60"
            accumulators[ou_name].add(event)
            bins[ou_name][str(event["residual_half_life_bin"])] += 1
    comparisons: dict[str, object] = {}
    for group in GROUPS:
        accumulator = accumulators[group]
        manifest_identity = str(manifest["event_manifest_identity"])
        summary: dict[str, object] = (
            build_aggregated_summary(accumulator, manifest_identity)
            if accumulator.event_count
            else {"event_count": 0}
        )
        _normalize_multik_summary_identity(summary, manifest_identity)
        k = int(group[1])
        raw_count = accumulators[f"K{k}_RAW"].event_count
        summary["retention_fraction_vs_same_k_raw"] = (
            accumulator.event_count / raw_count if raw_count else None
        )
        summary["half_life_bin_counts"] = {
            name: bins[group][name] for name in HALF_LIFE_BINS
        }
        comparisons[group] = summary
    document = {
        "study": STUDY_NAME,
        "scientific_status": STUDY_STATUS,
        "primary_promotion_status": BLOCKER,
        "h15_is_primary_fixed_horizon_diagnostic": True,
        "k1_k3_role": "ROBUSTNESS_ONLY; cannot rescue, replace, or redefine frozen K2",
        "study_identities": STUDY_IDENTITIES,
        "comparisons": comparisons,
        "cross_k_robustness": {
            f"K{k}_raw_events": accumulators[f"K{k}_RAW"].event_count for k in K_VALUES
        },
    }
    path = output_dir / "summary.json"
    _atomic_json(path, document)
    return path


def run(
    source_events: Path,
    output_dir: Path,
    *,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> None:
    if output_dir.exists():
        raise MultiKStudyError("refusing to overwrite existing output directory")
    if max_output_bytes <= 0:
        raise MultiKStudyError("max-output-bytes must be positive")
    source_sha = verify_source_hash(source_events)
    if PROCESS_ID != EXPECTED_PROCESS_ID:
        raise MultiKStudyError("frozen K2 process identity mismatch")
    registry = load_registry(REGISTRY)
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    datasets = {
        name: load_offline_corpus(authenticate_corpus(entries[name]))
        for name in INSTRUMENTS
    }
    built = build_panel(
        {name: datasets[name].bars for name in INSTRUMENTS},
        {name: str(entries[name]["assembled_dataset_id"]) for name in INSTRUMENTS},
        registry_identity=str(registry["registry_id"]),
    )
    events_dir = output_dir / "events"
    events_dir.mkdir(parents=True)
    manifest_path = output_dir / "manifest.json"
    counts = {str(k): 0 for k in K_VALUES}
    ou_counts = {str(k): 0 for k in K_VALUES}
    manifest: dict[str, object] = {
        "study": STUDY_NAME,
        "run_complete": False,
        "k2_frozen_replay_complete": False,
        "total_emitted_counts_by_k": counts,
        "total_ou_counts_by_k": ou_counts,
        "finalized_output_bytes": 0,
        "max_output_bytes": max_output_bytes,
        "shards": [],
        "study_identities": STUDY_IDENTITIES,
    }
    _atomic_json(manifest_path, manifest)
    writer = AtomicGzipShardWriter(
        events_dir, manifest, manifest_path, max_output_bytes=max_output_bytes
    )
    expected_stream = _source_events(source_events)
    expected = next(expected_stream, None)
    start = time.monotonic()
    processed = 0
    prior_timestamp = None
    try:
        for quality, panel_identity in iter_multik_with_panel_identities(
            built, MultiKPCAResidualEngine(FROZEN_PCA_CONFIG)
        ):
            row = quality.observation
            if row.timestamp != prior_timestamp:
                processed += 1
                prior_timestamp = row.timestamp
                if processed % PROGRESS_INTERVAL == 0:
                    print(
                        " ".join(
                            [f"processed_synchronized_rows={processed}"]
                            + [f"K{k}_events={counts[str(k)]}" for k in K_VALUES]
                            + [f"K{k}_ou_events={ou_counts[str(k)]}" for k in K_VALUES]
                            + [
                                f"finalized_output_bytes={manifest['finalized_output_bytes']}",
                                f"elapsed_seconds={time.monotonic() - start:.3f}",
                            ]
                        ),
                        flush=True,
                    )
            if not row.event_emitted:
                continue
            event = event_outcome(built, row, causal_panel_identity=panel_identity)
            if quality.pca_k == 2:
                if expected is None:
                    raise MultiKStudyError("regenerated K2 stream has extra events")
                verify_regenerated_event(expected, event)
                expected = next(expected_stream, None)
            else:
                event["event_id"] = _identity(
                    {
                        "base_event_id": event["event_id"],
                        "pca_k": quality.pca_k,
                        "study": STUDY_NAME,
                    }
                )
                apply_k_specific_provenance(event, quality.pca_k)
            enriched = enrich_event(event, quality)
            writer.write(enriched)
            counts[str(quality.pca_k)] += 1
            if enriched["residual_ou_eligible"]:
                ou_counts[str(quality.pca_k)] += 1
        writer.close()
        if expected is not None:
            raise MultiKStudyError("regenerated K2 stream ended before frozen source")
        manifest["k2_frozen_replay_complete"] = True
        manifest["event_manifest_identity"] = _identity(
            {
                "shards": manifest["shards"],
                "counts": counts,
                "study_identities": STUDY_IDENTITIES,
            }
        )
        manifest["run_complete"] = True
        _atomic_json(manifest_path, manifest)
    except BaseException:
        _atomic_json(manifest_path, manifest)
        raise
    audit = {
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_events_sha256": source_sha,
        "frozen_k2_process_identity": PROCESS_ID,
        "registry_identity": registry["registry_id"],
        "study_identities": STUDY_IDENTITIES,
        "scientific_status": STUDY_STATUS,
        "primary_promotion_status": BLOCKER,
        "authenticated_empirical_execution": True,
    }
    audit_path = output_dir / "execution-audit.json"
    _atomic_json(audit_path, audit)
    summary_path = postprocess(output_dir)
    print(
        f"elapsed_seconds={time.monotonic() - start:.3f} "
        f"raw_counts={counts} ou_counts={ou_counts} "
        f"compressed_event_bytes={manifest['finalized_output_bytes']} "
        f"shard_count={len(cast(list[object], manifest['shards']))} "
        f"manifest_path={manifest_path} "
        f"summary_path={summary_path} execution_audit_path={audit_path} "
        f"event_manifest_identity={manifest['event_manifest_identity']}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-events", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES
    )
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args(argv)
    if args.postprocess_only:
        postprocess(args.output_dir)
    elif args.source_events is None:
        parser.error("--source-events is required unless --postprocess-only is used")
    else:
        run(args.source_events, args.output_dir, max_output_bytes=args.max_output_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
