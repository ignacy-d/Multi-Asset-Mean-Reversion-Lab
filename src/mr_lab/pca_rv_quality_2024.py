"""Frozen follow-up-discovery enrichment for the 2024 RV-PCA event stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log
from pathlib import Path

import numpy as np

from mr_lab.pca_residual import PCAQualityObservation, PCAResidualEngine
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

SOURCE_EVENTS_SHA256 = (
    "sha256:300eef024df0ea9b43f0a3f2d2fe60eece36a09ecb4b25afe5b406ce47ebfa57"
)
EXPECTED_PROCESS_ID = (
    "sha256:98ecd95b287604833e7868fd11433c9ac9942f660501d52a60fb656468e68a21"
)
STUDY_STATUS = "2024_FOLLOW_UP_DISCOVERY_NOT_CONFIRMATION"
HALF_LIFE_LIMIT = 60.0
SUBSPACE_LAG = 60
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20240918
PROGRESS_INTERVAL = 10_000


class PCAQualityError(ValueError):
    """Raised when frozen inputs or replay identity fail closed."""


class StreamingJSONLWriter:
    """Exclusive, line-flushing writer that preserves completed expensive work."""

    def __init__(self, path: Path):
        self._target = path.open("x", encoding="utf-8")

    def write(self, value: Mapping[str, object]) -> None:
        self._target.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self._target.flush()

    def close(self) -> None:
        self._target.close()

    def __enter__(self) -> StreamingJSONLWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


QUALITY_SPEC = {
    "study": "PCA-RV-QUALITY-2024-v1",
    "status": STUDY_STATUS,
    "base_process_identity": PROCESS_ID,
    "residual_ar1": {
        "definition": "closed-form-ols-with-intercept:y=r[1:];x=r[:-1]",
        "prior_residual_window_rows": 256,
        "variance_denominator_policy": "invalid-if-nonfinite-or-le-variance-epsilon",
    },
    "residual_ou": {
        "kappa_definition": "-log(phi)-iff-0-lt-phi-lt-1",
        "half_life_definition": "log(2)/kappa",
        "eligibility": "0-lt-phi-lt-1-and-finite-half-life-rows-le-60",
        "half_life_limit_rows": HALF_LIFE_LIMIT,
    },
    "idio_ratio": {
        "definition": "abs(residual)/(abs(residual)+abs(factor_reconstruction))",
        "denominator_policy": "null-if-denominator-le-variance-epsilon",
    },
    "k2_explained_variance_ratio": (
        "(lambda1+lambda2)/sum(all-positive-finite-covariance-eigenvalues)"
    ),
    "k2_eigengap_ratio": (
        "(lambda2-lambda3)/sum(all-positive-finite-covariance-eigenvalues)"
    ),
    "subspace_distance": {
        "projection": "P=V.T@V",
        "definition": "frobenius-norm(P_t-P_t_minus_lag)/sqrt(2*K)",
        "components": 2,
        "lag_rows": SUBSPACE_LAG,
    },
    "cross_sectional_standardized_return_dispersion": (
        "sample-standard-deviation-ddof=1"
    ),
    "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
    "features_are_gates": ["residual_ou_eligible"],
}


def _identity(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


QUALITY_STUDY_ID = _identity(QUALITY_SPEC)


@dataclass(frozen=True, slots=True)
class AR1Quality:
    residual_ar1_phi: float | None
    residual_ar1_alpha: float | None
    residual_ou_kappa: float | None
    residual_half_life_rows: float | None
    residual_ou_eligible: bool
    residual_half_life_bin: str


def estimate_ar1(prior_residuals: Sequence[float], epsilon: float) -> AR1Quality:
    """Fit y=alpha+phi*x by deterministic closed form on prior values only."""
    values = np.asarray(prior_residuals, dtype=np.float64)
    if len(values) != 256 or not np.isfinite(values).all():
        return _invalid_ar()
    x, y = values[:-1], values[1:]
    x_mean, y_mean = float(x.mean()), float(y.mean())
    centered = x - x_mean
    denominator = float(centered @ centered)
    if not isfinite(denominator) or denominator <= epsilon:
        return _invalid_ar()
    phi = float(centered @ (y - y_mean) / denominator)
    alpha = float(y_mean - phi * x_mean)
    if not isfinite(phi) or not isfinite(alpha):
        return _invalid_ar()
    if not 0.0 < phi < 1.0:
        return AR1Quality(
            phi, alpha, None, None, False, "NON_MEAN_REVERTING_OR_INVALID"
        )
    kappa = -log(phi)
    half_life = log(2.0) / kappa
    eligible = isfinite(half_life) and half_life <= HALF_LIFE_LIMIT
    return AR1Quality(phi, alpha, kappa, half_life, eligible, half_life_bin(half_life))


def _invalid_ar() -> AR1Quality:
    return AR1Quality(None, None, None, None, False, "NON_MEAN_REVERTING_OR_INVALID")


def half_life_bin(value: float) -> str:
    if value <= 5:
        return "<=5"
    if value <= 15:
        return ">5_to_15"
    if value <= 30:
        return ">15_to_30"
    if value <= 60:
        return ">30_to_60"
    return ">60"


def idio_ratio(residual: float, reconstruction: float, epsilon: float) -> float | None:
    denominator = abs(residual) + abs(reconstruction)
    return None if denominator <= epsilon else abs(residual) / denominator


def projection_subspace_distance(
    current_components: np.ndarray, prior_components: np.ndarray
) -> float:
    """Sign-invariant projection distance normalized by sqrt(2*K)."""
    if current_components.shape != prior_components.shape:
        raise PCAQualityError("component matrices must have identical shape")
    k = current_components.shape[0]
    current = current_components.T @ current_components
    prior = prior_components.T @ prior_components
    return float(np.linalg.norm(current - prior, ord="fro") / np.sqrt(2 * k))


def verify_source_hash(path: Path) -> str:
    value = _file_hash(path)
    if value != SOURCE_EVENTS_SHA256:
        raise PCAQualityError("frozen source events SHA256 mismatch")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PCAQualityError("cannot read explicit source events path") from exc
    return "sha256:" + digest.hexdigest()


def _source_events(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PCAQualityError(
                    f"malformed source event line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise PCAQualityError(
                    f"source event line {line_number} is not an object"
                )
            yield value


IDENTITY_FIELDS = (
    "event_id",
    "timestamp",
    "instrument",
    "shock_sign",
    "fade_direction",
)
FLOAT_FIELDS = (
    "residual",
    "residual_z",
    *(
        f"h{h}_{suffix}"
        for h in (5, 15, 30, 60)
        for suffix in ("signed_raw_return", "signed_bps_return")
    ),
)


def verify_regenerated_event(
    source: Mapping[str, object], regenerated: Mapping[str, object]
) -> None:
    for field in IDENTITY_FIELDS:
        if source.get(field) != regenerated.get(field):
            raise PCAQualityError(f"regenerated event identity mismatch: {field}")
    for field in FLOAT_FIELDS:
        if source.get(field) != regenerated.get(field):
            raise PCAQualityError(f"regenerated event value mismatch: {field}")


class _PrefixPanelIdentities:
    """Incrementally reproduce Stage0 prefix identities in total O(N) work."""

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


def iter_quality_with_panel_identities(
    built: PanelBuild,
    engine: PCAResidualEngine,
    *,
    subspace_lag: int = SUBSPACE_LAG,
) -> Iterator[tuple[PCAQualityObservation, str]]:
    """Pair observations with one prefix identity update per synchronized row."""
    prefix_identities = _PrefixPanelIdentities(built.identity)
    warmup = (
        engine.config.pca_training_window + engine.config.residual_normalization_window
    )
    for timestamp in built.panel.timestamps[1 : warmup + 1]:
        prefix_identities.add(timestamp.isoformat())
    current_timestamp = None
    causal_identity = ""
    for quality in engine.iter_with_quality(built.panel, subspace_lag=subspace_lag):
        timestamp = quality.observation.timestamp
        if timestamp != current_timestamp:
            causal_identity = prefix_identities.add(timestamp.isoformat())
            current_timestamp = timestamp
        yield quality, causal_identity


def enrich_event(
    base: dict[str, object], quality: PCAQualityObservation
) -> dict[str, object]:
    observation = quality.observation
    if not observation.event_emitted or quality.prior_residuals is None:
        raise PCAQualityError("AR quality is available only for emitted events")
    ar = estimate_ar1(quality.prior_residuals, FROZEN_PCA_CONFIG.variance_epsilon)
    base.update(
        {
            "quality_study_identity": QUALITY_STUDY_ID,
            "quality_study_status": STUDY_STATUS,
            "standardized_return": observation.standardized_return,
            "factor_reconstruction": observation.factor_reconstruction,
            "idio_ratio": idio_ratio(
                observation.residual,
                observation.factor_reconstruction,
                FROZEN_PCA_CONFIG.variance_epsilon,
            ),
            "k2_explained_variance_ratio": quality.k2_explained_variance_ratio,
            "k2_eigengap_ratio": quality.k2_eigengap_ratio,
            "subspace_distance_60": quality.subspace_distance_60,
            "cross_sectional_standardized_return_dispersion": (
                quality.cross_sectional_standardized_return_dispersion
            ),
            **asdict(ar),
        }
    )
    return base


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def postprocess(
    enriched_path: Path, output_dir: Path, audit: Mapping[str, object]
) -> None:
    raw, eligible = EventAccumulator(), EventAccumulator()
    bins: Counter[str] = Counter()
    with enriched_path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            raw.add(event)
            bins[str(event["residual_half_life_bin"])] += 1
            if event["residual_ou_eligible"]:
                eligible.add(event)
    digest = _file_hash(enriched_path)
    raw_summary = build_aggregated_summary(raw, digest)
    eligible_summary = (
        build_aggregated_summary(eligible, digest) if eligible.event_count else None
    )
    summary = {
        "quality_study_identity": QUALITY_STUDY_ID,
        "scientific_status": STUDY_STATUS,
        "primary_promotion_status": BLOCKER,
        "comparisons": {
            "RAW_PCA": raw_summary,
            "RESIDUAL_OU_HL_LE_60": eligible_summary,
        },
        "retention_fraction": eligible.event_count / raw.event_count,
        "half_life_bin_diagnostics": dict(sorted(bins.items())),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "execution-audit.json", {**audit, "enriched_events_sha256": digest}
    )


def run(source_events: Path, output_dir: Path) -> None:
    targets = tuple(
        output_dir / name
        for name in ("enriched-events.jsonl", "summary.json", "execution-audit.json")
    )
    if output_dir.exists() or any(path.exists() for path in targets):
        raise PCAQualityError("refusing to overwrite quality-study artifacts")
    source_sha = verify_source_hash(source_events)
    if PROCESS_ID != EXPECTED_PROCESS_ID:
        raise PCAQualityError("base PCA process identity mismatch")
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
    output_dir.mkdir(parents=True)
    enriched_path = targets[0]
    sources = _source_events(source_events)
    expected = next(sources, None)
    emitted = 0
    processed_rows = 0
    previous_timestamp = None
    with StreamingJSONLWriter(enriched_path) as target:
        for quality, causal_identity in iter_quality_with_panel_identities(
            built,
            PCAResidualEngine(FROZEN_PCA_CONFIG),
            subspace_lag=SUBSPACE_LAG,
        ):
            row = quality.observation
            if row.timestamp != previous_timestamp:
                processed_rows += 1
                previous_timestamp = row.timestamp
                if processed_rows % PROGRESS_INTERVAL == 0:
                    print(
                        f"processed_synchronized_rows={processed_rows} "
                        f"emitted_enriched_events={emitted}",
                        flush=True,
                    )
            if row.event_emitted:
                if expected is None:
                    raise PCAQualityError("regenerated event stream has extra events")
                regenerated = event_outcome(
                    built, row, causal_panel_identity=causal_identity
                )
                verify_regenerated_event(expected, regenerated)
                target.write(enrich_event(regenerated, quality))
                emitted += 1
                expected = next(sources, None)
    if expected is not None:
        raise PCAQualityError("regenerated event stream ended before frozen source")
    audit = {
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_events_sha256": source_sha,
        "base_pca_process_identity": PROCESS_ID,
        "quality_study_identity": QUALITY_STUDY_ID,
        "registry_identity": registry["registry_id"],
        "activity_policy": ACTIVITY_POLICY,
        "frozen_pca_config": asdict(FROZEN_PCA_CONFIG),
        "half_life_limit_rows": HALF_LIFE_LIMIT,
        "subspace_lag_rows": SUBSPACE_LAG,
        "scientific_status": STUDY_STATUS,
        "primary_promotion_status": BLOCKER,
    }
    postprocess(enriched_path, output_dir, audit)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-events", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.source_events, args.output_dir)
    print(BLOCKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
