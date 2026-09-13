"""Post-hoc 2024 development quality ranking for frozen Trend Exhaustion events.

TE-Q1 is not a reclassification of Trend Exhaustion v1.  This module consumes
only the authenticated, explicit 2024 registry and regenerates primary events
from the frozen detector and machine-readable directional path diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from mr_lab.data import Timeframe, resample_bars
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.stage4a import (
    DirectionalPathDiagnostic,
    DirectionalPathRequest,
    diagnose_directional_paths,
)
from mr_lab.stage4a_runner import (
    FROZEN_INSTRUMENTS,
    load_corpus_registry,
    read_and_validate_manifest,
    validate_registry_entry,
)
from mr_lab.trend_exhaustion import (
    PRIMARY_THRESHOLD,
    TrendExhaustionEvent,
    TrendExhaustionSpec,
    detect_trend_exhaustion,
)

DEVELOPMENT_YEAR = 2024
CONTINUOUS_PREDICTORS = (
    "normalized_displacement",
    "efficiency_ratio",
    "extension_atr",
    "retained_progress_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "body_size_atr",
    "m15_range_atr",
    "wick_body_ratio",
    "minutes_from_session_open",
    "minutes_to_session_close",
)
CATEGORICAL_PREDICTORS = ("instrument", "direction", "session_label")
PREDICTORS = CONTINUOUS_PREDICTORS + CATEGORICAL_PREDICTORS
FOLDS = tuple((tuple(range(1, month)), month) for month in range(5, 13))
MODEL_ORDER = ("huber", "quantile", "additive_splines")
BOOTSTRAP_SEED = 20240913
BOOTSTRAP_REPLICATES = 2_000
CANONICAL_CORPUS_ROOT = Path("/mnt/e/mr-lab/frozen-2024")
CANONICAL_REGISTRY_PATH = Path("configs/stage4a-2024-corpus-registry.json")
REGISTRY_SCHEMA = "stage-4a-2024-corpus-registry-v1"
DEVELOPMENT_START = "2024-01-01"
DEVELOPMENT_END = "2024-12-31"
EXPECTED_PRIMARY_EVENT_COUNTS = {
    "AUDJPY": 958,
    "AUDUSD": 951,
    "EURUSD": 937,
    "GBPUSD": 961,
    "USDJPY": 1001,
}
EXPECTED_PRIMARY_EVENT_TOTAL = 4808


class QualityLabError(ValueError):
    """Raised when TE-Q1's frozen development contract is violated."""


@dataclass(frozen=True, slots=True)
class QualityRow:
    event_id: str
    signal_timestamp: str
    instrument: str
    direction: str
    session_label: str | None
    normalized_displacement: float
    efficiency_ratio: float
    extension_atr: float
    retained_progress_atr: float
    upper_wick_atr: float
    lower_wick_atr: float
    body_size_atr: float
    m15_range_atr: float
    wick_body_ratio: float | None
    minutes_from_session_open: int | None
    minutes_to_session_close: int | None
    volume_change: float | None
    y60_atr: float
    h60_pips: float
    mfe_pips: float | None
    mae_pips: float | None

    @property
    def month(self) -> int:
        return int(self.signal_timestamp[5:7])

    @property
    def quarter(self) -> str:
        return f"2024-Q{(self.month - 1) // 3 + 1}"


@dataclass(frozen=True, slots=True)
class Prediction:
    event_id: str
    family: str
    test_month: int
    score: float
    row: QualityRow


@dataclass(frozen=True, slots=True)
class DevelopmentDataset:
    rows: tuple[QualityRow, ...]
    primary_event_counts: dict[str, int]
    primary_event_total: int


def _same_explicit_path(given: Path, expected: Path) -> bool:
    """Compare path labels without resolving symlinks or inspecting the filesystem."""
    return given.expanduser().absolute() == expected.expanduser().absolute()


def validate_development_registry(registry: Mapping[str, object]) -> None:
    """Fail closed unless all registry metadata is the explicit 2024 contract."""
    if registry.get("registry_schema_version") != REGISTRY_SCHEMA:
        raise QualityLabError("registry is not the explicit 2024 development schema")
    entries = registry.get("instruments")
    if not isinstance(entries, dict) or set(entries) != set(FROZEN_INSTRUMENTS):
        raise QualityLabError(
            "registry does not contain the exact development universe"
        )
    for instrument in FROZEN_INSTRUMENTS:
        entry = entries[instrument]
        if not isinstance(entry, dict) or (
            entry.get("requested_start_date"),
            entry.get("requested_end_date"),
        ) != (DEVELOPMENT_START, DEVELOPMENT_END):
            raise QualityLabError("registry entry is outside the exact 2024 date range")
        validate_registry_entry(instrument, entry, require_verified=True)


def preflight_development_inputs(
    corpus_root: Path, registry_path: Path
) -> dict[str, object]:
    """Validate root label and complete registry before any corpus-path access."""
    if not _same_explicit_path(corpus_root, CANONICAL_CORPUS_ROOT):
        raise QualityLabError(
            f"corpus root must be the explicit development root {CANONICAL_CORPUS_ROOT}"
        )
    if not _same_explicit_path(registry_path, CANONICAL_REGISTRY_PATH):
        raise QualityLabError(
            "registry must be the canonical development registry "
            f"{CANONICAL_REGISTRY_PATH}"
        )
    registry = load_corpus_registry(registry_path)
    validate_development_registry(registry)
    return registry


def _validate_loaded_bars(bars: Sequence[Any]) -> None:
    """Defense in depth after an authenticated corpus has been loaded."""
    for bar in bars:
        for timestamp in (bar.open_time, bar.close_time, bar.available_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                raise QualityLabError("loaded corpus timestamps must be UTC")
        if bar.open_time.year != DEVELOPMENT_YEAR:
            raise QualityLabError(
                "loaded corpus contains an open outside development 2024"
            )
        if bar.close_time != bar.open_time + bar.timeframe.duration:
            raise QualityLabError(
                "loaded corpus bar close does not match its timeframe"
            )
        if bar.available_at < bar.close_time:
            raise QualityLabError("loaded corpus bar is available before completion")


def validate_primary_event_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    """Require exact v1 event-universe reproduction; this is not a promotion gate."""
    observed = {
        instrument: counts.get(instrument, 0) for instrument in FROZEN_INSTRUMENTS
    }
    total = sum(observed.values())
    if (
        observed != EXPECTED_PRIMARY_EVENT_COUNTS
        or total != EXPECTED_PRIMARY_EVENT_TOTAL
    ):
        raise QualityLabError(
            "primary-event reproducibility failure: "
            f"expected total={EXPECTED_PRIMARY_EVENT_TOTAL} "
            f"per_instrument={EXPECTED_PRIMARY_EVENT_COUNTS}; "
            f"observed total={total} per_instrument={observed}"
        )
    return {"total": total, "per_instrument": observed}


def _utc_2024(event: TrendExhaustionEvent) -> None:
    timestamp = event.signal_timestamp
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise QualityLabError("event timestamp must be timezone-aware UTC")
    if timestamp.year != DEVELOPMENT_YEAR:
        raise QualityLabError("event is outside the explicit 2024 development year")


def quality_row(
    event: TrendExhaustionEvent, path: DirectionalPathDiagnostic
) -> QualityRow:
    """Join a frozen event to its typed path without exposing outcomes as features."""
    _utc_2024(event)
    if event.displacement_threshold != PRIMARY_THRESHOLD:
        raise QualityLabError("TE-Q1 accepts only primary threshold 1.50 events")
    identity = (event.instrument, event.signal_timestamp, event.direction)
    if identity != (path.instrument, path.signal_timestamp, path.direction):
        raise QualityLabError("event and directional path identities disagree")
    h60 = next((item for item in path.horizons if item.horizon_minutes == 60), None)
    if h60 is None:
        raise QualityLabError("a machine-readable H60 outcome is required")
    if not all(
        math.isfinite(value)
        for value in (h60.signed_price_movement, h60.signed_return_pips)
    ):
        raise QualityLabError("H60 outcomes must be finite")
    atr = event.frozen_atr20
    if not math.isfinite(atr) or atr <= 0:
        raise QualityLabError("frozen ATR must be finite and positive")
    return QualityRow(
        event.event_id,
        event.signal_timestamp.isoformat(),
        event.instrument,
        event.direction.name.lower(),
        event.session_label,
        event.normalized_displacement,
        event.efficiency_ratio,
        event.exhaustion_extension / atr,
        event.retained_progress / atr,
        event.upper_wick / atr,
        event.lower_wick / atr,
        event.body_size / atr,
        event.m15_range / atr,
        event.wick_body_ratio,
        event.minutes_from_session_open,
        event.minutes_to_session_close,
        event.volume_change,
        h60.signed_price_movement / atr,
        h60.signed_return_pips,
        path.mfe_pips,
        path.mae_pips,
    )


def build_development_dataset(
    corpus_root: Path, registry_path: Path
) -> DevelopmentDataset:
    """Load only explicit authenticated 2024 corpus paths and regenerate TE-Q1 rows."""
    registry = preflight_development_inputs(corpus_root, registry_path)
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    rows: list[QualityRow] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for instrument in FROZEN_INSTRUMENTS:
        entry = validate_registry_entry(
            instrument, entries[instrument], require_verified=True
        )
        # Validate metadata before the offline loader may inspect corpus bytes.
        corpus_dir = corpus_root / instrument
        manifest = read_and_validate_manifest(corpus_dir, instrument)
        validate_registry_entry(
            instrument, entry, manifest=manifest, require_verified=True
        )
        dataset = load_offline_corpus(corpus_dir)
        if dataset.metadata.dataset_id != manifest["assembled_dataset_id"]:
            raise QualityLabError("loaded dataset identity disagrees with manifest")
        _validate_loaded_bars(dataset.bars)
        m15 = resample_bars(dataset.bars, Timeframe("15m")).bars
        by_time = {bar.available_at: bar for bar in m15}
        events = detect_trend_exhaustion(m15, TrendExhaustionSpec(PRIMARY_THRESHOLD))
        counts[instrument] = len(events)
        if any(event.event_id in seen for event in events):
            raise QualityLabError("duplicate primary opportunity")
        seen.update(event.event_id for event in events)
        requests = tuple(
            DirectionalPathRequest(
                instrument,
                event.signal_timestamp,
                event.direction,
                by_time[event.signal_timestamp].close,
            )
            for event in events
        )
        paths = diagnose_directional_paths(requests, dataset.bars)
        for event, path in zip(events, paths, strict=True):
            if any(item.horizon_minutes == 60 for item in path.horizons):
                rows.append(quality_row(event, path))
    diagnostics = validate_primary_event_counts(counts)
    return DevelopmentDataset(
        tuple(sorted(rows, key=lambda row: (row.signal_timestamp, row.event_id))),
        dict(diagnostics["per_instrument"]),
        int(diagnostics["total"]),
    )


def build_development_rows(
    corpus_root: Path, registry_path: Path
) -> tuple[QualityRow, ...]:
    """Compatibility facade returning rows from the authenticated dataset build."""
    return build_development_dataset(corpus_root, registry_path).rows


def _records(rows: Sequence[QualityRow]) -> list[dict[str, object]]:
    return [{name: getattr(row, name) for name in PREDICTORS} for row in rows]


def _matrix(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [[record[name] for name in PREDICTORS] for record in records], dtype=object
    )


def _quantiles(values: Sequence[float], probabilities: Sequence[float]) -> list[float]:
    result: Any = np.quantile(values, probabilities)
    return [float(result[index]) for index in range(len(probabilities))]


def _preprocessor(*, splines: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median"))
    ]
    if splines:
        numeric_steps.append(("spline", SplineTransformer(n_knots=5, degree=3)))
    numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        (
            (
                "continuous",
                Pipeline(numeric_steps),
                list(range(len(CONTINUOUS_PREDICTORS))),
            ),
            (
                "categorical",
                Pipeline(
                    (
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    )
                ),
                list(range(len(CONTINUOUS_PREDICTORS), len(PREDICTORS))),
            ),
        ),
        sparse_threshold=0.0,
    )


def _models() -> dict[str, tuple[Pipeline, ...]]:
    def pipeline(estimator: Any, *, splines: bool = False) -> Pipeline:
        return Pipeline(
            (("preprocess", _preprocessor(splines=splines)), ("model", estimator))
        )

    return {
        "huber": (pipeline(HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1_000)),),
        "quantile": tuple(
            pipeline(QuantileRegressor(quantile=q, alpha=0.1, solver="highs"))
            for q in (0.25, 0.50, 0.75)
        ),
        "additive_splines": (pipeline(Ridge(alpha=10.0), splines=True),),
    }


def walk_forward(rows: Sequence[QualityRow]) -> tuple[Prediction, ...]:
    """Fit pipelines within each chronological fold; return test predictions only."""
    ordered = tuple(sorted(rows, key=lambda row: (row.signal_timestamp, row.event_id)))
    output: list[Prediction] = []
    for train_months, test_month in FOLDS:
        train = tuple(row for row in ordered if row.month in train_months)
        test = tuple(row for row in ordered if row.month == test_month)
        if not train or not test:
            continue
        x_train, x_test = _matrix(_records(train)), _matrix(_records(test))
        y_train = np.asarray([row.y60_atr for row in train])
        for family, estimators in _models().items():
            scores = []
            for estimator in estimators:
                estimator.fit(x_train, y_train)
                scores.append(np.asarray(estimator.predict(x_test), dtype=float))
            if family == "quantile":
                for quantile, quantile_scores in zip(
                    ("q25", "q50", "q75"), scores, strict=True
                ):
                    output.extend(
                        Prediction(
                            row.event_id,
                            f"quantile_{quantile}",
                            test_month,
                            float(score),
                            row,
                        )
                        for row, score in zip(test, quantile_scores, strict=True)
                    )
            # Predeclared quantile-family score is the median of q25/q50/q75.
            combined = np.median(np.vstack(scores), axis=0)
            output.extend(
                Prediction(row.event_id, family, test_month, float(score), row)
                for row, score in zip(test, combined, strict=True)
            )
    return tuple(output)


def descriptive_eda(rows: Sequence[QualityRow]) -> dict[str, Any]:
    """Describe causal fields post hoc; this output never defines a filter."""

    def outcomes(selected: Sequence[QualityRow]) -> dict[str, Any]:
        return _summary(
            [Prediction(row.event_id, "eda", row.month, 0.0, row) for row in selected]
        )

    continuous: dict[str, Any] = {}
    for name in CONTINUOUS_PREDICTORS:
        observed = [
            (float(value), row)
            for row in rows
            if (value := getattr(row, name)) is not None
        ]
        observed.sort(key=lambda item: (item[0], item[1].event_id))
        values = [value for value, _ in observed]
        quantile_values = (
            _quantiles(values, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) if values else []
        )
        item: dict[str, Any] = {
            "count": len(values),
            "missing": len(rows) - len(values),
            "missing_rate": (len(rows) - len(values)) / len(rows) if rows else None,
            "quantiles": dict(
                zip(
                    ("p00", "p10", "p25", "p50", "p75", "p90", "p100"),
                    quantile_values,
                    strict=True,
                )
            )
            if values
            else {},
        }
        item["quintile_expectancy"] = [
            outcomes([row for _, row in bucket])
            for bucket in np.array_split(np.asarray(observed, dtype=object), 5)
        ]
        item["decile_expectancy"] = (
            [
                outcomes([row for _, row in bucket])
                for bucket in np.array_split(np.asarray(observed, dtype=object), 10)
            ]
            if len(observed) >= 10
            else []
        )
        continuous[name] = item
    grouped = {}
    for name in CATEGORICAL_PREDICTORS:
        keys = sorted({getattr(row, name) or "none" for row in rows})
        grouped[name] = {
            key: outcomes(
                [row for row in rows if (getattr(row, name) or "none") == key]
            )
            for key in keys
        }
    return {
        "label": "POST-HOC development EDA",
        "continuous": continuous,
        "categorical": grouped,
    }


def _summary(predictions: Sequence[Prediction]) -> dict[str, Any]:
    rows = [item.row for item in predictions]
    raw = [row.h60_pips for row in rows]
    normalized = [row.y60_atr for row in rows]
    losses = -sum(value for value in raw if value < 0)
    return {
        "n": len(rows),
        "mean_y60_atr": fmean(normalized) if rows else None,
        "median_y60_atr": median(normalized) if rows else None,
        "mean_h60_pips": fmean(raw) if rows else None,
        "median_h60_pips": median(raw) if rows else None,
        "profit_factor": sum(value for value in raw if value > 0) / losses
        if losses
        else None,
        "win_rate": sum(value > 0 for value in raw) / len(raw) if raw else None,
        "mfe_mean": fmean(v for v in (r.mfe_pips for r in rows) if v is not None)
        if any(r.mfe_pips is not None for r in rows)
        else None,
        "mae_mean": fmean(v for v in (r.mae_pips for r in rows) if v is not None)
        if any(r.mae_pips is not None for r in rows)
        else None,
    }


def evaluate_family(predictions: Sequence[Prediction]) -> dict[str, Any]:
    """Evaluate one family's pooled OOS-only scores using deterministic rank buckets."""
    if not predictions or len({item.family for item in predictions}) != 1:
        raise QualityLabError("evaluation requires one non-empty OOS model family")
    scores_finite = all(math.isfinite(item.score) for item in predictions)
    ranked = sorted(
        predictions,
        key=lambda item: (
            item.score if math.isfinite(item.score) else -math.inf,
            item.event_id,
        ),
    )
    chunks = np.array_split(np.asarray(ranked, dtype=object), 5)
    quintiles = [_summary(list(chunk)) for chunk in chunks]
    count = len(ranked)
    top20 = ranked[count - math.ceil(count * 0.20) :]
    top10 = ranked[count - math.ceil(count * 0.10) :]
    baseline = fmean(item.row.y60_atr for item in ranked)
    scores = [x.score for x in ranked]
    outcomes = [x.row.y60_atr for x in ranked]
    ranking_varies = len(set(scores)) > 1 and len(set(outcomes)) > 1
    raw_rho = (
        float(spearmanr(scores, outcomes).statistic)
        if scores_finite and ranking_varies
        else math.nan
    )
    rho = raw_rho if math.isfinite(raw_rho) else None
    stability = {}
    for key, getter in (
        ("instrument", lambda x: x.row.instrument),
        ("test_month", lambda x: str(x.test_month)),
        ("quarter", lambda x: x.row.quarter),
        ("direction", lambda x: x.row.direction),
        ("session", lambda x: x.row.session_label or "none"),
    ):
        counts = Counter(getter(item) for item in top20)
        stability[key] = {
            name: {"n": value, "share": value / len(top20)}
            for name, value in sorted(counts.items())
        }
    means = [item["mean_y60_atr"] for item in quintiles]
    result = {
        "oos_baseline": _summary(ranked),
        "quintiles": quintiles,
        "top_20": _summary(top20),
        "top_10": _summary(top10),
        "spearman": rho,
        "ranking_diagnostics_finite": scores_finite and rho is not None,
        "top_minus_bottom": means[-1] - means[0],
        "top_20_lift": _summary(top20)["mean_y60_atr"] - baseline,
        "top_10_lift": _summary(top10)["mean_y60_atr"] - baseline,
        "ordered_adjacent_quintile_pairs": sum(a <= b for a, b in pairwise(means)),
        "stability": stability,
    }
    result["month_block_bootstrap"] = month_block_bootstrap(ranked)
    return result


def month_block_bootstrap(predictions: Sequence[Prediction]) -> dict[str, list[float]]:
    """Return percentile CIs by resampling whole OOS test-month blocks."""
    groups: dict[int, list[Prediction]] = defaultdict(list)
    for item in predictions:
        groups[item.test_month].append(item)
    months = sorted(groups)
    if not months:
        return {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: tuple[list[float], list[float], list[float]] = ([], [], [])
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [
            item
            for month in rng.choice(months, len(months), replace=True)
            for item in groups[int(month)]
        ]
        ranked = sorted(sample, key=lambda item: (item.score, item.event_id))
        n = len(ranked)
        k = math.ceil(n * 0.2)
        top, bottom = ranked[n - k :], ranked[:k]
        values[0].append(fmean(x.row.h60_pips for x in top))
        values[1].append(fmean(x.row.y60_atr for x in top))
        values[2].append(
            fmean(x.row.y60_atr for x in top) - fmean(x.row.y60_atr for x in bottom)
        )
    names = ("top_20_mean_h60_pips", "top_20_mean_y60_atr", "top_minus_bottom_y60_atr")
    return {
        name: _quantiles(sample, (0.025, 0.975))
        for name, sample in zip(names, values, strict=True)
    }


def family_passes(report: Mapping[str, Any]) -> bool:
    top = report["top_20"]
    stability = report["stability"]
    positive_instruments = sum(
        isinstance(v["mean_h60_pips"], int | float) and v["mean_h60_pips"] > 0
        for v in report["top_20_by_instrument"].values()
    )

    def maximum(key: str) -> float:
        return max((v["share"] for v in stability[key].values()), default=1.0)

    return bool(
        report.get("ranking_diagnostics_finite") is True
        and isinstance(report.get("spearman"), int | float)
        and math.isfinite(report["spearman"])
        and report["oos_baseline"]["n"] >= 1000
        and top["n"] >= 200
        and top["mean_h60_pips"] > 0
        and top["median_h60_pips"] >= 0
        and top["profit_factor"] is not None
        and top["profit_factor"] >= 1.20
        and report["top_20_lift"] >= 0.05
        and report["spearman"] >= 0.05
        and report["ordered_adjacent_quintile_pairs"] >= 3
        and positive_instruments >= 3
        and maximum("instrument") <= 0.40
        and maximum("test_month") <= 0.30
        and maximum("quarter") <= 0.60
    )


def classify(reports: Mapping[str, Mapping[str, Any]]) -> str:
    if any(
        report["oos_baseline"]["n"] < 1000 or report["top_20"]["n"] < 200
        for report in reports.values()
    ):
        return "INCONCLUSIVE"
    return (
        "PROMOTE_TO_FROZEN_CANDIDATE"
        if sum(family_passes(r) for r in reports.values()) >= 2
        else "PARK"
    )


def select_model(reports: Mapping[str, Mapping[str, Any]]) -> str:
    def finite_metric(report: Mapping[str, Any], name: str) -> float:
        value = report.get(name)
        return (
            float(value)
            if isinstance(value, int | float) and math.isfinite(value)
            else -math.inf
        )

    leader = max(
        MODEL_ORDER,
        key=lambda name: (
            finite_metric(reports[name], "top_20_lift"),
            finite_metric(reports[name], "spearman"),
            -MODEL_ORDER.index(name),
        ),
    )
    leader_lift = finite_metric(reports[leader], "top_20_lift")
    leader_spearman = finite_metric(reports[leader], "spearman")
    for name in MODEL_ORDER:
        if (
            finite_metric(reports[name], "top_20_lift") >= leader_lift - 0.02
            and finite_metric(reports[name], "spearman") >= leader_spearman - 0.01
        ):
            return name
    return leader


def run(corpus_root: Path, registry_path: Path, output_dir: Path) -> Path:
    development = build_development_dataset(corpus_root, registry_path)
    rows = development.rows
    predictions = walk_forward(rows)
    reports = {}
    for family in MODEL_ORDER:
        selected = tuple(item for item in predictions if item.family == family)
        report = evaluate_family(selected)
        top = sorted(selected, key=lambda x: (x.score, x.event_id))[
            len(selected) - math.ceil(len(selected) * 0.2) :
        ]
        report["top_20_by_instrument"] = {
            name: _summary([x for x in top if x.row.instrument == name])
            for name in FROZEN_INSTRUMENTS
        }
        reports[family] = report
    quantile_diagnostics = {
        quantile: evaluate_family(
            tuple(item for item in predictions if item.family == f"quantile_{quantile}")
        )
        for quantile in ("q25", "q50", "q75")
    }
    payload = {
        "schema": "trend-exhaustion-quality-lab-v0",
        "research_status": "POST_HOC_DEVELOPMENT",
        "does_not_reclassify_trend_exhaustion_v1": True,
        "development_year": DEVELOPMENT_YEAR,
        "features": list(PREDICTORS),
        "target": "signed H60 price movement / frozen_atr20",
        "models": reports,
        "quantile_model_diagnostics": quantile_diagnostics,
        "eda": descriptive_eda(rows),
        "classification": classify(reports),
        "selected_model": select_model(reports),
        "row_count": len(rows),
        "primary_event_reproducibility": {
            "total": development.primary_event_total,
            "per_instrument": development.primary_event_counts,
            "matches_frozen_v1": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trend-exhaustion-quality-report.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run post-hoc 2024 TE-Q1 quality lab")
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=CANONICAL_REGISTRY_PATH,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(f"result={run(args.corpus_root, args.registry, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
