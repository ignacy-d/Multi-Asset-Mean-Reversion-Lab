"""TE-Q1 post-hoc quality ranking for frozen Trend Exhaustion opportunities.

This module consumes detector output; it neither imports nor changes the frozen
detector.  All learned transforms live inside sklearn pipelines fitted per
expanding chronological fold.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, SplineTransformer

METHODOLOGY_ID = "te-q1-post-hoc-development-v0"
PRIMARY_THRESHOLD = 1.50
CONTINUOUS_FEATURES = (
    "normalized_displacement",
    "efficiency_ratio",
    "extension_atr",
    "retained_progress_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "body_size_atr",
    "m15_range_atr",
    "wick_body_ratio",
    "frozen_atr20",
    "volume_change",
    "distance_to_session_open_minutes",
    "distance_to_session_close_minutes",
)
CATEGORICAL_FEATURES = ("instrument", "direction", "session_label")
FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
FORBIDDEN_FEATURES = frozenset(
    {
        "h15",
        "h30",
        "h60",
        "h120",
        "y60_atr",
        "h60_pips",
        "mfe",
        "mae",
        "time_to_mfe",
        "time_to_mae",
        "month",
        "quarter",
    }
)
FOLDS = tuple((month, tuple(range(1, month))) for month in range(5, 13))
MODEL_ORDER = ("huber", "quantile", "additive")


@dataclass(frozen=True)
class PromotionRules:
    """Preregistered before execution; changing these changes methodology ID."""

    minimum_oos_n: int = 1_000
    minimum_top20_n: int = 200
    minimum_top20_mean_pips: float = 0.0
    minimum_top20_median_pips: float = 0.0
    minimum_top20_pf: float = 1.20
    minimum_top20_lift_atr: float = 0.05
    minimum_spearman: float = 0.05
    minimum_adjacent_ordered_pairs: int = 3
    minimum_positive_instruments: int = 3
    maximum_instrument_share: float = 0.40
    maximum_month_share: float = 0.30
    maximum_quarter_share: float = 0.60
    minimum_supporting_families: int = 2


PROMOTION_RULES = PromotionRules()


def _utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("signal_timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def build_feature_row(event: Mapping[str, Any]) -> dict[str, Any]:
    """Build one causal row from one frozen primary event.

    Absolute volume is deliberately excluded: quote activity is instrument-local
    and not cross-instrument comparable.  Only the already-causal relative
    ``volume_change`` emitted at signal time may be used.
    """
    if not math.isclose(float(event["displacement_threshold"]), PRIMARY_THRESHOLD):
        raise ValueError("TE-Q1 accepts only primary 1.50 ATR opportunities")
    atr = float(event["frozen_atr20"])
    if not math.isfinite(atr) or atr <= 0:
        raise ValueError("frozen_atr20 must be positive and finite")
    row: dict[str, Any] = {
        "event_id": str(event["event_id"]),
        "signal_timestamp": _utc(event["signal_timestamp"]),
        "normalized_displacement": event.get("normalized_displacement"),
        "efficiency_ratio": event.get("efficiency_ratio"),
        "frozen_atr20": atr,
        "wick_body_ratio": event.get("wick_body_ratio"),
        "volume_change": event.get("volume_change"),
        "distance_to_session_open_minutes": event.get(
            "distance_to_session_open_minutes"
        ),
        "distance_to_session_close_minutes": event.get(
            "distance_to_session_close_minutes"
        ),
        "instrument": event.get("instrument"),
        "direction": event.get("direction"),
        "session_label": event.get("session_label"),
    }
    for source, target in (
        ("exhaustion_extension", "extension_atr"),
        ("retained_progress", "retained_progress_atr"),
        ("upper_wick", "upper_wick_atr"),
        ("lower_wick", "lower_wick_atr"),
        ("body_size", "body_size_atr"),
        ("m15_range", "m15_range_atr"),
    ):
        value = event.get(source)
        row[target] = None if value is None else float(value) / atr
    # Outcomes are carried for evaluation, never supplied to a model pipeline.
    signed_h60 = float(event["signed_h60_price_movement"])
    row.update(
        y60_atr=signed_h60 / atr,
        h60_pips=float(event["signed_h60_pips"]),
        h15=event.get("h15"),
        h30=event.get("h30"),
        h120=event.get("h120"),
        h60_positive=signed_h60 > 0,
        mfe=event.get("mfe"),
        mae=event.get("mae"),
    )
    return row


def build_dataset(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [build_feature_row(event) for event in events]
    ids = [row["event_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("each primary opportunity must be unique")
    return sorted(rows, key=lambda row: (row["signal_timestamp"], row["event_id"]))


def walk_forward_splits(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[list[int], list[int]]]:
    splits = []
    for test_month, train_months in FOLDS:
        train = [
            i
            for i, row in enumerate(rows)
            if row["signal_timestamp"].month in train_months
        ]
        test = [
            i
            for i, row in enumerate(rows)
            if row["signal_timestamp"].month == test_month
        ]
        if (
            train
            and test
            and max(rows[i]["signal_timestamp"] for i in train)
            >= min(rows[i]["signal_timestamp"] for i in test)
        ):
            raise ValueError("chronological fold boundary violated")
        splits.append((train, test))
    return splits


def _matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[row.get(name) for name in FEATURES] for row in rows], dtype=object
    )


def _preprocessor(*, splines: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median"))
    ]
    if splines:
        numeric_steps.append(("splines", SplineTransformer(n_knots=5, degree=3)))
    numeric_steps.append(("scale", RobustScaler()))
    return ColumnTransformer(
        [
            (
                "continuous",
                Pipeline(numeric_steps),
                list(range(len(CONTINUOUS_FEATURES))),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(range(len(CONTINUOUS_FEATURES), len(FEATURES))),
            ),
        ]
    )


def model_pipeline(name: str) -> Pipeline:
    if name == "huber":
        estimator: Any = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1_000)
        splines = False
    elif name == "additive":
        estimator, splines = Ridge(alpha=10.0), True
    elif name.startswith("quantile_"):
        estimator = QuantileRegressor(quantile=float(name.rsplit("_", 1)[1]), alpha=0.1)
        splines = False
    else:
        raise ValueError(f"unknown model: {name}")
    return Pipeline(
        [("preprocess", _preprocessor(splines=splines)), ("model", estimator)]
    )


def walk_forward_predictions(
    rows: Sequence[Mapping[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Return test-only predictions. Quantile score is the mean of q25/q50/q75."""
    output: list[dict[str, Any]] = []
    names = (
        ("quantile_0.25", "quantile_0.50", "quantile_0.75")
        if model == "quantile"
        else (model,)
    )
    for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(rows), start=1):
        if not train_idx or not test_idx:
            continue
        train, test = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
        x_train, x_test = _matrix(train), _matrix(test)
        y_train = np.asarray([row["y60_atr"] for row in train], dtype=float)
        components = []
        for name in names:
            pipeline = model_pipeline(name)
            pipeline.fit(x_train, y_train)
            components.append(pipeline.predict(x_test))
        for row, score in zip(test, np.mean(components, axis=0), strict=True):
            output.append(
                {
                    **row,
                    "model": model,
                    "fold": fold,
                    "score": float(score),
                    "is_oos": True,
                }
            )
    return output


def _pf(values: Sequence[float]) -> float:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return (
        math.inf if losses == 0 and gains > 0 else (gains / losses if losses else 0.0)
    )


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    y = [float(row["y60_atr"]) for row in rows]
    pips = [float(row["h60_pips"]) for row in rows]
    return {
        "n": len(rows),
        "mean_y60_atr": float(np.mean(y)) if y else None,
        "median_y60_atr": float(np.median(y)) if y else None,
        "mean_h60_pips": float(np.mean(pips)) if pips else None,
        "median_h60_pips": float(np.median(pips)) if pips else None,
        "pf": _pf(pips),
        "win_rate": sum(value > 0 for value in pips) / len(pips) if pips else None,
        "mfe_mean": float(np.mean([r["mfe"] for r in rows if r.get("mfe") is not None]))
        if any(r.get("mfe") is not None for r in rows)
        else None,
        "mae_mean": float(np.mean([r["mae"] for r in rows if r.get("mae") is not None]))
        if any(r.get("mae") is not None for r in rows)
        else None,
    }


def evaluate(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if any(not row.get("is_oos") for row in predictions):
        raise ValueError("evaluation accepts walk-forward OOS predictions only")
    ordered = sorted(predictions, key=lambda row: (row["score"], row["event_id"]))
    cuts = np.linspace(0, len(ordered), 6, dtype=int)
    buckets = {f"q{i + 1}": _summary(ordered[cuts[i] : cuts[i + 1]]) for i in range(5)}
    top10 = ordered[math.floor(0.9 * len(ordered)) :]
    buckets["top10"] = _summary(top10)
    means = [buckets[f"q{i}"]["mean_y60_atr"] for i in range(1, 6)]
    rho = spearmanr(
        [row["score"] for row in ordered], [row["y60_atr"] for row in ordered]
    ).statistic
    baseline = _summary(ordered)
    return {
        "baseline": baseline,
        "buckets": buckets,
        "spearman": float(rho),
        "top_minus_bottom_atr": means[-1] - means[0],
        "top20_lift_atr": means[-1] - baseline["mean_y60_atr"],
        "top10_lift_atr": buckets["top10"]["mean_y60_atr"] - baseline["mean_y60_atr"],
        "ordered_adjacent_pairs": sum(a <= b for a, b in pairwise(means)),
    }


def add_stability(
    report: dict[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> None:
    """Attach top-quintile stability and concentration diagnostics in place."""
    ordered = sorted(predictions, key=lambda row: (row["score"], row["event_id"]))
    selected = ordered[math.floor(0.8 * len(ordered)) :]
    dimensions = {
        "instrument": lambda row: str(row["instrument"]),
        "month": lambda row: f"{row['signal_timestamp'].month:02d}",
        "quarter": lambda row: f"Q{(row['signal_timestamp'].month - 1) // 3 + 1}",
        "direction": lambda row: str(row["direction"]),
        "session": lambda row: str(row["session_label"]),
    }
    report["stability"], report["concentration"] = {}, {}
    for dimension, key in dimensions.items():
        groups = {
            value: [row for row in selected if key(row) == value]
            for value in sorted({key(row) for row in selected})
        }
        report["stability"][dimension] = {
            value: _summary(group) for value, group in groups.items()
        }
        report["concentration"][dimension] = max(
            (len(group) / len(selected) for group in groups.values()), default=1.0
        )


def block_bootstrap(
    predictions: Sequence[Mapping[str, Any]],
    *,
    samples: int = 2_000,
    seed: int = 20240901,
) -> dict[str, list[float]]:
    """Month-block percentile intervals for the preregistered quantities."""
    rng = np.random.default_rng(seed)
    months = sorted({row["signal_timestamp"].month for row in predictions})
    by_month = {
        month: [row for row in predictions if row["signal_timestamp"].month == month]
        for month in months
    }
    values: dict[str, list[float]] = {
        "top20_mean_pips": [],
        "top20_mean_atr": [],
        "spread_atr": [],
    }
    for _ in range(samples):
        draw = [
            row
            for month in rng.choice(months, len(months), replace=True)
            for row in by_month[int(month)]
        ]
        report = evaluate(draw)
        values["top20_mean_pips"].append(report["buckets"]["q5"]["mean_h60_pips"])
        values["top20_mean_atr"].append(report["buckets"]["q5"]["mean_y60_atr"])
        values["spread_atr"].append(report["top_minus_bottom_atr"])
    return {
        name: [float(np.quantile(series, 0.025)), float(np.quantile(series, 0.975))]
        for name, series in values.items()
    }


def exploratory_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive, explicitly post-hoc development EDA; never used as a rule."""
    result: dict[str, Any] = {
        "label": "POST-HOC DEVELOPMENT EDA",
        "continuous": {},
        "groups": {},
    }
    for feature in CONTINUOUS_FEATURES:
        available = [
            row
            for row in rows
            if row.get(feature) is not None and math.isfinite(float(row[feature]))
        ]
        values = np.asarray([float(row[feature]) for row in available])
        item: dict[str, Any] = {
            "count": len(available),
            "missing": len(rows) - len(available),
            "quantiles": {
                str(q): float(np.quantile(values, q))
                for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
            }
            if len(values)
            else {},
        }
        for bins in (5, 10):
            item[f"{bins}_bins"] = []
            if len(available) >= bins * 10:
                ranked = sorted(
                    available, key=lambda row: (float(row[feature]), row["event_id"])
                )
                cuts = np.linspace(0, len(ranked), bins + 1, dtype=int)
                item[f"{bins}_bins"] = [
                    _summary(ranked[cuts[i] : cuts[i + 1]]) for i in range(bins)
                ]
        result["continuous"][feature] = item
    for dimension in CATEGORICAL_FEATURES:
        result["groups"][dimension] = {
            value: _summary([row for row in rows if str(row.get(dimension)) == value])
            for value in sorted({str(row.get(dimension)) for row in rows})
        }
    return result


def classify(evaluations: Mapping[str, Mapping[str, Any]]) -> str:
    """Apply frozen conservative promotion gates across all primary families."""
    passing = 0
    usable = 0
    for report in evaluations.values():
        baseline, top = report["baseline"], report["buckets"]["q5"]
        if baseline["n"] < PROMOTION_RULES.minimum_oos_n:
            continue
        usable += 1
        stability = report.get("stability", {})
        positive_instruments = sum(
            v["mean_h60_pips"] > 0 for v in stability.get("instrument", {}).values()
        )
        concentration = report.get("concentration", {})
        passed = (
            top["n"] >= PROMOTION_RULES.minimum_top20_n
            and top["mean_h60_pips"] > PROMOTION_RULES.minimum_top20_mean_pips
            and top["median_h60_pips"] >= PROMOTION_RULES.minimum_top20_median_pips
            and top["pf"] >= PROMOTION_RULES.minimum_top20_pf
            and report["top20_lift_atr"] >= PROMOTION_RULES.minimum_top20_lift_atr
            and report["spearman"] >= PROMOTION_RULES.minimum_spearman
            and report["ordered_adjacent_pairs"]
            >= PROMOTION_RULES.minimum_adjacent_ordered_pairs
            and positive_instruments >= PROMOTION_RULES.minimum_positive_instruments
            and concentration.get("instrument", 1.0)
            <= PROMOTION_RULES.maximum_instrument_share
            and concentration.get("month", 1.0) <= PROMOTION_RULES.maximum_month_share
            and concentration.get("quarter", 1.0)
            <= PROMOTION_RULES.maximum_quarter_share
        )
        passing += passed
    if usable == 0:
        return "INCONCLUSIVE"
    return (
        "PROMOTE_TO_FROZEN_CANDIDATE"
        if passing >= PROMOTION_RULES.minimum_supporting_families
        else "PARK"
    )


def select_model(evaluations: Mapping[str, Mapping[str, Any]]) -> str:
    """Predeclared simplicity rule: first model within fixed similarity margins."""
    eligible = [name for name in MODEL_ORDER if name in evaluations]
    if not eligible:
        raise ValueError("all three primary models must be reported")
    best = max(eligible, key=lambda name: evaluations[name]["top20_lift_atr"])
    for name in eligible:
        report, leader = evaluations[name], evaluations[best]
        if (
            report["top20_lift_atr"] >= leader["top20_lift_atr"] - 0.02 - 1e-12
            and report["spearman"] >= leader["spearman"] - 0.01 - 1e-12
        ):
            return name
    return best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered TE-Q1 on frozen detector JSONL"
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.events.open(encoding="utf-8") as handle:
        rows = build_dataset(json.loads(line) for line in handle if line.strip())
    evaluations = {}
    for name in MODEL_ORDER:
        predictions = walk_forward_predictions(rows, name)
        report = evaluate(predictions)
        add_stability(report, predictions)
        report["month_block_bootstrap_95_ci"] = block_bootstrap(predictions)
        evaluations[name] = report
    document = {
        "methodology_id": METHODOLOGY_ID,
        "promotion_rules": asdict(PROMOTION_RULES),
        "eda": exploratory_diagnostics(rows),
        "models": evaluations,
        "selected_model": select_model(evaluations),
        "classification": classify(evaluations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
