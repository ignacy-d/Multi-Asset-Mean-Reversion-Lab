from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mr_lab.research import Direction
from mr_lab.stage4a import DirectionalPathDiagnostic, ForwardOutcome
from mr_lab.trend_exhaustion import PRIMARY_THRESHOLD, TrendExhaustionEvent
from mr_lab.trend_exhaustion_quality import (
    CATEGORICAL_PREDICTORS,
    CONTINUOUS_PREDICTORS,
    FOLDS,
    MODEL_ORDER,
    PREDICTORS,
    QualityLabError,
    classify,
    quality_row,
    select_model,
)

CANONICAL = {  # Canonical PR #76 byte identities.
    "src/mr_lab/trend_exhaustion.py": "16e9e00b881d994479b446f70c15c90356b9145c75f6a99dc4c7dc9b2f573a5a",  # noqa: E501
    "src/mr_lab/trend_exhaustion_stage0.py": "3cb83bd9770333203450dc00f81a9524c6de49d72003215c71f204805fe9c63f",  # noqa: E501
    "src/mr_lab/trend_exhaustion_stage0_runner.py": "f93e46e2a6a02ba51006bdab92f79ffbab011741e3c10492f25e6466e3a282e3",  # noqa: E501
    "docs/trend-exhaustion-stage0-v1-preregistration.md": "dfab081069418bedcf003d7db7ca854cfacd179b1194ba797e5909ebea237a04",  # noqa: E501
}


def event(
    timestamp: datetime = datetime(2024, 5, 1, tzinfo=UTC),
) -> TrendExhaustionEvent:
    return TrendExhaustionEvent(
        "id",
        "trend-exhaustion",
        "spec",
        "EURUSD",
        timestamp,
        Direction.SHORT,
        PRIMARY_THRESHOLD,
        1,
        0.002,
        0.004,
        2.0,
        0.8,
        0.001,
        0.0001,
        0.0002,
        0.0001,
        0.0003,
        1.0,
        0.0006,
        100.0,
        5.0,
        "london",
        60,
        120,
        "session-spec",
    )


def path(
    timestamp: datetime = datetime(2024, 5, 1, tzinfo=UTC),
) -> DirectionalPathDiagnostic:
    return DirectionalPathDiagnostic(
        "EURUSD",
        timestamp,
        Direction.SHORT,
        1.1,
        True,
        (),
        (ForwardOutcome(60, 0.001, 0.001, 10.0, 10.0),),
        0.0005,
        5.0,
        5.0,
        0.0015,
        15.0,
        15.0,
        20,
        30,
    )


def test_canonical_v1_files_are_unchanged() -> None:
    for name, digest in CANONICAL.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest


def test_actual_event_contract_maps_session_fields_and_typed_h60() -> None:
    row = quality_row(event(), path())
    assert row.minutes_from_session_open == 60
    assert row.minutes_to_session_close == 120
    assert row.y60_atr == pytest.approx(0.5)
    assert row.h60_pips == 10.0


def test_predictors_are_causal_and_exclude_raw_atr_volume_and_outcomes() -> None:
    assert CONTINUOUS_PREDICTORS == (
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
    assert CATEGORICAL_PREDICTORS == ("instrument", "direction", "session_label")
    assert not (
        {
            "frozen_atr20",
            "volume_change",
            "y60_atr",
            "h60_pips",
            "mfe_pips",
            "mae_pips",
            "month",
            "quarter",
        }
        & set(PREDICTORS)
    )


@pytest.mark.parametrize(
    "bad", [datetime(2023, 12, 31, tzinfo=UTC), datetime(2024, 1, 1)]
)
def test_development_year_and_utc_guard_fails_closed(bad: datetime) -> None:
    with pytest.raises(QualityLabError):
        quality_row(event(bad), path(bad))


def test_only_primary_threshold_and_matching_unique_identity_are_accepted() -> None:
    with pytest.raises(QualityLabError, match="primary threshold"):
        quality_row(replace(event(), displacement_threshold=1.25), path())
    with pytest.raises(QualityLabError, match="identities"):
        quality_row(event(), replace(path(), instrument="GBPUSD"))


def test_frozen_chronological_fold_boundaries() -> None:
    assert tuple((tuple(range(1, month)), month) for month in range(5, 13)) == FOLDS
    assert all(max(train) < test and test not in train for train, test in FOLDS)


def report(*, n=1000, lift=0.05, rho=0.05, passes=True):
    mean = 1.0 if passes else -1.0
    return {
        "oos_baseline": {"n": n},
        "top_20": {
            "n": 200,
            "mean_h60_pips": mean,
            "median_h60_pips": 0,
            "profit_factor": 1.2,
        },
        "top_20_lift": lift,
        "spearman": rho,
        "ordered_adjacent_quintile_pairs": 3,
        "top_20_by_instrument": {str(i): {"mean_h60_pips": 1} for i in range(3)},
        "stability": {
            key: {"a": {"share": share}}
            for key, share in (
                ("instrument", 0.4),
                ("test_month", 0.3),
                ("quarter", 0.6),
            )
        },
    }


def test_classification_requires_two_independent_families_and_sample_gate() -> None:
    reports = {name: report(passes=index < 2) for index, name in enumerate(MODEL_ORDER)}
    assert classify(reports) == "PROMOTE_TO_FROZEN_CANDIDATE"
    reports["quantile"] = report(passes=False)
    assert classify(reports) == "PARK"
    reports["huber"] = report(n=999)
    assert classify(reports) == "INCONCLUSIVE"


def test_selection_prefers_simplest_within_both_tolerances() -> None:
    reports = {
        "huber": report(lift=0.09, rho=0.095),
        "quantile": report(lift=0.10, rho=0.10),
        "additive_splines": report(lift=0.11, rho=0.105),
    }
    assert select_model(reports) == "huber"
