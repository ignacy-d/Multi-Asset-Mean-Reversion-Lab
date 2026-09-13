from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from mr_lab.trend_exhaustion_quality import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURES,
    MODEL_ORDER,
    PROMOTION_RULES,
    add_stability,
    build_dataset,
    build_feature_row,
    classify,
    evaluate,
    model_pipeline,
    select_model,
    walk_forward_predictions,
    walk_forward_splits,
)


def event(month: int, number: int, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "event_id": f"e-{month}-{number}",
        "signal_timestamp": datetime(2024, month, min(number + 1, 28), tzinfo=UTC),
        "displacement_threshold": 1.5,
        "normalized_displacement": 1.5 + number / 100,
        "efficiency_ratio": number / 10,
        "exhaustion_extension": 2.0,
        "retained_progress": 1.0,
        "upper_wick": 0.5,
        "lower_wick": 0.25,
        "body_size": 0.75,
        "m15_range": 2.5,
        "wick_body_ratio": 1.0,
        "frozen_atr20": 2.0,
        "volume_change": None if number % 4 == 0 else number / 20,
        "distance_to_session_open_minutes": number,
        "distance_to_session_close_minutes": 60 - number,
        "instrument": "EURUSD" if number % 2 else "USDJPY",
        "direction": "long" if number % 2 else "short",
        "session_label": "London" if number % 3 else "New York",
        "signed_h60_price_movement": number / 10 - 0.5,
        "signed_h60_pips": number - 5,
        "mfe": number,
        "mae": -number,
    }
    value.update(changes)
    return value


def rows() -> list[dict[str, object]]:
    return build_dataset(
        event(month, number) for month in range(1, 13) for number in range(12)
    )


def test_primary_only_unique_and_target_contract() -> None:
    row = build_feature_row(event(1, 1))
    assert row["extension_atr"] == 1.0
    assert row["y60_atr"] == pytest.approx(-0.2)
    assert not set(FEATURES) & {"y60_atr", "h60_pips", "mfe", "mae"}
    with pytest.raises(ValueError, match="primary"):
        build_feature_row(event(1, 1, displacement_threshold=1.25))
    with pytest.raises(ValueError, match="unique"):
        build_dataset([event(1, 1), event(1, 1)])


def test_feature_causality_and_prefix_invariance() -> None:
    prefix = [event(1, number) for number in range(4)]
    complete = prefix + [event(2, number) for number in range(4)]
    assert build_dataset(prefix) == build_dataset(complete)[: len(prefix)]
    row = build_feature_row(event(1, 1, h120=999, future_volatility=999))
    assert row["h120"] == 999 and "future_volatility" not in row
    assert "h120" not in FEATURES


def test_frozen_detector_is_not_imported_or_modified() -> None:
    source = Path("src/mr_lab/trend_exhaustion_quality.py").read_text()
    assert "from mr_lab.trend_exhaustion" not in source
    assert "PRIMARY_THRESHOLD = 1.50" in source


def test_walk_forward_boundaries() -> None:
    data = rows()
    splits = walk_forward_splits(data)
    assert len(splits) == 8
    for expected_month, (train, test) in enumerate(splits, 5):
        assert {data[index]["signal_timestamp"].month for index in test} == {
            expected_month
        }
        assert max(data[index]["signal_timestamp"] for index in train) < min(
            data[index]["signal_timestamp"] for index in test
        )


def test_training_only_preprocessing_and_categories() -> None:
    data = rows()
    train = data[:24]
    test = [{**data[24], "instrument": "UNSEEN", "volume_change": None}]
    pipe = model_pipeline("huber")
    x_train = np.asarray(
        [[row.get(name) for name in FEATURES] for row in train], dtype=object
    )
    x_test = np.asarray(
        [[row.get(name) for name in FEATURES] for row in test], dtype=object
    )
    pipe.fit(x_train, [row["y60_atr"] for row in train])
    assert np.isfinite(pipe.predict(x_test)).all()
    assert len(CONTINUOUS_FEATURES) == 13 and len(CATEGORICAL_FEATURES) == 3


def test_predictions_are_deterministic_and_oos_only() -> None:
    data = rows()
    first = walk_forward_predictions(data, "additive")
    second = walk_forward_predictions(data, "additive")
    assert [row["score"] for row in first] == pytest.approx(
        [row["score"] for row in second]
    )
    assert all(row["is_oos"] and row["signal_timestamp"].month >= 5 for row in first)
    with pytest.raises(ValueError, match="OOS"):
        evaluate([{**first[0], "is_oos": False}])


def test_score_buckets_stability_and_selection() -> None:
    predictions = walk_forward_predictions(rows(), "huber")
    report = evaluate(predictions)
    assert set(report["buckets"]) == {"q1", "q2", "q3", "q4", "q5", "top10"}
    assert sum(
        report["buckets"][key]["n"] for key in ("q1", "q2", "q3", "q4", "q5")
    ) == len(predictions)
    add_stability(report, predictions)
    assert set(report["stability"]) == {
        "instrument",
        "month",
        "quarter",
        "direction",
        "session",
    }
    reports = {
        name: {**report, "top20_lift_atr": index / 100, "spearman": 0.02}
        for index, name in enumerate(MODEL_ORDER)
    }
    assert select_model(reports) == "huber"


def test_preregistered_classification_is_conservative() -> None:
    base = {"n": PROMOTION_RULES.minimum_oos_n, "mean_y60_atr": 0.0}
    top = {
        "n": PROMOTION_RULES.minimum_top20_n,
        "mean_h60_pips": 1.0,
        "median_h60_pips": 0.0,
        "pf": 1.3,
    }
    passing = {
        "baseline": base,
        "buckets": {"q5": top},
        "top20_lift_atr": 0.06,
        "spearman": 0.06,
        "ordered_adjacent_pairs": 3,
        "stability": {"instrument": {str(i): {"mean_h60_pips": 1.0} for i in range(3)}},
        "concentration": {"instrument": 0.4, "month": 0.3, "quarter": 0.6},
    }
    assert (
        classify({"huber": passing, "quantile": passing})
        == "PROMOTE_TO_FROZEN_CANDIDATE"
    )
    assert classify({"huber": passing}) == "PARK"
    assert classify({}) == "INCONCLUSIVE"


def test_sealed_year_is_absent_from_implementation() -> None:
    sealed = str(2024 + 1)
    source = Path("src/mr_lab/trend_exhaustion_quality.py").read_text()
    assert sealed not in source
