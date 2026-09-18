import json
import math
import statistics
from pathlib import Path

import pytest

import mr_lab.ou_bidirectional_runner as runner
from mr_lab.ou_bidirectional_runner import (
    StreamingMetrics,
    _consume,
    _with_combined,
    pair_orientation,
    usd_exposure,
)
from mr_lab.stage4c import CostProfile, transform_trade


def row(value, mfe, mae):
    return {
        "complete": True,
        "gross_return_pips_adverse_first": value,
        "mfe_pips_certain": mfe,
        "mae_pips_certain": mae,
    }


def trade_row(direction, value):
    return row(value, abs(value) + 1, abs(value) / 2) | {
        "instrument": "EURUSD",
        "session": "london",
        "direction": direction,
        "candidate_event_id": f"candidate-{direction}",
        "gross_return_pips_favorable_first": value,
    }


def test_streaming_metrics_equal_list_reference_exactly():
    rows = [row(-2.0, 1, 3), row(1.0, 2, 1), row(4.0, 5, 0.5), row(0, 1, 1)]
    stream = StreamingMetrics()
    for item in rows:
        stream.add(item)
    result = stream.result()
    values = [-2.0, 1.0, 4.0, 0.0]
    assert result == {
        "complete_trade_row_count": 4,
        "incomplete_trade_row_count": 0,
        "mean_pips": statistics.fmean(values),
        "median_pips": statistics.median(values),
        "win_rate": 0.5,
        "profit_factor": 2.5,
        "mean_mfe": 2.25,
        "mean_mae": 1.375,
    }


def test_combined_is_direct_union_and_partition_counts_hold():
    long = StreamingMetrics()
    short = StreamingMetrics()
    long.add(row(2, 3, 1))
    short.add(row(-1, 1, 2))
    metrics = _with_combined(
        {"mr_long_baseline_gross": long, "mr_short_baseline_gross": short}
    )
    combined = metrics["mr_combined_baseline_gross"].result()
    assert combined["complete_trade_row_count"] == 2
    assert combined["complete_trade_row_count"] == len(long.values) + len(short.values)
    assert combined["mean_pips"] == 0.5
    assert combined["profit_factor"] == 2.0


def test_all_combined_variants_are_direct_long_short_unions():
    bundle = {}
    for suffix in ("baseline_gross", "ou_gross", "baseline_net", "ou_net"):
        long = StreamingMetrics()
        short = StreamingMetrics()
        for value in (-3.0, 1.0):
            long.add(row(value, 4, 2))
        short.add(row(2.0, 3, 1))
        bundle[f"mr_long_{suffix}"] = long
        bundle[f"mr_short_{suffix}"] = short
    metrics = _with_combined(bundle)
    for suffix in ("baseline_gross", "ou_gross", "baseline_net", "ou_net"):
        combined = metrics[f"mr_combined_{suffix}"].result()
        assert combined["complete_trade_row_count"] == (
            len(metrics[f"mr_long_{suffix}"].values)
            + len(metrics[f"mr_short_{suffix}"].values)
        )
        assert combined["mean_pips"] == statistics.fmean((-3.0, 1.0, 2.0))
        assert combined["median_pips"] == 1.0
        assert combined["profit_factor"] == 1.0


def test_cost_overlay_matches_direct_stage4c_for_long_and_short():
    profile = CostProfile.load(Path("configs/stage4c-ftmo-cost-profile-v1.json"))
    bundle = runner._empty_bundle(True)
    for direction, gross in (("LONG", 4.0), ("SHORT", -2.0)):
        item = trade_row(direction, gross)
        expected = transform_trade(item, profile, "mean", 0.0)
        _consume(bundle, "baseline", item, profile, True)
        assert list(bundle[f"mr_{direction.lower()}_baseline_net"].values) == [
            expected["net_pips_adverse_first"]
        ]


def test_bounded_replay_scientific_output_is_deterministic(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text('{"bounded":"identity"}')
    authenticated = {
        "registry_schema_version": "fx-universe-2024-registry-v1",
        "registry_id": "sha256:bounded-registry",
        "instruments": {
            "EURUSD": {
                "corpus_path": "/explicit/2024/EURUSD",
                "corpus_id": "sha256:bounded-corpus",
                "assembled_dataset_id": "sha256:bounded-dataset",
            }
        },
    }
    monkeypatch.setattr(runner, "_load_stage4b_registry", lambda _: authenticated)

    def bounded_run(*args, eligibility_filter=None, trade_row_consumer=None, **kwargs):
        multiplier = 2.0 if eligibility_filter else 1.0
        for direction, value in (("LONG", 3.0), ("SHORT", -1.0)):
            trade_row_consumer(trade_row(direction, value * multiplier))

    monkeypatch.setattr(runner, "run", bounded_run)
    outputs = []
    for name in ("first", "second"):
        target = runner.replay(
            registry,
            Path("configs/stage4c-ftmo-cost-profile-v1.json"),
            tmp_path / name,
        )
        outputs.append(json.loads(target.read_text()))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("pair", "position", "family", "long", "short"),
    [
        ("EURUSD", "QUOTE", "XXXUSD", "SHORT_USD", "LONG_USD"),
        ("USDJPY", "BASE", "USDXXX", "LONG_USD", "SHORT_USD"),
        ("EURGBP", "NONE", "non-USD cross", "NO_USD", "NO_USD"),
    ],
)
def test_usd_orientation_is_descriptive(pair, position, family, long, short):
    assert pair_orientation(pair)[2:] == (position, family)
    assert usd_exposure(pair, "LONG") == long
    assert usd_exposure(pair, "SHORT") == short


def test_profit_factor_without_losses_is_explicit_infinity():
    metrics = StreamingMetrics()
    metrics.add(row(1, 1, 0))
    assert math.isinf(metrics.result()["profit_factor"])
