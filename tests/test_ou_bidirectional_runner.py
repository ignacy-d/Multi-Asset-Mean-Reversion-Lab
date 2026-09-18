import math
import statistics

import pytest

from mr_lab.ou_bidirectional_runner import (
    StreamingMetrics,
    _with_combined,
    pair_orientation,
    usd_exposure,
)


def row(value, mfe, mae):
    return {
        "complete": True,
        "gross_return_pips_adverse_first": value,
        "mfe_pips_certain": mfe,
        "mae_pips_certain": mae,
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
