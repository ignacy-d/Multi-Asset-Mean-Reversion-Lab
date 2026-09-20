from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mr_lab.cost_profile_builder import build_profile
from mr_lab.stage4c_v2 import (
    AVAILABLE,
    MISSING_CONVERSION,
    MISSING_SPREAD,
    CostProfileV2,
    EconomicAnalysisError,
    ResearchOutcome,
    analyze,
    break_even_additional_slippage,
    calendar_month_block_bootstrap,
    scenarios,
)


def event(instrument="EURUSD", entry=1.1, exit=1.1002, when=None):
    when = when or datetime(2024, 1, 1, tzinfo=UTC)
    return ResearchOutcome(
        "study",
        f"{instrument}-{when}",
        when,
        instrument,
        1,
        when,
        entry,
        when + timedelta(minutes=5),
        exit,
        5,
        2.0,
    )


def test_exact_pips_are_calculated_from_prices():
    assert event().gross_pips == pytest.approx(2)
    assert event("USDJPY", 150, 150.02).gross_pips == pytest.approx(2)


def profile(instrument, quote_per_usd):
    return CostProfileV2(
        {
            "schema_version": "stage4c-ftmo-cost-profile-v2",
            "commission_usd_round_turn": 5,
            "instruments": {
                instrument: {
                    "spread_profiles": {
                        "overall": {
                            "authenticated": True,
                            "mean_pips": 0.2,
                            "p75_pips": 0.3,
                            "p90_pips": 0.4,
                            "p95_pips": 0.5,
                        }
                    },
                    "commission_conversion": {
                        "authenticated": True,
                        "quote_currency_per_usd": quote_per_usd,
                    },
                    "conversion_adjustment": {"defined": True, "rate": 0},
                }
            },
        }
    )


@pytest.mark.parametrize(
    ("instrument", "rate", "expected"),
    [
        ("EURUSD", 1, 0.5),
        ("USDJPY", 150, 0.75),
        ("USDCAD", 1.4, 0.7),
        ("USDCHF", 0.9, 0.45),
        ("EURGBP", 0.8, 0.4),
    ],
)
def test_generalized_commission(instrument, rate, expected):
    assert profile(instrument, rate).costs(instrument, None, "mean")[
        1
    ] == pytest.approx(expected)


def test_coverage_fails_closed_and_has_no_pair_shortcut():
    template = CostProfileV2.load(Path("configs/stage4c-ftmo-cost-profile-v2.json"))
    assert template.coverage_status("USDJPY") == MISSING_SPREAD
    raw = profile("AUDJPY", 150).raw
    raw["instruments"]["AUDJPY"]["commission_conversion"]["authenticated"] = False
    assert CostProfileV2(raw).coverage_status("AUDJPY") == MISSING_CONVERSION
    with pytest.raises(EconomicAnalysisError, match=MISSING_CONVERSION):
        CostProfileV2(raw).costs("AUDJPY", None, "mean")


def test_scenarios_bootstrap_and_event_weighting():
    assert len(scenarios()) == 16
    rows = [
        event(when=datetime(2024, 1, day, tzinfo=UTC), exit=1.1001) for day in (1, 2, 3)
    ]
    rows.append(event(when=datetime(2024, 2, 1, tzinfo=UTC), exit=1.1009))
    assert calendar_month_block_bootstrap(
        rows, replicates=20, seed=7
    ) == calendar_month_block_bootstrap(rows, replicates=20, seed=7)
    values = set(calendar_month_block_bootstrap(rows, replicates=100, seed=3))
    assert any(
        value == pytest.approx(3) for value in values
    )  # one Jan block + one Feb block: event weighted
    report = analyze(rows, profile("EURUSD", 1), bootstrap_replicates=2)
    assert len(report["cost_scenarios"]) == 16


def test_break_even_zero_when_already_unprofitable():
    assert break_even_additional_slippage([-1], 0.2, 0.5, 0.007) == 0


def test_builder_provenance_statistics_and_clamp(tmp_path):
    csv_path = tmp_path / "explicit.csv"
    csv_path.write_text(
        "timestamp,bid,ask\n2024-01-02T09:00:00Z,1.0002,1.0001\n2024-01-02T09:01:00Z,1.0,1.0002\n"
    )
    result = build_profile({"EURUSD": csv_path})["instruments"]["EURUSD"]
    source, overall = result["raw_source"], result["spread_profiles"]["overall"]
    assert source["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert source["row_count"] == 2 and source["negative_spread_rows_raw"] == 1
    assert source["first_timestamp_utc"] == "2024-01-02T09:00:00Z"
    assert overall["mean_pips"] == pytest.approx(1)  # -1 clamped, +2 retained
    assert set(result["spread_profiles"]) <= {"overall", "asia", "london", "new_york"}


def test_builder_requires_explicit_files(tmp_path):
    (tmp_path / "unrequested.csv").write_text("not read")
    assert build_profile({})["instruments"] == {}


def test_available_constant():
    assert profile("EURUSD", 1).coverage_status("EURUSD") == AVAILABLE
