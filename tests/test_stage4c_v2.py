from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mr_lab.cost_profile_builder import build_profile
from mr_lab.stage4c_v2 import (
    AVAILABLE,
    COST_FLOOR_LABEL,
    MISSING_ADJUSTMENT,
    MISSING_CONVERSION,
    MISSING_SPREAD,
    CostProfileV2,
    EconomicAnalysisError,
    ResearchOutcome,
    analyze,
    bootstrap_summary,
    break_even_additional_slippage,
    calendar_month_block_bootstrap,
    scenarios,
)


def event(
    instrument="EURUSD",
    entry=1.1,
    exit=1.1002,
    when=None,
    *,
    entry_delay=0,
    cost_profile_session="overall",
    session_labels=(),
):
    when = when or datetime(2024, 1, 1, tzinfo=UTC)
    return ResearchOutcome(
        "study",
        f"{instrument}-{when}",
        when,
        instrument,
        1,
        when + timedelta(minutes=entry_delay),
        entry,
        when + timedelta(minutes=5),
        exit,
        5,
        2.0,
        cost_profile_session,
        session_labels,
    )


def test_exact_pips_are_calculated_from_prices():
    assert event().gross_pips == pytest.approx(2)
    assert event("USDJPY", 150, 150.02).gross_pips == pytest.approx(2)


def test_point_in_time_entry_ordering():
    assert event(entry_delay=0).entry_timestamp == event().timestamp
    assert event(entry_delay=2).entry_timestamp > event().timestamp
    with pytest.raises(EconomicAnalysisError, match="point-in-time"):
        event(entry_delay=-1)


def profile(instrument, quote_per_usd, *, sessions=("overall",), adjustment=None):
    quote = instrument[-3:]
    adjustment = (0 if quote == "USD" else 0.007) if adjustment is None else adjustment
    return CostProfileV2(
        {
            "schema_version": "stage4c-ftmo-cost-profile-v2",
            "null_session_cost_profile": "overall",
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
                        "profiles": {
                            session: {
                                "authenticated": True,
                                "quote_currency_per_usd": quote_per_usd,
                            }
                            for session in sessions
                        },
                    },
                    "conversion_adjustment": {
                        "authenticated": True,
                        "defined": True,
                        "rate": adjustment,
                    },
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


@pytest.mark.parametrize(
    ("instrument", "expected_rate"),
    [
        ("EURUSD", 0),
        ("USDJPY", 0.007),
        ("USDCAD", 0.007),
        ("USDCHF", 0.007),
        ("EURGBP", 0.007),
    ],
)
def test_metadata_driven_conversion_adjustment(instrument, expected_rate):
    assert profile(instrument, 1).costs(instrument, None, "mean")[2] == expected_rate


def test_commission_conversion_uses_same_explicit_session_and_fails_closed():
    item = profile("USDJPY", 150, sessions=("overall", "asia"))
    item.raw["instruments"]["USDJPY"]["commission_conversion"]["profiles"]["asia"][
        "quote_currency_per_usd"
    ] = 151
    item.raw["instruments"]["USDJPY"]["spread_profiles"]["asia"] = item.raw[
        "instruments"
    ]["USDJPY"]["spread_profiles"]["overall"]
    assert item.costs("USDJPY", "asia", "mean")[1] == pytest.approx(0.755)
    item.raw["instruments"]["USDJPY"]["spread_profiles"]["london"] = item.raw[
        "instruments"
    ]["USDJPY"]["spread_profiles"]["overall"]
    assert item.coverage_status("USDJPY", "london") == MISSING_CONVERSION


def test_coverage_fails_closed_and_has_no_pair_shortcut():
    template = CostProfileV2.load(Path("configs/stage4c-ftmo-cost-profile-v2.json"))
    assert template.coverage_status("USDJPY") == MISSING_SPREAD
    raw = profile("AUDJPY", 150).raw
    raw["instruments"]["AUDJPY"]["commission_conversion"]["profiles"]["overall"][
        "authenticated"
    ] = False
    assert CostProfileV2(raw).coverage_status("AUDJPY") == MISSING_CONVERSION
    with pytest.raises(EconomicAnalysisError, match=MISSING_CONVERSION):
        CostProfileV2(raw).costs("AUDJPY", None, "mean")
    adjusted = profile("AUDJPY", 150).raw
    adjusted["instruments"]["AUDJPY"]["conversion_adjustment"]["authenticated"] = False
    assert CostProfileV2(adjusted).coverage_status("AUDJPY") == MISSING_ADJUSTMENT


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
    floor = report["cost_scenarios"]["mean|0"]
    assert floor["n"] == 4
    assert (
        set(
            (
                "gross_mean_pips",
                "mean_spread_pips",
                "commission_pips",
                "slippage_pips",
                "mean_net_pips",
                "median_net_pips",
                "net_positive_fraction",
                "total_net_pips",
                "break_even_additional_slippage_pips",
            )
        )
        <= floor.keys()
    )
    assert floor["label"] == COST_FLOOR_LABEL
    assert report["cost_floor"]["net_mean_pips"] == floor["mean_net_pips"]


def test_bootstrap_summary_uses_same_replicates_and_frozen_type7_percentiles():
    summary = bootstrap_summary((0.0, 10.0, 20.0, 30.0), seed=41)
    assert summary | {"p2_5": 0.75, "p97_5": 29.25} == {
        "replicates": 4,
        "seed": 41,
        "bootstrap_mean": 15.0,
        "p2_5": 0.75,
        "median": 15.0,
        "p97_5": 29.25,
        "lower_2_5pct_gt_zero": True,
    }
    assert summary["p2_5"] == pytest.approx(0.75)
    assert summary["p97_5"] == pytest.approx(29.25)
    rows = [event(exit=1.1001), event(exit=1.1003)]
    report = analyze(rows, bootstrap_replicates=17, seed=123)
    replicates = report["bootstrap_event_weighted_mean_pips"]
    assert report["bootstrap_summary"] == bootstrap_summary(replicates, seed=123)


def test_diagnostic_labels_do_not_select_cost_session():
    item = profile("EURUSD", 1, sessions=("overall", "asia"))
    item.raw["instruments"]["EURUSD"]["spread_profiles"]["asia"] = {
        **item.raw["instruments"]["EURUSD"]["spread_profiles"]["overall"],
        "mean_pips": 9,
    }
    row = event(session_labels=("asia", "london"), cost_profile_session="overall")
    assert analyze([row], item, bootstrap_replicates=1)["cost_scenarios"]["mean|0"][
        "mean_spread_pips"
    ] == pytest.approx(0.2)


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
    assert (
        result["commission_conversion"]["profiles"]["asia"]["quote_currency_per_usd"]
        == 1
    )


def test_builder_explicit_session_references_and_non_usd_adjustment(tmp_path):
    direct = tmp_path / "usdjpy.csv"
    direct.write_text("timestamp,bid,ask\n2024-01-02T01:00:00Z,149.99,150.01\n")
    audjpy = tmp_path / "audjpy.csv"
    audjpy.write_text("timestamp,bid,ask\n2024-01-02T01:00:00Z,97.99,98.01\n")
    built = build_profile(
        {"AUDJPY": audjpy}, {"USDJPY": direct}, non_usd_adjustment_rate=0.007
    )["instruments"]["AUDJPY"]
    assert built["commission_conversion"]["profiles"]["asia"][
        "quote_currency_per_usd"
    ] == pytest.approx(150)
    assert built["conversion_adjustment"]["rate"] == 0.007


def test_builder_eurgbp_requires_explicit_gbpusd_reference(tmp_path):
    eurgbp = tmp_path / "eurgbp.csv"
    eurgbp.write_text("timestamp,bid,ask\n2024-01-02T09:00:00Z,.85,.8502\n")
    assert (
        build_profile({"EURGBP": eurgbp})["instruments"]["EURGBP"][
            "commission_conversion"
        ]["profiles"]
        == {}
    )
    gbpusd = tmp_path / "gbpusd.csv"
    gbpusd.write_text("timestamp,bid,ask\n2024-01-02T09:00:00Z,1.249,1.251\n")
    built = build_profile({"EURGBP": eurgbp}, {"GBPUSD": gbpusd})["instruments"][
        "EURGBP"
    ]
    assert built["commission_conversion"]["profiles"]["london"][
        "quote_currency_per_usd"
    ] == pytest.approx(0.8)


def test_builder_requires_explicit_files(tmp_path):
    (tmp_path / "unrequested.csv").write_text("not read")
    assert build_profile({})["instruments"] == {}


def test_available_constant():
    assert profile("EURUSD", 1).coverage_status("EURUSD") == AVAILABLE
