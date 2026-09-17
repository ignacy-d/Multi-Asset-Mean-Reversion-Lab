import json
from pathlib import Path

import pytest

from mr_lab.fx_replay import (
    _comparison_metrics,
    _cost_status,
    _is_mr_comparable_row,
    _metrics,
    _mr_comparable_baseline,
)
from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b_runner import _load_stage4b_registry
from mr_lab.stage4c import CostProfile


def test_authenticated_registry_is_exact_nine_pair_contract():
    registry = _load_stage4b_registry(Path("configs/fx-universe-2024-registry-v1.json"))
    assert registry["registry_schema_version"] == "fx-universe-2024-registry-v1"
    assert len(registry["instruments"]) == 9
    assert all(
        entry["verification_status"] == "verified"
        for entry in registry["instruments"].values()
    )


def test_metrics_are_deterministic_and_keep_incomplete_out():
    complete = {
        "complete": True,
        "gross_return_pips_adverse_first": 2.0,
        "mfe_pips_certain": 3.0,
        "mae_pips_certain": 1.0,
    }
    result = _metrics(
        [
            complete,
            complete | {"gross_return_pips_adverse_first": -1.0},
            {"complete": False},
        ]
    )
    assert result["n_trades"] == 2
    assert result["mean_pips"] == 0.5
    assert result["win_rate"] == 0.5
    assert result["profit_factor"] == 2.0


def test_missing_new_pair_cost_profile_fails_closed():
    profile = CostProfile.load(Path("configs/stage4c-ftmo-cost-profile-v1.json"))
    assert _cost_status(profile, "USDCAD", [{"session": "london"}]) == (
        "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
    )
    assert _cost_status(profile, "EURUSD", [{"session": "london"}]) == "AUTHENTICATED"
    assert _cost_status(profile, "USDCAD", []) == (
        "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
    )


def test_registry_rejects_non_exact_instrument_count(tmp_path):
    source = json.loads(Path("configs/fx-universe-2024-registry-v1.json").read_text())
    source["instruments"].pop("EURGBP")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_stage4b_registry(path)


OU_SPEC = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
COMPARABLE_ROW = {
    "complete": True,
    "gross_return_pips_adverse_first": 1.0,
    "mfe_pips_certain": 2.0,
    "mae_pips_certain": 0.5,
    "signal_timeframe": "15m",
    "session": "london",
    "direction": "SHORT",
    "benchmark_family": "vwap",
    "lookback": 20,
    "signal_threshold": 2.0,
    "filter_family": "none",
    "filter_spec_id": "none-v1",
    "entry_mode": "immediate",
    "tp_target_fraction": 0.75,
    "sl_extension_fraction": 0.25,
    "time_stop_minutes": 60,
}


@pytest.mark.parametrize(
    ("field", "out_of_scope"),
    (
        ("signal_timeframe", "5m"),
        ("session", "new_york"),
        ("direction", "LONG"),
        ("benchmark_family", "bollinger"),
        ("lookback", 10),
        ("signal_threshold", 1.5),
        ("filter_family", "ornstein-uhlenbeck"),
        ("filter_spec_id", "other"),
        ("entry_mode", "m1-reclaim-p0"),
        ("tp_target_fraction", 0.5),
        ("sl_extension_fraction", 1.0),
        ("time_stop_minutes", 30),
    ),
)
def test_comparable_baseline_matches_every_frozen_scope_and_grid_field(
    field, out_of_scope
):
    assert _is_mr_comparable_row(COMPARABLE_ROW, OU_SPEC)
    assert not _is_mr_comparable_row(COMPARABLE_ROW | {field: out_of_scope}, OU_SPEC)


def test_out_of_scope_rows_are_excluded_and_full_grid_is_unchanged():
    outside = COMPARABLE_ROW | {
        "signal_timeframe": "1h",
        "gross_return_pips_adverse_first": 9.0,
    }
    full_grid = [COMPARABLE_ROW, outside]
    before = _metrics(full_grid)

    comparable = _mr_comparable_baseline(full_grid, OU_SPEC)
    _, report = _comparison_metrics(
        full_grid, [COMPARABLE_ROW], OU_SPEC, None, "BLOCKED"
    )

    assert comparable == [COMPARABLE_ROW]
    assert report["mr_full_grid_gross"] == before
    assert report["mr_full_grid_trade_count"] == 2
    assert report["mr_comparable_baseline_trade_count"] == 1


def test_ou_retention_uses_comparable_baseline_not_full_grid():
    full_grid = [
        COMPARABLE_ROW,
        COMPARABLE_ROW | {"gross_return_pips_adverse_first": 2.0},
        COMPARABLE_ROW | {"signal_timeframe": "1h"},
        COMPARABLE_ROW | {"entry_mode": "m1-reclaim-p0"},
    ]
    _, report = _comparison_metrics(
        full_grid, [COMPARABLE_ROW], OU_SPEC, None, "BLOCKED"
    )
    assert report["ou_retention_ratio"] == 0.5
    assert report["mr_full_grid_trade_count"] == 4
