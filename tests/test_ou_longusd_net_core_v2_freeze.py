from __future__ import annotations

import json
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs" / "ou-longusd-net-core-v2.json"
PARENT_PATH = ROOT / "configs" / "ou-longusd-tradable-2024-v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_strict2_universe_and_exclusions_are_exact() -> None:
    freeze = load(FREEZE_PATH)
    assert freeze["primary_net_core"] == {"AUDUSD": "SHORT", "GBPUSD": "SHORT"}
    assert "EURUSD" not in freeze["primary_net_core"]
    assert "EURUSD" in freeze["excluded_by_post_cost_screen"]
    assert freeze["prior_auxiliary_excluded"] == {"EURGBP": "SHORT"}


def test_parent_london_alpha_execution_and_lifecycle_are_unchanged() -> None:
    freeze, parent = load(FREEZE_PATH), load(PARENT_PATH)
    assert freeze["inherited_alpha_methodology"] == {
        "signal": parent["signal"],
        "ou": parent["ou"],
    }
    assert freeze["inherited_execution_methodology"] == parent["execution"]
    assert freeze["inherited_lifecycle_methodology"] == parent["lifecycle"]


def test_cost_grid_is_the_frozen_sixteen_scenario_cross_product() -> None:
    freeze = load(FREEZE_PATH)
    grid = freeze["frozen_cost_scenarios"]
    scenarios = set(product(grid["spread_statistics"], grid["additional_slippage_pips"]))
    assert grid == {
        "spread_statistics": ["mean", "p75", "p90", "p95"],
        "additional_slippage_pips": [0.0, 0.1, 0.25, 0.5],
        "cross_product": True,
        "scenario_count": 16,
    }
    assert len(scenarios) == 16


def test_selection_governance_and_no_empirical_output() -> None:
    freeze = load(FREEZE_PATH)
    assert freeze["selection_dataset"] == 2024
    assert freeze["scientific_status"] == "2024_POST_COST_SELECTION_NOT_CONFIRMATION"
    assert freeze["selection_status"] == "post_cost_selection_not_confirmation"
    assert freeze["next_authorized_empirical_dataset"] == 2023
    assert freeze["next_authorized_phase"] == "2023_INDEPENDENT_REPLICATION"
    assert freeze["governance_only_freeze"] is True
    assert freeze["empirical_strategy_outputs"] == []


def test_freeze_sources_do_not_contain_forbidden_future_identifier() -> None:
    forbidden = str(2000 + 25)
    assert forbidden not in FREEZE_PATH.read_text(encoding="utf-8")
    assert forbidden not in Path(__file__).read_text(encoding="utf-8")


def test_new_york_study_is_separate_and_not_imported() -> None:
    freeze = load(FREEZE_PATH)
    assert freeze["separate_new_york_study_unchanged"] == (
        "OU-LONGUSD-TRADABLE-2024-NY-v1"
    )
    assert "new_york" not in freeze["inherited_alpha_methodology"]["signal"]
