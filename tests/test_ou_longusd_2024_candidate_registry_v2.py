from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "ou-longusd-2024-candidate-registry-v2.json"
PARENT_PATH = ROOT / "configs" / "ou-longusd-tradable-2024-v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_london_families_coexist_and_stage4b_is_not_replaced() -> None:
    families = load(REGISTRY_PATH)["families"]
    assert "FROZEN_STAGE4B_OU_LONDON" in families
    assert "OU_LONGUSD_TRADABLE_LONDON_2024_V1" in families
    stage4b = families["FROZEN_STAGE4B_OU_LONDON"]
    assert stage4b["replacement_policy"] == (
        "MUST_NOT_BE_REPLACED_BY_OU_LONGUSD_TRADABLE_LONDON_2024_V1"
    )
    assert stage4b["methodology"]["execution_grid"]["total_cells"] == 32


def test_tradable_london_methodology_is_exactly_the_parent_methodology() -> None:
    registry, parent = load(REGISTRY_PATH), load(PARENT_PATH)
    methodology = registry["families"]["OU_LONGUSD_TRADABLE_LONDON_2024_V1"][
        "methodology"
    ]
    assert methodology == {
        "signal": parent["signal"],
        "ou": parent["ou"],
        "execution": parent["execution"],
        "lifecycle": parent["lifecycle"],
    }


def test_all_three_instruments_remain_candidates() -> None:
    families = load(REGISTRY_PATH)["families"]
    tradable = families["OU_LONGUSD_TRADABLE_LONDON_2024_V1"][
        "authenticated_post_cost_evidence"
    ]
    assert tradable["EURUSD"]["status"] == "POSITIVE_CANDIDATE"
    assert tradable["GBPUSD"]["status"] == "STRONG_CANDIDATE"
    assert tradable["AUDUSD"]["status"] == "STRONG_CANDIDATE"


def test_audusd_evidence_strength_differs_between_london_families() -> None:
    families = load(REGISTRY_PATH)["families"]
    stage4b_status = families["FROZEN_STAGE4B_OU_LONDON"][
        "authenticated_historical_comparison"
    ]["evidence"]["AUDUSD"]["status"]
    tradable_status = families["OU_LONGUSD_TRADABLE_LONDON_2024_V1"][
        "authenticated_post_cost_evidence"
    ]["AUDUSD"]["status"]
    assert stage4b_status == "BORDERLINE_OR_WEAKER_CANDIDATE"
    assert tradable_status == "STRONG_CANDIDATE"


def test_new_york_family_is_parked() -> None:
    new_york = load(REGISTRY_PATH)["families"][
        "OU_LONGUSD_TRADABLE_NEW_YORK_2024_V1"
    ]
    assert new_york["role"] == "PARK_AFTER_COSTS"
    assert new_york["further_2024_tuning_authorized"] is False
    assert set(new_york["individual_cost_scenario_survival"].values()) == {"0/16"}


def test_replication_contract_is_exact_and_does_not_preselect_a_family() -> None:
    replication = load(REGISTRY_PATH)["independent_replication"]
    assert replication["dataset"] == 2023
    assert replication["instruments"] == ["AUDUSD", "EURUSD", "GBPUSD"]
    assert replication["families"] == [
        "FROZEN_STAGE4B_OU_LONDON",
        "OU_LONGUSD_TRADABLE_LONDON_2024_V1",
    ]
    assert replication["choose_family_before_replication"] is False
    assert replication["optimization_authorized"] is False
    assert replication["new_parameter_selection_authorized"] is False


def test_governance_task_creates_no_empirical_output() -> None:
    registry = load(REGISTRY_PATH)
    assert registry["scientific_status"] == (
        "2024_POST_COST_CANDIDATES_NOT_CONFIRMATION"
    )
    assert registry["governance_only"] is True
    assert registry["empirical_outputs_created"] == []
