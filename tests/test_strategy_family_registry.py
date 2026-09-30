import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "configs/strategy-family-registry-v1.json"
SCHEMA_PATH = ROOT / "configs/strategy-family-registry-schema-v1.json"
REQUIRED_INSTRUMENT_FIELDS = {
    "family_id",
    "session",
    "alpha_methodology_id",
    "execution_methodology_id",
    "instrument",
    "direction",
    "discovery_dataset",
    "gross_status",
    "net_status",
    "cost_robustness",
    "evidence_source",
    "evidence_artifacts",
    "evidence_verification_status",
    "replication_status",
    "notes",
}
EXPECTED_FAMILIES = {
    "FROZEN_STAGE4B_OU_LONDON",
    "OU_LONGUSD_TRADABLE_LONDON_2024_V1",
    "OU_LONGUSD_TRADABLE_NEW_YORK_2024_V1",
}


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def test_registry_schema_contract_and_instrument_integrity() -> None:
    registry = load_registry()
    schema = json.loads(SCHEMA_PATH.read_text())

    assert registry["schema_version"] == "strategy-family-registry-v1"
    assert registry["schema_path"] == "configs/strategy-family-registry-schema-v1.json"
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["family"]["additionalProperties"] is False
    assert schema["$defs"]["instrument"]["additionalProperties"] is False

    for family in registry["families"]:
        assert family["instruments"]
        for row in family["instruments"]:
            assert row.keys() >= REQUIRED_INSTRUMENT_FIELDS
            assert row["family_id"] == family["family_id"]
            assert row["session"] == family["session"]
            assert row["alpha_methodology_id"] == family["alpha_methodology_id"]
            assert row["execution_methodology_id"] == family["execution_methodology_id"]
            for artifact in row["evidence_artifacts"]:
                assert artifact.keys() == {"path", "sha256", "supports"}
                assert artifact["path"] is None or artifact["path"]
                assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
                assert artifact["supports"]


def test_family_ids_are_unique_and_required_history_is_preserved() -> None:
    registry = load_registry()
    family_ids = [family["family_id"] for family in registry["families"]]

    assert len(family_ids) == len(set(family_ids))
    assert set(family_ids) >= EXPECTED_FAMILIES


def test_status_values_are_allowed() -> None:
    registry = load_registry()
    allowed = set(registry["allowed_statuses"])
    schema_allowed = set(
        json.loads(SCHEMA_PATH.read_text())["properties"]["allowed_statuses"]["items"][
            "enum"
        ]
    )
    assert allowed == schema_allowed

    for family in registry["families"]:
        assert family["scientific_status"] in allowed
        for row in family["instruments"]:
            assert row["gross_status"] in allowed
            assert row["net_status"] in allowed
            assert row["evidence_verification_status"] in allowed
            assert row["replication_status"] in allowed
            status = row["cost_robustness"].get("status")
            assert status is None or status in allowed


def test_failed_after_costs_is_never_a_gross_status() -> None:
    registry = load_registry()

    assert all(
        row["gross_status"] != "FAILED_AFTER_COSTS"
        for family in registry["families"]
        for row in family["instruments"]
    )


def test_new_family_cannot_silently_replace_existing_family() -> None:
    registry = load_registry()

    assert registry["governance"]["history_policy"] == (
        "APPEND_ONLY_NO_SILENT_REPLACEMENT"
    )
    assert registry["governance"]["global_ranking"] is False
    assert registry["governance"]["authoritative_final_selection"] is False
    assert all(family["replaces_family_id"] is None for family in registry["families"])
    execution_ids = {
        family["execution_methodology_id"] for family in registry["families"]
    }
    assert len(execution_ids) == len(registry["families"])


def test_exact_candidates_target_independent_2023_replication() -> None:
    registry = load_registry()
    assert registry["governance"]["next_intended_validation_dataset"] == (
        "Independent 2023 replication for London candidates"
    )
    actual = {
        (family["family_id"], row["instrument"])
        for family in registry["families"]
        for row in family["instruments"]
        if row["replication_status"] == "INDEPENDENT_REPLICATION_PENDING"
    }
    expected = {
        (family_id, instrument)
        for family_id in {
            "FROZEN_STAGE4B_OU_LONDON",
            "OU_LONGUSD_TRADABLE_LONDON_2024_V1",
        }
        for instrument in {"AUDUSD", "EURUSD", "GBPUSD"}
    }
    assert actual == expected


def test_audited_stage4b_evidence_and_gross_only_eurgbp() -> None:
    registry = load_registry()
    family = next(
        item
        for item in registry["families"]
        if item["family_id"] == "FROZEN_STAGE4B_OU_LONDON"
    )
    rows = {row["instrument"]: row for row in family["instruments"]}

    assert rows["EURUSD"]["evidence_verification_status"] == "LOCAL_ARTIFACT_VERIFIED"
    assert rows["AUDUSD"]["evidence_verification_status"] == "LOCAL_ARTIFACT_VERIFIED"
    assert rows["EURGBP"]["net_status"] == "NOT_ASSESSED"
    assert rows["EURGBP"]["cost_robustness"]["status"] == (
        "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
    )
    assert "net_mean_pips" not in rows["EURGBP"]["metrics"]
