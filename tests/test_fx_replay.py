import json

import pytest

from mr_lab.fx_replay import _cost_status, _metrics
from mr_lab.stage4b_runner import _load_stage4b_registry
from mr_lab.stage4c import CostProfile


def test_authenticated_registry_is_exact_nine_pair_contract():
    registry = _load_stage4b_registry(
        __import__("pathlib").Path("configs/fx-universe-2024-registry-v1.json")
    )
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
    profile = CostProfile.load(
        __import__("pathlib").Path("configs/stage4c-ftmo-cost-profile-v1.json")
    )
    assert _cost_status(profile, "USDCAD", [{"session": "london"}]) == (
        "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
    )
    assert _cost_status(profile, "EURUSD", [{"session": "london"}]) == "AUTHENTICATED"


def test_registry_rejects_non_exact_instrument_count(tmp_path):
    source = json.loads(
        __import__("pathlib")
        .Path("configs/fx-universe-2024-registry-v1.json")
        .read_text()
    )
    source["instruments"].pop("EURGBP")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_stage4b_registry(path)
