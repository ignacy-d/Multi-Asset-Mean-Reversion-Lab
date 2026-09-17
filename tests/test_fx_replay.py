import hashlib
import inspect
import json
import statistics
from io import StringIO
from pathlib import Path

import pytest

from mr_lab.fx_replay import (
    OU_FILTER,
    StreamingMetrics,
    _compatible_completed_variant,
    _consume,
    _cost_status,
    _is_mr_comparable_row,
    _load_bundle,
    _new_bundle,
    _run_or_reuse_variant,
    _write_bundle,
)
from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4b_runner import OUTPUTS, _load_stage4b_registry, _write_trade_row, run
from mr_lab.stage4c import CostProfile, transform_trade

OU_SPEC = frozen_ou_eligibility_spec(OU_FILTER)
COMPARABLE_ROW = {
    "complete": True,
    "instrument": "EURUSD",
    "candidate_event_id": "event-1",
    "gross_return_pips_adverse_first": 1.0,
    "gross_return_pips_favorable_first": 1.5,
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


def _old_metrics(rows, field="gross_return_pips_adverse_first"):
    complete = [row for row in rows if row.get("complete")]
    values = [float(row[field]) for row in complete]
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return {
        "n_trades": len(values),
        "mean_pips": statistics.fmean(values) if values else None,
        "median_pips": statistics.median(values) if values else None,
        "win_rate": sum(value > 0 for value in values) / len(values) if values else 0.0,
        "profit_factor": gains / losses if losses else (float("inf") if gains else 0.0),
        "mean_mfe_pips": statistics.fmean(row["mfe_pips_certain"] for row in complete)
        if complete
        else None,
        "mean_mae_pips": statistics.fmean(row["mae_pips_certain"] for row in complete)
        if complete
        else None,
    }


def test_authenticated_registry_is_exact_nine_pair_contract():
    registry = _load_stage4b_registry(Path("configs/fx-universe-2024-registry-v1.json"))
    assert len(registry["instruments"]) == 9


def test_streaming_metrics_equal_previous_list_metrics_and_exact_median():
    rows = [
        COMPARABLE_ROW | {"gross_return_pips_adverse_first": value}
        for value in (9.0, -4.0, 2.0, 1.0)
    ] + [COMPARABLE_ROW | {"complete": False}]
    streaming = StreamingMetrics()
    for row in rows:
        streaming.add(row)
    assert streaming.result() == _old_metrics(rows)
    assert streaming.result()["median_pips"] == 1.5
    assert streaming.incomplete == 1


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


def test_streaming_comparable_filter_and_full_grid_are_independent():
    bundle = _new_bundle("mr", "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE")
    outside = COMPARABLE_ROW | {"signal_timeframe": "1h"}
    for row in (COMPARABLE_ROW, outside):
        _consume(
            bundle,
            "mr",
            row,
            OU_SPEC,
            None,
            "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE",
        )
    assert bundle["mr_full_grid_gross"].result()["n_trades"] == 2
    assert bundle["mr_comparable_baseline_gross"].result()["n_trades"] == 1


def test_streaming_net_cost_transform_is_identical():
    profile = CostProfile.load(Path("configs/stage4c-ftmo-cost-profile-v1.json"))
    bundle = _new_bundle("mr", "AUTHENTICATED")
    _consume(bundle, "mr", COMPARABLE_ROW, OU_SPEC, profile, "AUTHENTICATED")
    expected = transform_trade(COMPARABLE_ROW, profile, "mean", 0.0)
    assert bundle["mr_comparable_baseline_net"].result() == _old_metrics(
        [expected], "net_pips_adverse_first"
    )


def test_missing_cost_profile_fails_closed_even_without_rows():
    profile = CostProfile.load(Path("configs/stage4c-ftmo-cost-profile-v1.json"))
    assert (
        _cost_status(profile, "USDCAD") == "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE"
    )
    assert _cost_status(profile, "EURUSD") == "AUTHENTICATED"


def test_stage4b_trade_persistence_defaults_on_and_compact_mode_suppresses():
    default = StringIO()
    compact = StringIO()
    _write_trade_row(default, {"b": 1, "a": 2})
    _write_trade_row(compact, {"b": 1, "a": 2}, persist=False)
    assert default.getvalue() == '{"a":2,"b":1}\n'
    assert compact.getvalue() == ""
    assert inspect.signature(run).parameters["persist_trade_rows"].default is True


def test_replay_has_no_full_jsonl_list_materializer():
    source = inspect.getsource(__import__("mr_lab.fx_replay", fromlist=["*"]))
    assert "_read_jsonl" not in source
    assert ".readlines(" not in source


def _completed_variant(tmp_path, identity, entry):
    directory = tmp_path / "mr"
    directory.mkdir()
    hashes = {}
    for name in OUTPUTS:
        path = directory / name
        path.write_text("" if name == "trades.jsonl" else name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {
        "instrument": identity["instrument"],
        "corpus_id": entry["corpus_id"],
        "assembled_dataset_id": entry["assembled_dataset_id"],
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "output_sha256": hashes,
        "trades_jsonl_persisted": False,
    }
    (directory / "execution-audit.json").write_text(json.dumps(audit))
    hashes["execution-audit.json"] = hashlib.sha256(
        (directory / "execution-audit.json").read_bytes()
    ).hexdigest()
    (directory / "replay-identity.json").write_text(json.dumps(identity))
    bundle = _new_bundle("mr", "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE")
    _consume(
        bundle,
        "mr",
        COMPARABLE_ROW,
        OU_SPEC,
        None,
        "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE",
    )
    _write_bundle(directory, bundle)
    return directory


def test_completed_compatible_compact_artifact_is_reused(monkeypatch, tmp_path):
    profile = CostProfile.load(Path("configs/stage4c-ftmo-cost-profile-v1.json"))
    identity = {
        "registry_id": "registry",
        "registry_sha256": "sha256:registry",
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "ou_filter_spec_id": OU_SPEC.filter_spec_id,
        "cost_profile_sha256": profile.sha256,
        "instrument": "EURUSD",
        "variant": "mr",
    }
    entry = {"corpus_id": "corpus", "assembled_dataset_id": "dataset"}
    directory = _completed_variant(tmp_path, identity, entry)
    monkeypatch.setattr(
        "mr_lab.fx_replay.run", lambda *args, **kwargs: pytest.fail("must reuse")
    )
    reused = _run_or_reuse_variant(
        Path("unused"),
        directory,
        "EURUSD",
        Path("unused"),
        identity,
        "mr",
        OU_SPEC,
        profile,
        "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE",
        entry,
    )
    assert reused["mr_full_grid_gross"].result()["n_trades"] == 1


def test_completed_artifact_identity_mismatch_fails_closed(tmp_path):
    profile = CostProfile.load(Path("configs/stage4c-ftmo-cost-profile-v1.json"))
    identity = {
        "registry_id": "registry",
        "registry_sha256": "sha256:registry",
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "ou_filter_spec_id": OU_SPEC.filter_spec_id,
        "cost_profile_sha256": profile.sha256,
        "instrument": "EURUSD",
        "variant": "mr",
    }
    entry = {"corpus_id": "corpus", "assembled_dataset_id": "dataset"}
    directory = _completed_variant(tmp_path, identity, entry)
    with pytest.raises(RuntimeError, match="registry_id"):
        _compatible_completed_variant(
            directory, identity | {"registry_id": "different"}, entry, OU_SPEC
        )


def test_compact_aggregate_round_trip_uses_binary_values(tmp_path):
    bundle = _new_bundle("mr", "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE")
    _consume(
        bundle,
        "mr",
        COMPARABLE_ROW,
        OU_SPEC,
        None,
        "BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE",
    )
    _write_bundle(tmp_path, bundle)
    loaded = _load_bundle(tmp_path)
    assert (
        loaded["mr_full_grid_gross"].result() == bundle["mr_full_grid_gross"].result()
    )
    assert (tmp_path / "mr_full_grid_gross-values.f64").stat().st_size == 8
