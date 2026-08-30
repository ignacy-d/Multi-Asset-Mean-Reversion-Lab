from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import (
    CONFIG_FIELDS,
    SLIPPAGES,
    SPREAD_STATISTICS,
    CostProfile,
    Stage4CError,
    break_even_total_slippage,
    scenario_rows,
    transform_trade,
)
from mr_lab.stage4c_reducer import Stage4CReductionError, reduce_shards
from mr_lab.stage4c_runner import _group_owner, run_regenerated, run_rows

PROFILE = Path("configs/stage4c-ftmo-cost-profile-v1.json")


def trade(
    event="event-1", instrument="EURUSD", session="london", gross=2.0, favorable=2.0
):
    return {
        "instrument": instrument,
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "session": session,
        "direction": "LONG",
        "lookback": 20,
        "signal_threshold": 2.0,
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "entry_mode": "immediate",
        "tp_target_fraction": 0.5,
        "sl_extension_fraction": 0.5,
        "time_stop_minutes": 30,
        "candidate_event_id": event,
        "complete": True,
        "exit_ordering": "unambiguous",
        "holding_minutes": 4,
        "mae_pips_certain": 1.0,
        "mfe_pips_certain": 3.0,
        "gross_return_pips_adverse_first": gross,
        "gross_return_pips_favorable_first": favorable,
    }


@pytest.fixture
def profile():
    return CostProfile.load(PROFILE)


def test_one_bid_bid_spread_and_usd_commission(profile):
    row = transform_trade(trade(), profile, "mean", 0.0)
    assert row["commission_pips"] == 0.5
    assert row["net_pips_adverse_first"] == pytest.approx(
        2.0 - row["spread_pips"] - 0.5
    )


def test_audusd_commission_is_half_pip(profile):
    assert (
        transform_trade(trade(instrument="AUDUSD"), profile, "p75", 0)[
            "commission_pips"
        ]
        == 0.5
    )


def test_jpy_commission_and_conversion_adjustment(profile):
    positive = transform_trade(trade(instrument="USDJPY", gross=4), profile, "mean", 0)
    execution = 4 - positive["spread_pips"]
    assert positive["commission_pips"] == pytest.approx(0.795265089)
    assert positive["net_pips_adverse_first"] == pytest.approx(
        execution * 0.993 - positive["commission_pips"]
    )
    negative = transform_trade(trade(instrument="AUDJPY", gross=-4), profile, "mean", 0)
    assert negative["net_pips_adverse_first"] == pytest.approx(
        (-4 - negative["spread_pips"]) * 1.007 - negative["commission_pips"]
    )


def test_scenario_monotonicity_and_full_matrix(profile):
    rows = list(scenario_rows(trade(), profile))
    assert len(rows) == 16
    by_spread = {(r["spread_statistic"], r["slippage_pips"]): r for r in rows}
    for statistic in SPREAD_STATISTICS:
        values = [by_spread[statistic, x]["net_pips_adverse_first"] for x in SLIPPAGES]
        assert values == sorted(values, reverse=True)
    for slip in SLIPPAGES:
        # "Increasing spread" means the numeric frozen cost, not the statistic
        # label: an empirical mean can exceed p75 in a right-tailed sample.
        ordered = sorted(
            SPREAD_STATISTICS,
            key=lambda x: by_spread[x, slip]["spread_pips"],
        )
        values = [by_spread[x, slip]["net_pips_adverse_first"] for x in ordered]
        assert values == sorted(values, reverse=True)


def test_gross_and_ambiguity_are_preserved(profile):
    original = trade(gross=-2, favorable=3) | {
        "exit_ordering": "ambiguous_same_minute",
        "exit_reason": "sl",
    }
    snapshot = deepcopy(original)
    result = transform_trade(original, profile, "mean", 0.1)
    assert original == snapshot
    assert result["gross_return_pips_adverse_first"] == -2
    assert result["gross_return_pips_favorable_first"] == 3
    assert result["exit_ordering"] == "ambiguous_same_minute"
    assert result["net_pips_adverse_first"] < result["net_pips_favorable_first"]


def test_null_session_uses_overall(profile):
    row = transform_trade(trade(session=None), profile, "mean", 0)
    assert row["spread_pips"] == pytest.approx(0.160480694)


def test_streaming_matches_reference_and_sharded_equals_unsharded(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    rows = [trade("a", gross=2), trade("b", gross=-1)]
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
    }
    first, second = tmp_path / "one", tmp_path / "two"
    run_rows(
        iter(rows),
        first,
        PROFILE,
        source_mode="regenerated_stage4b",
        source_audit=audit,
        debug_raw=True,
    )
    run_rows(
        iter(rows[:1] + rows[1:]),
        second,
        PROFILE,
        source_mode="existing_stage4b_raw",
        source_audit=audit,
    )
    with (first / "stage4c-debug-trades.jsonl").open() as stream:
        actual = [json.loads(line) for line in stream]
    profile = CostProfile.load(PROFILE)
    expected = [result for row in rows for result in scenario_rows(row, profile)]
    assert actual == expected
    for name in (
        "stage4c-trade-matrix.csv",
        "stage4c-regime-breadth.csv",
        "stage4c-cost-scenarios.csv",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_usd_break_even_fixture(profile):
    spread, commission = profile.costs("EURUSD", "london", "mean")
    assert break_even_total_slippage(
        [2.0, 4.0], "EURUSD", spread, commission
    ) == pytest.approx(3.0 - spread - commission, abs=1e-9)


def test_mixed_sign_jpy_break_even_is_deterministic_and_spread_monotone(profile):
    values = (8.0, -1.0, 3.0)
    results = []
    for statistic in SPREAD_STATISTICS:
        spread, commission = profile.costs("USDJPY", "london", statistic)
        first = break_even_total_slippage(values, "USDJPY", spread, commission)
        assert first == break_even_total_slippage(values, "USDJPY", spread, commission)
        results.append(first)
    assert results == sorted(results, reverse=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1.0", None])
def test_nonfinite_or_malformed_gross_fails_closed(profile, bad):
    with pytest.raises(Stage4CError, match="gross"):
        transform_trade(trade(gross=bad), profile, "mean", 0)


def test_duplicate_and_wrong_methodology_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
    }
    with pytest.raises(Stage4CError, match="duplicate"):
        run_rows(
            iter([trade(), trade()]),
            tmp_path / "duplicate",
            PROFILE,
            source_mode="existing_stage4b_raw",
            source_audit=audit,
        )
    with pytest.raises(Stage4CError, match="methodology"):
        run_rows(
            iter([trade()]),
            tmp_path / "method",
            PROFILE,
            source_mode="existing_stage4b_raw",
            source_audit=audit | {"stage4b_methodology_id": "wrong"},
        )


def test_reports_required_metrics_and_no_2025_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
    }
    run_rows(
        iter([trade()]),
        tmp_path,
        PROFILE,
        source_mode="regenerated_stage4b",
        source_audit=audit,
    )
    with (tmp_path / "stage4c-trade-matrix.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert {
        "mean_net_pips",
        "median_net_pips",
        "net_pips_p10",
        "net_profit_factor",
        "cost_headroom_pips",
        "break_even_total_slippage_pips",
        "break_even_extra_slippage_pips",
    } <= set(row)
    assert "2025" not in Path("src/mr_lab/stage4c.py").read_text()
    assert "2025" not in Path("src/mr_lab/stage4c_runner.py").read_text()


def test_cost_profile_is_strict_json_and_malformed_fails(tmp_path):
    assert json.loads(PROFILE.read_text())["schema_version"] == (
        "stage4c-ftmo-cost-profile-v1"
    )
    malformed = tmp_path / "profile.json"
    malformed.write_text(PROFILE.read_text()[:-1])
    with pytest.raises(Stage4CError, match="malformed cost-profile JSON"):
        CostProfile.load(malformed)


def test_bounded_processing_releases_independent_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
    }
    rows = [trade(f"a-{index}") for index in range(20)] + [
        trade(f"b-{index}") | {"benchmark_family": "bollinger"} for index in range(3)
    ]
    import mr_lab.stage4c_runner as runner

    original = runner._cell_row
    observed_group_sizes = []

    def observe(key, group, profile, statistic, slippage):
        observed_group_sizes.append(len(group))
        return original(key, group, profile, statistic, slippage)

    monkeypatch.setattr(runner, "_cell_row", observe)
    run_rows(
        iter(rows),
        tmp_path,
        PROFILE,
        source_mode="existing_stage4b_raw",
        source_audit=audit,
    )
    assert set(observed_group_sizes) == {3, 20}
    assert not (tmp_path / ".stage4c-spool.sqlite3").exists()


def test_real_independent_shards_reduce_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "instrument": "EURUSD",
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
    }
    rows = [
        trade("a", gross=2),
        trade("b", gross=-1),
        trade("c", gross=4) | {"benchmark_family": "bollinger"},
        trade("d", gross=-3) | {"entry_mode": "m1-reclaim-p0"},
    ]
    unsharded = tmp_path / "unsharded"
    run_rows(
        iter(rows),
        unsharded,
        PROFILE,
        source_mode="existing_stage4b_raw",
        source_audit=audit,
    )
    shards = []
    for index in range(2):
        directory = tmp_path / f"shard-{index}"
        run_rows(
            iter(rows),
            directory,
            PROFILE,
            source_mode="existing_stage4b_raw",
            source_audit=audit,
            shard_index=index,
            shard_count=2,
        )
        shards.append(directory)
    reduced = tmp_path / "reduced"
    reduce_shards(shards, reduced, PROFILE, 2)
    for name in (
        "stage4c-trade-matrix.csv",
        "stage4c-regime-breadth.csv",
        "stage4c-cost-scenarios.csv",
    ):
        assert (reduced / name).read_bytes() == (unsharded / name).read_bytes()
    manifests = [
        json.loads((directory / "stage4c-shard-manifest.json").read_text())
        for directory in shards
    ]
    assert all(manifest["source_rows_read"] == 4 for manifest in manifests)
    assert all(manifest["source_complete_rows"] == 4 for manifest in manifests)
    assert sum(manifest["owned_complete_rows"] for manifest in manifests) == 4
    assert sum(manifest["scenario_evaluations"] for manifest in manifests) == 64
    reduced_audit = json.loads((reduced / "execution-audit.json").read_text())
    assert reduced_audit["source_mode"] == "existing_stage4b_raw"
    assert reduced_audit["row_counts"] == {
        "source_rows_read": 4,
        "source_complete_rows": 4,
        "owned_complete_rows": 4,
        "scenario_evaluations": 64,
    }
    assert reduced_audit["reduction"]["shard_state_commitments"] == [
        {
            "shard_index": manifest["shard_index"],
            "state_row_count": manifest["state_row_count"],
            "state_sha256": manifest["state_sha256"],
        }
        for manifest in manifests
    ]


def test_regenerated_shards_preserve_original_source_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "instrument": "EURUSD",
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"callback_stream": "c" * 64},
        "registry_identity": "registry",
    }
    rows = [trade("a"), trade("b") | {"benchmark_family": "bollinger"}]
    shards = []
    for index in range(2):
        directory = tmp_path / f"regenerated-{index}"
        run_rows(
            iter(rows),
            directory,
            PROFILE,
            source_mode="regenerated_stage4b",
            source_audit=audit,
            shard_index=index,
            shard_count=2,
        )
        shards.append(directory)
    output = tmp_path / "reduced"
    reduce_shards(shards, output, PROFILE, 2)
    final_audit = json.loads((output / "execution-audit.json").read_text())
    assert final_audit["source_mode"] == "regenerated_stage4b"
    assert final_audit["source_input_components"]["registry_identity"] == "registry"


def test_reducer_rejects_missing_duplicate_overlap_and_inconsistency(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "instrument": "EURUSD",
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
    }
    rows = [trade("a"), trade("b") | {"benchmark_family": "bollinger"}]
    shards = []
    for index in range(2):
        directory = tmp_path / f"s{index}"
        run_rows(
            iter(rows),
            directory,
            PROFILE,
            source_mode="existing_stage4b_raw",
            source_audit=audit,
            shard_index=index,
            shard_count=2,
        )
        shards.append(directory)
    with pytest.raises(Stage4CReductionError, match="missing"):
        reduce_shards(shards[:1], tmp_path / "missing", PROFILE, 2)
    manifest_path = shards[1] / "stage4c-shard-manifest.json"
    original = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(original | {"shard_index": 0}))
    with pytest.raises(Stage4CReductionError, match="duplicate"):
        reduce_shards(shards, tmp_path / "duplicate", PROFILE, 2)
    first_manifest_path = shards[0] / "stage4c-shard-manifest.json"
    first_original = json.loads(first_manifest_path.read_text())
    first_manifest_path.write_text(
        json.dumps(first_original | {"group_ownership": [["overlap"]]})
    )
    manifest_path.write_text(json.dumps(original | {"group_ownership": [["overlap"]]}))
    with pytest.raises(Stage4CReductionError, match="overlapping"):
        reduce_shards(shards, tmp_path / "overlap", PROFILE, 2)
    first_manifest_path.write_text(json.dumps(first_original))
    manifest_path.write_text(json.dumps(original | {"cost_profile_sha256": "wrong"}))
    with pytest.raises(Stage4CReductionError, match="inconsistent"):
        reduce_shards(shards, tmp_path / "inconsistent", PROFILE, 2)


def test_reducer_rejects_source_commitment_count_and_state_ownership(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    audit = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "instrument": "EURUSD",
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
    }
    rows = [trade("a"), trade("b") | {"benchmark_family": "bollinger"}]
    shards = []
    for index in range(2):
        directory = tmp_path / f"shard-{index}"
        run_rows(
            iter(rows),
            directory,
            PROFILE,
            source_mode="existing_stage4b_raw",
            source_audit=audit,
            shard_index=index,
            shard_count=2,
        )
        shards.append(directory)
    manifests = [directory / "stage4c-shard-manifest.json" for directory in shards]
    originals = [json.loads(path.read_text()) for path in manifests]

    manifests[1].write_text(
        json.dumps(originals[1] | {"source_input_commitment": "sha256:" + "0" * 64})
    )
    with pytest.raises(Stage4CReductionError, match="source_input_commitment"):
        reduce_shards(shards, tmp_path / "commitment", PROFILE, 2)
    manifests[1].write_text(json.dumps(originals[1] | {"shard_count": 3}))
    with pytest.raises(Stage4CReductionError, match="shard_count mismatch"):
        reduce_shards(shards, tmp_path / "count", PROFILE, 2)
    manifests[1].write_text(json.dumps(originals[1]))

    populated = next(
        index for index, value in enumerate(originals) if value["state_row_count"]
    )
    manifest_path = manifests[populated]
    state_path = shards[populated] / "stage4c-shard-state.jsonl"
    manifest = originals[populated]
    manifest_path.write_text(json.dumps(manifest | {"group_ownership": []}))
    with pytest.raises(Stage4CReductionError, match="absent from declared"):
        reduce_shards(shards, tmp_path / "absent", PROFILE, 2)

    state_original = state_path.read_text()
    state_rows = [json.loads(line) for line in state_original.splitlines()]
    changed = state_rows[0]
    suffix = 0
    while True:
        changed["benchmark_family"] = f"wrong-owner-{suffix}"
        group = tuple(changed.get(field) for field in CONFIG_FIELDS)
        if _group_owner(group, 2) != populated:
            break
        suffix += 1
    state_path.write_text("\n".join(json.dumps(row) for row in state_rows) + "\n")
    changed_manifest = manifest | {
        "group_ownership": [list(group)],
        "state_sha256": __import__("hashlib")
        .sha256(state_path.read_bytes())
        .hexdigest(),
    }
    manifest_path.write_text(json.dumps(changed_manifest))
    with pytest.raises(Stage4CReductionError, match="another deterministic shard"):
        reduce_shards(shards, tmp_path / "owner", PROFILE, 2)


def test_audit_authority_and_currency_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    hostile = {
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "source_trade_sha256": {"fixture": "f" * 64},
        "registry_identity": "registry",
        "schema_version": "hostile",
        "account_currency": "EUR",
        "volume_bands_modeled": True,
    }
    run_rows(
        iter([trade()]),
        tmp_path,
        PROFILE,
        source_mode="existing_stage4b_raw",
        source_audit=hostile,
    )
    audit = json.loads((tmp_path / "execution-audit.json").read_text())
    assert audit["schema_version"] == "stage-4c-report-v1"
    assert audit["account_currency"] == "USD"
    assert audit["volume_bands_modeled"] is False
    assert (
        "positive x 0.993" in audit["currency_conversion_adjustment"]["USDJPY/AUDJPY"]
    )


def test_regenerated_route_uses_stage4b_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE4C_SOURCE_COMMIT", "a" * 40)
    registry = tmp_path / "registry.json"
    registry.write_text("{}")

    def fake_run(_corpus, output, instrument, _registry, trade_row_consumer=None):
        output.mkdir()
        trade_row_consumer(trade(instrument=instrument))
        (output / "execution-audit.json").write_text(
            json.dumps(
                {
                    "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
                    "instrument": instrument,
                    "corpus_id": "corpus",
                    "assembled_dataset_id": "dataset",
                    "source_trade_sha256": {"fixture": "f" * 64},
                    "registry_identity": "registry",
                }
            )
        )

    monkeypatch.setattr("mr_lab.stage4b_runner.run", fake_run)
    output = tmp_path / "output"
    run_regenerated(tmp_path / "corpus", output, "EURUSD", registry, PROFILE)
    audit = json.loads((output / "execution-audit.json").read_text())
    assert audit["source_mode"] == "regenerated_stage4b"
    assert audit["corpus_id"] == "corpus"
