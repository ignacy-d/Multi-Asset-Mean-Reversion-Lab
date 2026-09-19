import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import mr_lab.pca_stage0_runner as stage0
from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.pca_residual import Direction, PCAResidualConfig, ResidualObservation
from mr_lab.pca_stage0_runner import (
    FROZEN_PCA_CONFIG,
    INSTRUMENTS,
    PROCESS_ID,
    REGISTRY,
    Stage0Error,
    build_panel,
    concentration,
    event_id,
    event_outcomes,
    is_padding,
    load_registry,
    month_block_bootstrap,
)


def bar(name, minute, close, *, open_=None, volume=1.0):
    start = datetime(2024, 1, 2, tzinfo=UTC) + timedelta(minutes=minute)
    open_ = close if open_ is None else open_
    return Bar(
        name,
        Timeframe("1m"),
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=1),
        open_,
        max(open_, close),
        min(open_, close),
        close,
        PriceBasis.BID,
        volume,
        VolumeSemantics.QUOTE_ACTIVITY,
    )


def panel_series(*, gap_name=None, padding_name=None):
    result = {}
    for j, name in enumerate(INSTRUMENTS):
        bars = [
            bar(
                name,
                i,
                100 + j + i * (0.01 + j / 10000),
                open_=100 + j + max(0, i - 1) * (0.01 + j / 10000),
            )
            for i in range(7)
        ]
        if name == gap_name:
            bars.pop(3)
        if name == padding_name:
            bars[3] = bar(name, 3, 101, volume=0)
        result[name] = bars
    return result


def long_panel_series(count=80):
    result = {}
    for j, name in enumerate(INSTRUMENTS):
        result[name] = [
            bar(name, i, 100 + j + 0.01 * i + 0.002 * np.sin(i / (j + 2)))
            for i in range(count)
        ]
    return result


def test_only_canonical_registry_path_is_accepted():
    with pytest.raises(Stage0Error, match="explicit 2024 registry"):
        load_registry(Path("anything.json"))
    assert set(load_registry(REGISTRY)["instruments"]) == set(INSTRUMENTS)


def test_mutated_registry_id_fails_before_corpus_access(tmp_path, monkeypatch):
    registry = json.loads(REGISTRY.read_text())
    registry["registry_id"] = "sha256:" + "0" * 64
    canonical = tmp_path / "configs" / REGISTRY.name
    canonical.parent.mkdir()
    canonical.write_text(json.dumps(registry))
    monkeypatch.chdir(tmp_path)
    accessed = False

    def forbidden_access(*_args, **_kwargs):
        nonlocal accessed
        accessed = True
        raise AssertionError("corpus access must not occur")

    monkeypatch.setattr(stage0, "authenticate_corpus", forbidden_access)
    with pytest.raises(Stage0Error, match="registry contract mismatch"):
        load_registry(REGISTRY)
    assert not accessed


def test_padding_requires_both_zero_activity_and_flat_ohlc():
    assert is_padding(bar("EURUSD", 0, 1, volume=0))
    assert not is_padding(bar("EURUSD", 0, 1.1, open_=1, volume=0))
    assert not is_padding(bar("EURUSD", 0, 1, volume=1))


def test_exact_intersection_does_not_bridge_gap_or_padding_or_forward_fill():
    datasets = {n: "id-" + n for n in INSTRUMENTS}
    complete = build_panel(panel_series(), datasets)
    gap = build_panel(panel_series(gap_name="EURUSD"), datasets)
    padded = build_panel(panel_series(padding_name="GBPUSD"), datasets)
    assert len(complete.panel.timestamps) == 7
    # Missing/padded minutes invalidate adjacent timestamps.
    assert len(gap.panel.timestamps) == 5
    assert len(padded.panel.timestamps) == 5
    assert gap.counts["EURUSD"]["invalid_or_gap_returns"] == 1
    assert padded.counts["GBPUSD"]["padding"] == 1


def test_incomplete_universe_fails_closed():
    series = panel_series()
    series.pop("EURGBP")
    with pytest.raises(Stage0Error, match="nine"):
        build_panel(series, {n: "x" for n in INSTRUMENTS})


@pytest.mark.parametrize("malformation", ["duplicate", "reverse", "instrument"])
def test_source_m1_identity_and_order_fail_closed(malformation):
    series = panel_series()
    if malformation == "duplicate":
        series["EURUSD"][3] = series["EURUSD"][2]
    elif malformation == "reverse":
        series["EURUSD"][2], series["EURUSD"][3] = (
            series["EURUSD"][3],
            series["EURUSD"][2],
        )
    else:
        original = series["EURUSD"][2]
        series["EURUSD"][2] = Bar(
            "GBPUSD",
            original.timeframe,
            original.open_time,
            original.close_time,
            original.available_at,
            original.open,
            original.high,
            original.low,
            original.close,
            original.price_basis,
            original.volume,
            original.volume_semantics,
        )
    with pytest.raises(Stage0Error):
        build_panel(series, {n: "x" for n in INSTRUMENTS})


def test_frozen_config_and_process_identity_are_explicit():
    assert (
        PCAResidualConfig(
            pca_training_window=1024,
            components=2,
            residual_normalization_window=256,
            shock_threshold=2.0,
            rearm_threshold=1.0,
            variance_epsilon=1e-12,
        )
        == FROZEN_PCA_CONFIG
    )
    changed = stage0._config_identity(FROZEN_PCA_CONFIG)
    changed["variance_epsilon"] = 1e-10
    assert stage0._sha({**stage0.STUDY_SPEC, "pca_config": changed}) != PROCESS_ID


def test_synthetic_adapter_exactly_reproduces_returns_and_timestamps(monkeypatch):
    series = panel_series(gap_name="EURUSD", padding_name="GBPUSD")
    built = build_panel(series, {n: "x" for n in INSTRUMENTS})
    expected = []
    for left, right in zip(
        built.panel.timestamps, built.panel.timestamps[1:], strict=False
    ):
        assert right > left
        expected.append(
            [
                np.log(
                    built.bars[name][right].close
                    / built.bars[name][right - timedelta(minutes=1)].close
                )
                for name in INSTRUMENTS
            ]
        )
    assert np.allclose(np.diff(np.log(built.panel.closes), axis=0), expected)

    tiny = PCAResidualConfig(2, 2, 2, 100.0, 1.0, 1e-14)
    monkeypatch.setattr(stage0, "FROZEN_PCA_CONFIG", tiny)
    observations = stage0.PCAResidualEngine(tiny).run(built.panel)
    if observations:
        expected_stamps = set(built.panel.timestamps[1:])
        assert all(row.timestamp in expected_stamps for row in observations)


def _fake_event(stamp):
    return ResidualObservation(
        stamp,
        "EURUSD",
        0.01,
        1.0,
        0.0,
        2.5,
        3.0,
        True,
        True,
        Direction.POSITIVE,
        Direction.NEGATIVE,
    )


def _outcomes(monkeypatch, series, stamp):
    built = build_panel(series, {n: "same-dataset" for n in INSTRUMENTS})

    class Engine:
        def __init__(self, config):
            assert config is FROZEN_PCA_CONFIG

        def run(self, panel):
            return (_fake_event(stamp),)

    monkeypatch.setattr(stage0, "PCAResidualEngine", Engine)
    return event_outcomes(built)


def test_events_survive_missing_execution_and_independent_outcomes(monkeypatch):
    base = datetime(2024, 1, 2, tzinfo=UTC)
    stamp = base + timedelta(minutes=10)
    complete = long_panel_series()
    baseline = _outcomes(monkeypatch, complete, stamp)[0]
    assert baseline["entry_complete"] and baseline["h15_complete"]

    missing_entry = long_panel_series()
    missing_entry["EURUSD"].pop(10)
    event = _outcomes(monkeypatch, missing_entry, stamp)[0]
    assert not event["entry_complete"]
    assert event["entry_incomplete_reason"] == "missing_bar"
    assert all(not event[f"h{h}_complete"] for h in stage0.HORIZONS)

    padding_entry = long_panel_series()
    padding_entry["EURUSD"][10] = bar("EURUSD", 10, 101, volume=0)
    event = _outcomes(monkeypatch, padding_entry, stamp)[0]
    assert not event["entry_complete"]
    assert event["entry_incomplete_reason"] == "provider_padding"

    for padding in (False, True):
        changed = long_panel_series()
        if padding:
            changed["EURUSD"][24] = bar("EURUSD", 24, 101, volume=0)
        else:
            changed["EURUSD"].pop(24)  # close timestamp t+15
        event = _outcomes(monkeypatch, changed, stamp)[0]
        assert not event["h15_complete"]
        assert event["h5_complete"]


def test_entry_uses_next_bar_open_at_completed_return_timestamp(monkeypatch):
    base = datetime(2024, 1, 2, 10, tzinfo=UTC)
    series = {}
    for j, name in enumerate(INSTRUMENTS):
        bars = []
        for minute in range(80):
            start = base + timedelta(minutes=minute)
            price = 100 + j + minute / 100
            bars.append(
                Bar(
                    name,
                    Timeframe("1m"),
                    start,
                    start + timedelta(minutes=1),
                    start + timedelta(minutes=1),
                    price,
                    price + 0.01,
                    price - 0.01,
                    price,
                    PriceBasis.BID,
                    1.0,
                    VolumeSemantics.QUOTE_ACTIVITY,
                )
            )
        series[name] = bars

    row_timestamp = base + timedelta(minutes=15)
    completed_bar = series["EURUSD"][14]
    entry_bar = series["EURUSD"][15]
    assert completed_bar.open_time == base + timedelta(minutes=14)
    assert completed_bar.close_time == row_timestamp
    assert entry_bar.open_time == row_timestamp
    assert entry_bar.close_time == base + timedelta(minutes=16)

    event = _outcomes(monkeypatch, series, row_timestamp)[0]
    assert event["entry_target_timestamp"] == row_timestamp.isoformat()
    assert event["entry_timestamp"] == row_timestamp.isoformat()
    assert event["entry_price"] == entry_bar.open
    for horizon in stage0.HORIZONS:
        assert (
            event[f"h{horizon}_exit_timestamp"]
            == (row_timestamp + timedelta(minutes=horizon)).isoformat()
        )


def test_future_h60_absence_changes_only_h60_completeness(monkeypatch):
    base = datetime(2024, 1, 2, tzinfo=UTC)
    stamp = base + timedelta(minutes=10)
    full = long_panel_series()
    short = {name: bars[:69] for name, bars in long_panel_series().items()}
    complete = _outcomes(monkeypatch, full, stamp)[0]
    incomplete = _outcomes(monkeypatch, short, stamp)[0]
    causal_fields = (
        "event_id",
        "timestamp",
        "instrument",
        "residual",
        "residual_z",
        "shock_sign",
        "fade_direction",
    )
    assert {key: complete[key] for key in causal_fields} == {
        key: incomplete[key] for key in causal_fields
    }
    assert complete["h60_complete"]
    assert not incomplete["h60_complete"]
    assert incomplete["h15_complete"]


def test_future_price_mutation_does_not_change_causal_event(monkeypatch):
    base = datetime(2024, 1, 2, tzinfo=UTC)
    stamp = base + timedelta(minutes=10)
    original = long_panel_series()
    changed = long_panel_series()
    for name in INSTRUMENTS:
        for i in range(11, len(changed[name])):
            old = changed[name][i]
            changed[name][i] = bar(name, i, old.close * 1.2, open_=old.open * 1.2)
    first = _outcomes(monkeypatch, original, stamp)[0]
    second = _outcomes(monkeypatch, changed, stamp)[0]
    for key in (
        "event_id",
        "instrument",
        "residual",
        "residual_z",
        "shock_sign",
        "fade_direction",
    ):
        assert first[key] == second[key]


def test_concentration_uses_all_events_but_positive_count_uses_complete_h15():
    events = [
        {
            "timestamp": "2024-01-01",
            "instrument": "A",
            "h15_complete": False,
            "h15_signed_bps_return": None,
        },
        {
            "timestamp": "2024-01-02",
            "instrument": "A",
            "h15_complete": True,
            "h15_signed_bps_return": 1,
        },
        {
            "timestamp": "2024-04-01",
            "instrument": "B",
            "h15_complete": True,
            "h15_signed_bps_return": -1,
        },
    ]
    assert concentration(events) == (2 / 3, 2 / 3, 1)


def test_existing_empirical_artifact_is_never_overwritten(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "events.jsonl"
    artifact.write_text("keep")
    with pytest.raises(Stage0Error, match="refusing to overwrite"):
        stage0.main(["--output-dir", str(output)])
    assert artifact.read_text() == "keep"


def test_event_ids_and_bootstrap_are_deterministic():
    stamp = datetime(2024, 1, 1, tzinfo=UTC)
    assert event_id(stamp, "EURUSD", "panel") == event_id(stamp, "EURUSD", "panel")
    events = [
        {
            "timestamp": "2024-01-02T00:00:00+00:00",
            "instrument": "EURUSD",
            "h15_signed_bps_return": 1.0,
        },
        {
            "timestamp": "2024-02-02T00:00:00+00:00",
            "instrument": "GBPUSD",
            "h15_signed_bps_return": -0.5,
        },
    ]
    assert (
        month_block_bootstrap(events, replicates=20, seed=7)
        == month_block_bootstrap(events, replicates=20, seed=7)
    ).all()


def test_concentration_arithmetic():
    events = [
        {"timestamp": "2024-01-01", "instrument": "A", "h15_signed_bps_return": 2},
        {"timestamp": "2024-02-01", "instrument": "A", "h15_signed_bps_return": 0},
        {"timestamp": "2024-04-01", "instrument": "B", "h15_signed_bps_return": -1},
        {"timestamp": "2024-07-01", "instrument": "C", "h15_signed_bps_return": 1},
    ]
    assert concentration(events) == (0.5, 0.5, 2)


def test_concentration_result_is_standard_json_serializable():
    events = [
        {
            "timestamp": "2024-01-01",
            "instrument": "A",
            "h15_signed_bps_return": np.float64(2),
        }
    ]
    result = concentration(events)
    assert type(result[0]) is float
    assert type(result[1]) is float
    assert type(result[2]) is int
    assert json.loads(json.dumps({"concentration": result}))["concentration"] == [
        1.0,
        1.0,
        1,
    ]
