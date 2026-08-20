import csv
import hashlib
import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.bollinger_benchmark import build_bollinger_features
from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.research import Direction, build_research_observations
from mr_lab.sessions import DEFAULT_SESSION_SPEC
from mr_lab.stage4a import (
    FrozenSignal,
    diagnose_event,
    diagnose_events,
    events_to_jsonl,
    frozen_signal_from_bollinger,
    reversion_fraction,
)
from mr_lab.stage4a_reporting import (
    MATRIX_GROUP_FIELDS,
    QUANTILE_CONVENTION,
    aggregate_stage4a_events,
    matrix_to_csv,
    render_report,
    write_stage4a_outputs,
)
from mr_lab.vwap_benchmark import build_vwap_features
from mr_lab.vwap_m1_robustness import build_canonical_m1_vwap_features

T = datetime(2024, 6, 3, 10, tzinfo=UTC)


def bar(
    available_at,
    close,
    *,
    timeframe="1m",
    instrument="EURUSD",
    high=None,
    low=None,
):
    duration = Timeframe(timeframe).duration
    high = close if high is None else high
    low = close if low is None else low
    return Bar(
        instrument,
        Timeframe(timeframe),
        available_at - duration,
        available_at,
        available_at,
        close,
        high,
        low,
        close,
        PriceBasis.BID,
        1.0,
        VolumeSemantics.QUOTE_ACTIVITY,
    )


def signal(direction=Direction.LONG):
    p0, e0, z = (
        (100.0, 101.0, -2.0) if direction is Direction.LONG else (100.0, 99.0, 2.0)
    )
    return FrozenSignal(
        "EURUSD",
        T,
        "native_timeframe_vwap",
        Timeframe("5m"),
        "london",
        20,
        1.5,
        direction,
        p0,
        e0,
        p0 - e0,
        z,
        "corpus:2024",
        "dataset:2024",
        "strategy:frozen",
    )


def path(prices=None, *, include_pre=True):
    prices = prices or {}
    items = []
    if include_pre:
        for minute in range(-30, 1):
            items.append(bar(T + timedelta(minutes=minute), 97.0 + (minute + 30) / 10))
    for minute in range(1, 121):
        items.append(bar(T + timedelta(minutes=minute), prices.get(minute, 100.0)))
    return tuple(items)


@pytest.mark.parametrize(
    ("direction", "price", "expected"),
    (
        (Direction.LONG, 100.5, 0.5),
        (Direction.SHORT, 99.5, 0.5),
        (Direction.LONG, 99.5, -0.5),
        (Direction.SHORT, 100.5, -0.5),
        (Direction.LONG, 101.0, 1.0),
        (Direction.SHORT, 99.0, 1.0),
        (Direction.LONG, 101.4, 1.4),
        (Direction.SHORT, 98.6, 1.4),
    ),
)
def test_direction_safe_reversion_fraction(direction, price, expected):
    assert reversion_fraction(signal(direction), price) == pytest.approx(expected)


def test_frozen_e0_first_passage_and_m1_path_not_snapshots():
    prices = {2: 100.25, 3: 100.50, 4: 100.75, 5: 100.0, 6: 101.0, 7: 99.5, 8: 101.4}
    event = diagnose_event(signal(), path(prices))

    assert event.signal.e0 == 101.0
    assert [(item.level, item.time_to_hit_minutes) for item in event.first_passage] == [
        (0.25, 2),
        (0.5, 3),
        (0.75, 4),
        (1.0, 6),
    ]
    assert event.horizons[0].reversion_fraction == 0.0
    assert event.max_reversion_fraction == pytest.approx(1.4)


@pytest.mark.parametrize(
    ("direction", "prices"),
    (
        (Direction.LONG, {1: 99.5, 4: 101.5}),
        (Direction.SHORT, {1: 100.5, 4: 98.5}),
    ),
)
def test_mae_mfe_and_times_for_long_and_short(direction, prices):
    event = diagnose_event(signal(direction), path(prices))
    assert event.mae_price == pytest.approx(0.5)
    assert event.mfe_price == pytest.approx(1.5)
    assert event.mae_fraction_d0 == pytest.approx(0.5)
    assert event.mfe_fraction_d0 == pytest.approx(1.5)
    assert event.time_to_mae_minutes == 1
    assert event.time_to_mfe_minutes == 4


@pytest.mark.parametrize(
    ("direction", "extreme_name", "extreme", "adverse_name", "adverse"),
    (
        (Direction.LONG, "high", 101.4, "low", 99.4),
        (Direction.SHORT, "low", 98.6, "high", 100.6),
    ),
)
def test_intraminute_extremes_drive_passage_max_mfe_and_mae(
    direction, extreme_name, extreme, adverse_name, adverse
):
    special = bar(
        T + timedelta(minutes=1),
        100.0,
        **{extreme_name: extreme, adverse_name: adverse},
    )
    bars = (*path()[0:31], special, *path()[32:])
    event = diagnose_event(signal(direction), bars)

    assert [item.time_to_hit_minutes for item in event.first_passage] == [1] * 4
    assert event.max_reversion_fraction == pytest.approx(1.4)
    assert event.mfe_price == pytest.approx(1.4)
    assert event.mae_price == pytest.approx(0.6)
    # Both extremes are known only to have occurred in this minute. Their order
    # is deliberately neither inferred nor represented.
    assert event.time_to_mfe_minutes == event.time_to_mae_minutes == 1


def test_fixed_horizon_uses_close_and_names_movement_and_return_explicitly():
    special = bar(T + timedelta(minutes=5), 101.0, high=110.0, low=90.0)
    bars = (*path()[0:35], special, *path()[36:])
    event = diagnose_event(signal(), bars)
    horizon = event.horizons[0]

    assert horizon.signed_price_movement == pytest.approx(1.0)
    assert horizon.signed_arithmetic_return == pytest.approx(0.01)
    assert horizon.signed_return_bps == pytest.approx(100.0)
    assert horizon.signed_price_movement != horizon.signed_arithmetic_return
    assert horizon.reversion_fraction == pytest.approx(1.0)


def test_jpy_pip_conversion_uses_verified_instrument_precision():
    jpy_signal = replace(
        signal(),
        instrument="USDJPY",
        p0=150.0,
        e0=151.0,
        d0=-1.0,
    )
    bars = tuple(
        bar(
            T + timedelta(minutes=minute),
            150.01 if minute == 5 else 150.0,
            instrument="USDJPY",
        )
        for minute in range(-30, 121)
        if minute != 0
    )
    event = diagnose_event(jpy_signal, bars)
    assert event.horizons[0].signed_return_pips == pytest.approx(1.0)


def test_presignal_movements_are_causal_and_impulse_shares_are_descriptive():
    event = diagnose_event(signal(), path())
    by_horizon = {item.horizon_minutes: item for item in event.presignal}
    assert by_horizon[5].raw_price_movement == pytest.approx(0.5)
    assert by_horizon[15].raw_price_movement == pytest.approx(1.5)
    assert by_horizon[30].raw_price_movement == pytest.approx(3.0)
    assert by_horizon[5].signed_price_movement == pytest.approx(0.5)
    assert [by_horizon[h].impulse_share for h in (5, 15, 30)] == pytest.approx(
        [0.5, 1.5, 3.0]
    )


def test_incomplete_future_path_is_explicit_and_not_silently_shortened():
    bars = tuple(
        item for item in path() if item.available_at != T + timedelta(minutes=17)
    )
    event = diagnose_event(signal(), bars)
    assert not event.future_path_complete
    assert event.missing_future_minutes == (17,)
    assert event.first_passage == ()
    assert event.max_reversion_fraction is None
    assert event.mae_price is event.mfe_price is None
    assert len(event.horizons) == 5


def reporting_event(
    *,
    direction=Direction.LONG,
    threshold=2.0,
    family="native_timeframe_vwap",
    complete=True,
    offset=0,
    prices=None,
):
    frozen = replace(
        signal(direction),
        signal_timestamp=T + timedelta(hours=offset),
        threshold=threshold,
        benchmark_family=family,
        normalized_deviation=-threshold if direction is Direction.LONG else threshold,
    )
    bars = path(prices)
    if offset:
        bars = tuple(
            replace(
                item,
                open_time=item.open_time + timedelta(hours=offset),
                close_time=item.close_time + timedelta(hours=offset),
                available_at=item.available_at + timedelta(hours=offset),
            )
            for item in bars
        )
    if not complete:
        bars = tuple(
            item
            for item in bars
            if item.available_at != frozen.signal_timestamp + timedelta(minutes=17)
        )
    return diagnose_event(frozen, bars)


def test_reporting_grouping_order_and_dimensions_never_pool():
    events = [
        reporting_event(
            direction=Direction.SHORT, threshold=2.5, family="bollinger", offset=3
        ),
        reporting_event(direction=Direction.LONG, threshold=1.5, offset=1),
        reporting_event(direction=Direction.SHORT, threshold=1.5, offset=2),
    ]
    rows = aggregate_stage4a_events(reversed(events))

    assert tuple(rows[0])[: len(MATRIX_GROUP_FIELDS)] == MATRIX_GROUP_FIELDS
    identities = [tuple(row[field] for field in MATRIX_GROUP_FIELDS) for row in rows]
    assert identities == sorted(identities)
    assert len(rows) == 3
    assert {
        (row["benchmark_family"], row["direction"], row["threshold"]) for row in rows
    } == {
        ("bollinger", "short", 2.5),
        ("native_timeframe_vwap", "long", 1.5),
        ("native_timeframe_vwap", "short", 1.5),
    }
    assert {row["stage4a_methodology_id"] for row in rows} == {
        events[0].stage4a_methodology_id
    }


def test_reporting_denominators_hits_missing_values_and_presignal():
    complete_hit = reporting_event(prices={5: 101.0, 10: 101.0})
    incomplete = reporting_event(complete=False, offset=1, prices={5: 101.0})
    # Remove the exact 15-minute snapshot while retaining the deliberately
    # incomplete event to prove it is excluded rather than zero-filled.
    incomplete = replace(
        incomplete,
        horizons=tuple(
            value for value in incomplete.horizons if value.horizon_minutes != 15
        ),
    )
    row = aggregate_stage4a_events((incomplete, complete_hit))[0]

    assert row["observation_count_total"] == 2
    assert row["observation_count_complete_path"] == 1
    assert row["observation_count_incomplete_path"] == 1
    assert row["complete_path_fraction"] == 0.5
    assert row["n_h5"] == 2 and row["n_h15"] == 1
    assert row["mean_pips_h15"] == complete_hit.horizons[1].signed_return_pips
    assert row["eligible_hit50_complete_path_count"] == 1
    assert row["hit50_count"] == 1 and row["hit50_rate"] == 1.0
    assert row["median_t50_minutes"] == 5.0
    assert row["n_presignal_h5"] == 2
    assert row["median_impulse_share_h5"] == pytest.approx(0.5)


def test_reporting_inclusive_quantiles_for_path_extremes():
    events = []
    for offset, adverse, favorable in (
        (0, 0.0, 0.0),
        (1, 1.0, 1.0),
        (2, 2.0, 2.0),
        (3, 3.0, 3.0),
    ):
        event = reporting_event(offset=offset)
        events.append(
            replace(
                event,
                mae_pips=adverse,
                mae_fraction_d0=adverse,
                mfe_pips=favorable,
                mfe_fraction_d0=favorable,
                max_reversion_fraction=favorable,
            )
        )
    row = aggregate_stage4a_events(events)[0]

    assert "statistics.quantiles" in QUANTILE_CONVENTION
    assert (
        row["median_mae_pips"],
        row["p75_mae_pips"],
        row["p90_mae_pips"],
    ) == pytest.approx((1.5, 2.25, 2.7))
    assert (
        row["median_mfe_pips"],
        row["p75_mfe_pips"],
        row["p90_mfe_pips"],
    ) == pytest.approx((1.5, 2.25, 2.7))
    assert (
        row["median_max_reversion_fraction"],
        row["p75_max_reversion_fraction"],
        row["p90_max_reversion_fraction"],
    ) == pytest.approx((1.5, 2.25, 2.7))


def test_report_structure_is_unranked_and_has_no_trade_metrics():
    events = tuple(
        reporting_event(threshold=value, offset=index)
        for index, value in enumerate((2.5, 1.0, 2.0, 1.5))
    )
    report = render_report(aggregate_stage4a_events(events))

    assert report.index("## Predeclared anchor") < report.index("## Threshold response")
    assert all(f"### |z| >= {value:.1f}" in report for value in (1.0, 1.5, 2.0, 2.5))
    assert "## Directional asymmetry" in report and "## Benchmark robustness" in report
    assert "not by outcomes" in report and "selected or optimized" in report
    for forbidden in ("Sharpe", "profit factor", "annualized return", "trade win rate"):
        assert forbidden not in report


def test_directional_report_uses_only_anchor_threshold_and_preserves_directions():
    rows = [
        dict(row)
        for row in aggregate_stage4a_events(
            (
                reporting_event(direction=Direction.LONG, threshold=2.0),
                reporting_event(direction=Direction.SHORT, threshold=2.0, offset=1),
                reporting_event(direction=Direction.LONG, threshold=2.5, offset=2),
            )
        )
    ]
    markers = {
        ("long", 2.0): "ANCHOR_LONG",
        ("short", 2.0): "ANCHOR_SHORT",
        ("long", 2.5): "HIGHER_THRESHOLD_LONG",
    }
    for row in rows:
        row["mean_pips_h60"] = markers[(row["direction"], row["threshold"])]

    report = render_report(rows)
    threshold_response = report.split("## Threshold response", 1)[1].split(
        "## Directional asymmetry", 1
    )[0]
    directional = report.split("## Directional asymmetry", 1)[1].split(
        "## Benchmark robustness", 1
    )[0]

    assert "ANCHOR_LONG" in directional
    assert "ANCHOR_SHORT" in directional
    assert "HIGHER_THRESHOLD_LONG" not in directional
    assert "HIGHER_THRESHOLD_LONG" in threshold_response


def test_four_outputs_are_byte_deterministic_and_hashes_match(tmp_path):
    events = (reporting_event(direction=Direction.SHORT, offset=1), reporting_event())
    first = tmp_path / "first"
    second = tmp_path / "second"
    paths = write_stage4a_outputs(reversed(events), first)
    write_stage4a_outputs(events, second)

    assert set(paths) == {"events.jsonl", "matrix.csv", "summary.json", "report.md"}
    for name in paths:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    summary = json.loads((first / "summary.json").read_text())
    assert summary["stage4a_methodology_ids"] == [events[0].stage4a_methodology_id]
    for name in ("events.jsonl", "matrix.csv", "report.md"):
        assert (
            summary["hashes"][name]
            == hashlib.sha256((first / name).read_bytes()).hexdigest()
        )
    records = [
        json.loads(line) for line in (first / "events.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2 and all(
        record["stage4a_methodology_id"] == events[0].stage4a_methodology_id
        for record in records
    )
    csv_rows = list(
        csv.DictReader(io.StringIO(matrix_to_csv(aggregate_stage4a_events(events))))
    )
    assert len(csv_rows) == 2


def test_insufficient_presignal_history_is_explicit_per_horizon():
    bars = tuple(
        bar(T + timedelta(minutes=minute), 100.0)
        for minute in range(-6, 121)
        if minute != 0
    )
    event = diagnose_event(signal(), bars)
    by_horizon = {item.horizon_minutes: item for item in event.presignal}
    assert by_horizon[5].impulse_share == 0.0
    assert by_horizon[15].impulse_share is None
    assert by_horizon[30].raw_price_movement is None


def test_future_values_cannot_change_frozen_signal_or_eligibility():
    frozen = signal()
    first = diagnose_event(frozen, path({1: 200.0}))
    second = diagnose_event(frozen, path({1: 1.0}))
    assert first.signal == second.signal == frozen
    assert first.signal.e0 == second.signal.e0
    assert first.signal.normalized_deviation == second.signal.normalized_deviation
    assert first.signal.direction is second.signal.direction


def test_stable_order_and_jsonl_serialization():
    later = replace(signal(), signal_timestamp=T + timedelta(minutes=5))
    events = diagnose_events((later, signal()), path())
    encoded = events_to_jsonl(events)
    assert events[0].signal.signal_timestamp == T
    assert encoded == events_to_jsonl(events)
    assert [
        json.loads(line)["signal"]["signal_timestamp"] for line in encoded.splitlines()
    ] == [T.isoformat(), (T + timedelta(minutes=5)).isoformat()]


def test_multi_event_diagnosis_prepares_m1_index_once(monkeypatch):
    import mr_lab.stage4a as stage4a

    calls = 0
    original = stage4a._prepare_m1_index

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stage4a, "_prepare_m1_index", counted)
    diagnose_events((signal(), replace(signal(), threshold=2.0)), path())
    assert calls == 1


def test_methodology_id_commits_to_each_semantic_constant(monkeypatch):
    import mr_lab.stage4a as stage4a

    original = stage4a._methodology_id()
    semantic_names = (
        "FROZEN_EQUILIBRIUM_RULE",
        "REVERSION_FRACTION_DEFINITION",
        "FIXED_HORIZON_PRICE_RULE",
        "HORIZON_DIAGNOSTIC_DEFINITION",
        "PATH_EXTREME_RULE",
        "FIRST_PASSAGE_DEFINITION",
        "MAE_DEFINITION",
        "MFE_DEFINITION",
        "PATH_TIME_RESOLUTION",
        "MISSING_FUTURE_PATH_POLICY",
        "PRESIGNAL_MOVEMENT_DEFINITION",
        "IMPULSE_SHARE_DEFINITION",
        "PIP_SIZE_RULE",
    )
    for name in semantic_names:
        with monkeypatch.context() as context:
            context.setattr(stage4a, name, f"{getattr(stage4a, name)}-changed")
            assert stage4a._methodology_id() != original


def _vwap_feature_inputs():
    research_bars = [
        bar(
            T - timedelta(minutes=5 * (20 - index)),
            100.0 + (0.01 if index % 2 else 0.0) if index < 20 else 96.0,
            timeframe="5m",
        )
        for index in range(21)
    ]
    m1_bars = [
        bar(T - timedelta(minutes=minute), 100.0) for minute in range(120, -1, -1)
    ]
    return (
        build_research_observations(research_bars, DEFAULT_SESSION_SPEC),
        build_research_observations(m1_bars, DEFAULT_SESSION_SPEC),
    )


@pytest.mark.parametrize(
    ("family", "canonical"),
    (("native_timeframe_vwap", False), ("canonical_m1_vwap", True)),
)
def test_existing_frozen_vwap_adapters_preserve_feature_values(family, canonical):
    research, m1 = _vwap_feature_inputs()
    if canonical:
        features = build_canonical_m1_vwap_features(
            m1, research, DEFAULT_SESSION_SPEC, 20
        )
    else:
        features = build_vwap_features(research, DEFAULT_SESSION_SPEC, 20)
    before = features[-1]
    from mr_lab.stage4a import frozen_signal_from_vwap

    frozen = frozen_signal_from_vwap(
        before,
        threshold=1.0,
        benchmark_family=family,
        source_corpus_id="corpus:2024",
        assembled_dataset_id="dataset:2024",
        strategy_spec_id=f"{family}:frozen",
    )
    assert frozen is not None
    assert features[-1] == before
    assert frozen.p0 == before.price
    assert frozen.e0 == before.vwap
    assert frozen.benchmark_family == family


def test_existing_frozen_bollinger_path_integrates_without_changing_features():
    closes = [100.0] * 19 + [96.0]
    observations = build_research_observations(
        [
            bar(T - timedelta(minutes=5 * (19 - index)), close, timeframe="5m")
            for index, close in enumerate(closes)
        ],
        DEFAULT_SESSION_SPEC,
    )
    features = build_bollinger_features(observations, 20)
    before = features[-1]
    frozen = frozen_signal_from_bollinger(
        before,
        threshold=1.0,
        session="london",
        source_corpus_id="corpus:2024",
        assembled_dataset_id="dataset:2024",
        strategy_spec_id="bollinger:frozen",
    )
    assert frozen is not None
    diagnose_event(frozen, path())
    assert features[-1] == before
    assert frozen.p0 == before.observation.bar.close
    assert frozen.e0 == before.middle
