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
