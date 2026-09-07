from __future__ import annotations

import lzma
import struct
from datetime import UTC, date, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, QuoteStatus, Timeframe, VolumeSemantics
from mr_lab.data.quotes import QuoteSynchronizationError, synchronize_quotes
from mr_lab.integrations.nautilus import FrozenSignal, export_frozen_signals
from mr_lab.providers.dukascopy import AcquisitionError, acquire, build_url
from mr_lab.providers.dukascopy_bi5 import (
    Bi5ParseError,
    build_side_dataset_metadata,
    parse_m1_ask_bars,
)
from mr_lab.spread import SpreadFeatureSpec, causal_spread_features
from mr_lab.stage4c_observed import (
    ObservedExecutionSpec,
    ObservedExecutionUnavailable,
    observed_execution_result,
)
from mr_lab.structure import StructureFeatureSpec, confirmed_swings, extreme_reclaims

T = datetime(2024, 1, 2, tzinfo=UTC)


def bar(index: int, close: float, side: PriceBasis, *, high=None, low=None) -> Bar:
    start = T + timedelta(minutes=index)
    return Bar(
        "EURUSD",
        Timeframe("1m"),
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=1),
        close,
        close if high is None else high,
        close if low is None else low,
        close,
        side,
        1.0,
        VolumeSemantics.QUOTE_ACTIVITY,
    )


def quotes(count=4):
    bids = tuple(bar(i, 1.1000 + i * 0.0001, PriceBasis.BID) for i in range(count))
    asks = tuple(bar(i, 1.1002 + i * 0.0001, PriceBasis.ASK) for i in range(count))
    return synchronize_quotes(bids, asks, pip_size=0.0001)[0]


def test_native_ask_url_parser_and_identity_are_explicit():
    assert build_url("EURUSD", date(2024, 1, 2), PriceBasis.ASK).endswith(
        "/ASK_candles_min_1.bi5"
    )
    decoded = struct.pack(">5if", 0, 110020, 110021, 110010, 110030, 2.0)
    payload = lzma.compress(decoded)
    parsed = parse_m1_ask_bars(payload, date(2024, 1, 2))
    assert parsed[0].price_basis is PriceBasis.ASK
    ask_id = build_side_dataset_metadata(
        payload, date(2024, 1, 2), "EURUSD", PriceBasis.ASK
    )
    bid_id = build_side_dataset_metadata(
        payload, date(2024, 1, 2), "EURUSD", PriceBasis.BID
    )
    assert ask_id.dataset_id != bid_id.dataset_id
    with pytest.raises(Bi5ParseError, match="partial"):
        parse_m1_ask_bars(lzma.compress(decoded + b"x"), date(2024, 1, 2))


def test_synchronizer_reports_missing_crossed_nonpositive_and_gaps():
    bids = (
        bar(0, 1.0, PriceBasis.BID),
        bar(1, 2.0, PriceBasis.BID),
        bar(4, 4.0, PriceBasis.BID),
    )
    asks = (
        bar(0, 1.1, PriceBasis.ASK),
        bar(1, 1.9, PriceBasis.ASK),
        bar(2, 3.0, PriceBasis.ASK),
        bar(4, 4.0, PriceBasis.ASK),
    )
    rows, diagnostics = synchronize_quotes(bids, asks, pip_size=0.1)
    assert [x.status for x in rows] == [
        QuoteStatus.SYNCHRONIZED,
        QuoteStatus.CROSSED,
        QuoteStatus.ASK_ONLY,
        QuoteStatus.NON_POSITIVE,
    ]
    assert diagnostics == type(diagnostics)(1, 0, 1, 1, 1, 1)
    assert rows[0].spread_price == pytest.approx(0.1)
    assert rows[0].spread_pips == pytest.approx(1)
    assert rows[1].spread_price is None


def test_synchronizer_rejects_wrong_side_and_does_not_fill():
    with pytest.raises(QuoteSynchronizationError, match="another side"):
        synchronize_quotes((bar(0, 1, PriceBasis.ASK),), (), pip_size=0.1)
    rows, _ = synchronize_quotes((bar(0, 1, PriceBasis.BID),), (), pip_size=0.1)
    assert rows[0].ask is None and rows[0].spread_pips is None


def test_spread_features_are_prefix_invariant_and_available_causally():
    source = quotes(5)
    spec = SpreadFeatureSpec(lookback=3)
    full = causal_spread_features(source, spec)
    prefix = causal_spread_features(source[:3], spec)
    assert full[:3] == prefix
    assert all(x.available_at == source[i].available_at for i, x in enumerate(full))
    altered_future = (
        source[:3]
        + synchronize_quotes(
            (bar(3, 1, PriceBasis.BID),),
            (bar(3, 1.01, PriceBasis.ASK),),
            pip_size=0.0001,
        )[0]
    )
    assert causal_spread_features(altered_future, spec)[:3] == prefix


def test_confirmed_swing_is_delayed_and_prefix_invariant():
    bars = tuple(
        bar(i, close, PriceBasis.BID, high=high, low=close - 0.1)
        for i, (close, high) in enumerate(
            ((1, 1), (2, 2), (3, 5), (2, 3), (1, 2), (4, 4))
        )
    )
    spec = StructureFeatureSpec(2, 2, 2)
    full = confirmed_swings(bars, spec)
    prefix = confirmed_swings(bars[:5], spec)
    event = next(x for x in full if x.kind == "confirmed_swing_high")
    assert event.event_time == bars[2].open_time
    assert event.available_at == bars[4].available_at > event.event_time
    assert full[: len(prefix)] == prefix


def test_reclaim_uses_only_trailing_completed_bars():
    bars = (
        bar(0, 2, PriceBasis.BID, high=3, low=1),
        bar(1, 2, PriceBasis.BID, high=3, low=1),
        bar(2, 2.5, PriceBasis.BID, high=4, low=2),
    )
    result = extreme_reclaims(bars, StructureFeatureSpec(1, 1, 2))
    assert [(x.kind, x.available_at) for x in result] == [
        ("high_liquidity_sweep_reclaim", bars[2].available_at)
    ]


def test_observed_execution_uses_correct_sides_and_is_deterministic():
    source = quotes(3)
    spec = ObservedExecutionSpec(commission_round_turn_pips=0.5)
    long = observed_execution_result(
        direction="LONG",
        entry_at=T,
        exit_at=T + timedelta(minutes=2),
        quotes=source,
        spec=spec,
    )
    short = observed_execution_result(
        direction="SHORT",
        entry_at=T,
        exit_at=T + timedelta(minutes=2),
        quotes=source,
        spec=spec,
    )
    assert long["entry_price"] == source[0].ask and long["exit_price"] == source[2].bid
    assert (
        short["entry_price"] == source[0].bid and short["exit_price"] == source[2].ask
    )
    assert long == observed_execution_result(
        direction="LONG",
        entry_at=T,
        exit_at=T + timedelta(minutes=2),
        quotes=source,
        spec=spec,
    )
    assert long["pathwise_tp_sl_status"] == "unavailable_not_resimulated"
    with pytest.raises(ObservedExecutionUnavailable):
        observed_execution_result(
            direction="LONG",
            entry_at=T,
            exit_at=T + timedelta(minutes=20),
            quotes=source,
            spec=spec,
        )


def test_ids_stable_and_semantic_changes_differ():
    first = SpreadFeatureSpec(60, 20, 2.0)
    assert first.identity == SpreadFeatureSpec(60, 20, 2.0).identity
    assert first.identity != SpreadFeatureSpec(61, 20, 2.0).identity


def test_acquisition_rejects_holdout_before_network(tmp_path):
    called = False

    def getter(url, timeout):
        nonlocal called
        called = True
        raise AssertionError("network must not be touched")

    with pytest.raises(AcquisitionError, match="holdout"):
        acquire(tmp_path, date(2025, 1, 1), getter=getter)
    assert not called


def test_nautilus_boundary_is_deterministic_and_dependency_free():
    signal = FrozenSignal(
        "s", "EURUSD", T, T + timedelta(minutes=1), "LONG", "strategy", "dataset"
    )
    assert export_frozen_signals((signal,)) == export_frozen_signals((signal,))
    assert "mr-lab-frozen-signals-v1" in export_frozen_signals((signal,))
