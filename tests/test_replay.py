from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics, resample_bars
from mr_lab.replay import (
    FrozenOuReplayEngine,
    IncrementalResampler,
    build_frozen_ou_decisions,
)
from mr_lab.replay.engine import ReplayError
from mr_lab.replay.parity import ParityError, compare_event_streams
from mr_lab.research import Direction
from mr_lab.signals import AlphaSignal
from mr_lab.stage4b import SignalState
from mr_lab.vwap_benchmark import VwapStrategySpec

START = datetime(2024, 2, 1, tzinfo=UTC)


def m1_bars(count=15 * 135, missing=frozenset()):
    bars = []
    for minute in range(count):
        if minute in missing:
            continue
        open_time = START + timedelta(minutes=minute)
        price = 1.1 + minute * 0.000001
        bars.append(
            Bar(
                "EURUSD",
                Timeframe("1m"),
                open_time,
                open_time + timedelta(minutes=1),
                open_time + timedelta(minutes=1),
                price,
                price + 0.00002,
                price - 0.00002,
                price + 0.000001,
                PriceBasis.BID,
                1.0,
                VolumeSemantics.QUOTE_ACTIVITY,
            )
        )
    return tuple(bars)


def state_builder(prefix):
    spec_id = VwapStrategySpec(20, 2.0).strategy_spec_id
    states = []
    for index, bar in enumerate(prefix.completed_m15):
        # A deterministic synthetic residual series.  Both prices and the
        # candidate decision use only the current completed prefix.
        residual = 0.0007 + 0.0001 * ((index % 11) - 5) + index * 0.0000001
        z = 2.1 if index in (129, 132) else 1.0
        states.append(
            SignalState(
                bar.instrument,
                bar.available_at,
                "vwap",
                bar.timeframe,
                "london",
                Direction.SHORT,
                20,
                bar.close,
                bar.close - residual,
                z,
                "synthetic-2024",
                "synthetic-2024-dataset",
                spec_id,
                z > 2.0,
            )
        )
    return tuple(states)


def test_incremental_m1_to_m15_exactly_equals_batch():
    bars = m1_bars(45)
    stream = IncrementalResampler()
    actual = tuple(output for bar in bars if (output := stream.push(bar)) is not None)
    assert stream.finish() == ()
    assert actual == resample_bars(bars, Timeframe("15m")).bars


def test_prefix_invariance_and_no_early_emission():
    bars = m1_bars(31)
    short = IncrementalResampler()
    prefix = tuple(x for bar in bars[:30] if (x := short.push(bar)) is not None)
    full = resample_bars(bars, Timeframe("15m")).bars
    assert prefix == full[:2]
    assert short.push(bars[30]) is None


def test_gap_and_trailing_incomplete_fail_closed():
    bars = m1_bars(30, frozenset({7}))
    stream = IncrementalResampler()
    output = tuple(x for bar in bars if (x := stream.push(bar)) is not None)
    gaps = stream.finish()
    assert output == resample_bars(bars, Timeframe("15m")).bars
    assert len(output) == 1 and len(gaps) == 1 and gaps[0].observed_components == 14


def test_duplicate_and_out_of_order_are_rejected():
    bars = m1_bars(2)
    stream = IncrementalResampler()
    stream.push(bars[0])
    with pytest.raises(ReplayError, match="duplicate"):
        stream.push(bars[0])
    stream.push(bars[1])
    with pytest.raises(ReplayError, match="chronological"):
        stream.push(bars[0])


def test_replay_exact_identity_ou_eligibility_and_rearm_parity():
    bars = m1_bars()
    batch_m15 = resample_bars(bars, Timeframe("15m")).bars
    from mr_lab.replay.engine import ReplayPrefix

    expected = build_frozen_ou_decisions(
        state_builder(ReplayPrefix(bars, batch_m15, batch_m15[-1].available_at))
    )
    actual = FrozenOuReplayEngine(state_builder).run(bars)
    report = compare_event_streams(expected, actual)
    assert report.passed and report.mismatch_count == 0
    assert report.reference_event_count == report.replay_event_count == 2
    assert [x.signal.source_event_id for x in actual] == [
        x.signal.source_event_id for x in expected
    ]
    assert [x.details.process_status for x in actual] == [
        x.details.process_status for x in expected
    ]
    assert [x.details.eligible for x in actual] == [
        x.details.eligible for x in expected
    ]


def test_parity_fails_on_first_field_mismatch_with_compact_categories():
    bars = m1_bars()
    record = FrozenOuReplayEngine(state_builder).run(bars)[0]
    changed = replace(record, signal=replace(record.signal, source_event_id="changed"))
    with pytest.raises(ParityError) as exc:
        compare_event_streams((record,), (changed,))
    assert exc.value.report.first_mismatch.field == "signal.source_event_id"
    assert exc.value.report.identity_mismatches == 1


def test_generic_alpha_signal_does_not_require_ou_fields():
    signal = AlphaSignal("bollinger-v1", "event-1", "EURUSD", START, "LONG")
    assert signal.module_id == "bollinger-v1" and signal.reference is None
    with pytest.raises(ValueError, match="source_event_id"):
        AlphaSignal("bollinger-v1", "", "EURUSD", START, "LONG")


def test_builder_receives_exact_causal_m1_and_m15_prefixes():
    bars = m1_bars(30)
    observed = []

    def capture(prefix):
        observed.append(prefix)
        return ()

    FrozenOuReplayEngine(capture).run(bars)
    assert [len(prefix.completed_m1) for prefix in observed] == [15, 30]
    assert [len(prefix.completed_m15) for prefix in observed] == [1, 2]
    assert all(
        prefix.completed_m1[-1].available_at == prefix.available_at
        for prefix in observed
    )


def test_future_builder_state_is_rejected():
    def future(prefix):
        state = state_builder(prefix)[-1]
        return (replace(state, timestamp=prefix.available_at + timedelta(minutes=15)),)

    with pytest.raises(ReplayError, match="future information"):
        FrozenOuReplayEngine(future).run(m1_bars(15))
