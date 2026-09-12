import math
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


def make_state(index, bar):
    spec_id = VwapStrategySpec(20, 2.0).strategy_spec_id
    # A deterministic synthetic residual series.  Both prices and the candidate
    # decision use only this completed observation and its monotonic index.
    residual = 0.0007 + 0.0001 * ((index % 11) - 5) + index * 0.0000001
    z = 2.1 if index in (129, 132) else 1.0
    return SignalState(
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


class SyntheticStateBuilder:
    def __init__(self):
        self.m1_count = 0
        self.m15_count = 0
        self.states = []

    def push_m1(self, bar):
        self.m1_count += 1

    def push_m15(self, bar):
        state = make_state(self.m15_count, bar)
        self.m15_count += 1
        self.states.append(state)
        return (state,)

    def advance(self, prefix):
        for bar in prefix.completed_m1:
            self.push_m1(bar)
        return tuple(
            state for bar in prefix.completed_m15 for state in self.push_m15(bar)
        )


def state_builder_for_batch(bars):
    states = []
    for index, bar in enumerate(bars):
        states.append(make_state(index, bar))
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
    expected = build_frozen_ou_decisions(state_builder_for_batch(batch_m15))
    actual = FrozenOuReplayEngine(SyntheticStateBuilder()).run(bars)
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
    record = FrozenOuReplayEngine(SyntheticStateBuilder()).run(bars)[0]
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


def test_builder_processes_each_m1_and_m15_exactly_once():
    bars = m1_bars(30)
    builder = SyntheticStateBuilder()
    engine = FrozenOuReplayEngine(builder)
    engine.run(bars)
    assert builder.m1_count == 30
    assert builder.m15_count == engine.completed_m15_count == 2


def test_future_builder_state_is_rejected():
    class Future(SyntheticStateBuilder):
        def push_m15(self, bar):
            state = super().push_m15(bar)[0]
            return (replace(state, timestamp=bar.available_at + timedelta(minutes=15)),)

    with pytest.raises(ReplayError, match="future information"):
        FrozenOuReplayEngine(Future()).run(m1_bars(15))


def test_frozen_ou_positive_eligibility_path():
    deviations = [0.0]
    seed = 1
    for _ in range(128):
        seed = (1103515245 * seed + 12345) % 2**31
        innovation = (seed / 2**31 - 0.5) * 0.0001
        deviations.append(0.8 * deviations[-1] + innovation)
    deviations[-1] = 0.0005
    states = tuple(
        SignalState(
            "EURUSD",
            START + timedelta(minutes=15 * (index + 1)),
            "vwap",
            Timeframe("15m"),
            "london",
            Direction.SHORT,
            20,
            1.0 + deviation,
            1.0,
            2.1 if index == 128 else 1.0,
            "synthetic-2024",
            "synthetic-2024-dataset",
            "strategy-v1",
            index == 128,
        )
        for index, deviation in enumerate(deviations)
    )
    details = build_frozen_ou_decisions(states)[0].details
    assert details.process_is_structurally_valid and details.process_status == "valid"
    assert math.isfinite(details.ou_score) and details.ou_score > 1.5
    assert details.half_life_minutes <= 120
    assert details.eligible and details.eligibility_reason == "eligible"

    class PositiveBuilder:
        index = 0

        def push_m1(self, bar):
            pass

        def push_m15(self, bar):
            state = states[self.index]
            self.index += 1
            assert state.timestamp == bar.available_at
            return (state,)

        def advance(self, prefix):
            for bar in prefix.completed_m1:
                self.push_m1(bar)
            return tuple(
                state for bar in prefix.completed_m15 for state in self.push_m15(bar)
            )

    replay = FrozenOuReplayEngine(PositiveBuilder()).run(m1_bars(15 * len(states)))
    assert compare_event_streams(build_frozen_ou_decisions(states), replay).passed
    assert replay[0].details.eligible
