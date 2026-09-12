from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics, resample_bars
from mr_lab.ornstein_uhlenbeck import (
    FrozenOuEligibilityFilter,
    ResidualObservation,
    build_candidate_ou_states,
    candidate_process_keys,
    frozen_ou_eligibility_spec,
)
from mr_lab.replay import FrozenOuReplayEngine, IncrementalResampler
from mr_lab.replay.engine import ReplayError
from mr_lab.replay.parity import ParityError, compare_event_streams
from mr_lab.research import Direction
from mr_lab.signals import AlphaSignal
from mr_lab.stage4b import SignalState, deduplicate_states
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


def state_builder(bars):
    spec_id = VwapStrategySpec(20, 2.0).strategy_spec_id
    states = []
    for index, bar in enumerate(bars):
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


def reference_signals(states):
    spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
    candidates = deduplicate_states(states)
    ou_states = build_candidate_ou_states(
        states, spec.process_spec, candidate_process_keys(candidates)
    )
    index = {(state.process_id, state.available_at): state for state in ou_states}
    gate = FrozenOuEligibilityFilter(spec, ou_states)
    output = []
    for candidate in candidates:
        signal = candidate.signal
        observation = ResidualObservation(
            signal.instrument,
            signal.benchmark_family,
            signal.signal_timeframe,
            signal.session,
            signal.lookback,
            signal.strategy_spec_id,
            signal.signal_timestamp,
            signal.p0,
            signal.e0,
        )
        state = index[(observation.process_id, signal.signal_timestamp)]
        decision = gate.evaluate(candidate)
        metadata = dict(decision.metadata)
        output.append(
            AlphaSignal(
                candidate.candidate_event_id,
                signal.signal_timestamp,
                signal.instrument,
                signal.direction.name,
                signal.benchmark_family,
                signal.lookback,
                signal.p0,
                signal.e0,
                signal.normalized_deviation,
                state.process_id,
                state.process_spec_id,
                state.status,
                state.is_structurally_valid,
                state.invalid_reason,
                state.ornstein_uhlenbeck_score,
                state.half_life_minutes,
                decision.eligible,
                metadata["eligibility_reason"],
                decision.filter_spec_id,
            )
        )
    return tuple(output)


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
    expected = reference_signals(state_builder(batch_m15))
    actual = FrozenOuReplayEngine(state_builder).run(bars)
    report = compare_event_streams(expected, actual)
    assert report.passed and report.mismatch_count == 0
    assert report.reference_event_count == report.replay_event_count == 2
    assert [x.source_event_id for x in actual] == [x.source_event_id for x in expected]
    assert [x.process_status for x in actual] == [x.process_status for x in expected]
    assert [x.eligible for x in actual] == [x.eligible for x in expected]


def test_parity_fails_on_first_field_mismatch_with_compact_categories():
    bars = m1_bars()
    signal = FrozenOuReplayEngine(state_builder).run(bars)[0]
    changed = AlphaSignal(
        "changed",
        *tuple(
            getattr(signal, field)
            for field in signal.__dataclass_fields__
            if field != "source_event_id"
        ),
    )
    with pytest.raises(ParityError) as exc:
        compare_event_streams((signal,), (changed,))
    assert exc.value.report.first_mismatch.field == "source_event_id"
    assert exc.value.report.identity_mismatches == 1
