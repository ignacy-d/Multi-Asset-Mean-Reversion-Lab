import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics, resample_bars
from mr_lab.replay import (
    FrozenOuReplayEngine,
    FrozenOuResearchBuilder,
    build_frozen_ou_decisions,
    compare_event_streams,
)
from mr_lab.stage4b_runner import assemble_signal_states

START = datetime(2024, 1, 8, tzinfo=UTC)
MANIFEST = {
    "corpus_id": "synthetic-2024-research-corpus",
    "assembled_dataset_id": "synthetic-2024-research-dataset",
}


def synthetic_research_m1():
    """Four UTC 00:00-17:00 prefixes with dislocations and quiet re-arms."""
    output = []
    sequence = 0
    for day in range(4):
        for minute in range(60 * 17):
            open_time = START + timedelta(days=day, minutes=minute)
            pulse = 0.0
            price = (
                1.1
                + sequence * 0.0000002
                + 0.000002 * math.sin(sequence)
                + pulse
                - 0.0000005 * max(minute - 700, 0)
                + 0.000001 * max(minute - 800, 0)
            )
            output.append(
                Bar(
                    "EURUSD",
                    Timeframe("1m"),
                    open_time,
                    open_time + timedelta(minutes=1),
                    open_time + timedelta(minutes=1),
                    price,
                    price + 0.00002,
                    price - 0.00002,
                    price,
                    PriceBasis.BID,
                    1.0,
                    VolumeSemantics.QUOTE_ACTIVITY,
                )
            )
            sequence += 1
    return tuple(output)


def frozen_m15_london_short(states):
    return tuple(
        state
        for state in states
        if str(state.signal_timeframe) == "15m"
        and state.session == "london"
        and state.direction.name == "SHORT"
        and state.benchmark_family in {"vwap", "vwap-canonical-m1"}
        and state.lookback in {20, 40}
    )


def test_actual_stage4b_research_assembly_exactly_matches_candle_replay():
    m1 = synthetic_research_m1()
    # The reference deliberately enters through the complete Stage4B research
    # assembly path, rather than through the replay builder.
    research_states = frozen_m15_london_short(
        assemble_signal_states(SimpleNamespace(bars=m1), MANIFEST)
    )
    reference = build_frozen_ou_decisions(research_states)
    builder = FrozenOuResearchBuilder(MANIFEST)
    engine = FrozenOuReplayEngine(builder)
    replay = engine.run(m1)

    report = compare_event_streams(reference, replay)
    assert report.mismatch_count == 0
    assert report.reference_event_count == report.replay_event_count > 0

    cells = {
        (record.details.benchmark_family, record.details.lookback)
        for record in reference
    }
    assert cells == {
        ("vwap", 20),
        ("vwap", 40),
        ("vwap-canonical-m1", 20),
        ("vwap-canonical-m1", 40),
    }
    assert len({record.signal.source_event_id for record in reference}) == len(
        reference
    )
    # Later re-armed candidates have a full 128-transition fit. This deterministic
    # path is structurally invalid, so the frozen eligibility gate must fail closed.
    assert any(record.details.process_status == "invalid" for record in reference)
    assert not any(record.details.eligible for record in reference)
    assert all(record.details.eligibility_reason for record in reference)

    # Replay emits the exact batch M15 count while retaining only bounded rolling
    # feature and OU state rather than either complete input history.
    batch = resample_bars(m1, Timeframe("15m"))
    assert engine.completed_m15_count == len(batch.bars)
    assert batch.incomplete_windows == ()
    assert builder.retained_rolling_values <= 20 + 40
    assert engine.retained_transition_count <= 4 * 128
