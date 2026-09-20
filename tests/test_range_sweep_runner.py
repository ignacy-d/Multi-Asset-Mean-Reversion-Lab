from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.range_sweep import (
    RangeSweepEvent,
    ReferenceFamily,
    ReferenceSide,
    SignalDirection,
)
from mr_lab.range_sweep_runner import (
    DECISION_CONTINUE,
    DECISION_PARK,
    RangeSweepRunnerError,
    aggregate_m1_to_m5,
    cost_profile_session,
    discovery_decision,
    execute_event,
    to_research_outcome,
)


def bar(minute: int, *, price: float = 1.1, volume: float = 1) -> Bar:
    start = datetime(2024, 1, 2, tzinfo=UTC) + timedelta(minutes=minute)
    return Bar(
        "EURUSD",
        Timeframe("1m"),
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=1),
        price,
        price + 0.001,
        price - 0.001,
        price + 0.0005,
        PriceBasis.BID,
        volume,
        VolumeSemantics.TICK,
    )


def event(timestamp: datetime, direction: SignalDirection) -> RangeSweepEvent:
    return RangeSweepEvent(
        "event-1",
        ReferenceFamily.PREVIOUS_FX_DAY,
        "instance",
        ReferenceSide.LOW if direction is SignalDirection.LONG else ReferenceSide.HIGH,
        1.1,
        direction,
        "EURUSD",
        timestamp,
        1.1,
        1.2,
        1.0,
        1.1,
        0.01,
        100.0,
        5.0,
        (),
        1.0,
    )


def test_safe_m5_aggregation_rejects_missing_and_padding_components() -> None:
    complete, audit = aggregate_m1_to_m5([bar(i) for i in range(5)])
    assert len(complete) == 1
    assert complete[0].open == pytest.approx(1.1)
    assert complete[0].close == pytest.approx(1.1005)
    assert complete[0].volume == 5
    assert audit.complete_windows == 1

    missing, missing_audit = aggregate_m1_to_m5([bar(i) for i in (0, 1, 3, 4)])
    assert not missing
    assert missing_audit.reasons == {"missing_or_noncontiguous_component": 1}

    padding_bar = bar(2, price=1.1, volume=0)
    padding_bar = Bar(
        padding_bar.instrument,
        padding_bar.timeframe,
        padding_bar.open_time,
        padding_bar.close_time,
        padding_bar.available_at,
        1.1,
        1.1,
        1.1,
        1.1,
        padding_bar.price_basis,
        0,
        padding_bar.volume_semantics,
    )
    aggregated, padding_audit = aggregate_m1_to_m5(
        [bar(0), bar(1), padding_bar, bar(3), bar(4)]
    )
    assert not aggregated
    assert padding_audit.reasons == {"provider_padding_component": 1}


@pytest.mark.parametrize(
    ("direction", "expected_sign"),
    [(SignalDirection.LONG, 1), (SignalDirection.SHORT, -1)],
)
def test_exact_execution_horizons_and_arithmetic(direction, expected_sign) -> None:
    bars = [bar(i, price=1.1 + i / 10_000) for i in range(61)]
    timestamp = bars[0].open_time
    record = execute_event(event(timestamp, direction), bars)
    assert record["entry_price"] == bars[0].open
    for horizon in (15, 30, 60):
        assert record[f"h{horizon}_complete"] is True
        assert record[f"h{horizon}_exit_price"] == bars[horizon - 1].close
    outcome = to_research_outcome(record, 15)
    expected = expected_sign * (bars[14].close - bars[0].open) / 0.0001
    assert outcome.gross_pips == pytest.approx(expected)


def test_execution_never_searches_forward_for_entry() -> None:
    bars = [bar(i) for i in range(1, 61)]
    record = execute_event(
        event(datetime(2024, 1, 2, tzinfo=UTC), SignalDirection.LONG), bars
    )
    assert record["entry_complete"] is False
    assert record["entry_incomplete_reason"] == "missing_exact_bar"
    assert all(
        record[f"h{h}_incomplete_reason"] == "entry_incomplete" for h in (15, 30, 60)
    )


def test_overlap_cost_session_uses_overall() -> None:
    # 14:00 UTC is simultaneously London and New York during January.
    assert cost_profile_session(datetime(2024, 1, 2, 14, tzinfo=UTC)) == "overall"
    assert cost_profile_session(datetime(2024, 1, 2, 1, tzinfo=UTC)) == "asia"


def test_discovery_decision_is_frozen() -> None:
    passed = {
        "previous_fx_day": {
            "n": 100,
            "gross_pips": {"mean": 1.0},
            "bootstrap_summary": {"p2_5": 0.01},
        }
    }
    assert discovery_decision(passed) == DECISION_CONTINUE
    passed["previous_fx_day"]["bootstrap_summary"]["p2_5"] = 0
    assert discovery_decision(passed) == DECISION_PARK


def test_aggregation_rejects_mixed_instruments() -> None:
    other = bar(1)
    other = Bar(
        "GBPUSD",
        other.timeframe,
        other.open_time,
        other.close_time,
        other.available_at,
        other.open,
        other.high,
        other.low,
        other.close,
        other.price_basis,
        other.volume,
        other.volume_semantics,
    )
    with pytest.raises(RangeSweepRunnerError, match="one instrument"):
        aggregate_m1_to_m5([bar(0), other])
