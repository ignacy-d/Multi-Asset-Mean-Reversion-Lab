"""Synthetic invariant tests for the provider-neutral range-sweep engine."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.range_sweep import (
    RangeSweepConfig,
    ReferenceFamily,
    ReferenceSide,
    SignalDirection,
    range_sweep_events,
)

M5 = Timeframe("5m")


def bar(
    stamp: datetime,
    *,
    open_: float = 1.1,
    high: float = 1.2,
    low: float = 1.0,
    close: float = 1.1,
    volume: float | None = None,
) -> Bar:
    semantics = (
        VolumeSemantics.NONE if volume is None else VolumeSemantics.QUOTE_ACTIVITY
    )
    return Bar(
        instrument="EURUSD",
        timeframe=M5,
        open_time=stamp,
        close_time=stamp + timedelta(minutes=5),
        available_at=stamp + timedelta(minutes=5),
        open=open_,
        high=high,
        low=low,
        close=close,
        price_basis=PriceBasis.BID,
        volume=volume,
        volume_semantics=semantics,
    )


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def family(events, family: ReferenceFamily):
    return tuple(event for event in events if event.reference_family is family)


def test_previous_fx_day_boundary_and_frozen_level() -> None:
    # Winter 17:00 New York is 22:00 UTC.  The 21:55 bar belongs to the old day;
    # the 22:00 bar is current-day data and cannot alter its 1.20 high.
    bars = [
        bar(utc("2024-01-02T21:55:00"), high=1.2),
        bar(utc("2024-01-02T22:00:00"), high=1.8, low=1.05, close=1.1),
        bar(utc("2024-01-02T22:05:00"), high=1.21, low=1.08, close=1.19),
    ]
    events = family(range_sweep_events(bars), ReferenceFamily.PREVIOUS_FX_DAY)
    assert len(events) == 1
    assert events[0].reference_price == 1.2
    assert events[0].signal_direction is SignalDirection.SHORT
    assert events[0].minutes_since_reference_valid == 5


def test_observation_unknown_at_day_completion_cannot_revise_level() -> None:
    delayed = replace(
        bar(utc("2024-01-02T21:50:00"), high=2.0),
        available_at=utc("2024-01-02T23:00:00"),
    )
    bars = [
        delayed,
        bar(utc("2024-01-02T21:55:00"), high=1.2),
        bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19),
    ]
    events = family(range_sweep_events(bars), ReferenceFamily.PREVIOUS_FX_DAY)
    assert len(events) == 1
    assert events[0].reference_price == 1.2


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("2024-03-08T21:55:00", "2024-03-08T22:00:00"),
        ("2024-03-11T20:55:00", "2024-03-11T21:00:00"),
    ],
)
def test_fx_day_dst_boundary(before: str, after: str) -> None:
    bars = [
        bar(utc(before), high=1.2),
        bar(utc(after), high=1.21, low=1.05, close=1.19),
    ]
    events = family(range_sweep_events(bars), ReferenceFamily.PREVIOUS_FX_DAY)
    assert [(event.reference_price, event.signal_direction) for event in events] == [
        (1.2, SignalDirection.SHORT)
    ]


def test_completed_asia_finalizes_only_after_exact_session() -> None:
    # Existing Asia session is 09:00-18:00 Tokyo, hence 00:00-09:00 UTC.
    bars = [
        bar(utc("2024-01-03T08:55:00"), high=1.2),
        bar(utc("2024-01-03T09:00:00"), high=1.21, low=1.05, close=1.19),
    ]
    events = family(range_sweep_events(bars), ReferenceFamily.COMPLETED_ASIA_SESSION)
    assert len(events) == 1
    assert events[0].reference_price == 1.2
    assert events[0].minutes_since_reference_valid == 5
    assert "asia" not in events[0].session_labels


def test_incomplete_current_asia_is_not_a_finalized_reference() -> None:
    bars = [
        bar(utc("2024-01-03T00:00:00"), high=1.2),
        bar(utc("2024-01-03T00:05:00"), high=1.21, close=1.19),
    ]
    assert not family(range_sweep_events(bars), ReferenceFamily.COMPLETED_ASIA_SESSION)


@pytest.mark.parametrize(
    ("high", "low", "close", "expected"),
    [
        (1.21, 1.05, 1.19, SignalDirection.SHORT),
        (1.15, 0.99, 1.01, SignalDirection.LONG),
        (1.20, 1.05, 1.19, None),  # high touch is not a sweep
        (1.21, 1.05, 1.20, None),  # close exactly on high is not a reclaim
        (1.15, 1.00, 1.01, None),  # low touch is not a sweep
        (1.15, 0.99, 1.00, None),  # close exactly on low is not a reclaim
    ],
)
def test_strict_signal_boundaries(
    high: float, low: float, close: float, expected: SignalDirection | None
) -> None:
    bars = [
        bar(utc("2024-01-02T21:55:00"), high=1.2, low=1.0),
        bar(utc("2024-01-02T22:00:00"), high=high, low=low, close=close),
    ]
    events = family(range_sweep_events(bars), ReferenceFamily.PREVIOUS_FX_DAY)
    assert [event.signal_direction for event in events] == (
        [] if expected is None else [expected]
    )


def test_first_sweep_only_and_independent_high_low() -> None:
    bars = [
        bar(utc("2024-01-02T21:55:00"), high=1.2, low=1.0),
        bar(utc("2024-01-02T22:00:00"), high=1.21, low=1.05, close=1.19),
        bar(utc("2024-01-02T22:05:00"), high=1.22, low=1.05, close=1.18),
        bar(utc("2024-01-02T22:10:00"), high=1.15, low=0.99, close=1.01),
    ]
    events = family(range_sweep_events(bars), ReferenceFamily.PREVIOUS_FX_DAY)
    assert [event.reference_side for event in events] == [
        ReferenceSide.HIGH,
        ReferenceSide.LOW,
    ]


def test_reference_families_are_independent() -> None:
    bars = [
        bar(utc("2024-01-02T21:55:00"), high=1.2, low=1.0),
        bar(utc("2024-01-03T08:55:00"), high=1.2, low=1.0),
        bar(utc("2024-01-03T09:00:00"), high=1.21, low=1.05, close=1.19),
    ]
    events = range_sweep_events(bars)
    assert {event.reference_family for event in events} == {
        ReferenceFamily.PREVIOUS_FX_DAY,
        ReferenceFamily.COMPLETED_ASIA_SESSION,
    }


def test_deterministic_identity_and_optional_pips() -> None:
    bars = [
        bar(utc("2024-01-02T21:55:00"), high=1.2),
        bar(utc("2024-01-02T22:00:00"), high=1.201, close=1.19),
    ]
    first = family(
        range_sweep_events(bars, RangeSweepConfig(pip_sizes={"EURUSD": 0.0001})),
        ReferenceFamily.PREVIOUS_FX_DAY,
    )[0]
    second = family(
        range_sweep_events(bars, RangeSweepConfig(pip_sizes={"EURUSD": 0.0001})),
        ReferenceFamily.PREVIOUS_FX_DAY,
    )[0]
    assert first.event_id == second.event_id
    assert first.overshoot_pips == pytest.approx(10)


def test_volume_has_no_signal_effect() -> None:
    plain = [
        bar(utc("2024-01-02T21:55:00"), high=1.2),
        bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19),
    ]
    quoted = [
        replace(item, volume=999, volume_semantics=VolumeSemantics.QUOTE_ACTIVITY)
        for item in plain
    ]
    plain_event = family(range_sweep_events(plain), ReferenceFamily.PREVIOUS_FX_DAY)[0]
    quoted_event = family(range_sweep_events(quoted), ReferenceFamily.PREVIOUS_FX_DAY)[
        0
    ]
    assert plain_event.event_id == quoted_event.event_id
    assert quoted_event.volume == 999


def test_prefix_and_future_mutation_invariance() -> None:
    prefix = [
        bar(utc("2024-01-02T21:55:00"), high=1.2),
        bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19),
    ]
    future = bar(utc("2024-01-02T22:05:00"), high=9.0, low=0.1, close=1.1)
    prefix_events = range_sweep_events(prefix)
    full_events = range_sweep_events([*prefix, future])
    mutated_events = range_sweep_events([*prefix, replace(future, high=10.0)])
    assert (
        tuple(
            event
            for event in full_events
            if event.signal_timestamp <= prefix[-1].close_time
        )
        == prefix_events
    )
    assert (
        tuple(
            event
            for event in mutated_events
            if event.signal_timestamp <= prefix[-1].close_time
        )
        == prefix_events
    )


def test_engine_has_no_empirical_io_surface() -> None:
    import mr_lab.range_sweep as module

    assert not {"open", "Path", "read_csv", "read_parquet"}.intersection(vars(module))
