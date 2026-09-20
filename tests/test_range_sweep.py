"""Synthetic invariant tests for the frozen RANGE-SWEEP-2024-v1 engine."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.range_sweep import (
    SCIENTIFIC_STATUS,
    STUDY_ID,
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
        close_time=stamp + M5.duration,
        available_at=stamp + M5.duration,
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


def interval(start: datetime, count: int) -> list[Bar]:
    return [bar(start + index * M5.duration) for index in range(count)]


def winter_fx_day() -> list[Bar]:
    return interval(utc("2024-01-01T22:00:00"), 288)


def asia_session() -> list[Bar]:
    return interval(utc("2024-01-03T00:00:00"), 108)


def events_for(events, family: ReferenceFamily):
    return tuple(event for event in events if event.reference_family is family)


def test_study_contract_is_explicit_and_v1_configuration_is_frozen() -> None:
    assert STUDY_ID == "RANGE-SWEEP-2024-v1"
    assert SCIENTIFIC_STATUS == "2024_DISCOVERY_NOT_CONFIRMATION"
    assert tuple(RangeSweepConfig.__dataclass_fields__) == ("pip_sizes",)


@pytest.mark.parametrize("bad_index", [0, 143, 287])
def test_fx_reference_requires_every_m5_component(bad_index: int) -> None:
    source = winter_fx_day()
    del source[bad_index]
    source.append(bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19))
    assert not events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY)


def test_delayed_fx_component_invalidates_entire_reference() -> None:
    source = winter_fx_day()
    source[100] = replace(source[100], available_at=utc("2024-01-02T22:05:00"))
    source.append(bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19))
    assert not events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY)


def test_only_one_bar_cannot_form_reference() -> None:
    source = [
        bar(utc("2024-01-02T21:55:00")),
        bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19),
    ]
    assert not events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY)


def test_provider_padding_invalidates_entire_reference() -> None:
    source = [
        replace(item, volume=1, volume_semantics=VolumeSemantics.QUOTE_ACTIVITY)
        for item in winter_fx_day()
    ]
    source[50] = replace(
        source[50],
        open=1.1,
        high=1.1,
        low=1.1,
        close=1.1,
        volume=0,
        volume_semantics=VolumeSemantics.QUOTE_ACTIVITY,
    )
    source.append(
        replace(
            bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19),
            volume=1,
            volume_semantics=VolumeSemantics.QUOTE_ACTIVITY,
        )
    )
    assert not events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY)


def test_asia_reference_requires_complete_session() -> None:
    complete = asia_session()
    signal = bar(utc("2024-01-03T09:00:00"), high=1.21, close=1.19)
    assert (
        len(
            events_for(
                range_sweep_events([*complete, signal]),
                ReferenceFamily.COMPLETED_ASIA_SESSION,
            )
        )
        == 1
    )
    assert not events_for(
        range_sweep_events([*complete[:-1], signal]),
        ReferenceFamily.COMPLETED_ASIA_SESSION,
    )


@pytest.mark.parametrize(
    ("start", "signal_at"),
    [
        ("2024-03-07T22:00:00", "2024-03-08T22:00:00"),
        ("2024-03-09T22:00:00", "2024-03-10T21:00:00"),
        ("2024-03-10T21:00:00", "2024-03-11T21:00:00"),
    ],
)
def test_new_york_boundary_obeys_dst(start: str, signal_at: str) -> None:
    component_count = (utc(signal_at) - utc(start)) // M5.duration
    source = interval(utc(start), component_count)
    source.append(bar(utc(signal_at), high=1.21, close=1.19))
    assert (
        len(events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY))
        == 1
    )


def test_fx_reference_expires_at_next_boundary_without_fallback() -> None:
    source = winter_fx_day()
    # No bars for the next FX day: its prior reference must still expire at 22:00.
    source.append(bar(utc("2024-01-03T22:00:00"), high=1.21, close=1.19))
    assert not events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY)


def test_asia_reference_expires_at_next_asia_start_without_fallback() -> None:
    source = asia_session()
    source.append(bar(utc("2024-01-04T00:00:00"), high=1.21, close=1.19))
    assert not events_for(
        range_sweep_events(source), ReferenceFamily.COMPLETED_ASIA_SESSION
    )


def test_delayed_signal_bar_is_stamped_when_actually_available() -> None:
    source = winter_fx_day()
    signal = replace(
        bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19),
        available_at=utc("2024-01-02T22:10:00"),
    )
    event = events_for(
        range_sweep_events([*source, signal]), ReferenceFamily.PREVIOUS_FX_DAY
    )[0]
    assert event.signal_timestamp == signal.available_at


@pytest.mark.parametrize(
    ("high", "low", "close", "expected"),
    [
        (1.21, 1.05, 1.19, SignalDirection.SHORT),
        (1.15, 0.99, 1.01, SignalDirection.LONG),
        (1.20, 1.05, 1.19, None),
        (1.21, 1.05, 1.20, None),
        (1.15, 1.00, 1.01, None),
        (1.15, 0.99, 1.00, None),
    ],
)
def test_strict_sweep_reclaim_boundaries(
    high: float, low: float, close: float, expected: SignalDirection | None
) -> None:
    source = winter_fx_day()
    source[10] = replace(source[10], high=1.2)
    source[11] = replace(source[11], low=1.0)
    source.append(bar(utc("2024-01-02T22:00:00"), high=high, low=low, close=close))
    events = events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY)
    assert [event.signal_direction for event in events] == (
        [] if expected is None else [expected]
    )


def test_first_sweep_only_with_independent_sides() -> None:
    source = winter_fx_day()
    source[10] = replace(source[10], high=1.2)
    source[11] = replace(source[11], low=1.0)
    source.extend(
        (
            bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19),
            bar(utc("2024-01-02T22:05:00"), high=1.22, close=1.18),
            bar(utc("2024-01-02T22:10:00"), low=0.99, close=1.01),
        )
    )
    events = events_for(range_sweep_events(source), ReferenceFamily.PREVIOUS_FX_DAY)
    assert [event.reference_side for event in events] == [
        ReferenceSide.HIGH,
        ReferenceSide.LOW,
    ]


def test_event_identity_binds_study_and_is_deterministic() -> None:
    source = winter_fx_day()
    source.append(bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19))
    event = events_for(
        range_sweep_events(source, RangeSweepConfig(pip_sizes={"EURUSD": 0.0001})),
        ReferenceFamily.PREVIOUS_FX_DAY,
    )[0]
    repeated = events_for(
        range_sweep_events(source, RangeSweepConfig(pip_sizes={"EURUSD": 0.0001})),
        ReferenceFamily.PREVIOUS_FX_DAY,
    )[0]
    assert event.event_id == repeated.event_id
    assert event.event_id.startswith("sha256:")
    assert event.overshoot_pips == pytest.approx(100)


def test_volume_is_metadata_not_filter() -> None:
    source = winter_fx_day()
    signal = bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19)
    plain = events_for(
        range_sweep_events([*source, signal]), ReferenceFamily.PREVIOUS_FX_DAY
    )[0]
    quoted_signal = replace(
        signal, volume=999, volume_semantics=VolumeSemantics.QUOTE_ACTIVITY
    )
    # Dataset volume semantics are homogeneous, so attach nonzero quote activity
    # throughout without changing any price or qualifying boundary.
    quoted_source = [
        replace(item, volume=1, volume_semantics=VolumeSemantics.QUOTE_ACTIVITY)
        for item in source
    ]
    quoted = events_for(
        range_sweep_events([*quoted_source, quoted_signal]),
        ReferenceFamily.PREVIOUS_FX_DAY,
    )[0]
    assert plain.event_id == quoted.event_id
    assert quoted.volume == 999


def test_prefix_and_future_mutation_invariance() -> None:
    source = winter_fx_day()
    signal = bar(utc("2024-01-02T22:00:00"), high=1.21, close=1.19)
    prefix = [*source, signal]
    future = bar(utc("2024-01-02T22:05:00"), high=9, low=0.1, close=1.1)
    expected = range_sweep_events(prefix)
    full = range_sweep_events([*prefix, future])
    mutated = range_sweep_events([*prefix, replace(future, high=10)])
    assert (
        tuple(event for event in full if event.signal_timestamp <= signal.close_time)
        == expected
    )
    assert (
        tuple(event for event in mutated if event.signal_timestamp <= signal.close_time)
        == expected
    )


def test_engine_has_no_empirical_io_surface() -> None:
    import mr_lab.range_sweep as module

    assert not {"open", "Path", "read_csv", "read_parquet"}.intersection(vars(module))
