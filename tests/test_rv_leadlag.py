from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data.models import Bar, DataContractError, PriceBasis, Timeframe
from mr_lab.rv_leadlag import (
    LeadLagParameters,
    detect_leadlag_events,
    leader_z_bin,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
M5 = Timeframe("5m")


def _bars(instrument: str, returns: list[float]) -> list[Bar]:
    closes = [1.0]
    for value in returns:
        closes.append(closes[-1] * math.exp(value))
    return [
        Bar(
            instrument=instrument,
            timeframe=M5,
            open_time=START + index * M5.duration,
            close_time=START + (index + 1) * M5.duration,
            available_at=START + (index + 1) * M5.duration,
            open=closes[index],
            high=max(closes[index], close),
            low=min(closes[index], close),
            close=close,
            price_basis=PriceBasis.MID,
        )
        for index, close in enumerate(closes)
    ]


def _history() -> list[float]:
    # Non-degenerate, exactly mean-zero population; sample deviation is stable.
    return [-1e-3, 1e-3] * 144


def _input(a_event: float = 0.003, b_event: float = 0.0005) -> list[Bar]:
    return _bars("EURUSD", [*_history(), a_event]) + _bars(
        "GBPUSD", [*_history(), b_event]
    )


def test_positive_event_contract_thresholds_and_order_invariance() -> None:
    bars = _input()
    event = detect_leadlag_events(bars)[0]
    reversed_event = detect_leadlag_events(reversed(bars))[0]
    assert event == reversed_event
    assert event.leader == "EURUSD" and event.laggard == "GBPUSD"
    assert event.direction == 1
    assert event.z_ratio < 0.5
    assert event.event_id == reversed_event.event_id
    assert event.leader_z_bin == "2.50 <= |z| < 3.00"


def test_negative_direction_and_laggard_identification() -> None:
    event = detect_leadlag_events(_input(-0.0004, -0.003))[0]
    assert (event.leader, event.laggard, event.direction) == ("GBPUSD", "EURUSD", -1)


def test_same_sign_and_tie_are_rejected() -> None:
    assert not detect_leadlag_events(_input(0.003, -0.0005))
    assert not detect_leadlag_events(_input(0.003, 0.003))


def test_exact_threshold_and_ratio_boundary_are_inclusive() -> None:
    sd = math.sqrt(288 / 287) * 1e-3
    events = detect_leadlag_events(_input(2 * sd, sd))
    assert len(events) == 1
    assert events[0].leader_abs_z == pytest.approx(2.0)
    assert events[0].z_ratio == pytest.approx(0.5)


def test_current_is_excluded_and_future_and_prefix_invariance_hold() -> None:
    bars = _input()
    prefix = detect_leadlag_events(bars)
    # A huge future observation cannot alter the already emitted event.
    extended = _bars("EURUSD", [*_history(), 0.003, 2.0]) + _bars(
        "GBPUSD", [*_history(), 0.0005, -2.0]
    )
    assert detect_leadlag_events(extended)[:1] == prefix
    # If current leaked into its own normalization, this deliberately huge shock
    # would be substantially damped rather than use the prior-only mean/std.
    huge = detect_leadlag_events(_input(0.1, 0.0005))[0]
    assert huge.leader_z > 90


def test_missing_or_overlapping_m5_observations_are_not_padded() -> None:
    bars = _input()
    missing_time = START + 100 * M5.duration
    missing = [
        bar
        for bar in bars
        if not (bar.instrument == "GBPUSD" and bar.open_time == missing_time)
    ]
    assert not detect_leadlag_events(missing)
    bad = replace(bars[0], open_time=bars[0].open_time + timedelta(minutes=1))
    with pytest.raises(DataContractError, match="complete M5"):
        detect_leadlag_events([bad, *bars[1:]])


def test_strict_synchronization_and_complete_availability() -> None:
    bars = _input()
    event_bar = next(
        index
        for index, bar in enumerate(bars)
        if bar.instrument == "EURUSD" and bar.close_time == START + 290 * M5.duration
    )
    delayed = replace(
        bars[event_bar], available_at=bars[event_bar].close_time + timedelta(seconds=1)
    )
    with pytest.raises(DataContractError, match="available"):
        detect_leadlag_events([*bars[:event_bar], delayed, *bars[event_bar + 1 :]])


def test_degenerate_variance_is_rejected_without_event() -> None:
    flat = [0.0] * 288
    bars = _bars("EURUSD", [*flat, 0.1]) + _bars("GBPUSD", [*flat, 0.01])
    assert not detect_leadlag_events(bars)


def test_relationship_cooldown_and_relationship_independence() -> None:
    history = _history()
    first = [0.003, 0.003, 0.003, 0.003]
    lag = [0.0005] * 4
    bars = _bars("EURUSD", history + first) + _bars("GBPUSD", history + lag)
    bars += _bars("AUDUSD", history + first) + _bars("NZDUSD", history + lag)
    events = detect_leadlag_events(bars)
    assert [(event.relationship_id, event.timestamp) for event in events] == [
        ("AUDUSD_NZDUSD", START + 290 * M5.duration),
        ("EURUSD_GBPUSD", START + 290 * M5.duration),
        ("AUDUSD_NZDUSD", START + 293 * M5.duration),
        ("EURUSD_GBPUSD", START + 293 * M5.duration),
    ]


def test_only_canonical_m5_and_frozen_universe_are_accepted() -> None:
    bar = _bars("EURUSD", [0.1])[0]
    with pytest.raises(DataContractError):
        detect_leadlag_events([replace(bar, instrument="EURJPY")])
    with pytest.raises(DataContractError):
        detect_leadlag_events([replace(bar, timeframe=Timeframe("1m"))])


@pytest.mark.parametrize(
    ("value", "label"),
    [(2.0, "2.00 <= |z| < 2.50"), (2.5, "2.50 <= |z| < 3.00"), (3.0, "|z| >= 3.00")],
)
def test_diagnostic_bin_boundaries(value: float, label: str) -> None:
    assert leader_z_bin(value) == label


def test_module_has_no_empirical_io_or_economic_integration() -> None:
    import mr_lab.rv_leadlag as module

    source = __import__("inspect").getsource(module)
    for forbidden in ("open(", "Path(", "2025", "entry_price", "pips", "costs="):
        assert forbidden not in source


def test_parameters_validate() -> None:
    with pytest.raises(ValueError):
        LeadLagParameters(lookback_returns=1)
