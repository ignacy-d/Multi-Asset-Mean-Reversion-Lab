from __future__ import annotations

import ast
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import exp
from pathlib import Path

import pytest

from mr_lab.data.models import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.vol_compression_breakout import (
    STUDY_ID,
    deterministic_percentile,
    generate_events,
    prior_realized_variance,
)

START = datetime(2024, 1, 1, tzinfo=UTC)


def make_bar(
    index: int,
    *,
    instrument: str = "EURUSD",
    close: float = 1.0,
    high: float | None = None,
    low: float | None = None,
    volume: float | None = None,
) -> Bar:
    open_time = START + index * timedelta(minutes=15)
    return Bar(
        instrument=instrument,
        timeframe=Timeframe("15m"),
        open_time=open_time,
        close_time=open_time + timedelta(minutes=15),
        available_at=open_time + timedelta(minutes=15),
        open=close,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        price_basis=PriceBasis.MID,
        volume=volume,
        volume_semantics=(
            VolumeSemantics.QUOTE_ACTIVITY
            if volume is not None
            else VolumeSemantics.NONE
        ),
    )


def first_eligible_history(*, instrument: str = "EURUSD") -> list[Bar]:
    # States evaluated at indices 5..1924 supply exactly 1920 prior states.
    return [make_bar(index, instrument=instrument) for index in range(1925)]


def breakout(
    index: int = 1925,
    *,
    instrument: str = "EURUSD",
    direction: str = "LONG",
    volume: float | None = None,
) -> Bar:
    close = 1.01 if direction == "LONG" else 0.99
    return make_bar(index, instrument=instrument, close=close, volume=volume)


def test_prior_only_four_return_rv_and_breakout_bar_excluded() -> None:
    rv_bars = [
        make_bar(index, close=exp(value))
        for index, value in enumerate([0, 0.1, 0.3, 0.6, 1.0])
    ]
    assert prior_realized_variance(rv_bars) == pytest.approx(0.30)

    event = generate_events([*first_eligible_history(), breakout()])[0]

    assert event.rv_1h_prior == 0.0
    assert event.breakout_bar_log_return > 0.0
    assert event.rv_p20_prior == 0.0


def test_rejects_mislabeled_ten_minute_bar() -> None:
    bar = make_bar(0)
    malformed = replace(
        bar,
        close_time=bar.open_time + timedelta(minutes=10),
        available_at=bar.open_time + timedelta(minutes=10),
    )

    with pytest.raises(ValueError, match="span exactly 15 minutes"):
        generate_events([malformed])


def test_rejects_non_m15_aligned_bar() -> None:
    bar = make_bar(0)
    malformed = replace(
        bar,
        open_time=bar.open_time + timedelta(minutes=1),
        close_time=bar.close_time + timedelta(minutes=1),
        available_at=bar.available_at + timedelta(minutes=1),
    )

    with pytest.raises(ValueError, match="epoch-aligned"):
        generate_events([malformed])


def test_accepts_aligned_m15_bar() -> None:
    assert generate_events([make_bar(0)]) == ()


def test_requires_exactly_1920_prior_states_and_excludes_current_state() -> None:
    bars = first_eligible_history()
    assert generate_events([*bars[:-1], breakout(1924)]) == ()

    event = generate_events([*bars, breakout()])[0]
    assert event.rv_percentile == 1.0  # all 1920 prior states equal current state


def test_future_mutation_and_prefix_invariance() -> None:
    prefix = [*first_eligible_history(), breakout()]
    expected = generate_events(prefix)
    future_a = [make_bar(index, close=1.01) for index in range(1926, 1932)]
    future_b = [make_bar(index, close=2.0) for index in range(1926, 1932)]

    assert generate_events([*prefix, *future_a])[:1] == expected
    assert generate_events([*prefix, *future_b])[:1] == expected


def test_deterministic_type7_percentile_and_exact_boundary() -> None:
    assert deterministic_percentile([4.0, 1.0, 3.0, 2.0], 0.20) == pytest.approx(1.6)
    assert deterministic_percentile(
        reversed([4.0, 1.0, 3.0, 2.0]), 0.20
    ) == pytest.approx(1.6)

    # Equality qualifies: both the current RV and its prior p20 are exactly zero.
    assert len(generate_events([*first_eligible_history(), breakout()])) == 1


def test_box_uses_only_prior_four_bars_and_touch_is_not_breakout() -> None:
    bars = first_eligible_history()
    bars[-5] = replace(bars[-5], high=5.0)  # RV seed bar is outside the box.
    event = generate_events([*bars, breakout()])[0]
    assert event.box_high == 1.0
    assert event.box_low == 1.0
    assert event.box_width_raw == 0.0

    assert generate_events([*first_eligible_history(), make_bar(1925)]) == ()


@pytest.mark.parametrize(
    ("direction", "expected"), [("LONG", "LONG"), ("SHORT", "SHORT")]
)
def test_directional_breakouts(direction: str, expected: str) -> None:
    event = generate_events([*first_eligible_history(), breakout(direction=direction)])[
        0
    ]
    assert event.direction == expected
    assert event.breakout_distance_raw == pytest.approx(0.01)


def test_no_event_outside_compression() -> None:
    bars = first_eligible_history()
    # Alter one of the four returns defining the current state, but not any of
    # the 1920 already-recorded zero reference states.
    bars[-2] = make_bar(1923, close=1.001, high=1.001)
    assert generate_events([*bars, breakout()]) == ()


def test_cooldown_is_exactly_60_minutes() -> None:
    # Alternation makes the reference RV comfortably exceed each later state,
    # while every rising candidate closes beyond its immediately prior box.
    bars = [
        make_bar(index, close=1.0 if index % 2 == 0 else 1.01) for index in range(1925)
    ]
    bars.extend(
        make_bar(index, close=1.011 + 0.001 * (index - 1925))
        for index in range(1925, 1930)
    )
    events = generate_events(bars)
    assert [event.timestamp for event in events] == [
        bars[1925].available_at,
        bars[1929].available_at,
    ]


def test_gap_breaks_current_state_continuity_but_keeps_prior_valid_states() -> None:
    bars = first_eligible_history()
    # Skip index 1925. The gap bar itself cannot use the pre-gap state. After
    # five new contiguous bars, the existing 1920 valid reference states apply.
    bars.extend(make_bar(index) for index in range(1926, 1931))
    candidate = breakout(1931)

    events = generate_events([*bars, candidate])
    assert [event.timestamp for event in events] == [candidate.available_at]


def test_independent_instruments_and_deterministic_identity() -> None:
    eur = [*first_eligible_history(), breakout()]
    gbp = [make_bar(i, instrument="GBPUSD") for i in range(1925)] + [
        breakout(instrument="GBPUSD")
    ]
    interleaved = [item for pair in zip(eur, gbp, strict=True) for item in pair]

    events = generate_events(interleaved)
    assert {event.instrument for event in events} == {"EURUSD", "GBPUSD"}
    assert events[0].event_id == generate_events(eur)[0].event_id
    assert events[0].event_id.startswith("sha256:")
    assert len(events[0].event_id.removeprefix("sha256:")) == 64
    assert events[0].study_id == STUDY_ID


def test_volume_is_diagnostic_only_and_never_filters() -> None:
    no_volume = generate_events([*first_eligible_history(), breakout()])[0]
    with_volume = generate_events(
        [*first_eligible_history(), breakout(volume=999_999.0)]
    )[0]
    assert no_volume.direction == with_volume.direction
    assert with_volume.current_quote_activity == 999_999.0


def test_module_has_no_empirical_io_or_year_specific_access() -> None:
    source_path = Path("src/mr_lab/vol_compression_breakout.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported_roots.isdisjoint({"pathlib", "pandas", "polars", "requests"})
    assert "2025" not in source_path.read_text(encoding="utf-8")
