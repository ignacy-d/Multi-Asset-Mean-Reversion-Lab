from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.research import (
    Direction,
    ResearchSpec,
    build_forward_outcomes,
    build_research_observations,
    is_no_activity_flat_bar,
)
from mr_lab.sessions import DEFAULT_SESSION_SPEC, classify_bar


def bar(
    minute: int,
    *,
    timeframe: str = "15m",
    price: float = 1.0,
    volume: float = 1.0,
    flat: bool = True,
) -> Bar:
    duration = Timeframe(timeframe).duration
    opened = datetime(2024, 1, 2, 8, tzinfo=UTC) + timedelta(minutes=minute)
    return Bar(
        "EURUSD",
        Timeframe(timeframe),
        opened,
        opened + duration,
        opened + duration,
        price,
        price + (0 if flat else 0.02),
        price,
        price + (0 if flat else 0.01),
        PriceBasis.BID,
        volume,
        VolumeSemantics.QUOTE_ACTIVITY,
    )


def test_activity_rule_uses_exact_current_bar_fields() -> None:
    inactive = bar(0, volume=0)
    moving_zero = bar(15, volume=0, flat=False)
    flat_nonzero = bar(30, volume=2)

    assert is_no_activity_flat_bar(inactive)
    assert not is_no_activity_flat_bar(moving_zero)
    assert not is_no_activity_flat_bar(flat_nonzero)
    assert is_no_activity_flat_bar(inactive) == is_no_activity_flat_bar(inactive)


def test_view_preserves_bars_and_reuses_sessions_prefix_invariantly() -> None:
    bars = (bar(0), bar(15, volume=0), bar(30))
    full = build_research_observations(bars, DEFAULT_SESSION_SPEC)
    prefix = build_research_observations(bars[:2], DEFAULT_SESSION_SPEC)
    changed = build_research_observations(
        (*bars[:2], bar(30, price=9)), DEFAULT_SESSION_SPEC
    )

    assert tuple(item.bar for item in full) == bars
    assert len(full) == len(bars)
    assert [item.bar.open_time for item in full] == [item.open_time for item in bars]
    assert prefix == full[:2] == changed[:2]
    assert full[0].sessions.session_membership["london"]
    assert full[0].sessions == classify_bar(bars[0], DEFAULT_SESSION_SPEC)


@pytest.mark.parametrize("timeframe", ["5m", "15m", "1h"])
def test_supported_research_timeframes_work_generically(timeframe: str) -> None:
    observations = build_research_observations(
        (bar(0, timeframe=timeframe),), DEFAULT_SESSION_SPEC
    )
    assert observations[0].bar.timeframe == Timeframe(timeframe)


def test_exact_clock_outcomes_missing_and_inactive_targets() -> None:
    observations = build_research_observations(
        (bar(0, price=1), bar(30, price=1.1, volume=0), bar(45, price=1.2)),
        DEFAULT_SESSION_SPEC,
    )
    outcomes = build_forward_outcomes(
        observations, ResearchSpec((timedelta(minutes=15), timedelta(minutes=30)))
    )
    source = [item for item in outcomes if item.source is observations[0]]

    assert not source[0].is_available  # no exact +15 target; +30 is not substituted
    assert not source[1].is_available  # exact +30 target is provider-inactive
    assert source[0].target_available_at == observations[0].available_at + timedelta(
        minutes=15
    )


def test_future_target_enables_return_without_changing_source_semantics() -> None:
    source, target = bar(0, price=1), bar(30, price=1.1)
    spec = ResearchSpec((timedelta(minutes=30),))
    prefix = build_research_observations((source,), DEFAULT_SESSION_SPEC)
    full = build_research_observations((source, target), DEFAULT_SESSION_SPEC)

    assert prefix[0] == full[0]
    assert not build_forward_outcomes(prefix, spec)[0].is_available
    outcome = build_forward_outcomes(full, spec)[0]
    assert outcome.forward_return == pytest.approx(0.1)
    assert outcome.signed_forward_return(Direction.LONG) == pytest.approx(0.1)
    assert outcome.signed_forward_return(Direction.SHORT) == pytest.approx(-0.1)


def test_research_spec_is_canonical_separate_and_method_sensitive() -> None:
    forward = ResearchSpec((timedelta(hours=1), timedelta(minutes=15)))
    reverse = ResearchSpec((timedelta(minutes=15), timedelta(hours=1)))
    changed = replace(forward, activity_rule_version="future-rule-v2")

    assert forward.to_json() == reverse.to_json()
    assert forward.research_spec_id == reverse.research_spec_id
    assert changed.research_spec_id != forward.research_spec_id
    assert "dataset_id" not in forward.to_json()
    assert "session_spec_id" not in forward.to_json()
    assert forward.research_spec_id.startswith("sha256:")
