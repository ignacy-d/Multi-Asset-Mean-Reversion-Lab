"""Unambiguous causal definitions for initial structure experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from mr_lab.data import Bar
from mr_lab.identity import methodology_id


@dataclass(frozen=True, slots=True)
class StructureFeatureSpec:
    swing_left_bars: int = 2
    swing_confirmation_bars: int = 2
    reclaim_lookback_bars: int = 20
    version: str = "structure-features-v1"

    def __post_init__(self) -> None:
        if min(self.swing_left_bars, self.swing_confirmation_bars) < 1:
            raise ValueError("swing sides must contain at least one bar")
        if self.reclaim_lookback_bars < 2:
            raise ValueError("reclaim lookback must be at least two")

    @property
    def identity(self) -> str:
        return methodology_id("structure-feature-spec-v1", asdict(self))


@dataclass(frozen=True, slots=True)
class StructureEvent:
    kind: str
    direction: str
    event_time: datetime
    available_at: datetime
    reference_price: float
    feature_spec_id: str


def confirmed_swings(
    bars: tuple[Bar, ...], spec: StructureFeatureSpec
) -> tuple[StructureEvent, ...]:
    """Emit a candidate only on the close of its final confirmation bar."""
    output: list[StructureEvent] = []
    left, right = spec.swing_left_bars, spec.swing_confirmation_bars
    for available_index in range(left + right, len(bars)):
        candidate_index = available_index - right
        candidate = bars[candidate_index]
        before = bars[candidate_index - left : candidate_index]
        after = bars[candidate_index + 1 : available_index + 1]
        available_at = bars[available_index].available_at
        if candidate.high > max(x.high for x in (*before, *after)):
            output.append(
                StructureEvent(
                    "confirmed_swing_high",
                    "SHORT",
                    candidate.open_time,
                    available_at,
                    candidate.high,
                    spec.identity,
                )
            )
        if candidate.low < min(x.low for x in (*before, *after)):
            output.append(
                StructureEvent(
                    "confirmed_swing_low",
                    "LONG",
                    candidate.open_time,
                    available_at,
                    candidate.low,
                    spec.identity,
                )
            )
    return tuple(output)


def extreme_reclaims(
    bars: tuple[Bar, ...], spec: StructureFeatureSpec
) -> tuple[StructureEvent, ...]:
    """Detect same-bar sweep-and-close-reclaim of a trailing completed-bar extreme."""
    output: list[StructureEvent] = []
    for index in range(spec.reclaim_lookback_bars, len(bars)):
        bar = bars[index]
        trailing = bars[index - spec.reclaim_lookback_bars : index]
        prior_high, prior_low = (
            max(x.high for x in trailing),
            min(x.low for x in trailing),
        )
        if bar.high > prior_high and bar.close < prior_high:
            output.append(
                StructureEvent(
                    "high_liquidity_sweep_reclaim",
                    "SHORT",
                    bar.open_time,
                    bar.available_at,
                    prior_high,
                    spec.identity,
                )
            )
        if bar.low < prior_low and bar.close > prior_low:
            output.append(
                StructureEvent(
                    "low_liquidity_sweep_reclaim",
                    "LONG",
                    bar.open_time,
                    bar.available_at,
                    prior_low,
                    spec.identity,
                )
            )
    return tuple(output)
