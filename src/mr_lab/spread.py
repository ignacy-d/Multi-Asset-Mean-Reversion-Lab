"""Causal spread features and descriptive, deliberately unranked summaries."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite

from mr_lab.data.quotes import QuoteStatus, SynchronizedQuote
from mr_lab.identity import methodology_id


@dataclass(frozen=True, slots=True)
class SpreadFeatureSpec:
    lookback: int = 60
    volatility_lookback: int = 20
    shock_multiple: float = 2.0
    version: str = "spread-features-v1"

    def __post_init__(self) -> None:
        if self.lookback < 2 or self.volatility_lookback < 2:
            raise ValueError("rolling lookbacks must be at least two")
        if not isfinite(self.shock_multiple) or self.shock_multiple <= 0:
            raise ValueError("shock_multiple must be finite and positive")

    @property
    def identity(self) -> str:
        return methodology_id("spread-feature-spec-v1", asdict(self))


@dataclass(frozen=True, slots=True)
class SpreadFeature:
    observed_at: datetime
    available_at: datetime
    spread_pips: float | None
    percentile: float | None
    relative_to_median: float | None
    expansion_ratio: float | None
    volatility_shock: bool | None
    feature_spec_id: str


def causal_spread_features(
    quotes: tuple[SynchronizedQuote, ...],
    spec: SpreadFeatureSpec,
    *,
    causal_returns: tuple[float | None, ...] | None = None,
) -> tuple[SpreadFeature, ...]:
    """Compute prefix-only features; a full-history suffix cannot alter a prefix."""
    if causal_returns is not None and len(causal_returns) != len(quotes):
        raise ValueError("causal_returns must align one-for-one with quotes")
    history: list[float] = []
    return_history: list[float] = []
    output: list[SpreadFeature] = []
    for index, quote in enumerate(quotes):
        spread = quote.spread_pips
        if quote.status is not QuoteStatus.SYNCHRONIZED or spread is None:
            output.append(
                SpreadFeature(
                    quote.observed_at,
                    quote.available_at,
                    None,
                    None,
                    None,
                    None,
                    None,
                    spec.identity,
                )
            )
            continue
        window = [*history, spread][-spec.lookback :]
        percentile = sum(value <= spread for value in window) / len(window)
        median = statistics.median(window)
        previous = history[-1] if history else None
        shock: bool | None = None
        if causal_returns is not None:
            value = causal_returns[index]
            if value is not None and isfinite(value):
                baseline = return_history[-spec.volatility_lookback :]
                shock = bool(baseline) and abs(
                    value
                ) >= spec.shock_multiple * statistics.median(abs(x) for x in baseline)
                return_history.append(value)
        output.append(
            SpreadFeature(
                quote.observed_at,
                quote.available_at,
                spread,
                percentile,
                None if median == 0 else spread / median,
                None if previous in (None, 0) else spread / previous,
                shock,
                spec.identity,
            )
        )
        history.append(spread)
    return tuple(output)


def summarize_spreads(
    features: tuple[SpreadFeature, ...], *, instrument: str, session: str | None
) -> dict[str, object]:
    values = [x.spread_pips for x in features if x.spread_pips is not None]
    return {
        "instrument": instrument,
        "session": session,
        "sample_count": len(values),
        "unavailable_count": len(features) - len(values),
        "coverage_start": features[0].observed_at.isoformat() if features else None,
        "coverage_end": features[-1].observed_at.isoformat() if features else None,
        "feature_spec_id": features[0].feature_spec_id if features else None,
        "mean_spread_pips": statistics.fmean(values) if values else None,
        "median_spread_pips": statistics.median(values) if values else None,
    }
