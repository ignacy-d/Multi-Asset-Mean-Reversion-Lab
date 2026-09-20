"""Strategy-neutral economic analysis and generalized Stage 4C-v2 costs."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

METHODOLOGY_ID = "stage4c-economic-analysis-v2"
PROFILE_SCHEMA = "stage4c-ftmo-cost-profile-v2"
SPREAD_STATISTICS = ("mean", "p75", "p90", "p95")
SLIPPAGES = (0.0, 0.1, 0.25, 0.5)
COST_FLOOR_LABEL = "COST_FLOOR_ONLY_NOT_LIVE_EXPECTANCY"
AVAILABLE = "AUTHENTICATED_COST_OVERLAY_AVAILABLE"
MISSING_SPREAD = "BLOCKED_MISSING_SPREAD_PROFILE"
MISSING_CONVERSION = "BLOCKED_MISSING_COMMISSION_CONVERSION"
MISSING_REFERENCE = "BLOCKED_MISSING_REFERENCE_RATE"


class EconomicAnalysisError(ValueError):
    """An input violates the v2 economic-analysis contract."""


@dataclass(frozen=True, slots=True)
class FXInstrument:
    base_currency: str
    quote_currency: str
    pip_size: float
    standard_lot_size: int = 100_000


FX_INSTRUMENTS: Mapping[str, FXInstrument] = MappingProxyType(
    {
        "EURUSD": FXInstrument("EUR", "USD", 0.0001),
        "GBPUSD": FXInstrument("GBP", "USD", 0.0001),
        "AUDUSD": FXInstrument("AUD", "USD", 0.0001),
        "NZDUSD": FXInstrument("NZD", "USD", 0.0001),
        "USDJPY": FXInstrument("USD", "JPY", 0.01),
        "AUDJPY": FXInstrument("AUD", "JPY", 0.01),
        "USDCAD": FXInstrument("USD", "CAD", 0.0001),
        "USDCHF": FXInstrument("USD", "CHF", 0.0001),
        "EURGBP": FXInstrument("EUR", "GBP", 0.0001),
    }
)


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EconomicAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise EconomicAnalysisError(
            f"{name} must be finite{' and positive' if positive else ''}"
        )
    return result


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    study_id: str
    event_id: str
    timestamp: datetime
    instrument: str
    direction: int
    entry_timestamp: datetime
    entry_price: float
    exit_timestamp: datetime
    exit_price: float
    horizon_minutes: int
    gross_signed_bps_return: float
    session_labels: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.study_id or not self.event_id:
            raise EconomicAnalysisError("study_id and event_id are required")
        if self.instrument not in FX_INSTRUMENTS:
            raise EconomicAnalysisError(f"unsupported instrument: {self.instrument}")
        if self.direction not in (-1, 1):
            raise EconomicAnalysisError("direction must be -1 or +1")
        for name in ("timestamp", "entry_timestamp", "exit_timestamp"):
            value = getattr(self, name)
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() != UTC.utcoffset(value)
            ):
                raise EconomicAnalysisError(f"{name} must be timezone-aware UTC")
        if (
            self.entry_timestamp > self.timestamp
            or self.exit_timestamp < self.entry_timestamp
        ):
            raise EconomicAnalysisError("timestamps violate point-in-time ordering")
        _finite(self.entry_price, "entry_price", positive=True)
        _finite(self.exit_price, "exit_price", positive=True)
        _finite(self.gross_signed_bps_return, "gross_signed_bps_return")
        if isinstance(self.horizon_minutes, bool) or self.horizon_minutes <= 0:
            raise EconomicAnalysisError("horizon_minutes must be positive")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def gross_price_move(self) -> float:
        return self.direction * (self.exit_price - self.entry_price)

    @property
    def gross_pips(self) -> float:
        result = self.gross_price_move / FX_INSTRUMENTS[self.instrument].pip_size
        return _finite(result, "gross_pips")


@dataclass(frozen=True, slots=True)
class CostProfileV2:
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> CostProfileV2:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != PROFILE_SCHEMA:
            raise EconomicAnalysisError("unexpected cost-profile schema")
        scenarios = raw.get("cost_scenarios", {})
        if (
            tuple(scenarios.get("spread_statistic", ())) != SPREAD_STATISTICS
            or tuple(scenarios.get("slippage_round_turn_pips", ())) != SLIPPAGES
        ):
            raise EconomicAnalysisError("v2 requires exactly the frozen 16 scenarios")
        return cls(raw)

    def coverage_status(self, instrument: str, session: str | None = None) -> str:
        key = session or "overall"
        profile = self.raw.get("instruments", {}).get(instrument)
        if not profile or not profile.get("spread_profiles", {}).get(key, {}).get(
            "authenticated", False
        ):
            return MISSING_SPREAD
        conversion = profile.get("commission_conversion", {})
        if not conversion.get("authenticated", False):
            return MISSING_CONVERSION
        if conversion.get("quote_currency_per_usd") is None:
            return MISSING_REFERENCE
        adjustment = profile.get("conversion_adjustment", {})
        if not adjustment.get("defined", False):
            return MISSING_REFERENCE
        return AVAILABLE

    def costs(
        self, instrument: str, session: str | None, statistic: str
    ) -> tuple[float, float, float]:
        if instrument not in FX_INSTRUMENTS or statistic not in SPREAD_STATISTICS:
            raise EconomicAnalysisError("unsupported instrument or spread statistic")
        status = self.coverage_status(instrument, session)
        if status != AVAILABLE:
            raise EconomicAnalysisError(status)
        data = self.raw["instruments"][instrument]
        spread = _finite(
            data["spread_profiles"][session or "overall"][f"{statistic}_pips"], "spread"
        )
        conversion = _finite(
            data["commission_conversion"]["quote_currency_per_usd"],
            "quote_currency_per_usd",
            positive=True,
        )
        meta = FX_INSTRUMENTS[instrument]
        commission = (
            _finite(self.raw["commission_usd_round_turn"], "commission_usd_round_turn")
            * conversion
            / (meta.standard_lot_size * meta.pip_size)
        )
        return (
            spread,
            commission,
            _finite(data["conversion_adjustment"]["rate"], "adjustment rate"),
        )


def adjusted_execution_pips(value: float, rate: float) -> float:
    """Apply the explicitly configured sign-dependent quote-currency adjustment."""
    value, rate = _finite(value, "value"), _finite(rate, "rate")
    if not 0 <= rate < 1:
        raise EconomicAnalysisError("adjustment rate must be in [0, 1)")
    return value * (1 - rate if value > 0 else 1 + rate if value < 0 else 1)


def net_pips(
    gross: float,
    spread: float,
    slippage: float,
    commission: float,
    adjustment_rate: float,
) -> float:
    values = [
        _finite(x, name)
        for x, name in (
            (gross, "gross"),
            (spread, "spread"),
            (slippage, "slippage"),
            (commission, "commission"),
        )
    ]
    if any(x < 0 for x in values[1:]):
        raise EconomicAnalysisError("costs must be non-negative")
    return (
        adjusted_execution_pips(values[0] - values[1] - values[2], adjustment_rate)
        - values[3]
    )


def scenarios() -> tuple[tuple[str, float], ...]:
    return tuple(
        (statistic, slip) for statistic in SPREAD_STATISTICS for slip in SLIPPAGES
    )


def break_even_additional_slippage(
    gross_values: Sequence[float],
    spread: float,
    commission: float,
    adjustment_rate: float,
) -> float:
    if not gross_values:
        raise EconomicAnalysisError("break-even requires observations")

    def mean_at(slip: float) -> float:
        return statistics.fmean(
            net_pips(x, spread, slip, commission, adjustment_rate) for x in gross_values
        )

    if mean_at(0) <= 0:
        return 0.0
    low, high = 0.0, 1.0
    while mean_at(high) > 0:
        high *= 2
    for _ in range(200):
        mid = (low + high) / 2
        if mean_at(mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _trimmed_mean(values: Sequence[float]) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * 0.05)
    return statistics.fmean(ordered[trim : len(ordered) - trim] if trim else ordered)


def calendar_month_block_bootstrap(
    outcomes: Sequence[ResearchOutcome], *, replicates: int = 10_000, seed: int = 0
) -> tuple[float, ...]:
    """Resample whole observed calendar-month blocks; each result is event weighted."""
    if not outcomes or replicates <= 0:
        raise EconomicAnalysisError(
            "bootstrap requires outcomes and positive replicates"
        )
    blocks: dict[tuple[int, int], list[float]] = defaultdict(list)
    for item in outcomes:
        blocks[(item.timestamp.year, item.timestamp.month)].append(item.gross_pips)
    months = sorted(blocks)
    rng = random.Random(seed)
    return tuple(
        statistics.fmean(
            x for month in rng.choices(months, k=len(months)) for x in blocks[month]
        )
        for _ in range(replicates)
    )


def analyze(
    outcomes: Sequence[ResearchOutcome],
    profile: CostProfileV2 | None = None,
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if not outcomes:
        raise EconomicAnalysisError("analysis requires observations")
    pips = [item.gross_pips for item in outcomes]
    bps = [item.gross_signed_bps_return for item in outcomes]
    months = Counter((x.timestamp.year, x.timestamp.month) for x in outcomes)
    instruments = Counter(x.instrument for x in outcomes)
    result: dict[str, Any] = {
        "methodology_id": METHODOLOGY_ID,
        "n": len(outcomes),
        "gross_pips": {
            "mean": statistics.fmean(pips),
            "median": statistics.median(pips),
            "trimmed_mean_5pct": _trimmed_mean(pips),
            "positive_fraction": sum(x > 0 for x in pips) / len(pips),
            "total": sum(pips),
        },
        "gross_bps": {
            "mean": statistics.fmean(bps),
            "median": statistics.median(bps),
            "trimmed_mean_5pct": _trimmed_mean(bps),
        },
        "calendar_month_breadth": len(months),
        "instrument_breadth": len(instruments),
        "concentration": {
            "largest_month_fraction": max(months.values()) / len(outcomes),
            "largest_instrument_fraction": max(instruments.values()) / len(outcomes),
        },
        "bootstrap_event_weighted_mean_pips": calendar_month_block_bootstrap(
            outcomes, replicates=bootstrap_replicates, seed=seed
        ),
        "cost_floor_label": COST_FLOOR_LABEL,
        "cost_scenarios": {},
    }
    if profile is not None:
        for statistic, slip in scenarios():
            supported, blocked = [], Counter[str]()
            for event in outcomes:
                status = profile.coverage_status(
                    event.instrument,
                    event.session_labels[0] if event.session_labels else None,
                )
                if status != AVAILABLE:
                    blocked[status] += 1
                    continue
                spread, commission, rate = profile.costs(
                    event.instrument,
                    event.session_labels[0] if event.session_labels else None,
                    statistic,
                )
                supported.append(
                    net_pips(event.gross_pips, spread, slip, commission, rate)
                )
            summary: dict[str, Any] = {
                "supported_n": len(supported),
                "blocked": dict(blocked),
            }
            if supported:
                summary |= {
                    "mean_net_pips": statistics.fmean(supported),
                    "median_net_pips": statistics.median(supported),
                    "net_positive_fraction": sum(x > 0 for x in supported)
                    / len(supported),
                    "total_net_pips": sum(supported),
                }
            result["cost_scenarios"][f"{statistic}|{slip:g}"] = summary
    return result
