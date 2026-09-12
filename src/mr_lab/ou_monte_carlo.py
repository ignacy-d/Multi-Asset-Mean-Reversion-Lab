"""Provisional, 2024-only OU portfolio Monte Carlo primitives.

This module deliberately consumes complete economic trades rather than Stage 4B's
analytical returns.  In particular, risk is never inferred from ``d0`` or an
execution-policy fraction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


class MonteCarloInputError(ValueError):
    """An input cannot safely support the requested economic calculation."""


def reject_sealed_path(path: str | Path) -> Path:
    raw = str(path).replace("\\", "/")
    if "/2025/" in f"/{raw.strip('/')}/":
        raise MonteCarloInputError("sealed OOS path rejected")
    return Path(path)


def parse_utc(value: str, field: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise MonteCarloInputError(f"invalid {field}") from error
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise MonteCarloInputError(f"{field} must be timezone-aware UTC")
    return result.astimezone(UTC)


def r_multiple(net_pips: float, initial_stop_distance_pips: float) -> float:
    values = (net_pips, initial_stop_distance_pips)
    if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in values):
        raise MonteCarloInputError("R inputs must be numeric")
    if (
        not all(math.isfinite(float(x)) for x in values)
        or initial_stop_distance_pips <= 0
    ):
        raise MonteCarloInputError("initial stop distance must be finite and positive")
    return float(net_pips) / float(initial_stop_distance_pips)


def account_risk_amount(equity: float, risk_fraction: float) -> float:
    if not math.isfinite(equity) or equity <= 0 or not 0 < risk_fraction < 1:
        raise MonteCarloInputError("invalid equity or risk fraction")
    return equity * risk_fraction


@dataclass(frozen=True, slots=True)
class PortfolioEvent:
    instrument: str
    candidate_event_id: str
    signal_timestamp: datetime
    entry_timestamp: datetime
    exit_timestamp: datetime
    policy_id: str
    gross_pips: float
    transaction_cost_pips: float
    net_pips: float
    initial_stop_distance_pips: float
    r: float
    ou_provenance: str


def deduplicate_events(events: list[PortfolioEvent]) -> list[PortfolioEvent]:
    """Collapse identical representations and reject incompatible duplicates."""
    found: dict[tuple[str, str, str], PortfolioEvent] = {}
    for event in events:
        key = (event.instrument, event.candidate_event_id, event.policy_id)
        previous = found.get(key)
        if previous is not None and previous != event:
            raise MonteCarloInputError(f"incompatible duplicate economic trade: {key}")
        found[key] = event
    return sorted(
        found.values(),
        key=lambda e: (e.entry_timestamp, e.instrument, e.candidate_event_id),
    )


def block_bootstrap_indices(
    day_count: int, horizon_days: int, block_days: int, seed: int
) -> list[int]:
    if min(day_count, horizon_days, block_days) <= 0:
        raise MonteCarloInputError("bootstrap dimensions must be positive")
    rng = random.Random(seed)
    result: list[int] = []
    while len(result) < horizon_days:
        start = rng.randrange(day_count)
        result.extend((start + offset) % day_count for offset in range(block_days))
    return result[:horizon_days]


@dataclass(frozen=True, slots=True)
class AccountRules:
    starting_balance: float = 100_000.0
    profit_target: float = 0.10
    max_daily_loss: float = 0.05
    max_total_loss: float = 0.10
    daily_reset_timezone: str = "UTC"


@dataclass(frozen=True, slots=True)
class ReplayResult:
    returns: tuple[float, ...]
    target_day: int | None
    daily_breach: bool
    total_breach: bool
    max_drawdown: float
    max_losing_streak: int
    max_losing_days: int


def replay_r_days(
    day_trades: list[list[tuple[datetime, float]]],
    risk_fraction: float,
    max_portfolio_risk: float,
    rules: AccountRules,
) -> ReplayResult:
    """Replay close-time R using a labelled realized-equity daily approximation."""
    zone = ZoneInfo(rules.daily_reset_timezone)
    equity = peak = rules.starting_balance
    returns: list[float] = []
    target_day = None
    daily_breach = total_breach = False
    max_dd = 0.0
    losing_streak = max_losing_streak = losing_days = max_losing_days = 0
    for day_number, trades in enumerate(day_trades, 1):
        day_start = equity
        day_lost = False
        # Deterministic ordering; simultaneous positions share a fixed risk budget.
        ordered = sorted(trades, key=lambda value: value[0])
        simultaneous: dict[datetime, int] = {}
        for timestamp, _ in ordered:
            simultaneous[timestamp] = simultaneous.get(timestamp, 0) + 1
        for timestamp, r_value in ordered:
            allowed = min(risk_fraction, max_portfolio_risk / simultaneous[timestamp])
            pnl = account_risk_amount(equity, allowed) * r_value
            equity += pnl
            if pnl < 0:
                losing_streak += 1
                day_lost = True
            else:
                losing_streak = 0
            max_losing_streak = max(max_losing_streak, losing_streak)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            if equity <= rules.starting_balance * (1 - rules.max_total_loss):
                total_breach = True
            # Close timestamps define reset days; no intratrade path is asserted.
            _ = timestamp.astimezone(zone).date()
            if equity <= day_start - rules.starting_balance * rules.max_daily_loss:
                daily_breach = True
        losing_days = losing_days + 1 if day_lost else 0
        max_losing_days = max(max_losing_days, losing_days)
        returns.append(equity / rules.starting_balance - 1)
        if target_day is None and (
            equity / rules.starting_balance - 1 >= rules.profit_target - 1e-12
        ):
            target_day = day_number
    return ReplayResult(
        tuple(returns),
        target_day,
        daily_breach,
        total_breach,
        max_dd,
        max_losing_streak,
        max_losing_days,
    )
