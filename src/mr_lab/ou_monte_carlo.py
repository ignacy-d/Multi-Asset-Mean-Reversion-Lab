"""Provisional, 2024-only OU portfolio Monte Carlo primitives.

This module deliberately consumes complete economic trades rather than Stage 4B's
analytical returns.  In particular, risk is never inferred from ``d0`` or an
execution-policy fraction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
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
    terminal_outcome: str
    terminal_day: int | None
    daily_breach_day: int | None
    total_breach_day: int | None


@dataclass(frozen=True, slots=True)
class SimTrade:
    entry_timestamp: datetime
    exit_timestamp: datetime
    r: float
    identity: str = ""


def research_calendar_2024() -> tuple[date, ...]:
    """The explicit provisional Monday-Friday 2024 research calendar."""
    current = date(2024, 1, 1)
    end = date(2024, 12, 31)
    result = []
    while current <= end:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return tuple(result)


def replay_events(
    trades: list[SimTrade],
    calendar: list[date] | tuple[date, ...],
    risk_fraction: float,
    max_portfolio_risk: float,
    rules: AccountRules,
) -> ReplayResult:
    """Replay positions, reserving risk while open and realizing P/L at exit."""
    if not calendar:
        raise MonteCarloInputError("replay calendar must not be empty")
    zone = ZoneInfo(rules.daily_reset_timezone)
    start = rules.starting_balance
    equity = peak = start
    open_positions: dict[str, tuple[float, float]] = {}
    realized: dict[date, list[float]] = {}
    events = []
    for number, trade in enumerate(trades):
        if trade.entry_timestamp >= trade.exit_timestamp:
            raise MonteCarloInputError("trade exit must be strictly after entry")
        identity = trade.identity or str(number)
        # Existing exits release risk/equity before same-time new entries.
        events.append((trade.entry_timestamp, 1, "entry", identity, trade.r))
        events.append((trade.exit_timestamp, 0, "exit", identity, trade.r))
    for timestamp, _, kind, identity, r_value in sorted(events):
        if kind == "entry":
            reserved = sum(value[0] for value in open_positions.values())
            allocation = max(0.0, min(risk_fraction, max_portfolio_risk - reserved))
            open_positions[identity] = (allocation, equity * allocation)
        else:
            allocation, amount = open_positions.pop(identity)
            pnl = amount * r_value
            equity += pnl
            local_day = timestamp.astimezone(zone).date()
            realized.setdefault(local_day, []).append(pnl)

    equity = peak = start
    returns = []
    target_day = daily_day = total_day = terminal_day = None
    terminal = "UNRESOLVED"
    losing_streak = max_losing_streak = losing_days = max_losing_days = 0
    max_dd = 0.0
    for day_number, day in enumerate(calendar, 1):
        day_start = equity
        day_has_loss = False
        for pnl in realized.get(day, ()):
            equity += pnl
            if pnl < 0:
                losing_streak += 1
                day_has_loss = True
            else:
                losing_streak = 0
            max_losing_streak = max(max_losing_streak, losing_streak)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            if daily_day is None and equity <= day_start - start * rules.max_daily_loss:
                daily_day = day_number
            if total_day is None and equity <= start * (1 - rules.max_total_loss):
                total_day = day_number
            if terminal == "UNRESOLVED":
                if equity >= start * (1 + rules.profit_target) - 1e-9:
                    target_day = terminal_day = day_number
                    terminal = "PASS"
                elif daily_day == day_number or total_day == day_number:
                    terminal_day = day_number
                    terminal = "BREACH"
        losing_days = losing_days + 1 if day_has_loss else 0
        max_losing_days = max(max_losing_days, losing_days)
        returns.append(equity / start - 1)
    return ReplayResult(
        tuple(returns),
        target_day,
        daily_day is not None,
        total_day is not None,
        max_dd,
        max_losing_streak,
        max_losing_days,
        terminal,
        terminal_day,
        daily_day,
        total_day,
    )


def replay_r_days(
    day_trades: list[list[tuple[datetime, float]]],
    risk_fraction: float,
    max_portfolio_risk: float,
    rules: AccountRules,
) -> ReplayResult:
    """Compatibility helper for tests expressed as daily close-time R values."""
    calendar = [date(2024, 1, 1) + timedelta(days=n) for n in range(len(day_trades))]
    trades = []
    for day, values in zip(calendar, day_trades, strict=True):
        for number, (timestamp, r_value) in enumerate(values):
            mapped = datetime.combine(day, time(timestamp.hour, timestamp.minute), UTC)
            trades.append(
                SimTrade(
                    mapped,
                    mapped + timedelta(microseconds=1),
                    r_value,
                    f"{day}:{number}",
                )
            )
    return replay_events(trades, calendar, risk_fraction, max_portfolio_risk, rules)
