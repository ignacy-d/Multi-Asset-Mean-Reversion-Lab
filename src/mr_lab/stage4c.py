"""Frozen Stage 4C-A transaction-cost transform.

This module deliberately knows nothing about signal construction.  Its input is the
immutable, complete trade-row contract emitted by Stage 4B.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID

SCHEMA_VERSION = "stage-4c-report-v1"
STAGE4B_SOURCE_COMMIT = "3090682b61090de1b4a30efc18a7547e92fa262e"
INSTRUMENTS = ("EURUSD", "USDJPY", "AUDUSD", "AUDJPY")
SPREAD_STATISTICS = ("mean", "p75", "p90", "p95")
SLIPPAGES = (0.0, 0.1, 0.25, 0.5)
CONFIG_FIELDS = (
    "instrument",
    "benchmark_family",
    "signal_timeframe",
    "session",
    "direction",
    "lookback",
    "signal_threshold",
    "filter_family",
    "filter_spec_id",
    "entry_mode",
    "tp_target_fraction",
    "sl_extension_fraction",
    "time_stop_minutes",
)
REGIME_FIELDS = ("instrument", "signal_timeframe", "session", "direction", "entry_mode")


class Stage4CError(ValueError):
    """An input violates the frozen Stage 4C contract."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class CostProfile:
    raw: dict
    sha256: str

    @classmethod
    def load(cls, path: Path):
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise Stage4CError("malformed cost-profile JSON") from error
        if raw.get("schema_version") != "stage4c-ftmo-cost-profile-v1":
            raise Stage4CError("unexpected cost-profile schema")
        matrix = raw.get("cost_scenarios", {})
        if (
            tuple(matrix.get("spread_statistic", ())) != SPREAD_STATISTICS
            or tuple(matrix.get("slippage_round_turn_pips", ())) != SLIPPAGES
        ):
            raise Stage4CError("inconsistent scenario matrix")
        return cls(raw, sha256_file(path))

    def costs(self, instrument: str, session: str | None, statistic: str):
        if instrument not in INSTRUMENTS:
            raise Stage4CError(f"unexpected instrument: {instrument}")
        if statistic not in SPREAD_STATISTICS:
            raise Stage4CError(f"unexpected spread statistic: {statistic}")
        key = session if session is not None else "overall"
        try:
            spread = self.raw["spread_measurement"]["profiles"][instrument][key][
                f"{statistic}_pips"
            ]
        except KeyError as error:
            raise Stage4CError(f"missing cost profile: {instrument}/{key}") from error
        commission = (
            self.raw["commission"]["usd_quote_pairs"][instrument][
                "commission_round_turn_pips"
            ]
            if instrument in ("EURUSD", "AUDUSD")
            else self.raw["commission"]["jpy_quote_pairs"]["profiles"][key][
                "commission_round_turn_pips"
            ]
        )
        if not all(
            math.isfinite(float(x)) and float(x) >= 0 for x in (spread, commission)
        ):
            raise Stage4CError("costs must be finite and non-negative")
        return float(spread), float(commission)


def adjusted_execution_pips(value: float, instrument: str) -> float:
    if instrument not in ("USDJPY", "AUDJPY") or value == 0:
        return value
    return value * (0.993 if value > 0 else 1.007)


def net_pips(
    gross: float, instrument: str, spread: float, slippage: float, commission: float
):
    if not all(math.isfinite(float(x)) for x in (gross, spread, slippage, commission)):
        raise Stage4CError("gross and net inputs must be finite")
    return adjusted_execution_pips(gross - spread - slippage, instrument) - commission


def validate_trade(row: dict):
    if row.get("instrument") not in INSTRUMENTS:
        raise Stage4CError(f"unexpected instrument: {row.get('instrument')}")
    if not row.get("candidate_event_id"):
        raise Stage4CError("missing candidate identity")
    if not row.get("complete"):
        return False
    for field in (
        "gross_return_pips_adverse_first",
        "gross_return_pips_favorable_first",
    ):
        value = row.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise Stage4CError(f"malformed/non-finite gross value: {field}")
    return True


def trade_identity(row: dict):
    return (*(row.get(field) for field in CONFIG_FIELDS), row["candidate_event_id"])


def transform_trade(row: dict, profile: CostProfile, statistic: str, slippage: float):
    """Return a new row; the caller-owned Stage 4B row is never mutated."""
    if not validate_trade(row):
        raise Stage4CError("only complete Stage 4B rows can be transformed")
    spread, commission = profile.costs(row["instrument"], row.get("session"), statistic)
    adverse = net_pips(
        row["gross_return_pips_adverse_first"],
        row["instrument"],
        spread,
        slippage,
        commission,
    )
    favorable = net_pips(
        row["gross_return_pips_favorable_first"],
        row["instrument"],
        spread,
        slippage,
        commission,
    )
    if not all(math.isfinite(x) for x in (adverse, favorable)):
        raise Stage4CError("malformed/non-finite net value")
    return row | {
        "spread_statistic": statistic,
        "spread_pips": spread,
        "slippage_pips": slippage,
        "commission_pips": commission,
        "modeled_cost_before_conversion_adjustment": spread + slippage + commission,
        "net_pips_adverse_first": adverse,
        "net_pips_favorable_first": favorable,
    }


def scenario_rows(row: dict, profile: CostProfile):
    for statistic in SPREAD_STATISTICS:
        for slippage in SLIPPAGES:
            yield transform_trade(row, profile, statistic, slippage)


def break_even_total_slippage(
    gross_values, instrument, spread, commission, *, tolerance=1e-10
):
    """Deterministic bisection of the exact, sign-dependent per-trade transform."""
    values = tuple(float(x) for x in gross_values)
    if not values or not all(math.isfinite(x) for x in values):
        raise Stage4CError("break-even requires finite gross observations")

    def mean_at(slippage):
        return sum(
            net_pips(x, instrument, spread, slippage, commission) for x in values
        ) / len(values)

    if mean_at(0.0) <= 0:
        return 0.0
    low, high = 0.0, 1.0
    while mean_at(high) > 0:
        high *= 2
    for _ in range(200):
        if high - low <= tolerance:
            break
        mid = (low + high) / 2
        if mean_at(mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def assert_stage4b_methodology(value):
    if value != STAGE4B_METHODOLOGY_ID:
        raise Stage4CError("inconsistent Stage 4B methodology")
