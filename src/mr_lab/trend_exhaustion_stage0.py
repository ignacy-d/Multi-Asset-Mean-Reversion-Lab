"""Frozen deterministic aggregation and classification for Trend Exhaustion v1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean, median, quantiles
from typing import Any

from mr_lab.stage4a import DirectionalPathDiagnostic
from mr_lab.trend_exhaustion import (
    DISPLACEMENT_THRESHOLDS,
    PRIMARY_THRESHOLD,
    TrendExhaustionEvent,
)

HORIZONS = (15, 30, 60, 120)
OVERLAP_WINDOWS = (0, 15, 30, 60)
MIN_AGGREGATE_EVENTS = 100
MIN_INSTRUMENT_EVENTS = 20
MIN_REPLICATING_INSTRUMENTS = 2
MAX_INSTRUMENT_CONCENTRATION = 0.60
MAX_QUARTER_CONCENTRATION = 0.50


@dataclass(frozen=True, slots=True)
class Observation:
    event: TrendExhaustionEvent
    path: DirectionalPathDiagnostic

    def __post_init__(self) -> None:
        if (
            self.event.instrument,
            self.event.signal_timestamp,
            self.event.direction,
        ) != (self.path.instrument, self.path.signal_timestamp, self.path.direction):
            raise ValueError("event and path identities disagree")


@dataclass(frozen=True, slots=True)
class ModuleAReference:
    instrument: str
    signal_timestamp: datetime


def _quarter(value: datetime) -> str:
    return f"{value.year}-Q{(value.month - 1) // 3 + 1}"


def _summary(items: Sequence[Observation]) -> dict[str, Any]:
    instruments: dict[str, list[float]] = defaultdict(list)
    months: dict[str, list[float]] = defaultdict(list)
    quarters: dict[str, list[float]] = defaultdict(list)
    row: dict[str, Any] = {"event_count": len(items)}
    for horizon in HORIZONS:
        values = [
            out.signed_return_pips
            for item in items
            for out in item.path.horizons
            if out.horizon_minutes == horizon
        ]
        prefix = f"h{horizon}"
        qs = quantiles(values, n=100, method="inclusive") if len(values) > 1 else None
        losses = -sum(value for value in values if value < 0)
        row |= {
            f"n_{prefix}": len(values),
            f"mean_{prefix}": fmean(values) if values else None,
            f"median_{prefix}": median(values) if values else None,
            f"p10_{prefix}": qs[9] if qs else (values[0] if values else None),
            f"p25_{prefix}": qs[24] if qs else (values[0] if values else None),
            f"p75_{prefix}": qs[74] if qs else (values[0] if values else None),
            f"p90_{prefix}": qs[89] if qs else (values[0] if values else None),
            f"win_rate_{prefix}": sum(v > 0 for v in values) / len(values)
            if values
            else None,
            f"profit_factor_{prefix}": sum(v for v in values if v > 0) / losses
            if losses
            else None,
        }
    complete = [item for item in items if item.path.future_path_complete]
    row["mfe_pips_mean"] = (
        fmean(item.path.mfe_pips for item in complete if item.path.mfe_pips is not None)
        if complete
        else None
    )
    row["mae_pips_mean"] = (
        fmean(item.path.mae_pips for item in complete if item.path.mae_pips is not None)
        if complete
        else None
    )
    for item in items:
        outcome = next(
            (
                x.signed_return_pips
                for x in item.path.horizons
                if x.horizon_minutes == 60
            ),
            None,
        )
        if outcome is not None:
            instruments[item.event.instrument].append(outcome)
            months[item.event.signal_timestamp.strftime("%Y-%m")].append(outcome)
            quarters[_quarter(item.event.signal_timestamp)].append(outcome)
    row["instrument_counts"] = {
        k: sum(i.event.instrument == k for i in items)
        for k in sorted({i.event.instrument for i in items})
    }
    row["quarter_counts"] = {
        k: sum(_quarter(i.event.signal_timestamp) == k for i in items)
        for k in sorted({_quarter(i.event.signal_timestamp) for i in items})
    }
    row["monthly_counts"] = {
        key: sum(item.event.signal_timestamp.strftime("%Y-%m") == key for item in items)
        for key in sorted(
            {item.event.signal_timestamp.strftime("%Y-%m") for item in items}
        )
    }
    row["instrument_expectancy_h60"] = {
        k: fmean(v) for k, v in sorted(instruments.items())
    }
    row["monthly_expectancy_h60"] = {k: fmean(v) for k, v in sorted(months.items())}
    row["quarterly_expectancy_h60"] = {k: fmean(v) for k, v in sorted(quarters.items())}
    row["max_instrument_concentration"] = (
        max(row["instrument_counts"].values(), default=0) / len(items)
        if items
        else None
    )
    row["max_quarter_concentration"] = (
        max(row["quarter_counts"].values(), default=0) / len(items) if items else None
    )
    return row


def aggregate(
    observations: Iterable[Observation], instruments: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    items = tuple(observations)
    rows = []
    for threshold in DISPLACEMENT_THRESHOLDS:
        selected = tuple(
            x for x in items if x.event.displacement_threshold == threshold
        )
        for instrument in instruments:
            rows.append(
                {
                    "scope": "instrument",
                    "threshold": threshold,
                    "instrument": instrument,
                }
                | _summary(
                    tuple(x for x in selected if x.event.instrument == instrument)
                )
            )
        rows.append(
            {"scope": "threshold", "threshold": threshold, "instrument": None}
            | _summary(selected)
        )
        if threshold == PRIMARY_THRESHOLD:
            rows.append(
                {"scope": "family", "threshold": threshold, "instrument": None}
                | _summary(selected)
            )
    return tuple(rows)


def calculate_overlap(
    observations: Iterable[Observation], references: Iterable[ModuleAReference]
) -> tuple[dict[str, object], ...]:
    refs: dict[str, set[datetime]] = defaultdict(set)
    for ref in references:
        refs[ref.instrument].add(ref.signal_timestamp)
    items = tuple(
        x for x in observations if x.event.displacement_threshold == PRIMARY_THRESHOLD
    )
    output = []
    for instrument in (None, *sorted({x.event.instrument for x in items})):
        selected = tuple(
            x for x in items if instrument is None or x.event.instrument == instrument
        )
        row: dict[str, object] = {
            "instrument": instrument,
            "event_count": len(selected),
        }
        for window in OVERLAP_WINDOWS:
            count = sum(
                any(
                    abs((clock - x.event.signal_timestamp).total_seconds())
                    <= window * 60
                    for clock in refs[x.event.instrument]
                )
                for x in selected
            )
            row[f"overlap_count_w{window}"] = count
            row[f"overlap_rate_w{window}"] = count / len(selected) if selected else None
        output.append(row)
    return tuple(output)


def plateau_acceptance(rows: Iterable[dict[str, Any]]) -> bool:
    means = {
        row["threshold"]: row["mean_h60"] for row in rows if row["scope"] == "threshold"
    }
    primary = means.get(PRIMARY_THRESHOLD)
    return (
        primary is not None
        and primary > 0
        and all(
            means.get(t) is not None and means[t] >= 0.5 * primary
            for t in DISPLACEMENT_THRESHOLDS
            if t != PRIMARY_THRESHOLD
        )
    )


def classify(rows: Iterable[dict[str, Any]]) -> str:
    rows = tuple(rows)
    primary = next(
        row
        for row in rows
        if row["scope"] == "threshold" and row["threshold"] == PRIMARY_THRESHOLD
    )
    adequate = (
        primary["event_count"] >= MIN_AGGREGATE_EVENTS
        and primary["n_h60"] >= MIN_AGGREGATE_EVENTS
    )
    evidence = (
        adequate
        and sum(
            v >= MIN_INSTRUMENT_EVENTS for v in primary["instrument_counts"].values()
        )
        >= 2
        and primary["mean_h60"] > 0
        and primary["median_h60"] >= 0
        and sum(v > 0 for v in primary["instrument_expectancy_h60"].values())
        >= MIN_REPLICATING_INSTRUMENTS
        and primary["max_instrument_concentration"] <= MAX_INSTRUMENT_CONCENTRATION
        and primary["max_quarter_concentration"] <= MAX_QUARTER_CONCENTRATION
        and plateau_acceptance(rows)
    )
    return "PASS" if evidence else "KILL" if adequate else "INCONCLUSIVE"
