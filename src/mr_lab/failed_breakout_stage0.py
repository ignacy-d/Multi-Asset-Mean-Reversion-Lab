"""Deterministic Stage-0 evidence for the standalone failed-breakout family."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median, quantiles

from mr_lab.failed_breakout import (
    FAILED_BREAKOUT_FAMILY,
    PREREGISTERED_MINIMUM_DEPTHS,
    FailedBreakoutEvent,
)
from mr_lab.stage4a import DirectionalPathDiagnostic

FORWARD_HORIZONS = (15, 30, 60, 120)
OVERLAP_WINDOWS = (0, 15, 30, 60)
PRIMARY_HORIZON = 60
STAGE0_SCHEMA_VERSION = "failed-breakout-stage0-v1"
MIN_AGGREGATE_EVENTS = 100
MIN_INSTRUMENT_EVENTS = 20
MAX_CONCENTRATION = 0.60
MAX_QUARTER_CONCENTRATION = 0.50


class FailedBreakoutStage0Error(ValueError):
    """Raised when Stage-0 evidence violates its preregistered contract."""


@dataclass(frozen=True, slots=True)
class FailedBreakoutObservation:
    event: FailedBreakoutEvent
    minimum_depth_fraction: float
    path: DirectionalPathDiagnostic

    def __post_init__(self) -> None:
        if self.minimum_depth_fraction not in PREREGISTERED_MINIMUM_DEPTHS:
            raise FailedBreakoutStage0Error("depth is outside the preregistered grid")
        if (
            self.event.instrument != self.path.instrument
            or self.event.signal_timestamp != self.path.signal_timestamp
            or self.event.direction is not self.path.direction
        ):
            raise FailedBreakoutStage0Error("event and path identities disagree")

    def as_dict(self) -> dict[str, object]:
        return {
            "event": self.event.as_dict(),
            "minimum_depth_fraction": self.minimum_depth_fraction,
            "path": asdict(self.path),
        }


@dataclass(frozen=True, slots=True)
class ModuleAReference:
    instrument: str
    signal_timestamp: datetime


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[percentile - 1]


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def _profit_factor(values: Sequence[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else None


def _quarter(timestamp: datetime) -> str:
    return f"{timestamp.year}-Q{(timestamp.month - 1) // 3 + 1}"


def _physical_key(item: FailedBreakoutObservation) -> tuple[object, ...]:
    event = item.event
    return (
        event.instrument,
        event.anchor_family,
        event.level_id,
        event.breakout_timestamp,
        event.reclaim_timestamp,
        event.direction,
    )


def _aggregate(items: Sequence[FailedBreakoutObservation]) -> dict[str, object]:
    timestamps = {item.event.signal_timestamp for item in items}
    physical = {_physical_key(item) for item in items}
    months = sorted({timestamp.strftime("%Y-%m") for timestamp in timestamps})
    counts_by_month = {
        month: sum(
            item.event.signal_timestamp.strftime("%Y-%m") == month for item in items
        )
        for month in months
    }
    counts_by_instrument = {
        instrument: sum(item.event.instrument == instrument for item in items)
        for instrument in sorted({item.event.instrument for item in items})
    }
    counts_by_quarter = {
        quarter: sum(_quarter(item.event.signal_timestamp) == quarter for item in items)
        for quarter in sorted({_quarter(item.event.signal_timestamp) for item in items})
    }
    row: dict[str, object] = {
        "event_count": len(items),
        "unique_physical_opportunities": len(physical),
        "unique_global_signal_clocks": len(timestamps),
        "events_per_month": counts_by_month,
        "active_months": len(months),
        "instrument_event_counts": counts_by_instrument,
        "quarter_event_counts": counts_by_quarter,
        "max_instrument_concentration": max(counts_by_instrument.values(), default=0)
        / len(items)
        if items
        else None,
        "max_quarter_concentration": max(counts_by_quarter.values(), default=0)
        / len(items)
        if items
        else None,
    }
    for horizon in FORWARD_HORIZONS:
        values = [
            outcome.signed_return_pips
            for item in items
            for outcome in item.path.horizons
            if outcome.horizon_minutes == horizon
        ]
        prefix = f"forward_pips_h{horizon}"
        row |= {
            f"n_h{horizon}": len(values),
            f"{prefix}_mean": _mean(values),
            f"{prefix}_median": _median(values),
            f"{prefix}_p10": _percentile(values, 10),
            f"{prefix}_p25": _percentile(values, 25),
            f"{prefix}_p75": _percentile(values, 75),
            f"{prefix}_p90": _percentile(values, 90),
            f"{prefix}_win_rate": sum(value > 0 for value in values) / len(values)
            if values
            else None,
            f"{prefix}_forward_return_profit_factor": _profit_factor(values),
        }
    complete = [item for item in items if item.path.future_path_complete]
    mfe = [item.path.mfe_pips for item in complete if item.path.mfe_pips is not None]
    mae = [item.path.mae_pips for item in complete if item.path.mae_pips is not None]
    row |= {
        "complete_path_count": len(complete),
        "median_mfe_pips": _median(mfe),
        "median_mae_pips": _median(mae),
        "median_mfe_to_mae_ratio": (
            median(mfe) / median(mae) if mfe and mae and median(mae) > 0 else None
        ),
        "monthly_expectancy_h60": _period_expectancy(
            items, lambda value: value.strftime("%Y-%m")
        ),
        "quarterly_expectancy_h60": _period_expectancy(items, _quarter),
        "instrument_expectancy_h60": _category_expectancy(
            items, lambda item: item.event.instrument
        ),
    }
    return row


def _period_expectancy(items, key_fn) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for outcome in item.path.horizons:
            if outcome.horizon_minutes == PRIMARY_HORIZON:
                groups[key_fn(item.event.signal_timestamp)].append(
                    outcome.signed_return_pips
                )
    return {key: fmean(groups[key]) for key in sorted(groups)}


def _category_expectancy(items, key_fn) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for outcome in item.path.horizons:
            if outcome.horizon_minutes == PRIMARY_HORIZON:
                groups[key_fn(item)].append(outcome.signed_return_pips)
    return {key: fmean(groups[key]) for key in sorted(groups)}


def aggregate_observations(
    observations: Iterable[FailedBreakoutObservation], instruments: Sequence[str]
) -> tuple[dict[str, object], ...]:
    """Return every preregistered anchor/depth/instrument cell plus family rows."""
    items = tuple(observations)
    anchors = ("previous-day", "asia-session", "london-or60")
    rows = []
    for anchor in anchors:
        for depth in PREREGISTERED_MINIMUM_DEPTHS:
            for instrument in instruments:
                selected = [
                    item
                    for item in items
                    if item.event.anchor_family == anchor
                    and item.minimum_depth_fraction == depth
                    and item.event.instrument == instrument
                ]
                rows.append(
                    {
                        "scope": "cell",
                        "family": FAILED_BREAKOUT_FAMILY,
                        "anchor_family": anchor,
                        "minimum_depth_fraction": depth,
                        "instrument": instrument,
                    }
                    | _aggregate(selected)
                )
    for depth in PREREGISTERED_MINIMUM_DEPTHS:
        depth_selected = tuple(
            {
                _physical_key(item): item
                for item in items
                if item.minimum_depth_fraction == depth
            }.values()
        )
        rows.append(
            {
                "scope": "depth",
                "family": FAILED_BREAKOUT_FAMILY,
                "anchor_family": None,
                "minimum_depth_fraction": depth,
                "instrument": None,
            }
            | _aggregate(depth_selected)
        )
    physical: dict[tuple[object, ...], FailedBreakoutObservation] = {}
    for item in items:
        physical.setdefault(_physical_key(item), item)
    rows.append(
        {
            "scope": "family",
            "family": FAILED_BREAKOUT_FAMILY,
            "anchor_family": None,
            "minimum_depth_fraction": None,
            "instrument": None,
        }
        | _aggregate(tuple(physical.values()))
    )
    return tuple(rows)


def calculate_overlap(
    observations: Iterable[FailedBreakoutObservation],
    module_a: Iterable[ModuleAReference],
) -> tuple[dict[str, object], ...]:
    """Compare distinct failed-breakout opportunities to frozen Module A clocks."""
    observations = tuple(observations)
    module_by_instrument: dict[str, set[datetime]] = defaultdict(set)
    for reference in module_a:
        module_by_instrument[reference.instrument].add(reference.signal_timestamp)
    scopes: list[tuple[str, str | None, str | None, set[tuple[str, datetime]]]] = []
    all_keys = {
        (item.event.instrument, item.event.signal_timestamp) for item in observations
    }
    scopes.append(("aggregate", None, None, all_keys))
    for instrument in sorted({item.event.instrument for item in observations}):
        scopes.append(
            (
                "instrument",
                instrument,
                None,
                {key for key in all_keys if key[0] == instrument},
            )
        )
    for anchor in sorted({item.event.anchor_family for item in observations}):
        keys = {
            (item.event.instrument, item.event.signal_timestamp)
            for item in observations
            if item.event.anchor_family == anchor
        }
        scopes.append(("anchor", None, anchor, keys))
    rows = []
    for scope, scope_instrument, scope_anchor, keys in scopes:
        row: dict[str, object] = {
            "scope": scope,
            "instrument": scope_instrument,
            "anchor_family": scope_anchor,
            "failed_breakout_physical_opportunity_count": len(keys),
        }
        for window in OVERLAP_WINDOWS:
            overlap = sum(
                any(
                    abs((candidate - timestamp).total_seconds()) <= window * 60
                    for candidate in module_by_instrument[name]
                )
                for name, timestamp in keys
            )
            row |= {
                f"overlap_count_w{window}": overlap,
                f"overlap_percentage_w{window}": overlap / len(keys) if keys else None,
                f"unique_to_module_a_count_w{window}": len(keys) - overlap,
                f"unique_to_module_a_percentage_w{window}": (len(keys) - overlap)
                / len(keys)
                if keys
                else None,
            }
        rows.append(row)
    return tuple(rows)


def classify_family(rows, overlap_rows) -> dict[str, str]:
    family = next(row for row in rows if row["scope"] == "family")
    depth_rows = [row for row in rows if row["scope"] == "depth"]
    count = int(family["event_count"])
    meaningful = [
        count
        for count in family["instrument_event_counts"].values()
        if count >= MIN_INSTRUMENT_EVENTS
    ]
    primary_mean = family["forward_pips_h60_mean"]
    primary_median = family["forward_pips_h60_median"]
    depth_means = [row["forward_pips_h60_mean"] for row in depth_rows]
    evidence = (
        count >= MIN_AGGREGATE_EVENTS
        and family["n_h60"] >= MIN_AGGREGATE_EVENTS
        and len(meaningful) >= 2
        and sum(value > 0 for value in family["instrument_expectancy_h60"].values())
        >= 2
        and primary_mean is not None
        and primary_mean > 0
        and primary_median is not None
        and primary_median >= 0
        and family["max_instrument_concentration"] <= MAX_CONCENTRATION
        and family["max_quarter_concentration"] <= MAX_QUARTER_CONCENTRATION
        and all(value is not None and value > 0 for value in depth_means)
    )
    adequate = count >= MIN_AGGREGATE_EVENTS and family["n_h60"] >= MIN_AGGREGATE_EVENTS
    classification = "PASS" if evidence else "KILL" if adequate else "INCONCLUSIVE"
    aggregate_overlap = next(row for row in overlap_rows if row["scope"] == "aggregate")
    unique = aggregate_overlap["unique_to_module_a_percentage_w30"]
    if unique is None or not adequate:
        independence = "INSUFFICIENT EVIDENCE"
    elif unique >= 0.5 and evidence:
        independence = "INDEPENDENT CANDIDATE"
    elif unique >= 0.5:
        independence = "PARTIALLY ORTHOGONAL"
    else:
        independence = "LIKELY MODULE-A FILTER/TIMING LAYER"
    return {
        "family_classification": classification,
        "independence_status": independence,
    }


def render_report(rows, overlap_rows, classification, *, execution_status: str) -> str:
    family = next(row for row in rows if row["scope"] == "family")
    lines = [
        "# Failed Breakout Stage-0 deterministic report",
        "",
        "## A. Family summary",
        "",
        f"- Execution status: **{execution_status}**",
        f"- Event observations: {family['event_count']}",
        f"- Unique physical opportunities: {family['unique_physical_opportunities']}",
        f"- Unique global signal clocks: {family['unique_global_signal_clocks']}",
        "",
        "## OBSERVED RESULT",
        "",
        f"- Primary 60-minute mean (pips): {family['forward_pips_h60_mean']}",
        f"- Primary 60-minute median (pips): {family['forward_pips_h60_median']}",
        "- Median MFE / MAE (pips): "
        f"{family['median_mfe_pips']} / {family['median_mae_pips']}",
        "",
        "## B. Event counts",
        "",
        f"- Events per month: {json.dumps(family['events_per_month'], sort_keys=True)}",
        f"- Active months: {family['active_months']}",
        "",
        "## C. Anchor/depth matrix",
        "",
        "See `anchor-depth-instrument.csv`; every preregistered cell is included.",
        "",
        "## D. Cross-asset matrix",
        "",
        "- Instrument counts: "
        f"{json.dumps(family['instrument_event_counts'], sort_keys=True)}",
        "- Instrument expectancy at 60m: "
        f"{json.dumps(family['instrument_expectancy_h60'], sort_keys=True)}",
        f"- Maximum instrument concentration: {family['max_instrument_concentration']}",
        "",
        "## E. Forward outcomes: 15 / 30 / 60 / 120 minutes",
        "",
        *(
            f"- {horizon}m: n={family[f'n_h{horizon}']}, "
            f"mean={family[f'forward_pips_h{horizon}_mean']}, "
            f"median={family[f'forward_pips_h{horizon}_median']}, "
            f"p10/p25/p75/p90="
            f"{family[f'forward_pips_h{horizon}_p10']}/"
            f"{family[f'forward_pips_h{horizon}_p25']}/"
            f"{family[f'forward_pips_h{horizon}_p75']}/"
            f"{family[f'forward_pips_h{horizon}_p90']}, "
            f"win rate={family[f'forward_pips_h{horizon}_win_rate']}, "
            "forward-return profit factor="
            f"{family[f'forward_pips_h{horizon}_forward_return_profit_factor']}"
            for horizon in FORWARD_HORIZONS
        ),
        "",
        "## F. MFE / MAE diagnostics",
        "",
        f"- Complete paths: {family['complete_path_count']}",
        f"- Median MFE / MAE ratio: {family['median_mfe_to_mae_ratio']}",
        "",
        "## G. Monthly stability",
        "",
        "- 60m expectancy: "
        f"{json.dumps(family['monthly_expectancy_h60'], sort_keys=True)}",
        "",
        "## H. Quarterly stability",
        "",
        "- 60m expectancy: "
        f"{json.dumps(family['quarterly_expectancy_h60'], sort_keys=True)}",
        f"- Maximum quarter concentration: {family['max_quarter_concentration']}",
        "",
        "## I. Overlap with Module A: exact / ±15 / ±30 / ±60 minutes",
        "",
    ]
    for row in overlap_rows:
        lines.append(
            f"- {row['scope']} {row['instrument'] or row['anchor_family'] or 'all'}: "
            + ", ".join(
                f"±{window}m overlap={row[f'overlap_count_w{window}']}, "
                "unique to Module A="
                f"{row[f'unique_to_module_a_percentage_w{window}']}"
                for window in OVERLAP_WINDOWS
            )
        )
    lines += ["", "## J. Unique-event percentages", ""]
    for row in overlap_rows:
        lines.append(
            f"- {row['scope']} {row['instrument'] or row['anchor_family'] or 'all'}: "
            + ", ".join(
                f"±{window}m={row[f'unique_to_module_a_percentage_w{window}']}"
                for window in OVERLAP_WINDOWS
            )
        )
    lines += [
        "",
        "## INTERPRETATION",
        "",
        "## K. Family classification",
        "",
        f"- Family classification: **{classification['family_classification']}**",
        "",
        "## L. Independence classification",
        "",
        f"- Independence status: **{classification['independence_status']}**",
        "",
        "## M. Limitations",
        "",
        "- This is a discovery diagnostic, not a trade, cost, or portfolio backtest.",
        "- Missing exact future clocks are excluded rather than substituted.",
        "- Classification thresholds are fixed research-triage rules, not "
        "optimized parameters.",
        "- No empirical conclusion is valid when execution status is pending.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    observations,
    module_a,
    instruments,
    output_dir: Path,
    *,
    execution_status="complete",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = tuple(
        sorted(
            observations,
            key=lambda item: (
                item.event.instrument,
                item.event.signal_timestamp,
                item.minimum_depth_fraction,
                item.event.candidate_event_id,
            ),
        )
    )
    rows = aggregate_observations(ordered, instruments)
    overlaps = calculate_overlap(ordered, module_a)
    classification = classify_family(rows, overlaps)
    family = next(row for row in rows if row["scope"] == "family")
    payload = {
        "schema_version": STAGE0_SCHEMA_VERSION,
        "execution_status": execution_status,
        "classification": classification,
        "metrics": rows,
        "overlap": overlaps,
    }
    paths = {
        name: output_dir / name
        for name in (
            "events.jsonl",
            "anchor-depth-instrument.csv",
            "metrics.json",
            "overlap.json",
            "report.md",
            "summary.json",
        )
    }
    paths["events.jsonl"].write_text(
        "".join(
            json.dumps(
                item.as_dict(), sort_keys=True, separators=(",", ":"), default=str
            )
            + "\n"
            for item in ordered
        )
    )
    with paths["anchor-depth-instrument.csv"].open("w", newline="") as stream:
        flat = [
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in row.items()
            }
            for row in rows
        ]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    paths["metrics.json"].write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    paths["overlap.json"].write_text(
        json.dumps(overlaps, sort_keys=True, separators=(",", ":")) + "\n"
    )
    paths["report.md"].write_text(
        render_report(rows, overlaps, classification, execution_status=execution_status)
    )
    hashes = {
        name: sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if name != "summary.json"
    }
    paths["summary.json"].write_text(
        json.dumps(
            {
                "schema_version": STAGE0_SCHEMA_VERSION,
                "execution_status": execution_status,
                "event_count": family["event_count"],
                "grid_observation_count": len(ordered),
                "classification": classification,
                "hashes": hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return paths


__all__ = [
    "FORWARD_HORIZONS",
    "FailedBreakoutObservation",
    "ModuleAReference",
    "aggregate_observations",
    "calculate_overlap",
    "classify_family",
    "render_report",
    "write_outputs",
]
