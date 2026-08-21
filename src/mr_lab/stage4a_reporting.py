"""Deterministic aggregation and reporting for frozen Stage 4A events.

This module only reduces :class:`Stage4AEvent` values.  It deliberately does not
reconstruct signals or path mathematics.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median, quantiles
from tempfile import NamedTemporaryFile

from mr_lab.providers.instruments import get_instrument_spec
from mr_lab.stage4a import Stage4AEvent

STAGE4A_REPORT_SCHEMA_VERSION = "stage-4a-report-v1"
MATRIX_GROUP_FIELDS = (
    "instrument",
    "benchmark_family",
    "signal_timeframe",
    "session",
    "direction",
    "lookback",
    "threshold",
    "stage4a_methodology_id",
)
QUANTILE_CONVENTION = (
    "Python statistics.quantiles(method='inclusive'): linear interpolation at "
    "(n-1)*p; a singleton returns its sole value"
)
OBSERVATION_WARNING = (
    "Stage 4A rows are conditional observations and may overlap heavily. "
    "Observation counts are NOT independent trade counts. No statistical test "
    "should assume row-level event independence."
)
HORIZONS = (5, 15, 30, 60, 120)
LEVELS = ((25, 0.25), (50, 0.5), (75, 0.75), (100, 1.0))
PRESIGNAL_HORIZONS = (5, 15, 30)
REPORT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)


def _event_key(event: Stage4AEvent) -> tuple[object, ...]:
    signal = event.signal
    return (
        signal.instrument,
        signal.benchmark_family,
        signal.signal_timeframe.value,
        signal.session or "",
        signal.direction.name,
        signal.lookback,
        signal.threshold,
        event.stage4a_methodology_id,
        signal.signal_timestamp,
        signal.source_corpus_id,
        signal.assembled_dataset_id or "",
        signal.strategy_spec_id,
    )


def _group_key(event: Stage4AEvent) -> tuple[object, ...]:
    signal = event.signal
    return (
        signal.instrument,
        signal.benchmark_family,
        signal.signal_timeframe.value,
        signal.session or "",
        signal.direction.name.lower(),
        signal.lookback,
        signal.threshold,
        event.stage4a_methodology_id,
    )


def _percentile(values: Sequence[float | int], percentile: int) -> float | None:
    """Use the documented inclusive standard-library quantile convention."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return quantiles(values, n=100, method="inclusive")[percentile - 1]


def _median(values: Sequence[float | int]) -> float | None:
    return float(median(values)) if values else None


def _mean(values: Sequence[float | int]) -> float | None:
    return fmean(values) if values else None


def aggregate_stage4a_events(
    events: Iterable[Stage4AEvent],
) -> tuple[dict[str, object], ...]:
    """Aggregate the complete supplied event set into stable setup-context rows."""
    groups: dict[tuple[object, ...], list[Stage4AEvent]] = defaultdict(list)
    for event in events:
        groups[_group_key(event)].append(event)

    rows = []
    for key in sorted(groups):
        items = groups[key]
        complete = [event for event in items if event.future_path_complete]
        row = dict(zip(MATRIX_GROUP_FIELDS, key, strict=True))
        row.update(
            observation_count_total=len(items),
            observation_count_complete_path=len(complete),
            observation_count_incomplete_path=len(items) - len(complete),
            complete_path_fraction=len(complete) / len(items),
        )

        for horizon in HORIZONS:
            diagnostics = [
                diagnostic
                for event in items
                for diagnostic in event.horizons
                if diagnostic.horizon_minutes == horizon
            ]
            row[f"n_h{horizon}"] = len(diagnostics)
            for name, attribute in (
                ("bps", "signed_return_bps"),
                ("pips", "signed_return_pips"),
                ("reversion_fraction", "reversion_fraction"),
            ):
                values = [getattr(value, attribute) for value in diagnostics]
                row[f"mean_{name}_h{horizon}"] = _mean(values)
                row[f"median_{name}_h{horizon}"] = _median(values)

        row["eligible_complete_path_count"] = len(complete)
        for label, level in LEVELS:
            passages = [
                next(value for value in event.first_passage if value.level == level)
                for event in complete
            ]
            hits = [value for value in passages if value.hit]
            times = [value.time_to_hit_minutes for value in hits]
            row[f"eligible_hit{label}_complete_path_count"] = len(complete)
            row[f"hit{label}_count"] = len(hits)
            row[f"hit{label}_rate"] = len(hits) / len(complete) if complete else None
            row[f"median_t{label}_minutes"] = _median(times)
            row[f"mean_t{label}_minutes"] = _mean(times)

        for prefix in ("mae", "mfe"):
            for suffix in ("pips", "fraction_d0"):
                values = [getattr(event, f"{prefix}_{suffix}") for event in complete]
                row[f"median_{prefix}_{suffix}"] = _median(values)
                row[f"p75_{prefix}_{suffix}"] = _percentile(values, 75)
                row[f"p90_{prefix}_{suffix}"] = _percentile(values, 90)
            bps = [getattr(event, f"{prefix}_bps") for event in complete]
            times = [getattr(event, f"time_to_{prefix}_minutes") for event in complete]
            row[f"median_{prefix}_bps"] = _median(bps)
            row[f"median_time_to_{prefix}_minutes"] = _median(times)

        max_reversion = [event.max_reversion_fraction for event in complete]
        row["median_max_reversion_fraction"] = _median(max_reversion)
        row["p75_max_reversion_fraction"] = _percentile(max_reversion, 75)
        row["p90_max_reversion_fraction"] = _percentile(max_reversion, 90)

        pip_size = 10 ** -(
            get_instrument_spec(items[0].signal.instrument).price_precision - 1
        )
        for horizon in PRESIGNAL_HORIZONS:
            diagnostics = [
                next(
                    value
                    for value in event.presignal
                    if value.horizon_minutes == horizon
                )
                for event in items
            ]
            available = [
                value for value in diagnostics if value.impulse_share is not None
            ]
            raw = [value.raw_price_movement / pip_size for value in available]
            signed = [value.signed_price_movement / pip_size for value in available]
            impulse = [value.impulse_share for value in available]
            row[f"n_presignal_h{horizon}"] = len(available)
            row[f"median_presignal_raw_pips_h{horizon}"] = _median(raw)
            row[f"median_presignal_signed_pips_h{horizon}"] = _median(signed)
            row[f"median_impulse_share_h{horizon}"] = _median(impulse)
            row[f"p25_impulse_share_h{horizon}"] = _percentile(impulse, 25)
            row[f"p75_impulse_share_h{horizon}"] = _percentile(impulse, 75)
        rows.append(row)
    return tuple(rows)


def matrix_to_csv(rows: Sequence[dict[str, object]]) -> str:
    """Serialize matrix rows with a stable schema and empty missing values."""
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(
    rows: Sequence[dict[str, object]], columns: Sequence[tuple[str, str]]
) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_display(row.get(field)) for field, _ in columns) + " |"
        for row in rows
    )
    if not rows:
        lines.append("| " + " | ".join("—" for _ in columns) + " |")
    return lines


def render_report(rows: Sequence[dict[str, object]]) -> str:
    """Render deterministic, identity-ordered descriptive views (never rankings)."""
    identity = (
        ("instrument", "Instrument"),
        ("benchmark_family", "Family"),
        ("signal_timeframe", "TF"),
        ("session", "Session"),
        ("direction", "Direction"),
        ("lookback", "Lookback"),
    )
    anchor_columns = (
        *identity,
        ("observation_count_complete_path", "N complete"),
        ("mean_pips_h15", "Mean 15m pips"),
        ("mean_pips_h30", "Mean 30m pips"),
        ("mean_pips_h60", "Mean 60m pips"),
        ("mean_pips_h120", "Mean 120m pips"),
        ("hit50_rate", "Hit50"),
        ("hit100_rate", "Hit100"),
        ("median_t50_minutes", "Median T50"),
        ("p75_mae_pips", "p75 MAE pips"),
        ("median_max_reversion_fraction", "Median max reversion"),
    )
    lines = [
        "# Stage 4A deterministic event report",
        "",
        f"> **Caution:** {OBSERVATION_WARNING}",
        "",
        "The tables are sorted by setup identity, not by outcomes. No "
        "configuration is ranked or selected.",
        "",
        "## Predeclared anchor: |z| >= 2.0",
        "",
        *_markdown_table(
            [row for row in rows if row["threshold"] == 2.0], anchor_columns
        ),
        "",
        "## Threshold response",
        "",
        "Threshold response is shown descriptively; no threshold is selected or "
        "optimized.",
        "",
    ]
    response_columns = (
        *identity,
        ("threshold", "Threshold"),
        ("observation_count_complete_path", "N complete"),
        ("mean_pips_h60", "Mean 60m pips"),
        ("mean_pips_h120", "Mean 120m pips"),
        ("hit50_rate", "Hit50"),
        ("hit100_rate", "Hit100"),
        ("p75_mae_pips", "p75 MAE pips"),
        ("median_max_reversion_fraction", "Median max reversion"),
    )
    for threshold in REPORT_THRESHOLDS:
        lines.extend((f"### |z| >= {threshold:.1f}", ""))
        lines.extend(
            _markdown_table(
                [row for row in rows if row["threshold"] == threshold], response_columns
            )
        )
        lines.append("")
    lines.extend(
        (
            "## Directional asymmetry at |z| >= 2.0",
            "",
            "LONG and SHORT remain separate conditional-observation rows; neither "
            "is designated best.",
            "",
        )
    )
    direction_columns = (
        *identity,
        ("observation_count_complete_path", "N complete"),
        ("mean_pips_h60", "Mean 60m pips"),
        ("mean_pips_h120", "Mean 120m pips"),
        ("hit50_rate", "Hit50"),
        ("hit100_rate", "Hit100"),
        ("median_mae_pips", "Median MAE pips"),
        ("median_max_reversion_fraction", "Median max reversion"),
    )
    lines.extend(
        _markdown_table(
            [row for row in rows if row["threshold"] == 2.0], direction_columns
        )
    )
    lines.extend(
        (
            "",
            "## Benchmark robustness at |z| >= 2.0",
            "",
            "Naturally aligned setup identities are displayed without a composite "
            "score.",
            "",
        )
    )
    lines.extend(
        _markdown_table(
            [row for row in rows if row["threshold"] == 2.0], anchor_columns
        )
    )
    lines.extend(
        (
            "",
            "## Methodology notes",
            "",
            f"- Reporting schema: `{STAGE4A_REPORT_SCHEMA_VERSION}`.",
            f"- Quantiles: {QUANTILE_CONVENTION}.",
            "- Fixed-horizon aggregates use only events containing that exact "
            "snapshot; `n_h*` exposes each denominator.",
            "- First-passage rates and all pathwise excursion aggregates use "
            "complete 120-minute paths only; hit times use hits only.",
            "- Missing values are excluded, never replaced with zero.",
            "- This report contains no Stage 4B trade construction, costs, ranking, "
            "optimization, or independence-based statistics.",
            "",
        )
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _atomic_stream_events(path: Path, events: Iterable[Stage4AEvent]) -> str:
    """Atomically stream the frozen JSONL representation and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = sha256()
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            for event in events:
                encoded = event.to_json().encode()
                temporary.write(encoded)
                temporary.write(b"\n")
                digest.update(encoded)
                digest.update(b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def write_stage4a_outputs(
    events: Iterable[Stage4AEvent], output_dir: str | Path
) -> dict[str, Path]:
    """Write exactly the four deterministic Stage 4A reporting artifacts."""
    ordered = tuple(sorted(events, key=_event_key))
    print(f"STAGE4A_REPORTING ordering_complete event_count={len(ordered)}", flush=True)
    rows = aggregate_stage4a_events(ordered)
    print(
        f"STAGE4A_REPORTING aggregation_complete matrix_row_count={len(rows)}",
        flush=True,
    )
    matrix_bytes = matrix_to_csv(rows).encode()
    report_bytes = render_report(rows).encode()
    paths = {
        name: Path(output_dir) / name
        for name in ("events.jsonl", "matrix.csv", "summary.json", "report.md")
    }
    print("STAGE4A_REPORTING events_stream_start", flush=True)
    events_digest = _atomic_stream_events(paths["events.jsonl"], ordered)
    print("STAGE4A_REPORTING events_stream_complete", flush=True)
    _atomic_write(paths["matrix.csv"], matrix_bytes)
    print("STAGE4A_REPORTING matrix_write_complete", flush=True)
    _atomic_write(paths["report.md"], report_bytes)
    print("STAGE4A_REPORTING report_write_complete", flush=True)

    summary = {
        "benchmark_families": sorted(
            {event.signal.benchmark_family for event in ordered}
        ),
        "complete_path_total": sum(event.future_path_complete for event in ordered),
        "event_total": len(ordered),
        "hashes": {
            "events.jsonl": events_digest,
            "matrix.csv": sha256(matrix_bytes).hexdigest(),
            "report.md": sha256(report_bytes).hexdigest(),
        },
        "incomplete_path_total": sum(
            not event.future_path_complete for event in ordered
        ),
        "instruments": sorted({event.signal.instrument for event in ordered}),
        "lookbacks": sorted({event.signal.lookback for event in ordered}),
        "matrix_group_fields": list(MATRIX_GROUP_FIELDS),
        "matrix_row_count": len(rows),
        "observation_warning": OBSERVATION_WARNING,
        "quantile_convention": QUANTILE_CONVENTION,
        "report_schema_version": STAGE4A_REPORT_SCHEMA_VERSION,
        "sessions": sorted({event.signal.session or "" for event in ordered}),
        "stage4a_methodology_ids": sorted(
            {event.stage4a_methodology_id for event in ordered}
        ),
        "thresholds": sorted({event.signal.threshold for event in ordered}),
        "timeframes": sorted(
            {event.signal.signal_timeframe.value for event in ordered}
        ),
    }
    summary_bytes = (
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    _atomic_write(paths["summary.json"], summary_bytes)
    print("STAGE4A_REPORTING summary_write_complete", flush=True)
    return paths


__all__ = [
    "MATRIX_GROUP_FIELDS",
    "QUANTILE_CONVENTION",
    "STAGE4A_REPORT_SCHEMA_VERSION",
    "aggregate_stage4a_events",
    "matrix_to_csv",
    "render_report",
    "write_stage4a_outputs",
]
