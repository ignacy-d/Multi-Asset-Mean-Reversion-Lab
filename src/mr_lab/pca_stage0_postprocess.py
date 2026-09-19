"""Deterministic recovery reporting from an existing Stage0 events JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import numpy as np

HORIZONS = (5, 15, 30, 60)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20240918
BLOCKER = "BLOCKED_MISSING_FROZEN_ATR_NORMALIZATION_DEFINITION"
POSTPROCESSOR_IDENTITY = "mr-lab-pca-stage0-events-postprocessor-v1"
REQUIRED_FIELDS = frozenset(
    {
        "timestamp",
        "instrument",
        "entry_complete",
        "process_identity",
        "activity_policy",
        *(f"h{h}_complete" for h in HORIZONS),
        *(f"h{h}_signed_bps_return" for h in HORIZONS),
    }
)


class PostprocessError(ValueError):
    """Raised when an events artifact cannot be safely summarized."""


def _timestamp(value: object, line_number: int) -> datetime:
    if not isinstance(value, str):
        raise PostprocessError(f"line {line_number}: timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PostprocessError(f"line {line_number}: invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(None):
        raise PostprocessError(f"line {line_number}: timestamp must be UTC-aware")
    return parsed


def _validate_event(value: object, line_number: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PostprocessError(f"line {line_number}: event must be an object")
    missing = REQUIRED_FIELDS - value.keys()
    if missing:
        raise PostprocessError(
            f"line {line_number}: missing required fields: {', '.join(sorted(missing))}"
        )
    _timestamp(value["timestamp"], line_number)
    for field in ("instrument", "process_identity", "activity_policy"):
        if not isinstance(value[field], str) or not value[field]:
            raise PostprocessError(f"line {line_number}: {field} must be non-empty")
    if not isinstance(value["entry_complete"], bool):
        raise PostprocessError(f"line {line_number}: entry_complete must be boolean")
    for horizon in HORIZONS:
        complete_field = f"h{horizon}_complete"
        metric = f"h{horizon}_signed_bps_return"
        if not isinstance(value[complete_field], bool):
            raise PostprocessError(
                f"line {line_number}: {complete_field} must be boolean"
            )
        outcome = value[metric]
        if value[complete_field] and (
            isinstance(outcome, bool)
            or not isinstance(outcome, int | float)
            or not isfinite(float(outcome))
        ):
            raise PostprocessError(
                f"line {line_number}: complete {metric} must be finite"
            )
    return value


def read_events(path: Path) -> tuple[list[dict[str, object]], str]:
    """Stream and validate events while hashing their exact source bytes."""
    events: list[dict[str, object]] = []
    digest = hashlib.sha256()
    process_identity: str | None = None
    activity_policy: str | None = None
    try:
        with path.open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                digest.update(raw)
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PostprocessError(
                        f"line {line_number}: malformed JSON"
                    ) from exc
                event = _validate_event(value, line_number)
                current_process = str(event["process_identity"])
                current_policy = str(event["activity_policy"])
                if process_identity is not None and current_process != process_identity:
                    raise PostprocessError("inconsistent process_identity")
                if activity_policy is not None and current_policy != activity_policy:
                    raise PostprocessError("inconsistent activity_policy")
                process_identity = current_process
                activity_policy = current_policy
                events.append(event)
    except OSError as exc:
        raise PostprocessError(f"cannot read explicit events path: {path}") from exc
    if not events:
        raise PostprocessError("events file must contain at least one event")
    return events, "sha256:" + digest.hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            key: None
            for key in (
                "arithmetic_mean_bps",
                "median_bps",
                "p10_bps",
                "p25_bps",
                "p75_bps",
                "p90_bps",
                "trimmed_mean_5pct_bps",
                "positive_fraction",
                "average_positive_outcome_bps",
                "average_negative_outcome_bps",
            )
        } | {"n": 0}
    array = np.asarray(values, dtype=float)
    ordered = np.sort(array)
    trim = int(len(ordered) * 0.05)
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    positive = array[array > 0]
    negative = array[array < 0]
    return {
        "n": len(array),
        "arithmetic_mean_bps": float(np.mean(array)),
        "median_bps": float(np.median(array)),
        "p10_bps": float(np.percentile(array, 10)),
        "p25_bps": float(np.percentile(array, 25)),
        "p75_bps": float(np.percentile(array, 75)),
        "p90_bps": float(np.percentile(array, 90)),
        "trimmed_mean_5pct_bps": float(np.mean(trimmed)),
        "positive_fraction": float(np.mean(array > 0)),
        "average_positive_outcome_bps": (
            float(np.mean(positive)) if len(positive) else None
        ),
        "average_negative_outcome_bps": (
            float(np.mean(negative)) if len(negative) else None
        ),
    }


def _outcome(event: Mapping[str, object], horizon: int) -> float:
    value = event[f"h{horizon}_signed_bps_return"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PostprocessError(f"complete H{horizon} outcome must be numeric")
    return float(value)


def _group_h15(
    events: Sequence[Mapping[str, object]], key
) -> dict[str, dict[str, int | float | None]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event["h15_complete"]:
            groups[key(event)].append(_outcome(event, 15))
    return {name: _distribution(groups[name]) for name in sorted(groups)}


def _bootstrap(events: Sequence[Mapping[str, object]]) -> dict[str, int | float | bool]:
    groups: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event["h15_complete"]:
            stamp = _timestamp(event["timestamp"], 0)
            groups[stamp.strftime("%Y-%m")].append(_outcome(event, 15))
    if not groups:
        raise PostprocessError("bootstrap requires at least one complete H15 outcome")
    blocks = tuple(groups[name] for name in sorted(groups))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = float(
            np.mean(
                [value for block_index in selected for value in blocks[block_index]]
            )
        )
    return {
        "observed_calendar_month_blocks": len(blocks),
        "replicates": int(BOOTSTRAP_REPLICATES),
        "seed": int(BOOTSTRAP_SEED),
        "bootstrap_mean_bps": float(np.mean(samples)),
        "p2_5_bps": float(np.percentile(samples, 2.5)),
        "median_bps": float(np.median(samples)),
        "p97_5_bps": float(np.percentile(samples, 97.5)),
        "lower_2_5pct_gt_zero": bool(np.percentile(samples, 2.5) > 0),
    }


def build_summary(
    events: Sequence[Mapping[str, object]], source_sha256: str
) -> dict[str, object]:
    stamps = [_timestamp(event["timestamp"], 0) for event in events]
    event_count = len(events)
    complete_h15 = [event for event in events if event["h15_complete"]]
    instrument_counts = Counter(str(event["instrument"]) for event in events)
    quarter_counts = Counter(
        f"{stamp.year:04d}-Q{(stamp.month - 1) // 3 + 1}" for stamp in stamps
    )
    contributions: dict[str, float] = defaultdict(float)
    instrument_values: dict[str, list[float]] = defaultdict(list)
    for event in complete_h15:
        name = str(event["instrument"])
        outcome = _outcome(event, 15)
        contributions[name] += outcome
        instrument_values[name].append(outcome)
    contributions = dict(sorted(contributions.items()))
    absolute_total = sum(abs(value) for value in contributions.values())
    executable_count = int(sum(bool(event["entry_complete"]) for event in events))
    return {
        "postprocessor_identity": POSTPROCESSOR_IDENTITY,
        "source_events_sha256": source_sha256,
        "process_identity": str(events[0]["process_identity"]),
        "activity_policy": str(events[0]["activity_policy"]),
        "primary_promotion_status": BLOCKER,
        "h15_diagnostic_scope": "raw-H15 discovery diagnostic; not ATR-normalized",
        "event_count": int(event_count),
        "first_event_timestamp": min(stamps).isoformat(),
        "last_event_timestamp": max(stamps).isoformat(),
        "executable_entry_count": executable_count,
        "executable_entry_fraction": float(executable_count / event_count),
        "horizon_statistics": {
            f"h{h}": _distribution(
                [_outcome(event, h) for event in events if event[f"h{h}_complete"]]
            )
            for h in HORIZONS
        },
        "h15_by_instrument": _group_h15(events, lambda event: str(event["instrument"])),
        "h15_by_calendar_month": _group_h15(
            events, lambda event: str(event["timestamp"])[:7]
        ),
        "h15_by_instrument_calendar_quarter": _group_h15(
            events,
            lambda event: (
                f"{event['instrument']}|{str(event['timestamp'])[:4]}-Q"
                f"{(int(str(event['timestamp'])[5:7]) - 1) // 3 + 1}"
            ),
        ),
        "event_count_concentration": {
            "maximum_instrument_fraction": float(
                max(instrument_counts.values()) / event_count
            ),
            "maximum_calendar_quarter_fraction": float(
                max(quarter_counts.values()) / event_count
            ),
        },
        "instruments_with_positive_mean_complete_h15": int(
            sum(np.mean(values) > 0 for values in instrument_values.values())
        ),
        "signed_h15_contribution_bps_by_instrument": contributions,
        "absolute_signed_contribution_concentration": {
            "definition": (
                "maximum absolute instrument signed-H15 sum divided by the sum "
                "of absolute instrument signed-H15 sums"
            ),
            "value": (
                float(max(map(abs, contributions.values())) / absolute_total)
                if absolute_total
                else 0.0
            ),
        },
        "h15_calendar_month_block_bootstrap": _bootstrap(events),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def postprocess(events_path: Path, output_dir: Path) -> Path:
    output = output_dir / "summary.json"
    if output_dir.exists():
        raise PostprocessError("refusing to overwrite existing output directory")
    events, digest = read_events(events_path)
    summary = build_summary(events, digest)
    _atomic_json(output, summary)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    postprocess(args.events, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
