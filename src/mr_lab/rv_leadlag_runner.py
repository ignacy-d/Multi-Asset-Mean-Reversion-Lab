"""Fail-closed local empirical runner for frozen RV-LEADLAG-2024-v1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mr_lab.data import Bar, Timeframe
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.providers.fx_universe_2024 import FxUniverseError, validate_registry
from mr_lab.rv_leadlag import (
    DEFAULT_PARAMETERS,
    RELATIONSHIPS,
    SCIENTIFIC_STATUS,
    STUDY_ID,
    LeadLagEvent,
    detect_leadlag_events,
)
from mr_lab.sessions import DEFAULT_SESSION_SPEC, classify_timestamp
from mr_lab.stage4c_v2 import CostProfileV2, ResearchOutcome, analyze

REGISTRY_PATH = Path("configs/fx-universe-2024-registry-v1.json")
COST_PROFILE_PATH = Path("configs/stage4c-ftmo-cost-profile-v2.json")
OUTPUT_DIR = Path("results/rv-leadlag-2024-v1")
INSTRUMENTS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF")
HORIZONS = (5, 15, 30, 60)
PRIMARY_HORIZON = 15
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20240920
ACTIVITY_POLICY = "volume == 0 AND open == high == low == close"
REGISTRY_ID = "sha256:16288d516f88d58e243864ef2c8a572f639bc92f7b623412939faff298a9eab2"
REGISTRY_FILE_SHA256 = (
    "sha256:61d1876ebe12224ac5c668a1b61c8ffb7b080859395d02f237e13f452b254850"
)
COST_PROFILE_FILE_SHA256 = (
    "sha256:bab7cea933329aba5182ed05364414828277bf74046af3cf4ebf539cb8323316"
)
METHODOLOGY = {
    "study_id": STUDY_ID,
    "scientific_status": SCIENTIFIC_STATUS,
    "signal_parameters": asdict(DEFAULT_PARAMETERS),
    "relationships": [asdict(item) for item in RELATIONSHIPS],
    "instruments": INSTRUMENTS,
    "activity_policy": ACTIVITY_POLICY,
    "m5_policy": "five-contiguous-active-M1-bars-no-fill-repair-or-gap-bridging",
    "entry": "exact-M1-open-at-event-timestamp",
    "exit": "exact-M1-close-with-close-time-equal-event-plus-horizon",
    "horizons_minutes": HORIZONS,
    "primary_horizon_minutes": PRIMARY_HORIZON,
    "bootstrap": {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "blocks": "whole-calendar-month",
    },
    "session_spec_id": DEFAULT_SESSION_SPEC.session_spec_id,
}


class LeadLagRunnerError(ValueError):
    """An empirical input or output violates the frozen runner contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


METHODOLOGY_ID = _sha(_canonical(METHODOLOGY))


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, object]:
    """Validate the complete frozen registry before any corpus is touched."""
    if path != REGISTRY_PATH:
        raise LeadLagRunnerError("only the explicit frozen registry is authorized")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        validated = validate_registry(value)
    except (OSError, json.JSONDecodeError, FxUniverseError) as exc:
        raise LeadLagRunnerError("registry contract validation failed") from exc
    if _sha(raw) != REGISTRY_FILE_SHA256 or validated.get("registry_id") != REGISTRY_ID:
        raise LeadLagRunnerError("authenticated registry identity mismatch")
    return validated


def authenticate_corpus(entry: Mapping[str, object], instrument: str) -> Path:
    """Authenticate only the manifest explicitly named for one required instrument."""
    if entry.get("instrument") != instrument or instrument not in INSTRUMENTS:
        raise LeadLagRunnerError("unexpected corpus entry")
    path = Path(str(entry.get("corpus_path")))
    manifest_path = path / "corpus-manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LeadLagRunnerError(f"invalid declared manifest for {instrument}") from exc
    if _sha(raw) != entry.get("manifest_sha256"):
        raise LeadLagRunnerError(f"manifest hash mismatch for {instrument}")
    for field in (
        "instrument",
        "corpus_id",
        "assembled_dataset_id",
        "requested_start_date",
        "requested_end_date",
    ):
        if manifest.get(field) != entry.get(field):
            raise LeadLagRunnerError(
                f"manifest identity mismatch for {instrument}: {field}"
            )
    return path


def is_padding(bar: Bar) -> bool:
    return bar.volume == 0 and bar.open == bar.high == bar.low == bar.close


def _valid_m1(bar: Bar, instrument: str) -> bool:
    return (
        bar.instrument == instrument
        and bar.timeframe == Timeframe("1m")
        and bar.open_time.tzinfo is not None
        and bar.open_time.utcoffset() == timedelta(0)
        and bar.close_time == bar.open_time + timedelta(minutes=1)
        and bar.available_at <= bar.close_time
        and all(
            math.isfinite(x) and x > 0 for x in (bar.open, bar.high, bar.low, bar.close)
        )
        and not is_padding(bar)
    )


@dataclass(frozen=True, slots=True)
class PreparedData:
    m5_bars: tuple[Bar, ...]
    m1_by_instrument: Mapping[str, Mapping[datetime, Bar]]
    activity_counts: Mapping[str, Mapping[str, int]]


def prepare_data(series: Mapping[str, Sequence[Bar]]) -> PreparedData:
    """Build M5 bars solely from five exact, active, valid M1 components."""
    if set(series) != set(INSTRUMENTS):
        raise LeadLagRunnerError(
            "all and only the explicit six instruments are required"
        )
    all_m5: list[Bar] = []
    indexes: dict[str, dict[datetime, Bar]] = {}
    counts: dict[str, dict[str, int]] = {}
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    for instrument in INSTRUMENTS:
        bars = tuple(series[instrument])
        by_open: dict[datetime, Bar] = {}
        for bar in bars:
            if bar.open_time in by_open:
                raise LeadLagRunnerError(f"duplicate M1 open time for {instrument}")
            by_open[bar.open_time] = bar
        indexes[instrument] = by_open
        padding = sum(is_padding(bar) for bar in bars)
        candidate_starts = sorted(
            stamp
            for stamp in by_open
            if (stamp - epoch) % timedelta(minutes=5) == timedelta(0)
        )
        built = 0
        for start in candidate_starts:
            components = [by_open.get(start + timedelta(minutes=i)) for i in range(5)]
            if any(bar is None for bar in components):
                continue
            complete = [bar for bar in components if bar is not None]
            if not all(_valid_m1(bar, instrument) for bar in complete):
                continue
            all_m5.append(
                Bar(
                    instrument=instrument,
                    timeframe=Timeframe("5m"),
                    open_time=start,
                    close_time=start + timedelta(minutes=5),
                    available_at=start + timedelta(minutes=5),
                    open=complete[0].open,
                    high=max(x.high for x in complete),
                    low=min(x.low for x in complete),
                    close=complete[-1].close,
                    price_basis=complete[0].price_basis,
                    volume=sum(float(x.volume or 0) for x in complete),
                    volume_semantics=complete[0].volume_semantics,
                )
            )
            built += 1
        counts[instrument] = {"m1": len(bars), "padding_m1": padding, "valid_m5": built}
    return PreparedData(
        tuple(sorted(all_m5, key=lambda x: (x.close_time, x.instrument))),
        indexes,
        counts,
    )


def cost_profile_session(active_sessions: Sequence[str]) -> str:
    """Use a session cost only for exactly one active major session."""
    unique = set(active_sessions)
    return next(iter(unique)) if len(unique) == 1 else "overall"


def _reason(bar: Bar | None, instrument: str, price: str) -> str | None:
    if bar is None:
        return f"MISSING_EXACT_{price.upper()}_BAR"
    if is_padding(bar):
        return f"PADDING_{price.upper()}_BAR"
    if not _valid_m1(bar, instrument):
        return f"INVALID_{price.upper()}_BAR"
    return None


def execute_event(
    event: LeadLagEvent, prepared: PreparedData
) -> tuple[dict[str, Any], tuple[ResearchOutcome, ...]]:
    """Apply exact entry and exit timing without searching for substitutes."""
    bars = prepared.m1_by_instrument[event.laggard]
    entry = bars.get(event.timestamp)
    entry_reason = _reason(entry, event.laggard, "entry")
    classification = classify_timestamp(event.timestamp, DEFAULT_SESSION_SPEC)
    session = cost_profile_session(classification.active_sessions)
    record: dict[str, Any] = {
        **asdict(event),
        "timestamp": event.timestamp.isoformat(),
        "session_labels": list(classification.active_sessions),
        "session_regime": classification.regime,
        "cost_profile_session": session,
        "entry_complete": entry_reason is None,
        "entry_incomplete_reason": entry_reason,
        "horizons": {},
    }
    outcomes: list[ResearchOutcome] = []
    for horizon in HORIZONS:
        exit_time = event.timestamp + timedelta(minutes=horizon)
        exit_bar = bars.get(exit_time - timedelta(minutes=1))
        exit_reason = _reason(exit_bar, event.laggard, "exit")
        reason = entry_reason or exit_reason
        detail: dict[str, Any] = {
            "complete": reason is None,
            "exit_timestamp": exit_time.isoformat(),
            "incomplete_reason": reason,
        }
        if reason is None and entry is not None and exit_bar is not None:
            signed = event.direction * math.log(exit_bar.close / entry.open)
            bps = signed * 10_000
            outcome = ResearchOutcome(
                study_id=STUDY_ID,
                event_id=event.event_id,
                timestamp=event.timestamp,
                instrument=event.laggard,
                direction=event.direction,
                entry_timestamp=event.timestamp,
                entry_price=entry.open,
                exit_timestamp=exit_time,
                exit_price=exit_bar.close,
                horizon_minutes=horizon,
                gross_signed_bps_return=bps,
                cost_profile_session=session,
                session_labels=classification.active_sessions,
                metadata={
                    "relationship_id": event.relationship_id,
                    "leader": event.leader,
                    "leader_z_bin": event.leader_z_bin,
                    "session_regime": classification.regime,
                },
            )
            detail |= {
                "entry_price": entry.open,
                "exit_price": exit_bar.close,
                "signed_log_return": signed,
                "gross_signed_bps": bps,
                "gross_pips": outcome.gross_pips,
            }
            outcomes.append(outcome)
        record["horizons"][f"H{horizon}"] = detail
    return record, tuple(outcomes)


def _screen(analysis: dict[str, Any], cost_available: bool) -> list[str]:
    labels = [
        "PASSES_GROSS_1PIP_DISCOVERY_SCREEN"
        if analysis["gross_pips"]["mean"] >= 1.0
        else "FAILS_GROSS_1PIP_SCREEN"
    ]
    floor = analysis["cost_floor"]["net_mean_pips"]
    if not cost_available or floor is None:
        labels.append("BLOCKED_MISSING_COST_PROFILE")
    else:
        labels.append(
            "SURVIVES_AUTHENTICATED_COST_FLOOR_DISCOVERY_ONLY"
            if floor >= 0
            else "FAILS_AUTHENTICATED_COST_FLOOR"
        )
    return labels


def economic_summary(
    outcomes: Sequence[ResearchOutcome], profile: CostProfileV2 | None
) -> dict[str, Any]:
    """Run shared v2 analysis for only the frozen, predeclared segments."""
    primary = [x for x in outcomes if x.horizon_minutes == PRIMARY_HORIZON]
    segment_values: dict[str, list[ResearchOutcome]] = defaultdict(list)
    relationships = {item.relationship_id for item in RELATIONSHIPS}
    for outcome in primary:
        meta = outcome.metadata
        keys = [
            "ALL",
            f"relationship:{meta['relationship_id']}",
            f"leader:{meta['leader']}",
            f"regime:{meta['session_regime']}",
            f"leader_z:{meta['leader_z_bin']}",
        ]
        keys.extend(f"session:{label}" for label in outcome.session_labels)
        for key in keys:
            segment_values[key].append(outcome)
    segments = {}
    for key in sorted(segment_values):
        result = analyze(
            segment_values[key],
            profile,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        result["screen_labels"] = _screen(result, profile is not None)
        segments[key] = result
    for relationship in RELATIONSHIPS:
        key = f"relationship:{relationship.relationship_id}"
        if key not in segments:
            segments[key] = {
                "methodology_id": "stage4c-economic-analysis-v2",
                "n": 0,
                "gross_pips": None,
                "gross_bps": None,
                "bootstrap_summary": None,
                "calendar_month_breadth": 0,
                "cost_floor": None,
                "screen_labels": ["FAILS_GROSS_1PIP_SCREEN"],
            }
    passing = [
        segments[f"relationship:{name}"]
        for name in relationships
        if segments[f"relationship:{name}"]["n"]
    ]
    family = None
    if not any(
        x["gross_pips"]["mean"] >= 1.0 and x["bootstrap_summary"]["p2_5"] > 0
        for x in passing
    ):
        family = "PARK_NO_ECONOMICALLY_LARGE_V1_EFFECT"
    return {"primary_horizon": "H15", "segments": segments, "family_status": family}


def _code_revision() -> str:
    """Record the revision when available, without requiring a particular SHA."""
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _encode_events(records: Sequence[Mapping[str, Any]]) -> bytes:
    payload = b"".join(_canonical(record) + b"\n" for record in records)
    out = __import__("io").BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as stream:
        stream.write(payload)
    return out.getvalue()


def write_outputs(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    """Atomically publish all deterministic artifacts, refusing any overwrite."""
    names = ("events.jsonl.gz", "summary.json", "execution-audit.json")
    if any((output_dir / name).exists() for name in names):
        raise LeadLagRunnerError("refusing to overwrite empirical artifacts")
    encoded_summary = (
        json.dumps(summary, indent=2, sort_keys=True, default=str).encode() + b"\n"
    )
    event_bytes = _encode_events(records)
    final_audit = dict(audit) | {
        "artifact_hashes": {
            "events.jsonl.gz": _sha(event_bytes),
            "summary.json": _sha(encoded_summary),
        }
    }
    payloads = (
        event_bytes,
        encoded_summary,
        json.dumps(final_audit, indent=2, sort_keys=True, default=str).encode() + b"\n",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary: list[tuple[Path, Path]] = []
    try:
        for name, payload in zip(names, payloads, strict=True):
            fd, raw_path = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
            temp = Path(raw_path)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.append((temp, output_dir / name))
        for temp, target in temporary:
            os.replace(temp, target)
    finally:
        for temp, _ in temporary:
            temp.unlink(missing_ok=True)


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    registry = load_registry()
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    datasets = {}
    for instrument in INSTRUMENTS:
        entry = entries[instrument]
        assert isinstance(entry, dict)
        datasets[instrument] = load_offline_corpus(
            authenticate_corpus(entry, instrument)
        )
    prepared = prepare_data({name: datasets[name].bars for name in INSTRUMENTS})
    events = detect_leadlag_events(prepared.m5_bars)
    records: list[dict[str, Any]] = []
    outcomes: list[ResearchOutcome] = []
    for event in events:
        record, event_outcome_rows = execute_event(event, prepared)
        records.append(record)
        outcomes.extend(event_outcome_rows)
    cost_profile_raw = COST_PROFILE_PATH.read_bytes()
    if _sha(cost_profile_raw) != COST_PROFILE_FILE_SHA256:
        raise LeadLagRunnerError("authenticated cost-profile identity mismatch")
    profile = CostProfileV2.load(COST_PROFILE_PATH)
    summary = {
        "study_id": STUDY_ID,
        "methodology_id": METHODOLOGY_ID,
        "registry_id": registry["registry_id"],
        "event_count": len(events),
        "activity_counts": prepared.activity_counts,
        "economic_analysis": economic_summary(outcomes, profile),
    }
    audit = {
        "methodology": METHODOLOGY,
        "methodology_id": METHODOLOGY_ID,
        "registry_id": registry["registry_id"],
        "dataset_ids": {
            name: entries[name]["assembled_dataset_id"] for name in INSTRUMENTS
        },
        "cost_profile_sha256": _sha(COST_PROFILE_PATH.read_bytes()),
        "runner_source_sha256": _sha(Path(__file__).read_bytes()),
        "code_revision": _code_revision(),
    }
    write_outputs(output_dir, records, summary, audit)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir)["economic_analysis"]["family_status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
