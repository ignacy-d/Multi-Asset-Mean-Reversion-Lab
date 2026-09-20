"""Authenticated local runner for VOL-COMPRESSION-BREAKOUT-2024-v1.

The signal implementation lives in :mod:`mr_lab.vol_compression_breakout` and
is intentionally not duplicated here.  This module is only the reproducible
data, execution-outcome, economic-analysis, and artifact boundary.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mr_lab.data.models import Bar, Timeframe
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.providers.fx_universe_2024 import FxUniverseError, validate_registry
from mr_lab.sessions import DEFAULT_SESSION_SPEC, classify_timestamp
from mr_lab.stage4c_v2 import CostProfileV2, ResearchOutcome, analyze
from mr_lab.vol_compression_breakout import (
    STUDY_ID,
    VolCompressionEvent,
    generate_events,
)

REGISTRY_SCHEMA = "fx-universe-2024-registry-v1"
INSTRUMENTS = (
    "AUDJPY",
    "AUDUSD",
    "EURGBP",
    "EURUSD",
    "GBPUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
)
HORIZONS = (15, 30, 60)
PRIMARY_HORIZON = 30
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20240920
DEFAULT_REGISTRY = Path("configs/fx-universe-2024-registry-v1.json")
DEFAULT_OUTPUT = Path("results/vol-compression-breakout-2024-v1")


class VolCompressionRunnerError(ValueError):
    """A frozen runner contract or authenticated input was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VolCompressionRunnerError(f"cannot read explicit path: {path}") from exc
    return "sha256:" + digest.hexdigest()


def load_authenticated_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load only the explicit, frozen nine-pair discovery registry."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VolCompressionRunnerError("invalid explicit registry") from exc
    try:
        raw = validate_registry(raw)
    except FxUniverseError as exc:
        raise VolCompressionRunnerError("registry authentication failed") from exc
    if raw.get("registry_schema_version") != REGISTRY_SCHEMA:
        raise VolCompressionRunnerError("unexpected registry schema")
    if (raw.get("requested_start_date"), raw.get("requested_end_date")) != (
        "2024-01-01",
        "2024-12-31",
    ):
        raise VolCompressionRunnerError(
            "registry is not bounded to discovery year 2024"
        )
    entries = raw.get("instruments")
    if not isinstance(entries, dict) or tuple(sorted(entries)) != INSTRUMENTS:
        raise VolCompressionRunnerError(
            "registry must contain exactly nine allowed pairs"
        )
    for instrument in INSTRUMENTS:
        entry = entries[instrument]
        required = (
            "corpus_path",
            "corpus_id",
            "assembled_dataset_id",
            "manifest_sha256",
        )
        if (
            entry.get("instrument") != instrument
            or entry.get("verification_status") != "verified"
            or entry.get("native_timeframe") != "1m"
            or any(not isinstance(entry.get(key), str) for key in required)
        ):
            raise VolCompressionRunnerError(
                f"unauthenticated registry entry: {instrument}"
            )
    return raw


def authenticate_corpus(entry: Mapping[str, Any]) -> Path:
    """Authenticate an exact manifest path without looking for alternatives."""
    corpus = Path(str(entry["corpus_path"]))
    manifest_path = corpus / "corpus-manifest.json"
    if _sha256(manifest_path) != entry["manifest_sha256"]:
        raise VolCompressionRunnerError("corpus manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VolCompressionRunnerError(
            "invalid authenticated corpus manifest"
        ) from exc
    for field in ("corpus_id", "assembled_dataset_id", "instrument"):
        if manifest.get(field) != entry[field]:
            raise VolCompressionRunnerError(f"corpus manifest mismatch: {field}")
    if "2025" in json.dumps(manifest, sort_keys=True):
        raise VolCompressionRunnerError("sealed-year reference in corpus manifest")
    return corpus


def is_provider_padding(bar: Bar) -> bool:
    return bar.volume == 0 and bar.open == bar.high == bar.low == bar.close


def invalid_m1_reason(bar: Bar) -> str | None:
    if bar.timeframe != Timeframe("1m"):
        return "invalid_timeframe"
    if bar.close_time - bar.open_time != timedelta(minutes=1):
        return "invalid_duration"
    if is_provider_padding(bar):
        return "provider_padding"
    prices = (bar.open, bar.high, bar.low, bar.close)
    if not all(math.isfinite(value) and value > 0 for value in prices):
        return "invalid_ohlc"
    return None


def canonical_m15_bars(bars: Iterable[Bar]) -> tuple[tuple[Bar, ...], dict[str, int]]:
    """Aggregate only exact, contiguous groups of fifteen valid M1 observations."""
    groups: dict[datetime, list[Bar]] = defaultdict(list)
    audit: Counter[str] = Counter()
    ordered = sorted(bars, key=lambda item: item.open_time)
    for bar in ordered:
        start = bar.open_time.replace(
            minute=(bar.open_time.minute // 15) * 15, second=0, microsecond=0
        )
        groups[start].append(bar)
    result: list[Bar] = []
    for start in sorted(groups):
        window = groups[start]
        if len(window) != 15 or any(
            item.open_time != start + timedelta(minutes=index)
            for index, item in enumerate(window)
        ):
            audit["missing_observation"] += 1
            continue
        reasons = [invalid_m1_reason(item) for item in window]
        reason = next((item for item in reasons if item is not None), None)
        if reason is not None:
            audit[reason] += 1
            continue
        first, last = window[0], window[-1]
        volumes = [item.volume for item in window]
        volume = sum(item for item in volumes if item is not None)
        if all(item is None for item in volumes):
            volume = None
        result.append(
            Bar(
                instrument=first.instrument,
                timeframe=Timeframe("15m"),
                open_time=start,
                close_time=start + timedelta(minutes=15),
                available_at=start + timedelta(minutes=15),
                open=first.open,
                high=max(item.high for item in window),
                low=min(item.low for item in window),
                close=last.close,
                price_basis=first.price_basis,
                volume=volume,
                volume_semantics=first.volume_semantics,
            )
        )
    audit["valid_m15"] = len(result)
    return tuple(result), dict(sorted(audit.items()))


def cost_session(timestamp: datetime) -> tuple[str, tuple[str, ...], str]:
    classification = classify_timestamp(timestamp, DEFAULT_SESSION_SPEC)
    active = classification.active_sessions
    return (active[0] if len(active) == 1 else "overall", active, classification.regime)


def _lookup_reason(bar: Bar | None, price_field: str) -> str | None:
    if bar is None:
        return "missing_observation"
    if is_provider_padding(bar):
        return "provider_padding"
    value = getattr(bar, price_field)
    if not math.isfinite(value) or value <= 0:
        return f"invalid_{price_field}"
    return None


def construct_outcomes(
    event: VolCompressionEvent, bars: Sequence[Bar]
) -> tuple[dict[int, ResearchOutcome], dict[str, Any]]:
    """Apply exact timestamp entry/exits; never search forward."""
    opens = {bar.open_time: bar for bar in bars}
    closes = {bar.close_time: bar for bar in bars}
    entry = opens.get(event.timestamp)
    entry_reason = _lookup_reason(entry, "open")
    audit: dict[str, Any] = {"entry_reason": entry_reason, "horizons": {}}
    outcomes: dict[int, ResearchOutcome] = {}
    if entry_reason is not None or entry is None:
        for horizon in HORIZONS:
            audit["horizons"][str(horizon)] = {"reason": "entry_" + str(entry_reason)}
        return outcomes, audit
    session, active, regime = cost_session(event.timestamp)
    for horizon in HORIZONS:
        exit_time = event.timestamp + timedelta(minutes=horizon)
        exit_bar = closes.get(exit_time)
        reason = _lookup_reason(exit_bar, "close")
        audit["horizons"][str(horizon)] = {"reason": reason}
        if reason is not None or exit_bar is None:
            continue
        direction = 1 if event.direction == "LONG" else -1
        signed_raw = direction * (exit_bar.close / entry.open - 1.0)
        outcomes[horizon] = ResearchOutcome(
            study_id=STUDY_ID,
            event_id=event.event_id,
            timestamp=event.timestamp,
            instrument=event.instrument,
            direction=direction,
            entry_timestamp=event.timestamp,
            entry_price=entry.open,
            exit_timestamp=exit_time,
            exit_price=exit_bar.close,
            horizon_minutes=horizon,
            gross_signed_bps_return=signed_raw * 10_000,
            cost_profile_session=session,
            session_labels=active,
            metadata={"diagnostic_regime": regime},
        )
    return outcomes, audit


def _segments(outcomes: Sequence[ResearchOutcome]) -> dict[str, list[ResearchOutcome]]:
    groups: dict[str, list[ResearchOutcome]] = {"ALL": list(outcomes)}
    for item in outcomes:
        groups.setdefault(f"instrument:{item.instrument}", []).append(item)
        groups.setdefault(
            "direction:" + ("LONG" if item.direction == 1 else "SHORT"), []
        ).append(item)
        regime = str(item.metadata["diagnostic_regime"])
        groups.setdefault(f"regime:{regime}", []).append(item)
        for session in item.session_labels:
            groups.setdefault(f"session:{session}", []).append(item)
    return dict(sorted(groups.items()))


def _rv_bucket(event: VolCompressionEvent) -> str:
    if event.rv_percentile <= 0.05:
        return "0-5"
    if event.rv_percentile <= 0.10:
        return "5-10"
    return "10-20"


def summarize(
    events: Sequence[VolCompressionEvent],
    outcomes: Mapping[int, Sequence[ResearchOutcome]],
    profile: CostProfileV2 | None,
) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for horizon in HORIZONS:
        rows = outcomes[horizon]
        horizons[f"H{horizon}"] = {
            key: analyze(
                value,
                profile,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED,
            )
            for key, value in _segments(rows).items()
            if value
        }
    primary = horizons["H30"]
    passes = {
        key: value["gross_pips"]["mean"] >= 1.0
        and value["bootstrap_summary"]["p2_5"] > 0.0
        for key, value in primary.items()
    }
    rv_counts = Counter(_rv_bucket(event) for event in events)
    return {
        "study_id": STUDY_ID,
        "primary_horizon": "H30",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "calendar-month",
        },
        "event_count": len(events),
        "horizons": horizons,
        "diagnostic_rv_percentile_buckets": {
            key: rv_counts.get(key, 0) for key in ("0-5", "5-10", "10-20")
        },
        "economic_discovery_screen": {
            "segment_passes": passes,
            "mean_h30_gross_pips_min": 1.0,
            "bootstrap_p2_5_strictly_positive": True,
        },
        "status": "CONTINUE_VOL_COMPRESSION_BREAKOUT_V1"
        if any(passes.values())
        else "PARK_VOL_COMPRESSION_BREAKOUT_V1",
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _event_record(
    event: VolCompressionEvent,
    found: Mapping[int, ResearchOutcome],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    row = asdict(event)
    row["timestamp"] = event.timestamp.isoformat()
    row["rv_percentile_bucket"] = _rv_bucket(event)
    row["execution"] = audit
    row["outcomes"] = {
        f"H{horizon}": {
            "entry_timestamp": item.entry_timestamp.isoformat(),
            "entry_price": item.entry_price,
            "exit_timestamp": item.exit_timestamp.isoformat(),
            "exit_price": item.exit_price,
            "gross_pips": item.gross_pips,
            "gross_signed_bps_return": item.gross_signed_bps_return,
            "cost_profile_session": item.cost_profile_session,
        }
        for horizon, item in sorted(found.items())
    }
    return row


def _write_artifacts(
    target: Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    audit: dict[str, Any],
) -> None:
    events_path = target / "events.jsonl.gz"
    with (
        events_path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
    ):
        for record in records:
            compressed.write(_json_bytes(record))
    (target / "summary.json").write_bytes(_json_bytes(summary))
    audit["artifact_sha256"] = {
        "events.jsonl.gz": _sha256(events_path),
        "summary.json": _sha256(target / "summary.json"),
    }
    (target / "execution-audit.json").write_bytes(_json_bytes(audit))


def run(
    registry_path: Path = DEFAULT_REGISTRY,
    output_dir: Path = DEFAULT_OUTPUT,
    cost_profile_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    registry = load_authenticated_registry(registry_path)
    all_events: list[VolCompressionEvent] = []
    all_outcomes: dict[int, list[ResearchOutcome]] = {h: [] for h in HORIZONS}
    records: list[dict[str, Any]] = []
    corpus_audit: dict[str, Any] = {}
    for instrument in INSTRUMENTS:
        entry = registry["instruments"][instrument]
        corpus = authenticate_corpus(entry)
        m1 = tuple(load_offline_corpus(corpus).bars)
        m15, aggregation_audit = canonical_m15_bars(m1)
        events = generate_events(m15)
        corpus_audit[instrument] = {
            "manifest_sha256": entry["manifest_sha256"],
            "aggregation": aggregation_audit,
        }
        for event in events:
            found, execution = construct_outcomes(event, m1)
            all_events.append(event)
            for horizon, outcome in found.items():
                all_outcomes[horizon].append(outcome)
            records.append(_event_record(event, found, execution))
    order = sorted(
        range(len(all_events)),
        key=lambda i: (
            all_events[i].timestamp,
            all_events[i].instrument,
            all_events[i].event_id,
        ),
    )
    all_events = [all_events[i] for i in order]
    records = [records[i] for i in order]
    profile = CostProfileV2.load(cost_profile_path) if cost_profile_path else None
    summary = summarize(all_events, all_outcomes, profile)
    audit = {
        "study_id": STUDY_ID,
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "registry_id": registry["registry_id"],
        "session_spec_id": DEFAULT_SESSION_SPEC.session_spec_id,
        "corpora": corpus_audit,
        "execution_policy": "exact-M1-no-forward-search",
        "horizons_minutes": list(HORIZONS),
        "cost_profile_path": str(cost_profile_path) if cost_profile_path else None,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_artifacts(temporary, records, summary, audit)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-profile", type=Path)
    args = parser.parse_args(argv)
    run(args.registry, args.output_dir, args.cost_profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
