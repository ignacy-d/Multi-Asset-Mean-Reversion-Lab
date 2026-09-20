"""Fail-closed local empirical runner for ``RANGE-SWEEP-2024-v1``.

The runner deliberately owns data authentication, conservative M1 aggregation,
execution alignment, and reporting.  Signal qualification remains exclusively
in :mod:`mr_lab.range_sweep`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mr_lab.data import Bar, Timeframe
from mr_lab.pca_stage0_runner import (
    INSTRUMENTS,
    REGISTRY,
    authenticate_corpus,
    load_registry,
)
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.range_sweep import (
    SCIENTIFIC_STATUS,
    STUDY_ID,
    RangeSweepConfig,
    RangeSweepEvent,
    ReferenceFamily,
    ReferenceSide,
    SignalDirection,
    range_sweep_events,
)
from mr_lab.sessions import DEFAULT_SESSION_SPEC, classify_timestamp
from mr_lab.stage4c_v2 import (
    FX_INSTRUMENTS,
    METHODOLOGY_ID,
    CostProfileV2,
    ResearchOutcome,
    analyze,
)

HORIZONS = (15, 30, 60)
PRIMARY_HORIZON = 15
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20240920
DEFAULT_OUTPUT = Path("results/range-sweep-2024-v1")
DEFAULT_COST_PROFILE = Path("configs/stage4c-ftmo-cost-profile-v2.json")
M5_POLICY = "five-contiguous-valid-nonpadding-m1-v1"
DECISION_CONTINUE = "CONTINUE_RANGE_SWEEP_V1"
DECISION_PARK = "PARK_RANGE_SWEEP_V1"
STUDY_SPEC = {
    "study_id": STUDY_ID,
    "instruments": INSTRUMENTS,
    "m5_policy": M5_POLICY,
    "horizons_minutes": HORIZONS,
    "primary_horizon_minutes": PRIMARY_HORIZON,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_unit": "whole_calendar_month",
    "discovery_screen": {"minimum_n": 100, "minimum_mean_pips": 1.0, "p2_5": ">0"},
    "cost_session_policy": "one_major_else_overall",
}


class RangeSweepRunnerError(ValueError):
    """An empirical input or output violates the frozen runner contract."""


@dataclass(frozen=True, slots=True)
class AggregationAudit:
    source_bars: int
    complete_windows: int
    incomplete_windows: int
    reasons: Mapping[str, int]


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def is_provider_padding(bar: Bar) -> bool:
    """Return the frozen provider-padding predicate (and no broader one)."""
    return bar.volume == 0 and bar.open == bar.high == bar.low == bar.close


def aggregate_m1_to_m5(bars: Sequence[Bar]) -> tuple[tuple[Bar, ...], AggregationAudit]:
    """Build epoch-aligned M5 bars only from five safe contiguous M1 bars."""
    grouped: dict[datetime, list[Bar]] = {}
    instruments = {bar.instrument for bar in bars}
    if len(instruments) > 1:
        raise RangeSweepRunnerError("M5 aggregation requires one instrument")
    for bar in bars:
        if bar.timeframe != Timeframe("1m"):
            raise RangeSweepRunnerError("M5 aggregation accepts M1 bars only")
        seconds = int(bar.open_time.timestamp())
        start = datetime.fromtimestamp(seconds - seconds % 300, tz=UTC)
        grouped.setdefault(start, []).append(bar)
    result: list[Bar] = []
    reasons: Counter[str] = Counter()
    for start in sorted(grouped):
        components = sorted(grouped[start], key=lambda item: item.open_time)
        reason = None
        if len(components) != 5 or any(
            bar.open_time != start + timedelta(minutes=i)
            or bar.close_time != start + timedelta(minutes=i + 1)
            for i, bar in enumerate(components)
        ):
            reason = "missing_or_noncontiguous_component"
        elif any(is_provider_padding(bar) for bar in components):
            reason = "provider_padding_component"
        elif any(
            not math.isfinite(price) or price <= 0
            for bar in components
            for price in (bar.open, bar.high, bar.low, bar.close)
        ):
            reason = "invalid_ohlc_component"
        if reason is not None:
            reasons[reason] += 1
            continue
        first, last = components[0], components[-1]
        volume = (
            None if first.volume is None else sum(bar.volume or 0 for bar in components)
        )
        result.append(
            Bar(
                instrument=first.instrument,
                timeframe=Timeframe("5m"),
                open_time=start,
                close_time=start + timedelta(minutes=5),
                available_at=max(
                    start + timedelta(minutes=5), *(b.available_at for b in components)
                ),
                open=first.open,
                high=max(b.high for b in components),
                low=min(b.low for b in components),
                close=last.close,
                price_basis=first.price_basis,
                volume=volume,
                volume_semantics=first.volume_semantics,
            )
        )
    audit = AggregationAudit(
        len(bars), len(result), sum(reasons.values()), dict(reasons)
    )
    return tuple(result), audit


def _price_reason(bar: Bar | None, field: str) -> str | None:
    if bar is None:
        return "missing_exact_bar"
    if is_provider_padding(bar):
        return "provider_padding"
    value = getattr(bar, field)
    if not math.isfinite(value) or value <= 0:
        return f"invalid_{field}_price"
    return None


def cost_profile_session(timestamp: datetime) -> str:
    """Map exactly-one major session to it; overlaps and gaps to overall."""
    active = classify_timestamp(timestamp, DEFAULT_SESSION_SPEC).active_sessions
    return active[0] if len(active) == 1 else "overall"


def execute_event(event: RangeSweepEvent, m1_bars: Sequence[Bar]) -> dict[str, Any]:
    """Attach exact-time entry and close-time exits without forward searching."""
    opens = {bar.open_time: bar for bar in m1_bars}
    closes = {bar.close_time: bar for bar in m1_bars}
    entry = opens.get(event.signal_timestamp)
    entry_reason = _price_reason(entry, "open")
    classification = classify_timestamp(event.signal_timestamp, DEFAULT_SESSION_SPEC)
    record: dict[str, Any] = {
        "event_id": event.event_id,
        "study_id": STUDY_ID,
        "instrument": event.instrument,
        "timestamp": event.signal_timestamp.isoformat(),
        "reference_family": event.reference_family.value,
        "reference_instance": event.reference_instance,
        "reference_side": event.reference_side.value,
        "reference_price": event.reference_price,
        "signal_direction": event.signal_direction.value,
        "session_labels": list(event.session_labels),
        "major_sessions": list(classification.active_sessions),
        "session_regime": classification.regime,
        "cost_profile_session": cost_profile_session(event.signal_timestamp),
        "overshoot_raw_price": event.overshoot_raw_price,
        "overshoot_pips": event.overshoot_pips,
        "minutes_since_reference_valid": event.minutes_since_reference_valid,
        "volume": event.volume,
        "entry_target_timestamp": event.signal_timestamp.isoformat(),
        "entry_complete": entry_reason is None,
        "entry_incomplete_reason": entry_reason,
        "entry_price": entry.open
        if entry_reason is None and entry is not None
        else None,
    }
    direction = 1 if event.signal_direction is SignalDirection.LONG else -1
    for horizon in HORIZONS:
        target = event.signal_timestamp + timedelta(minutes=horizon)
        exit_bar = closes.get(target)
        reason = (
            "entry_incomplete" if entry_reason else _price_reason(exit_bar, "close")
        )
        complete = reason is None
        exit_price = exit_bar.close if complete and exit_bar is not None else None
        gross_bps = (
            direction * (exit_price / entry.open - 1) * 10_000
            if complete and entry is not None and exit_price is not None
            else None
        )
        record.update(
            {
                f"h{horizon}_target_timestamp": target.isoformat(),
                f"h{horizon}_complete": complete,
                f"h{horizon}_incomplete_reason": reason,
                f"h{horizon}_exit_price": exit_price,
                f"h{horizon}_gross_signed_bps_return": gross_bps,
            }
        )
    return record


def to_research_outcome(record: Mapping[str, Any], horizon: int) -> ResearchOutcome:
    if horizon not in HORIZONS or not record[f"h{horizon}_complete"]:
        raise RangeSweepRunnerError(
            "only complete frozen-horizon outcomes are convertible"
        )
    timestamp = datetime.fromisoformat(str(record["timestamp"]))
    direction = 1 if record["signal_direction"] == "long" else -1
    return ResearchOutcome(
        study_id=STUDY_ID,
        event_id=str(record["event_id"]),
        timestamp=timestamp,
        instrument=str(record["instrument"]),
        direction=direction,
        entry_timestamp=timestamp,
        entry_price=float(record["entry_price"]),
        exit_timestamp=timestamp + timedelta(minutes=horizon),
        exit_price=float(record[f"h{horizon}_exit_price"]),
        horizon_minutes=horizon,
        gross_signed_bps_return=float(record[f"h{horizon}_gross_signed_bps_return"]),
        cost_profile_session=str(record["cost_profile_session"]),
        session_labels=tuple(record["session_labels"]),
        metadata={
            "reference_family": record["reference_family"],
            "reference_side": record["reference_side"],
            "session_regime": record["session_regime"],
            "major_sessions": tuple(record["major_sessions"]),
            "overshoot_pips": record["overshoot_pips"],
        },
    )


def discovery_decision(reference_reports: Mapping[str, Mapping[str, Any]]) -> str:
    passes = any(
        report.get("n", 0) >= 100
        and report.get("gross_pips", {}).get("mean", -math.inf) >= 1.0
        and report.get("bootstrap_summary", {}).get("p2_5", -math.inf) > 0
        for report in reference_reports.values()
    )
    return DECISION_CONTINUE if passes else DECISION_PARK


def _segment_reports(
    outcomes: Sequence[ResearchOutcome], profile: CostProfileV2
) -> dict[str, Any]:
    selectors: dict[str, Any] = {"ALL": lambda _: True}
    for family in (item.value for item in ReferenceFamily):
        selectors[family] = (
            lambda x, family=family: x.metadata["reference_family"] == family
        )
    for side in (item.value for item in ReferenceSide):
        selectors[side.upper() + "_sweep"] = (
            lambda x, side=side: x.metadata["reference_side"] == side
        )
    for instrument in INSTRUMENTS:
        selectors[f"instrument/{instrument}"] = (
            lambda x, instrument=instrument: x.instrument == instrument
        )
    for session in (window.name for window in DEFAULT_SESSION_SPEC.major_sessions):
        selectors[f"session/{session}"] = (
            lambda x, session=session: session in x.metadata["major_sessions"]
        )
    regimes = sorted({str(x.metadata["session_regime"]) for x in outcomes})
    for regime in regimes:
        selectors[f"session_regime/{regime}"] = (
            lambda x, regime=regime: x.metadata["session_regime"] == regime
        )
    reports = {}
    for label, selector in selectors.items():
        selected = [item for item in outcomes if selector(item)]
        if selected:
            report = analyze(
                selected,
                profile,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED,
            )
            # The frozen report requires the bootstrap summary, not all 10,000
            # intermediate draws.  Omitting draws also keeps the JSON compact.
            report.pop("bootstrap_event_weighted_mean_pips")
            reports[label] = report
        else:
            reports[label] = {"n": 0, "status": "NO_COMPLETE_H15_OUTCOMES"}
    return reports


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    registry_path: Path = REGISTRY,
    cost_profile_path: Path = DEFAULT_COST_PROFILE,
) -> dict[str, Any]:
    """Authenticate, run, and atomically publish the frozen 2024 study."""
    if registry_path != REGISTRY:
        raise RangeSweepRunnerError(
            "only the explicit authenticated 2024 registry is authorized"
        )
    if output_dir.exists():
        raise RangeSweepRunnerError("refusing to overwrite empirical artifacts")
    registry_raw = registry_path.read_bytes()
    registry = load_registry(registry_path)
    entries = registry.get("instruments")
    if (
        not isinstance(entries, dict)
        or set(entries) != set(INSTRUMENTS)
        or len(entries) != 9
    ):
        raise RangeSweepRunnerError(
            "all and only the registered nine FX instruments are required"
        )
    profile_raw = cost_profile_path.read_bytes()
    profile = CostProfileV2.load(cost_profile_path)
    all_records: list[dict[str, Any]] = []
    aggregation: dict[str, Any] = {}
    corpus_identities: dict[str, Any] = {}
    authenticated: dict[str, Path] = {}
    # Authenticate the complete allowlist before loading even the first corpus.
    # This prevents a partially authorized empirical run if a later manifest is
    # absent or altered.
    for instrument in INSTRUMENTS:
        entry = entries[instrument]
        if not isinstance(entry, dict):
            raise RangeSweepRunnerError("invalid registry instrument entry")
        authenticated[instrument] = authenticate_corpus(entry)
    for instrument in INSTRUMENTS:
        entry = entries[instrument]
        assert isinstance(entry, dict)  # Established in authentication pass.
        dataset = load_offline_corpus(authenticated[instrument])
        bars = tuple(dataset.bars)
        if any(
            bar.instrument != instrument or bar.open_time.year != 2024 for bar in bars
        ):
            raise RangeSweepRunnerError(
                "corpus contains wrong instrument or non-2024 data"
            )
        m5, audit = aggregate_m1_to_m5(bars)
        pip_size = FX_INSTRUMENTS[instrument].pip_size
        events = range_sweep_events(m5, RangeSweepConfig({instrument: pip_size}))
        all_records.extend(execute_event(event, bars) for event in events)
        aggregation[instrument] = asdict(audit)
        corpus_identities[instrument] = {
            "corpus_id": entry["corpus_id"],
            "assembled_dataset_id": entry["assembled_dataset_id"],
            "manifest_sha256": entry["manifest_sha256"],
        }
    all_records.sort(key=lambda x: (x["timestamp"], x["instrument"], x["event_id"]))
    outcomes = [
        to_research_outcome(row, PRIMARY_HORIZON)
        for row in all_records
        if row["h15_complete"]
    ]
    segments = _segment_reports(outcomes, profile)
    references = {family.value: segments[family.value] for family in ReferenceFamily}
    summary = {
        "study_id": STUDY_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "decision": discovery_decision(references),
        "primary_horizon_minutes": PRIMARY_HORIZON,
        "secondary_horizons_minutes": [30, 60],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "whole_calendar_month",
        },
        "event_count": len(all_records),
        "complete_h15_count": len(outcomes),
        "reference_family_screen": references,
        "economic_segments": segments,
        "methodology": {
            "signal": STUDY_ID,
            "economic": METHODOLOGY_ID,
            "m5_policy": M5_POLICY,
            "session_spec_id": DEFAULT_SESSION_SPEC.session_spec_id,
        },
        "identities": {
            "registry_id": registry["registry_id"],
            "registry_sha256": _sha_bytes(registry_raw),
            "cost_profile_sha256": _sha_bytes(profile_raw),
            "study_spec_sha256": _sha_bytes(_canonical(STUDY_SPEC)),
            "code_revision": _git_revision(),
            "corpora": corpus_identities,
        },
    }
    audit = {
        "study_id": STUDY_ID,
        "instruments": list(INSTRUMENTS),
        "aggregation": aggregation,
        "execution": {
            "entry_incomplete_reasons": dict(
                Counter(
                    x["entry_incomplete_reason"]
                    for x in all_records
                    if x["entry_incomplete_reason"]
                )
            ),
            "horizon_incomplete_reasons": {
                f"H{h}": dict(
                    Counter(
                        x[f"h{h}_incomplete_reason"]
                        for x in all_records
                        if x[f"h{h}_incomplete_reason"]
                    )
                )
                for h in HORIZONS
            },
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        lines = b"".join(_canonical(record) + b"\n" for record in all_records)
        (temporary / "events.jsonl.gz").write_bytes(gzip.compress(lines, mtime=0))
        (temporary / "summary.json").write_bytes(
            json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n"
        )
        (temporary / "execution-audit.json").write_bytes(
            json.dumps(audit, indent=2, sort_keys=True).encode() + b"\n"
        )
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--cost-profile", type=Path, default=DEFAULT_COST_PROFILE)
    args = parser.parse_args(argv)
    summary = run(
        output_dir=args.output_dir,
        registry_path=args.registry,
        cost_profile_path=args.cost_profile,
    )
    print(summary["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
