"""Fail-closed 2024 Stage0 orchestration for RV-PCA-SHOCK-1.

Only the explicit authenticated nine-instrument registry is accepted.  This
module never searches for corpora and never repairs absent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from math import isfinite, log
from pathlib import Path

import numpy as np

from mr_lab.data import Bar
from mr_lab.pca_residual import (
    PCAResidualConfig,
    PCAResidualEngine,
    SynchronizedClosePanel,
)
from mr_lab.providers.dukascopy_range import load_offline_corpus

REGISTRY = Path("configs/fx-universe-2024-registry-v1.json")
INSTRUMENTS = (
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "USDJPY",
    "AUDJPY",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURGBP",
)
REGISTRY_SCHEMA = "fx-universe-2024-registry-v1"
ACTIVITY_POLICY = "zero-activity-and-flat-ohlc-padding-v1"
PROCESS_ID = "RV-PCA-SHOCK-1/2024-Stage0;W=1024;ddof=1;K=2;RW=256;shock=2;rearm=1"
BLOCKER = "BLOCKED_MISSING_FROZEN_ATR_NORMALIZATION_DEFINITION"
HORIZONS = (5, 15, 30, 60)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20240918


class Stage0Error(ValueError):
    """Raised before unsafe or incomplete research input can be used."""


@dataclass(frozen=True, slots=True)
class PanelBuild:
    panel: SynchronizedClosePanel
    counts: Mapping[str, Mapping[str, int]]
    identity: str
    bars: Mapping[str, Mapping[datetime, Bar]]


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def load_registry(path: Path = REGISTRY) -> dict[str, object]:
    """Read exactly the canonical registry path; no fallback or discovery."""
    if path != REGISTRY:
        raise Stage0Error("only the explicit 2024 registry is authorized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0Error("invalid or missing explicit 2024 registry") from exc
    entries = value.get("instruments") if isinstance(value, dict) else None
    if (
        value.get("registry_schema_version") != REGISTRY_SCHEMA
        or value.get("requested_start_date") != "2024-01-01"
        or value.get("requested_end_date") != "2024-12-31"
        or not isinstance(entries, dict)
        or set(entries) != set(INSTRUMENTS)
    ):
        raise Stage0Error("registry contract mismatch")
    for name in INSTRUMENTS:
        entry = entries[name]
        if (
            not isinstance(entry, dict)
            or entry.get("instrument") != name
            or entry.get("verification_status") != "verified"
        ):
            raise Stage0Error(f"malformed registry entry: {name}")
        if any(
            entry.get(k) in (None, "")
            for k in (
                "corpus_path",
                "corpus_id",
                "assembled_dataset_id",
                "manifest_sha256",
            )
        ):
            raise Stage0Error(f"incomplete corpus identity: {name}")
        if (entry.get("requested_start_date"), entry.get("requested_end_date")) != (
            "2024-01-01",
            "2024-12-31",
        ):
            raise Stage0Error("only the exact 2024 range is authorized")
    return value


def authenticate_corpus(entry: Mapping[str, object]) -> Path:
    """Authenticate the one manifest named by an explicit registry entry."""
    path = Path(str(entry["corpus_path"]))
    manifest_path = path / "corpus-manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0Error("invalid or missing declared corpus manifest") from exc
    if "sha256:" + hashlib.sha256(raw).hexdigest() != entry["manifest_sha256"]:
        raise Stage0Error("corpus manifest hash mismatch")
    for field in (
        "instrument",
        "assembled_dataset_id",
        "corpus_id",
        "requested_start_date",
        "requested_end_date",
    ):
        if manifest.get(field) != entry.get(field):
            raise Stage0Error(f"registry/corpus identity mismatch: {field}")
    return path


def is_padding(bar: Bar) -> bool:
    return bar.volume == 0 and bar.open == bar.high == bar.low == bar.close


def build_panel(
    series: Mapping[str, Sequence[Bar]], dataset_ids: Mapping[str, str]
) -> PanelBuild:
    """Create the exact nine-way intersection of valid, contiguous M1 returns."""
    if set(series) != set(INSTRUMENTS) or set(dataset_ids) != set(INSTRUMENTS):
        raise Stage0Error("all and only nine instruments are required")
    valid: dict[str, dict[datetime, tuple[float, Bar]]] = {}
    indexed: dict[str, dict[datetime, Bar]] = {}
    counts: dict[str, dict[str, int]] = {}
    for name in INSTRUMENTS:
        bars = tuple(series[name])
        indexed[name] = {b.close_time: b for b in bars}
        if len(indexed[name]) != len(bars) or any(b.instrument != name for b in bars):
            raise Stage0Error(f"invalid bar identity/order: {name}")
        padding = sum(is_padding(b) for b in bars)
        returns: dict[datetime, tuple[float, Bar]] = {}
        for previous, current in pairwise(bars):
            if (
                not is_padding(previous)
                and not is_padding(current)
                and current.close_time - previous.close_time == timedelta(minutes=1)
                and all(isfinite(x) and x > 0 for x in (previous.close, current.close))
            ):
                returns[current.close_time] = (
                    log(current.close / previous.close),
                    current,
                )
        valid[name] = returns
        counts[name] = {
            "bars": len(bars),
            "padding": padding,
            "invalid_or_gap_returns": max(0, len(bars) - 1 - len(returns)),
            "valid_returns": len(returns),
        }
    common = sorted(set.intersection(*(set(valid[n]) for n in INSTRUMENTS)))
    if not common:
        raise Stage0Error("no complete nine-pair synchronized rows")
    # Synthetic closes encode already validated returns; the causal engine then
    # operates unchanged and timestamps are the true synchronized row stamps.
    returns = np.array([[valid[n][t][0] for n in INSTRUMENTS] for t in common])
    closes = np.vstack((np.ones(len(INSTRUMENTS)), np.exp(np.cumsum(returns, axis=0))))
    first = common[0] - timedelta(minutes=1)
    panel = SynchronizedClosePanel((first, *common), INSTRUMENTS, closes)
    identity = _sha(
        {
            "datasets": dict(dataset_ids),
            "timestamps": [x.isoformat() for x in common],
            "policy": ACTIVITY_POLICY,
        }
    )
    return PanelBuild(panel, counts, identity, indexed)


def event_id(timestamp: datetime, instrument: str, panel_identity: str) -> str:
    return _sha(
        {
            "process": PROCESS_ID,
            "panel": panel_identity,
            "timestamp": timestamp.isoformat(),
            "instrument": instrument,
        }
    )


def event_outcomes(built: PanelBuild) -> list[dict[str, object]]:
    observations = PCAResidualEngine(PCAResidualConfig()).run(built.panel)
    events = []
    for row in observations:
        if not row.event_emitted:
            continue
        bars = built.bars[row.instrument]
        entry_time = row.timestamp + timedelta(minutes=1)
        entry = bars.get(entry_time)
        exits = {h: bars.get(row.timestamp + timedelta(minutes=h)) for h in HORIZONS}
        if entry is None or any(x is None for x in exits.values()):
            continue
        assert all(x is not None for x in exits.values())
        record: dict[str, object] = {
            "event_id": event_id(row.timestamp, row.instrument, built.identity),
            "timestamp": row.timestamp.isoformat(),
            "instrument": row.instrument,
            "residual": row.residual,
            "residual_z": row.residual_zscore,
            "shock_sign": int(row.residual_shock_sign),
            "fade_direction": int(row.fade_direction),
            "entry_timestamp": entry_time.isoformat(),
            "entry_price": entry.open,
            "process_identity": PROCESS_ID,
            "panel_identity": built.identity,
            "pca_window": 1024,
            "residual_window": 256,
            "activity_policy": ACTIVITY_POLICY,
        }
        for h, bar in exits.items():
            assert bar is not None
            signed = int(row.fade_direction) * log(bar.close / entry.open)
            record.update(
                {
                    f"h{h}_exit_timestamp": bar.close_time.isoformat(),
                    f"h{h}_exit_price": bar.close,
                    f"h{h}_signed_raw_return": signed,
                    f"h{h}_signed_bps_return": signed * 10_000,
                }
            )
        events.append(record)
    return events


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage0Error(f"{field} must be numeric")
    return float(value)


def month_block_bootstrap(
    events: Sequence[Mapping[str, object]],
    metric: str = "h15_signed_bps_return",
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    groups: dict[str, list[float]] = {}
    for event in events:
        month = str(event["timestamp"])[:7]
        groups.setdefault(month, []).append(_number(event[metric], metric))
    if not groups:
        raise Stage0Error("bootstrap requires events")
    blocks = tuple(groups.values())
    rng = np.random.default_rng(seed)
    result = np.empty(replicates)
    for i in range(replicates):
        sample = [blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]
        result[i] = float(np.mean([x for block in sample for x in block]))
    return result


def concentration(events: Sequence[Mapping[str, object]]) -> tuple[float, float, int]:
    if not events:
        return 0.0, 0.0, 0
    instruments = Counter(str(x["instrument"]) for x in events)
    quarters = Counter(
        str(x["timestamp"])[:4]
        + "Q"
        + str((int(str(x["timestamp"])[5:7]) - 1) // 3 + 1)
        for x in events
    )
    positive = sum(
        np.mean(
            [
                _number(x["h15_signed_bps_return"], "h15_signed_bps_return")
                for x in events
                if x["instrument"] == name
            ]
        )
        > 0
        for name in instruments
    )
    return (
        max(instruments.values()) / len(events),
        max(quarters.values()) / len(events),
        positive,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    registry = load_registry()
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    datasets = {
        n: load_offline_corpus(authenticate_corpus(entries[n])) for n in INSTRUMENTS
    }
    built = build_panel(
        {n: datasets[n].bars for n in INSTRUMENTS},
        {n: str(entries[n]["assembled_dataset_id"]) for n in INSTRUMENTS},
    )
    events = event_outcomes(built)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "events.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in events), encoding="utf-8"
    )
    inst_conc, quarter_conc, positive = concentration(events)
    summary = {
        "event_count": len(events),
        "activity_counts": built.counts,
        "process_identity": PROCESS_ID,
        "panel_identity": built.identity,
        "activity_policy": ACTIVITY_POLICY,
        "primary_decision": BLOCKER,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "max_instrument_concentration": inst_conc,
        "max_quarter_concentration": quarter_conc,
        "positive_instruments": positive,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(BLOCKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
