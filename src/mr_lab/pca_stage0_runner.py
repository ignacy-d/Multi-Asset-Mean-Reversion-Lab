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
from datetime import UTC, datetime, timedelta
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
from mr_lab.providers.fx_universe_2024 import FxUniverseError, validate_registry

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
ACTIVITY_POLICY = "zero-activity-and-flat-ohlc-padding-v1"
BLOCKER = "BLOCKED_MISSING_FROZEN_ATR_NORMALIZATION_DEFINITION"
HORIZONS = (5, 15, 30, 60)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20240918
FROZEN_PCA_CONFIG = PCAResidualConfig(
    pca_training_window=1024,
    components=2,
    residual_normalization_window=256,
    shock_threshold=2.0,
    rearm_threshold=1.0,
    variance_epsilon=1e-12,
)


class Stage0Error(ValueError):
    """Raised before unsafe or incomplete research input can be used."""


@dataclass(frozen=True, slots=True)
class PanelBuild:
    panel: SynchronizedClosePanel
    counts: Mapping[str, Mapping[str, int]]
    identity: str
    bars: Mapping[str, Mapping[datetime, Bar]]
    open_bars: Mapping[str, Mapping[datetime, Bar]]


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _config_identity(config: PCAResidualConfig) -> dict[str, object]:
    return {
        "pca_training_window": config.pca_training_window,
        "components": config.components,
        "residual_normalization_window": config.residual_normalization_window,
        "shock_threshold": config.shock_threshold,
        "rearm_threshold": config.rearm_threshold,
        "variance_epsilon": config.variance_epsilon,
    }


STUDY_SPEC = {
    "study_version": "RV-PCA-SHOCK-1/2024-Stage0-v2",
    "pca_config": _config_identity(FROZEN_PCA_CONFIG),
    "standardization": "strictly-prior-window-ddof-1",
    "residual_normalization": "strictly-prior-window-ddof-1",
    "return_definition": "adjacent-active-M1-log-close-return",
    "synchronization": "exact-nine-way-valid-return-timestamp-intersection",
    "activity_policy": ACTIVITY_POLICY,
    "instruments": INSTRUMENTS,
}
PROCESS_ID = _sha(STUDY_SPEC)


def load_registry(path: Path = REGISTRY) -> dict[str, object]:
    """Read exactly the canonical registry path; no fallback or discovery."""
    if path != REGISTRY:
        raise Stage0Error("only the explicit 2024 registry is authorized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0Error("invalid or missing explicit 2024 registry") from exc
    if not isinstance(value, dict):
        raise Stage0Error("registry must be a JSON object")
    try:
        return validate_registry(value)
    except FxUniverseError as exc:
        raise Stage0Error("registry contract mismatch") from exc


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
    series: Mapping[str, Sequence[Bar]],
    dataset_ids: Mapping[str, str],
    *,
    registry_identity: str = "unregistered-test-input",
) -> PanelBuild:
    """Create the exact nine-way intersection of valid, contiguous M1 returns."""
    if set(series) != set(INSTRUMENTS) or set(dataset_ids) != set(INSTRUMENTS):
        raise Stage0Error("all and only nine instruments are required")
    valid: dict[str, dict[datetime, tuple[float, Bar]]] = {}
    indexed: dict[str, dict[datetime, Bar]] = {}
    open_indexed: dict[str, dict[datetime, Bar]] = {}
    counts: dict[str, dict[str, int]] = {}
    for name in INSTRUMENTS:
        bars = tuple(series[name])
        for bar in bars:
            if (
                bar.instrument != name
                or bar.timeframe.value != "1m"
                or bar.open_time.tzinfo is None
                or bar.open_time.utcoffset() != UTC.utcoffset(None)
                or bar.close_time.tzinfo is None
                or bar.close_time.utcoffset() != UTC.utcoffset(None)
                or bar.close_time != bar.open_time + timedelta(minutes=1)
            ):
                raise Stage0Error(f"invalid bar identity/timeframe: {name}")
        if any(right.close_time <= left.close_time for left, right in pairwise(bars)):
            raise Stage0Error(f"bar timestamps must be strictly increasing: {name}")
        indexed[name] = {b.close_time: b for b in bars}
        open_indexed[name] = {b.open_time: b for b in bars}
        if len(indexed[name]) != len(bars) or len(open_indexed[name]) != len(bars):
            raise Stage0Error(f"duplicate bar timestamp: {name}")
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
            "registry": registry_identity,
            "datasets": dict(dataset_ids),
            "policy": ACTIVITY_POLICY,
        }
    )
    return PanelBuild(panel, counts, identity, indexed, open_indexed)


def event_id(timestamp: datetime, instrument: str, panel_identity: str) -> str:
    return _sha(
        {
            "process": PROCESS_ID,
            "panel": panel_identity,
            "timestamp": timestamp.isoformat(),
            "instrument": instrument,
        }
    )


def panel_identity_at(built: PanelBuild, timestamp: datetime | None = None) -> str:
    """Identify the synchronized panel prefix available at ``timestamp``."""
    stamps = built.panel.timestamps[1:]
    if timestamp is not None:
        stamps = tuple(stamp for stamp in stamps if stamp <= timestamp)
    return _sha(
        {
            "source_identity": built.identity,
            "synchronized_timestamps": [stamp.isoformat() for stamp in stamps],
            "activity_policy": ACTIVITY_POLICY,
        }
    )


def event_outcomes(built: PanelBuild) -> list[dict[str, object]]:
    observations = PCAResidualEngine(FROZEN_PCA_CONFIG).run(built.panel)
    events = []
    for row in observations:
        if not row.event_emitted:
            continue
        bars = built.bars[row.instrument]
        open_bars = built.open_bars[row.instrument]
        # The completed return stamped at ``row.timestamp`` is known when the
        # following M1 bar opens.  That bar's open time is numerically equal to
        # the completed bar's close time, so this is Open[t+1] in bar-index
        # semantics rather than same-bar execution.
        entry_time = row.timestamp
        entry = open_bars.get(entry_time)
        exits = {h: bars.get(row.timestamp + timedelta(minutes=h)) for h in HORIZONS}
        entry_reason = _price_incomplete_reason(entry, "open")
        entry_complete = entry_reason is None
        causal_panel_identity = panel_identity_at(built, row.timestamp)
        record: dict[str, object] = {
            "event_id": event_id(row.timestamp, row.instrument, causal_panel_identity),
            "timestamp": row.timestamp.isoformat(),
            "instrument": row.instrument,
            "residual": row.residual,
            "residual_z": row.residual_zscore,
            "shock_sign": int(row.residual_shock_sign),
            "fade_direction": int(row.fade_direction),
            "entry_target_timestamp": entry_time.isoformat(),
            "entry_complete": entry_complete,
            "entry_incomplete_reason": entry_reason,
            "entry_timestamp": entry_time.isoformat() if entry_complete else None,
            "entry_price": entry.open if entry_complete and entry is not None else None,
            "process_identity": PROCESS_ID,
            "panel_identity": causal_panel_identity,
            "pca_config": _config_identity(FROZEN_PCA_CONFIG),
            "activity_policy": ACTIVITY_POLICY,
        }
        for h, bar in exits.items():
            exit_reason = _price_incomplete_reason(bar, "close")
            reason = "entry_incomplete" if not entry_complete else exit_reason
            complete = reason is None
            signed = (
                int(row.fade_direction) * log(bar.close / entry.open)
                if complete and bar is not None and entry is not None
                else None
            )
            record.update(
                {
                    f"h{h}_complete": complete,
                    f"h{h}_incomplete_reason": reason,
                    f"h{h}_exit_timestamp": bar.close_time.isoformat()
                    if complete and bar is not None
                    else None,
                    f"h{h}_exit_price": bar.close
                    if complete and bar is not None
                    else None,
                    f"h{h}_signed_raw_return": signed,
                    f"h{h}_signed_bps_return": signed * 10_000
                    if signed is not None
                    else None,
                }
            )
        events.append(record)
    return events


def _price_incomplete_reason(bar: Bar | None, field: str) -> str | None:
    if bar is None:
        return "missing_bar"
    if is_padding(bar):
        return "provider_padding"
    price = getattr(bar, field)
    if not isfinite(price) or price <= 0:
        return f"invalid_{field}_price"
    return None


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
    complete = [x for x in events if x.get("h15_complete", True)]
    # ``np.mean(...) > 0`` is a numpy.bool_; summing those values produces a
    # numpy.int64, which the standard-library JSON encoder rejects.
    positive = int(
        sum(
            np.mean(
                [
                    _number(x["h15_signed_bps_return"], "h15_signed_bps_return")
                    for x in complete
                    if x["instrument"] == name
                ]
            )
            > 0
            for name in {str(x["instrument"]) for x in complete}
        )
    )
    return (
        float(max(instruments.values()) / len(events)),
        float(max(quarters.values()) / len(events)),
        positive,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact_paths = (
        args.output_dir / "events.jsonl",
        args.output_dir / "summary.json",
    )
    if any(path.exists() for path in artifact_paths):
        raise Stage0Error("refusing to overwrite existing Stage0 empirical artifacts")
    registry = load_registry()
    entries = registry["instruments"]
    assert isinstance(entries, dict)
    datasets = {
        n: load_offline_corpus(authenticate_corpus(entries[n])) for n in INSTRUMENTS
    }
    built = build_panel(
        {n: datasets[n].bars for n in INSTRUMENTS},
        {n: str(entries[n]["assembled_dataset_id"]) for n in INSTRUMENTS},
        registry_identity=str(registry["registry_id"]),
    )
    events = event_outcomes(built)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "events.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in events), encoding="utf-8"
    )
    inst_conc, quarter_conc, positive = concentration(events)
    summary = {
        "emitted_event_count": len(events),
        "executable_entry_count": int(sum(bool(x["entry_complete"]) for x in events)),
        "complete_h15_outcome_count": int(sum(bool(x["h15_complete"]) for x in events)),
        "activity_counts": built.counts,
        "process_identity": PROCESS_ID,
        "panel_identity": panel_identity_at(built),
        "activity_policy": ACTIVITY_POLICY,
        "primary_decision": BLOCKER,
        "bootstrap_seed": int(BOOTSTRAP_SEED),
        "bootstrap_replicates": int(BOOTSTRAP_REPLICATES),
        "max_instrument_concentration": float(inst_conc),
        "max_quarter_concentration": float(quarter_conc),
        "positive_instruments": int(positive),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(BLOCKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
