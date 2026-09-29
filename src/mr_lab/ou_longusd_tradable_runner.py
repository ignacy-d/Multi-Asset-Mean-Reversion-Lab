"""Frozen runner for the OU-LONGUSD-TRADABLE-2024-v1 follow-up study.

This module deliberately contains the scientifically new execution and portfolio
semantics.  It reuses Stage 4B signal assembly/deduplication and the frozen OU
estimator without changing either historical methodology.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from random import Random
from typing import Any
from zoneinfo import ZoneInfo

from mr_lab.data import Bar, Timeframe
from mr_lab.ornstein_uhlenbeck import (
    OrnsteinUhlenbeckProcessSpec,
    build_candidate_ou_states,
    candidate_process_keys,
)
from mr_lab.pca_stage0_runner import authenticate_corpus, load_registry
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.research import Direction
from mr_lab.sessions import DEFAULT_SESSION_SPEC, classify_timestamp
from mr_lab.stage4b import SignalState, deduplicate_states, pip_size
from mr_lab.stage4b_runner import assemble_signal_states
from mr_lab.stage4c_v2 import AVAILABLE, SLIPPAGES, SPREAD_STATISTICS, CostProfileV2

STUDY_ID = "OU-LONGUSD-TRADABLE-2024-v1"
SCIENTIFIC_STATUS = "2024_FOLLOW_UP_DISCOVERY_NOT_CONFIRMATION"
REGISTRY = Path("configs/fx-universe-2024-registry-v1.json")
DEFAULT_COST_PROFILE = Path("configs/stage4c-ftmo-cost-profile-v2.json")
DEFAULT_OUTPUT = Path("results/ou-longusd-tradable-2024-v1")
CORE_DIRECTIONS: Mapping[str, Direction] = {
    "AUDUSD": Direction.SHORT,
    "EURUSD": Direction.SHORT,
    "GBPUSD": Direction.SHORT,
    "NZDUSD": Direction.SHORT,
    "USDCAD": Direction.LONG,
    "USDCHF": Direction.LONG,
    "USDJPY": Direction.LONG,
}
AUX_DIRECTIONS: Mapping[str, Direction] = {"EURGBP": Direction.SHORT}
STUDY_DIRECTIONS = CORE_DIRECTIONS | AUX_DIRECTIONS
BENCHMARKS = ("vwap", "vwap-canonical-m1")
LOOKBACKS = (20, 40)
TP_FRACTIONS = (0.50, 0.75, 1.00)
STOP_FRACTION = 0.50
OU_SCORE_THRESHOLD = 1.5
OU_HALF_LIFE_CAP = 120.0
OU_WINDOW = 128
BOOTSTRAP_SEED = 20241024
BOOTSTRAP_REPLICATES = 10_000
DECISION_CONTINUE = "CONTINUE_OU_LONGUSD_TRADABLE_V1"
DECISION_PARK = "PARK_OU_LONGUSD_TRADABLE_V1"
DECISION_INCONCLUSIVE = "INCONCLUSIVE_OU_LONGUSD_TRADABLE_V1"

STUDY_SPEC = {
    "study_id": STUDY_ID,
    "scientific_status": SCIENTIFIC_STATUS,
    "core_directions": {k: v.name for k, v in CORE_DIRECTIONS.items()},
    "aux_directions": {k: v.name for k, v in AUX_DIRECTIONS.items()},
    "signal": {
        "timeframe": "15m",
        "session": "london",
        "benchmarks": BENCHMARKS,
        "lookbacks": LOOKBACKS,
        "stage4b_qualification": "frozen",
    },
    "ou": {
        "window_transitions": OU_WINDOW,
        "directional_score": ">1.5",
        "half_life_minutes": "<=120",
    },
    "execution": {
        "entry": "exact-signal-time-m1-open",
        "tp_fractions": TP_FRACTIONS,
        "stop_fraction": STOP_FRACTION,
        "same_bar": "adverse-first",
    },
    "deadlines": {
        "safe_london_end": "16:45 Europe/London",
        "primary_minutes": 180,
        "robustness": "min(180,floor(2/3*remaining-minutes))",
    },
    "portfolio_modes": ("UNCONSTRAINED", "ONE_PER_INSTRUMENT"),
    "bootstrap": {
        "unit": "calendar-month",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
    },
    "continuation_screen": {
        "minimum_trades": 200,
        "minimum_mean_r": 0.10,
        "bootstrap_mean_r_p2_5": ">0",
    },
}
METHODOLOGY_ID = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            STUDY_SPEC, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()
)


class OuTradableError(ValueError):
    """An input violates the frozen study contract."""


@dataclass(frozen=True, slots=True)
class Component:
    event_id: str
    instrument: str
    timestamp: datetime
    direction: Direction
    benchmark_family: str
    lookback: int
    p0: float
    e0: float
    directional_ou_score: float
    half_life_minutes: float
    source_corpus_id: str
    assembled_dataset_id: str | None


@dataclass(frozen=True, slots=True)
class EnsembleSignal:
    signal_id: str
    instrument: str
    timestamp: datetime
    direction: Direction
    scope: str
    p0: float
    ensemble_e0: float
    d0: float
    components: tuple[Component, ...]
    bollinger_diagnostics: Mapping[str, Any] | None = None

    @property
    def model_vote_count(self) -> int:
        return len(self.components)


@dataclass(frozen=True, slots=True)
class ExactM1Index:
    by_open: Mapping[datetime, Bar]
    by_close: Mapping[datetime, Bar]

    @classmethod
    def build(cls, instrument: str, bars: Sequence[Bar]) -> ExactM1Index:
        by_open: dict[datetime, Bar] = {}
        by_close: dict[datetime, Bar] = {}
        for bar in bars:
            reason = _bar_reason(bar, instrument, bar.open_time)
            if reason is not None:
                raise OuTradableError(f"invalid authenticated M1 corpus: {reason}")
            if bar.open_time in by_open or bar.close_time in by_close:
                raise OuTradableError("ambiguous duplicate M1 data")
            by_open[bar.open_time] = bar
            by_close[bar.close_time] = bar
        return cls(by_open, by_close)


def scope_for(instrument: str) -> str:
    if instrument in CORE_DIRECTIONS:
        return "CORE_LONG_USD"
    if instrument in AUX_DIRECTIONS:
        return "AUX_EURGBP_SHORT"
    raise OuTradableError(
        f"instrument is outside the preregistered universe: {instrument}"
    )


def required_direction(instrument: str) -> Direction:
    try:
        return STUDY_DIRECTIONS[instrument]
    except KeyError as exc:
        raise OuTradableError(
            f"instrument is outside the preregistered universe: {instrument}"
        ) from exc


def ou_eligible(score: float | None, half_life: float | None) -> bool:
    return bool(
        score is not None
        and half_life is not None
        and math.isfinite(score)
        and math.isfinite(half_life)
        and score > OU_SCORE_THRESHOLD
        and half_life <= OU_HALF_LIFE_CAP
    )


def ensemble_components(
    components: Sequence[Component], diagnostics=None
) -> tuple[EnsembleSignal, ...]:
    """Union eligible component votes without making vote count a gate."""
    groups: dict[tuple[str, datetime, Direction], list[Component]] = defaultdict(list)
    for component in components:
        if component.direction is not required_direction(component.instrument):
            raise OuTradableError(
                "component direction violates semantic study exposure"
            )
        if not ou_eligible(component.directional_ou_score, component.half_life_minutes):
            raise OuTradableError("ineligible OU component supplied to ensemble")
        groups[(component.instrument, component.timestamp, component.direction)].append(
            component
        )
    result = []
    for key, votes in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][0], item[0][2].name)
    ):
        votes.sort(key=lambda c: (c.benchmark_family, c.lookback, c.event_id))
        if not 1 <= len(votes) <= 4:
            raise OuTradableError("ensemble vote count must be in 1..4")
        if len({(v.benchmark_family, v.lookback) for v in votes}) != len(votes):
            raise OuTradableError("duplicate benchmark/lookback vote")
        if any(v.p0 != votes[0].p0 for v in votes[1:]):
            raise OuTradableError("component p0 values disagree")
        e0 = statistics.median(v.e0 for v in votes)
        raw = json.dumps(
            (
                key[0],
                key[1].isoformat(),
                key[2].name,
                tuple(v.event_id for v in votes),
                e0,
            ),
            separators=(",", ":"),
        )
        result.append(
            EnsembleSignal(
                "ou-longusd-" + hashlib.sha256(raw.encode()).hexdigest()[:24],
                key[0],
                key[1],
                key[2],
                scope_for(key[0]),
                votes[0].p0,
                e0,
                votes[0].p0 - e0,
                tuple(votes),
                diagnostics,
            )
        )
    return tuple(result)


def safe_london_end(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise OuTradableError("timestamp must be timezone-aware UTC")
    london = ZoneInfo("Europe/London")
    local = timestamp.astimezone(london)
    return datetime.combine(local.date(), time(16, 45), london).astimezone(UTC)


def lifecycle_deadline(entry: datetime, lifecycle: str) -> datetime:
    end = safe_london_end(entry)
    remaining_minutes = math.floor((end - entry).total_seconds() / 60)
    if remaining_minutes <= 0:
        raise OuTradableError("entry_at_or_after_safe_london_end")
    if lifecycle == "PRIMARY":
        hold = min(180, remaining_minutes)
    elif lifecycle == "ROBUSTNESS":
        hold = min(180, math.floor(2 * remaining_minutes / 3))
    else:
        raise OuTradableError("unsupported lifecycle")
    if hold <= 0:
        raise OuTradableError("non_positive_remaining_tradable_time")
    return entry + timedelta(minutes=hold)


def _bar_reason(bar: Bar | None, instrument: str, timestamp: datetime) -> str | None:
    if bar is None:
        return "missing_exact_m1_entry"
    if bar.instrument != instrument:
        return "entry_instrument_mismatch"
    if bar.timeframe != Timeframe("1m"):
        return "entry_wrong_timeframe"
    if bar.open_time != timestamp or bar.close_time != timestamp + timedelta(minutes=1):
        return "entry_wrong_exact_duration"
    if any(
        t.tzinfo is None or t.utcoffset() != timedelta(0)
        for t in (bar.open_time, bar.close_time, bar.available_at)
    ):
        return "entry_non_utc"
    if bar.volume == 0 and bar.open == bar.high == bar.low == bar.close:
        return "entry_provider_padding"
    if any(
        not math.isfinite(x) or x <= 0 for x in (bar.open, bar.high, bar.low, bar.close)
    ):
        return "entry_invalid_ohlc"
    return None


def geometry(signal: EnsembleSignal) -> tuple[tuple[float, ...], float]:
    distance = abs(signal.d0)
    if not math.isfinite(distance) or distance <= 0:
        raise OuTradableError("signal displacement must be finite and positive")
    direction = int(signal.direction)
    return (
        tuple(signal.p0 + direction * fraction * distance for fraction in TP_FRACTIONS),
        signal.p0 - direction * STOP_FRACTION * distance,
    )


def execute_signal(
    signal: EnsembleSignal, bars: Sequence[Bar] | ExactM1Index, lifecycle: str
) -> dict[str, Any]:
    """Execute three independent equal tranches with conservative first passage."""
    index = (
        bars
        if isinstance(bars, ExactM1Index)
        else ExactM1Index.build(signal.instrument, bars)
    )
    indexed = index.by_open
    close_index = index.by_close
    entry_bar = indexed.get(signal.timestamp)
    reason = _bar_reason(entry_bar, signal.instrument, signal.timestamp)
    base = {
        "signal_id": signal.signal_id,
        "instrument": signal.instrument,
        "scope": signal.scope,
        "timestamp": signal.timestamp.isoformat(),
        "lifecycle": lifecycle,
        "executed": False,
        "incomplete_reason": reason,
    }
    if reason is not None or entry_bar is None:
        return base
    try:
        deadline = lifecycle_deadline(signal.timestamp, lifecycle)
    except OuTradableError as exc:
        return base | {"incomplete_reason": str(exc)}
    targets, stop = geometry(signal)
    direction = int(signal.direction)
    # The open is the only value known at fill. Crossing either original boundary
    # means the intended trade was already stale; no retroactive award is allowed.
    if (
        direction * (entry_bar.open - targets[0]) >= 0
        or direction * (entry_bar.open - stop) <= 0
    ):
        return base | {"incomplete_reason": "entry_already_past_target_or_stop"}
    risk = abs(entry_bar.open - stop)
    if not math.isfinite(risk) or risk <= 0:
        return base | {"incomplete_reason": "invalid_actual_entry_risk"}
    open_tranches = set(range(3))
    exits: list[dict[str, Any] | None] = [None, None, None]
    mfe = mae = 0.0
    minute_count = int((deadline - signal.timestamp).total_seconds() / 60)
    for minute in range(minute_count):
        ts = signal.timestamp + timedelta(minutes=minute)
        bar = indexed.get(ts)
        invalid = _bar_reason(bar, signal.instrument, ts)
        if invalid is not None:
            return base | {"incomplete_reason": f"invalid_m1_path:{invalid}"}
        assert bar is not None
        favorable = max(
            direction * (bar.high - entry_bar.open),
            direction * (bar.low - entry_bar.open),
        )
        adverse = max(
            -direction * (bar.high - entry_bar.open),
            -direction * (bar.low - entry_bar.open),
        )
        mfe, mae = max(mfe, favorable), max(mae, adverse)
        stop_hit = bar.low <= stop if direction == 1 else bar.high >= stop
        hits = {
            i
            for i in open_tranches
            if (bar.high >= targets[i] if direction == 1 else bar.low <= targets[i])
        }
        # Adverse first: if both boundaries occur in a minute every still-open
        # tranche is stopped, including one whose target also occurred.
        if stop_hit:
            for i in tuple(open_tranches):
                exits[i] = {
                    "tranche": i + 1,
                    "reason": "stop",
                    "timestamp": bar.close_time.isoformat(),
                    "price": stop,
                    "holding_minutes": int(
                        (bar.close_time - signal.timestamp).total_seconds() / 60
                    ),
                }
                open_tranches.remove(i)
        else:
            for i in sorted(hits):
                exits[i] = {
                    "tranche": i + 1,
                    "reason": f"tp{i + 1}",
                    "timestamp": bar.close_time.isoformat(),
                    "price": targets[i],
                    "holding_minutes": int(
                        (bar.close_time - signal.timestamp).total_seconds() / 60
                    ),
                }
                open_tranches.remove(i)
        if not open_tranches:
            break
    if open_tranches:
        deadline_bar = close_index.get(deadline)
        invalid = _bar_reason(
            deadline_bar, signal.instrument, deadline - timedelta(minutes=1)
        )
        if invalid is not None or deadline_bar is None:
            return base | {"incomplete_reason": "missing_exact_deadline_m1_close"}
        for i in tuple(open_tranches):
            exits[i] = {
                "tranche": i + 1,
                "reason": "deadline",
                "timestamp": deadline.isoformat(),
                "price": deadline_bar.close,
                "holding_minutes": int(
                    (deadline - signal.timestamp).total_seconds() / 60
                ),
            }
            open_tranches.remove(i)
    completed = [x for x in exits if x is not None]
    for item in completed:
        item["gross_r"] = direction * (item["price"] - entry_bar.open) / risk
        item["gross_pips"] = (
            direction * (item["price"] - entry_bar.open) / pip_size(signal.instrument)
        )
    exit_timestamp = max(datetime.fromisoformat(x["timestamp"]) for x in completed)
    return base | {
        "executed": True,
        "incomplete_reason": None,
        "direction": signal.direction.name,
        "entry_price": entry_bar.open,
        "entry_timestamp": signal.timestamp.isoformat(),
        "p0": signal.p0,
        "ensemble_e0": signal.ensemble_e0,
        "d0": signal.d0,
        "targets": targets,
        "stop": stop,
        "initial_risk_distance": risk,
        "deadline": deadline.isoformat(),
        "tranches": completed,
        "gross_r": statistics.fmean(x["gross_r"] for x in completed),
        "gross_pips": statistics.fmean(x["gross_pips"] for x in completed),
        "exit_timestamp": exit_timestamp.isoformat(),
        "holding_minutes": int(
            (exit_timestamp - signal.timestamp).total_seconds() / 60
        ),
        "mfe_pips": mfe / pip_size(signal.instrument),
        "mae_pips": mae / pip_size(signal.instrument),
    }


def portfolio(
    trades: Sequence[Mapping[str, Any]], mode: str
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    ordered = sorted(
        (dict(t) for t in trades if t.get("executed")),
        key=lambda t: (t["entry_timestamp"], t["instrument"], t["signal_id"]),
    )
    if mode == "UNCONSTRAINED":
        selected, skipped = ordered, []
    elif mode == "ONE_PER_INSTRUMENT":
        selected, skipped, occupied = [], [], {}
        for trade in ordered:
            start = datetime.fromisoformat(trade["entry_timestamp"])
            if (
                occupied.get(trade["instrument"], datetime.min.replace(tzinfo=UTC))
                > start
            ):
                skipped.append(trade["signal_id"])
                continue
            selected.append(trade)
            occupied[trade["instrument"]] = datetime.fromisoformat(
                trade["exit_timestamp"]
            )
    else:
        raise OuTradableError("unsupported portfolio mode")
    points = sorted(
        {
            datetime.fromisoformat(t[k])
            for t in selected
            for k in ("entry_timestamp", "exit_timestamp")
        }
    )
    occupancy = []
    for point in points:
        count = sum(
            t["scope"] == "CORE_LONG_USD"
            and datetime.fromisoformat(t["entry_timestamp"])
            <= point
            < datetime.fromisoformat(t["exit_timestamp"])
            for t in selected
        )
        occupancy.append({"timestamp": point.isoformat(), "core_positions": count})
    counts = Counter(x["core_positions"] for x in occupancy)
    audit = {
        "skipped_due_to_open_instrument": len(skipped),
        "skipped_signal_ids": skipped,
        "max_concurrent_core_positions": max(counts, default=0),
        "mean_concurrent_core_positions": statistics.fmean(
            x["core_positions"] for x in occupancy
        )
        if occupancy
        else 0,
        "concurrent_core_distribution": dict(sorted(counts.items())),
        "occupancy": occupancy,
    }
    return tuple(selected), audit


def _bootstrap(
    trades: Sequence[Mapping[str, Any]], field: str
) -> dict[str, float | None]:
    by_month: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        by_month[trade["entry_timestamp"][:7]].append(float(trade[field]))
    months = sorted(by_month)
    if not months:
        return {"p2_5": None, "median": None, "p97_5": None}
    random = Random(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [
            value
            for _month in range(len(months))
            for value in by_month[random.choice(months)]
        ]
        values.append(statistics.fmean(sample))
    values.sort()
    return {
        "p2_5": values[int(0.025 * (len(values) - 1))],
        "median": values[len(values) // 2],
        "p97_5": values[int(0.975 * (len(values) - 1))],
    }


def summarize(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(trades)
    n = len(rows)
    if not rows:
        return {
            "executed_trade_count": 0,
            "bootstrap_mean_gross_r": _bootstrap(rows, "gross_r"),
            "bootstrap_mean_gross_pips": _bootstrap(rows, "gross_pips"),
        }
    rs = [float(x["gross_r"]) for x in rows]
    pips = [float(x["gross_pips"]) for x in rows]
    winners = [x for x in rs if x > 0]
    losers = [x for x in rs if x < 0]
    reasons = Counter(t["reason"] for x in rows for t in x["tranches"])
    monthly = defaultdict(list)
    instruments = defaultdict(list)
    for x in rows:
        monthly[x["entry_timestamp"][:7]].append(x)
        instruments[x["instrument"]].append(x)
    pf = sum(winners) / abs(sum(losers)) if losers else (math.inf if winners else None)
    return {
        "executed_trade_count": n,
        "trades_per_calendar_month": {k: len(v) for k, v in sorted(monthly.items())},
        "mean_gross_pips": statistics.fmean(pips),
        "median_gross_pips": statistics.median(pips),
        "mean_gross_r": statistics.fmean(rs),
        "median_gross_r": statistics.median(rs),
        "win_rate": len(winners) / n,
        "profit_factor_r": pf,
        "average_winner_r": statistics.fmean(winners) if winners else None,
        "average_loser_r": statistics.fmean(losers) if losers else None,
        **{f"tp{i}_hit_rate": reasons[f"tp{i}"] / (3 * n) for i in range(1, 4)},
        "stop_hit_rate": reasons["stop"] / (3 * n),
        "deadline_exit_rate": reasons["deadline"] / (3 * n),
        "mean_holding_minutes": statistics.fmean(x["holding_minutes"] for x in rows),
        "median_holding_minutes": statistics.median(x["holding_minutes"] for x in rows),
        "p90_holding_minutes": sorted(x["holding_minutes"] for x in rows)[
            math.ceil(0.9 * n) - 1
        ],
        "mean_mfe_pips": statistics.fmean(x["mfe_pips"] for x in rows),
        "mean_mae_pips": statistics.fmean(x["mae_pips"] for x in rows),
        "instrument_breakdown": {
            k: {
                "n": len(v),
                "mean_r": statistics.fmean(x["gross_r"] for x in v),
                "mean_pips": statistics.fmean(x["gross_pips"] for x in v),
            }
            for k, v in sorted(instruments.items())
        },
        "monthly_breakdown": {
            k: {
                "n": len(v),
                "mean_r": statistics.fmean(x["gross_r"] for x in v),
                "mean_pips": statistics.fmean(x["gross_pips"] for x in v),
            }
            for k, v in sorted(monthly.items())
        },
        "positive_month_fraction": sum(
            statistics.fmean(x["gross_r"] for x in v) > 0 for v in monthly.values()
        )
        / len(monthly),
        "positive_instrument_fraction": sum(
            statistics.fmean(x["gross_r"] for x in v) > 0 for v in instruments.values()
        )
        / len(instruments),
        "bootstrap_mean_gross_r": _bootstrap(rows, "gross_r"),
        "bootstrap_mean_gross_pips": _bootstrap(rows, "gross_pips"),
    }


def decision(summary: Mapping[str, Any], *, inputs_complete: bool = True) -> str:
    if not inputs_complete or summary.get("executed_trade_count", 0) == 0:
        return DECISION_INCONCLUSIVE
    passed = (
        summary["executed_trade_count"] >= 200
        and summary["mean_gross_r"] >= 0.10
        and summary["bootstrap_mean_gross_r"]["p2_5"] > 0
    )
    return DECISION_CONTINUE if passed else DECISION_PARK


def _components(states: Sequence[SignalState]) -> tuple[Component, ...]:
    events = tuple(
        e
        for e in deduplicate_states(states)
        if e.signal.instrument in STUDY_DIRECTIONS
        and e.signal.direction is required_direction(e.signal.instrument)
        and str(e.signal.signal_timeframe) == "15m"
        and e.signal.session == "london"
        and e.signal.benchmark_family in BENCHMARKS
        and e.signal.lookback in LOOKBACKS
    )
    spec = OrnsteinUhlenbeckProcessSpec(OU_WINDOW)
    ou_states = build_candidate_ou_states(states, spec, candidate_process_keys(events))
    by_key = {(s.process_id, s.available_at): s for s in ou_states}
    result = []
    from mr_lab.ornstein_uhlenbeck import ResidualObservation

    for event in events:
        s = event.signal
        obs = ResidualObservation(
            s.instrument,
            s.benchmark_family,
            s.signal_timeframe,
            s.session,
            s.lookback,
            s.strategy_spec_id,
            s.signal_timestamp,
            s.p0,
            s.e0,
        )
        state = by_key.get((obs.process_id, s.signal_timestamp))
        raw = None if state is None else state.ornstein_uhlenbeck_score
        score = None if raw is None else raw if s.direction is Direction.SHORT else -raw
        half_life = None if state is None else state.half_life_minutes
        if (
            state is not None
            and state.status == "valid"
            and ou_eligible(score, half_life)
        ):
            result.append(
                Component(
                    event.candidate_event_id,
                    s.instrument,
                    s.signal_timestamp,
                    s.direction,
                    s.benchmark_family,
                    s.lookback,
                    s.p0,
                    s.e0,
                    score,
                    half_life,
                    s.source_corpus_id,
                    s.assembled_dataset_id,
                )
            )
    return tuple(result)


def _json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    ).encode()


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return gzip.compress(b"".join(_json(x) for x in rows), mtime=0)


def run(
    output_dir: Path = DEFAULT_OUTPUT, cost_profile: Path = DEFAULT_COST_PROFILE
) -> dict[str, Any]:
    """Authenticate the explicit registry, execute, and atomically publish artifacts."""
    if output_dir.exists():
        raise OuTradableError(
            "refusing to overwrite existing empirical output directory"
        )
    registry = load_registry(REGISTRY)
    states = []
    bars_by_instrument = {}
    provenance = {}
    for instrument in STUDY_DIRECTIONS:
        entry = registry["instruments"].get(instrument)
        if entry is None:
            raise OuTradableError(
                f"missing preregistered registry instrument: {instrument}"
            )
        corpus_path = authenticate_corpus(entry)
        dataset = load_offline_corpus(corpus_path)
        bars_by_instrument[instrument] = ExactM1Index.build(
            instrument, tuple(dataset.bars)
        )
        states.extend(assemble_signal_states(dataset, entry))
        provenance[instrument] = {
            k: entry[k]
            for k in ("corpus_id", "assembled_dataset_id", "manifest_sha256")
        }
    components = _components(states)
    signals = ensemble_components(components)
    trades = {
        life: [
            execute_signal(s, bars_by_instrument[s.instrument], life) for s in signals
        ]
        for life in ("PRIMARY", "ROBUSTNESS")
    }
    cost_sha = None
    profile = None
    if cost_profile.exists():
        cost_sha = "sha256:" + hashlib.sha256(cost_profile.read_bytes()).hexdigest()
        profile = CostProfileV2.load(cost_profile)
    for rows in trades.values():
        for row in rows:
            session = (
                None
                if not row.get("executed")
                else (
                    classify_timestamp(
                        datetime.fromisoformat(row["entry_timestamp"]),
                        DEFAULT_SESSION_SPEC,
                    ).active_sessions
                )
            )
            session_key = session[0] if session and len(session) == 1 else "overall"
            row["cost_status"] = (
                "BLOCKED_MISSING_COST_PROFILE"
                if profile is None
                else profile.coverage_status(row["instrument"], session_key)
            )
            if (
                row.get("executed")
                and profile is not None
                and row["cost_status"] == AVAILABLE
            ):
                row["net_pips_cost_scenarios"] = {}
                for statistic in SPREAD_STATISTICS:
                    spread, commission, adjustment = profile.costs(
                        row["instrument"], session_key, statistic
                    )
                    for slippage in SLIPPAGES:
                        raw_net = row["gross_pips"] - spread - commission - slippage
                        row["net_pips_cost_scenarios"][
                            f"{statistic}/slippage_{slippage:g}"
                        ] = raw_net * (
                            1 - adjustment if raw_net > 0 else 1 + adjustment
                        )
    reports = {}
    audits = {}
    for life, rows in trades.items():
        for mode in ("UNCONSTRAINED", "ONE_PER_INSTRUMENT"):
            selected, audit = portfolio(rows, mode)
            audits[f"{life}/{mode}"] = audit
            for scope in ("CORE_LONG_USD", "AUX_EURGBP_SHORT", "CORE_PLUS_EURGBP"):
                scope_signals = (
                    signals
                    if scope == "CORE_PLUS_EURGBP"
                    else tuple(x for x in signals if x.scope == scope)
                )
                scope_rows = (
                    rows
                    if scope == "CORE_PLUS_EURGBP"
                    else tuple(x for x in rows if x["scope"] == scope)
                )
                scoped = (
                    selected
                    if scope == "CORE_PLUS_EURGBP"
                    else tuple(x for x in selected if x["scope"] == scope)
                )
                report = summarize(scoped)
                report.update(
                    {
                        "raw_component_candidate_count": sum(
                            x.model_vote_count for x in scope_signals
                        ),
                        "unique_ensemble_signal_count": len(scope_signals),
                        "incomplete_unexecuted_count": sum(
                            not x.get("executed") for x in scope_rows
                        ),
                        "incomplete_unexecuted_reasons": dict(
                            sorted(
                                Counter(
                                    x["incomplete_reason"]
                                    for x in scope_rows
                                    if not x.get("executed")
                                ).items()
                            )
                        ),
                        "concurrent_exposure": audit,
                    }
                )
                reports[f"{life}/{mode}/{scope}"] = report
    headline = reports["PRIMARY/ONE_PER_INSTRUMENT/CORE_LONG_USD"]
    summary = {
        "study_id": STUDY_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "methodology_id": METHODOLOGY_ID,
        "raw_component_candidate_count": len(components),
        "unique_ensemble_signal_count": len(signals),
        "reports": reports,
        "headline_decision": decision(headline),
        "cost_profile_status": "AVAILABLE" if profile else "MISSING",
        "provenance": {
            "registry_id": registry["registry_id"],
            "registry_sha": "sha256:"
            + hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
            "corpora": provenance,
            "session_spec_id": DEFAULT_SESSION_SPEC.session_spec_id,
            "cost_profile_sha": cost_sha,
            "code_revision": subprocess.run(
                ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
    }
    signal_rows = [
        asdict(s)
        | {
            "timestamp": s.timestamp.isoformat(),
            "direction": s.direction.name,
            "components": [
                asdict(c)
                | {"timestamp": c.timestamp.isoformat(), "direction": c.direction.name}
                for c in s.components
            ],
        }
        for s in signals
    ]
    files = {
        "summary.json": _json(summary),
        "signals.jsonl.gz": _gzip_jsonl(signal_rows),
        "trades-primary.jsonl.gz": _gzip_jsonl(trades["PRIMARY"]),
        "trades-robustness.jsonl.gz": _gzip_jsonl(trades["ROBUSTNESS"]),
        "execution-audit.json": _json(
            {"portfolio_audits": audits, "provenance": summary["provenance"]}
        ),
        "report.md": (
            f"# {STUDY_ID}\n\nStatus: `{SCIENTIFIC_STATUS}`\n\n"
            f"Headline decision: **{summary['headline_decision']}**\n\n"
            f"Signals: {len(signals)}; components: {len(components)}.\n"
        ).encode(),
    }
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        for name, payload in files.items():
            (temp / name).write_bytes(payload)
        hashes = {
            name: "sha256:" + hashlib.sha256((temp / name).read_bytes()).hexdigest()
            for name in files
            if name != "execution-audit.json"
        }
        audit = json.loads((temp / "execution-audit.json").read_text())
        audit["artifact_hashes"] = hashes
        (temp / "execution-audit.json").write_bytes(_json(audit))
        os.replace(temp, output_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-profile", type=Path, default=DEFAULT_COST_PROFILE)
    args = parser.parse_args(argv)
    run(args.output_dir, args.cost_profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
