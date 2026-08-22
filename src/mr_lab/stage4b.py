"""Frozen Stage 4B point-in-time signal state and gross-BID trade engine."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Protocol

from mr_lab.data import Bar, Timeframe
from mr_lab.providers.instruments import get_instrument_spec
from mr_lab.research import Direction
from mr_lab.stage4a import FrozenSignal

SIGNAL_THRESHOLD = 2.0
ENTRY_MODES = ("immediate", "m1-reclaim-p0", "extension25-then-reclaim-p0")
TP_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SL_FRACTIONS = (0.25, 0.50, 1.00, None)
TIME_STOPS_MINUTES = (30, 60, 120)
DIAGNOSTIC_HORIZONS = (5, 15, 30)
STAGE4B_REPORT_SCHEMA_VERSION = "stage-4b-report-v1"

SEMANTICS = {
    "threshold_qualification": "existing-stage3b-signal-direction-abs-z-ge-2-v1",
    "dedup_rearm": "per-spec-first-qualifier-rearm-only-valid-abs-z-lt-2-v2",
    "reversion_fraction": "direction-times-price-minus-p0-over-abs-d0-v1",
    "entry_modes": ENTRY_MODES,
    "entry_windows_minutes": {
        "immediate": 0,
        "m1-reclaim-p0": 15,
        "extension25-then-reclaim-p0": 30,
    },
    "extension25": (
        "m1-extreme-r-le-minus-0.25-then-same-or-later-completed-close-reclaim-v2"
    ),
    "diagnostics": {"horizons": DIAGNOSTIC_HORIZONS, "barriers": (0.25, 0.50)},
    "take_profits_original_displacement": TP_FRACTIONS,
    "stop_extensions_original_displacement": SL_FRACTIONS,
    "time_stops_from_entry_exact_m1_close": TIME_STOPS_MINUTES,
    "target_already_passed": "ineligible-when-r-entry-ge-tp-v1",
    "m1_completion": "available-at-close-and-strictly-after-entry-v1",
    "joint_first_exit": "first-m1-tp-sl-or-exact-time-stop-v1",
    "same_minute_ambiguity": "adverse-first-headline-favorable-first-bound-v1",
    "exit_bar_mae_mfe": (
        "certain-prior-bars-plus-exit-price-bound-no-full-exit-bar-extrema-v2"
    ),
    "pip_rule": "verified-instrument-precision-minus-one-v1",
    "baseline_filter": {"family": "none", "spec": "none-v1"},
    "report_schema": STAGE4B_REPORT_SCHEMA_VERSION,
}
STAGE4B_METHODOLOGY_ID = (
    "sha256:" + sha256(json.dumps(SEMANTICS, sort_keys=True).encode()).hexdigest()
)


class Stage4BError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SignalState:
    """One valid completed research observation, qualifying or below threshold."""

    instrument: str
    timestamp: datetime
    benchmark_family: str
    signal_timeframe: Timeframe
    session: str | None
    direction: Direction
    lookback: int
    p0: float
    e0: float
    z: float
    source_corpus_id: str
    assembled_dataset_id: str | None
    strategy_spec_id: str
    qualifying: bool

    def __post_init__(self):
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timedelta(0):
            raise Stage4BError("state timestamp must be UTC")
        if not math.isfinite(self.z):
            raise Stage4BError("state z must be finite")


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    eligible: bool = True
    filter_family: str = "none"
    filter_spec_id: str = "none-v1"
    metadata: tuple[tuple[str, str], ...] = ()


class EntryEligibilityFilter(Protocol):
    def evaluate(self, event: CandidateEvent) -> EligibilityDecision: ...


class NoEntryEligibilityFilter:
    def evaluate(self, event: CandidateEvent) -> EligibilityDecision:
        return EligibilityDecision()


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    candidate_event_id: str
    signal: FrozenSignal

    def as_dict(self):
        value = asdict(self)
        value["signal"]["signal_timestamp"] = self.signal.signal_timestamp.isoformat()
        value["signal"]["signal_timeframe"] = str(self.signal.signal_timeframe)
        value["signal"]["direction"] = self.signal.direction.name
        return value


@dataclass(frozen=True, slots=True)
class Entry:
    mode: str
    executed: bool
    timestamp: datetime | None = None
    price: float | None = None
    wait_minutes: int | None = None
    r_at_entry: float | None = None
    pre_entry_max_favorable: float | None = None
    pre_entry_max_adverse: float | None = None
    time_to_extension25: int | None = None
    time_from_extension25_to_entry: int | None = None


@dataclass(frozen=True, slots=True)
class ExitResult:
    complete: bool
    exit_timestamp: datetime | None = None
    exit_reason: str | None = None
    exit_ordering: str = "unambiguous"
    exit_price_adverse_first: float | None = None
    exit_price_favorable_first: float | None = None
    gross_return_price_adverse_first: float | None = None
    gross_return_price_favorable_first: float | None = None
    gross_return_pips_adverse_first: float | None = None
    gross_return_pips_favorable_first: float | None = None
    gross_return_bp_adverse_first: float | None = None
    gross_return_bp_favorable_first: float | None = None
    gross_return_fraction_d0_adverse_first: float | None = None
    gross_return_fraction_d0_favorable_first: float | None = None
    holding_minutes: int | None = None
    mae_price_certain: float | None = None
    mae_price_upper_bound: float | None = None
    mae_pips_certain: float | None = None
    mae_bp_certain: float | None = None
    mae_fraction_d0_certain: float | None = None
    mfe_price_certain: float | None = None
    mfe_price_upper_bound: float | None = None
    mfe_pips_certain: float | None = None
    mfe_bp_certain: float | None = None
    mfe_fraction_d0_certain: float | None = None
    time_to_mae_certain: int | None = None
    time_to_mfe_certain: int | None = None
    exit_bar_path_ambiguous: bool = False


class M1Index:
    """Validated canonical M1 exact-time index built once per corpus."""

    def __init__(self, bars):
        selected = sorted(
            (b for b in bars if b.timeframe == Timeframe("1m")),
            key=lambda b: b.available_at,
        )
        self._by_time = {b.available_at: b for b in selected}
        if len(self._by_time) != len(selected):
            raise Stage4BError("duplicate M1 availability time")
        self.instrument = selected[0].instrument if selected else None

    def path(self, start: datetime, minutes: int) -> tuple[Bar, ...]:
        return tuple(
            b
            for n in range(1, minutes + 1)
            if (b := self._by_time.get(start + timedelta(minutes=n))) is not None
        )

    def exact(self, timestamp: datetime):
        return self._by_time.get(timestamp)


def _state_key(s):
    return (
        s.instrument,
        s.benchmark_family,
        str(s.signal_timeframe),
        s.session,
        s.direction,
        s.lookback,
    )


def _state_sort_key(s):
    """Total ordering for state processing without changing state identity."""
    return (
        s.instrument,
        s.benchmark_family,
        str(s.signal_timeframe),
        s.session is not None,
        s.session or "",
        s.direction.name,
        s.lookback,
        s.timestamp,
    )


def _event_id(s):
    raw = json.dumps(
        (*_state_key(s), s.timestamp.isoformat(), s.p0, s.e0, s.z, SIGNAL_THRESHOLD),
        default=str,
        separators=(",", ":"),
    )
    return "stage4b-" + sha256(raw.encode()).hexdigest()[:24]


def deduplicate_states(states) -> tuple[CandidateEvent, ...]:
    armed = {}
    events = []
    for state in sorted(states, key=_state_sort_key):
        key = _state_key(state)
        if not state.qualifying:
            if abs(state.z) < SIGNAL_THRESHOLD:
                armed[key] = True
            continue
        if abs(state.z) < SIGNAL_THRESHOLD:
            raise Stage4BError("qualifying state below threshold")
        if armed.get(key, True):
            d0 = state.p0 - state.e0
            signal = FrozenSignal(
                state.instrument,
                state.timestamp,
                state.benchmark_family,
                state.signal_timeframe,
                state.session,
                state.lookback,
                SIGNAL_THRESHOLD,
                state.direction,
                state.p0,
                state.e0,
                d0,
                state.z,
                state.source_corpus_id,
                state.assembled_dataset_id,
                state.strategy_spec_id,
            )
            events.append(CandidateEvent(_event_id(state), signal))
            armed[key] = False
    return tuple(
        sorted(events, key=lambda e: (e.signal.signal_timestamp, e.candidate_event_id))
    )


def reversion_fraction(signal, price):
    return int(signal.direction) * (price - signal.p0) / abs(signal.d0)


def _extremes(signal, bar):
    x = (reversion_fraction(signal, bar.high), reversion_fraction(signal, bar.low))
    return max(x), min(x)


def pip_size(instrument):
    return 10 ** -(get_instrument_spec(instrument).price_precision - 1)


def construct_entry(event, index, mode):
    if mode not in ENTRY_MODES:
        raise Stage4BError("unsupported entry mode")
    s = event.signal
    if mode == "immediate":
        return Entry(mode, True, s.signal_timestamp, s.p0, 0, 0.0, 0.0, 0.0)
    path = index.path(s.signal_timestamp, 15 if mode == "m1-reclaim-p0" else 30)
    favorable = adverse = 0.0
    ext_time = None
    ext_wait = None
    for bar in path:
        fav, adv = _extremes(s, bar)
        favorable = max(favorable, fav)
        adverse = min(adverse, adv)
        wait = int((bar.available_at - s.signal_timestamp).total_seconds() / 60)
        if mode.startswith("extension") and ext_time is None and adv <= -0.25:
            ext_time = bar.available_at
            ext_wait = wait
        reclaim = int(s.direction) * (bar.close - s.p0) > 0
        # The completed close is known after every intrabar extreme in this M1 bar.
        if reclaim and (mode == "m1-reclaim-p0" or ext_time is not None):
            return Entry(
                mode,
                True,
                bar.available_at,
                bar.close,
                wait,
                reversion_fraction(s, bar.close),
                favorable,
                adverse,
                ext_wait,
                None
                if ext_time is None
                else int((bar.available_at - ext_time).total_seconds() / 60),
            )
    return Entry(
        mode,
        False,
        pre_entry_max_favorable=favorable,
        pre_entry_max_adverse=adverse,
        time_to_extension25=ext_wait,
    )


def construct_eligible_entries(event, index, eligibility_filter):
    """Invoke eligibility exactly once before constructing any entry path."""
    decision = eligibility_filter.evaluate(event)
    entries = (
        {mode: construct_entry(event, index, mode) for mode in ENTRY_MODES}
        if decision.eligible
        else {}
    )
    return decision, entries


def path_diagnostic(event, index):
    full = index.path(event.signal.signal_timestamp, 30)
    rows = []
    for horizon in DIAGNOSTIC_HORIZONS:
        hits = {
            k: None
            for k in ("reversion25", "extension25", "reversion50", "extension50")
        }
        for bar in full:
            minute = int(
                (bar.available_at - event.signal.signal_timestamp).total_seconds() / 60
            )
            if minute > horizon:
                break
            fav, adv = _extremes(event.signal, bar)
            for level, suffix in ((0.25, "25"), (0.5, "50")):
                if hits["reversion" + suffix] is None and fav >= level:
                    hits["reversion" + suffix] = minute
                if hits["extension" + suffix] is None and adv <= -level:
                    hits["extension" + suffix] = minute
        row = {"horizon_minutes": horizon}
        for suffix in ("25", "50"):
            r, e = hits["reversion" + suffix], hits["extension" + suffix]
            order = (
                "none"
                if r is None and e is None
                else "reversion"
                if e is None or (r is not None and r < e)
                else "extension"
                if r is None or e < r
                else "ambiguous_same_minute"
            )
            row |= {
                f"reversion{suffix}_hit": r is not None,
                f"extension{suffix}_hit": e is not None,
                f"time_to_reversion{suffix}": r,
                f"time_to_extension{suffix}": e,
                f"ordering_{suffix}": order,
                f"same_minute_ambiguity_{suffix}": order == "ambiguous_same_minute",
            }
        rows.append(row)
    return tuple(rows)


def target_already_passed(entry, tp):
    return bool(
        entry.executed and entry.r_at_entry is not None and entry.r_at_entry >= tp
    )


def simulate_exit(event, entry, path, *, tp_fraction, sl_fraction, time_stop_minutes):
    if not entry.executed or entry.timestamp is None or entry.price is None:
        raise Stage4BError("exit requires executed entry")
    if target_already_passed(entry, tp_fraction):
        raise Stage4BError("target_already_passed_at_entry")
    direction = int(event.signal.direction)
    certain_mae = certain_mfe = 0.0
    tmae = tmfe = 0
    deadline = entry.timestamp + timedelta(minutes=time_stop_minutes)
    for bar in path:
        if bar.available_at > deadline:
            break
        fav_r, adv_r = _extremes(event.signal, bar)
        tp = fav_r >= tp_fraction
        sl = sl_fraction is not None and adv_r <= -sl_fraction
        minute = int((bar.available_at - entry.timestamp).total_seconds() / 60)
        if tp or sl:
            tp_price = event.signal.p0 + direction * tp_fraction * abs(event.signal.d0)
            sl_price = (
                None
                if sl_fraction is None
                else event.signal.p0 - direction * sl_fraction * abs(event.signal.d0)
            )
            ambiguous = tp and sl
            adverse_price = sl_price if sl else tp_price
            favorable_price = tp_price if tp else sl_price
            # Exit-bar full extrema are bounds only; the barrier exit price is certain.
            exit_fav = (
                max(0.0, direction * (tp_price - entry.price))
                if tp and not ambiguous
                else 0.0
            )
            exit_adv = (
                max(0.0, -direction * (sl_price - entry.price))
                if sl and not ambiguous
                else 0.0
            )
            full_fav = max(
                0.0,
                direction * (bar.high - entry.price),
                direction * (bar.low - entry.price),
            )
            full_adv = max(
                0.0,
                -direction * (bar.high - entry.price),
                -direction * (bar.low - entry.price),
            )
            return _result(
                event,
                entry,
                bar.available_at,
                "ambiguous" if ambiguous else "tp" if tp else "sl",
                adverse_price,
                favorable_price,
                max(certain_mae, exit_adv),
                max(certain_mfe, exit_fav),
                max(certain_mae, exit_adv, full_adv),
                max(certain_mfe, exit_fav, full_fav),
                minute if exit_adv > certain_mae else tmae,
                minute if exit_fav > certain_mfe else tmfe,
                True,
                "ambiguous_same_minute" if ambiguous else "unambiguous",
            )
        fav = max(
            0.0,
            direction * (bar.high - entry.price),
            direction * (bar.low - entry.price),
        )
        adv = max(
            0.0,
            -direction * (bar.high - entry.price),
            -direction * (bar.low - entry.price),
        )
        if fav > certain_mfe:
            certain_mfe, tmfe = fav, minute
        if adv > certain_mae:
            certain_mae, tmae = adv, minute
        if bar.available_at == deadline:
            return _result(
                event,
                entry,
                deadline,
                "time_stop",
                bar.close,
                bar.close,
                certain_mae,
                certain_mfe,
                certain_mae,
                certain_mfe,
                tmae,
                tmfe,
                False,
                "unambiguous",
            )
    return ExitResult(
        False,
        mae_price_certain=certain_mae,
        mfe_price_certain=certain_mfe,
        time_to_mae_certain=tmae,
        time_to_mfe_certain=tmfe,
    )


def _result(
    event,
    entry,
    ts,
    reason,
    adverse_price,
    favorable_price,
    mae,
    mfe,
    mae_bound,
    mfe_bound,
    tmae,
    tmfe,
    path_ambiguous,
    ordering,
):
    direction = int(event.signal.direction)
    scale = pip_size(event.signal.instrument)

    def values(price):
        raw = direction * (price - entry.price)
        return raw, raw / scale, raw / entry.price * 10000, raw / abs(event.signal.d0)

    a = values(adverse_price)
    f = values(favorable_price)
    return ExitResult(
        complete=True,
        exit_timestamp=ts,
        exit_reason=reason,
        exit_ordering=ordering,
        exit_price_adverse_first=adverse_price,
        exit_price_favorable_first=favorable_price,
        gross_return_price_adverse_first=a[0],
        gross_return_pips_adverse_first=a[1],
        gross_return_bp_adverse_first=a[2],
        gross_return_fraction_d0_adverse_first=a[3],
        gross_return_price_favorable_first=f[0],
        gross_return_pips_favorable_first=f[1],
        gross_return_bp_favorable_first=f[2],
        gross_return_fraction_d0_favorable_first=f[3],
        holding_minutes=int((ts - entry.timestamp).total_seconds() / 60),
        mae_price_certain=mae,
        mae_price_upper_bound=mae_bound,
        mae_pips_certain=mae / scale,
        mae_bp_certain=mae / entry.price * 10_000,
        mae_fraction_d0_certain=mae / abs(event.signal.d0),
        mfe_price_certain=mfe,
        mfe_price_upper_bound=mfe_bound,
        mfe_pips_certain=mfe / scale,
        mfe_bp_certain=mfe / entry.price * 10_000,
        mfe_fraction_d0_certain=mfe / abs(event.signal.d0),
        time_to_mae_certain=tmae,
        time_to_mfe_certain=tmfe,
        exit_bar_path_ambiguous=path_ambiguous,
    )
