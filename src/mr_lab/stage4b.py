"""Frozen Stage 4B gross-BID trade construction.

The functions in this module are deliberately independent of signal generation and
reporting.  They consume point-in-time Stage 4A signals and completed canonical M1
bars, making the engine reusable by a later eligibility filter without changing
trade semantics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Protocol

from mr_lab.data import Bar
from mr_lab.providers.instruments import get_instrument_spec
from mr_lab.stage4a import FrozenSignal

SIGNAL_THRESHOLD = 2.0
ENTRY_MODES = ("immediate", "m1-reclaim-p0", "extension25-then-reclaim-p0")
TP_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SL_FRACTIONS = (0.25, 0.50, 1.00, None)
TIME_STOPS_MINUTES = (30, 60, 120)
DIAGNOSTIC_HORIZONS = (5, 15, 30)
STAGE4B_REPORT_SCHEMA_VERSION = "stage-4b-report-v1"


class Stage4BError(ValueError):
    """Raised when input would violate the frozen methodology."""


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    eligible: bool = True
    filter_family: str = "none"
    filter_spec_id: str = "none-v1"
    metadata: tuple[tuple[str, str], ...] = ()


class EntryEligibilityFilter(Protocol):
    def evaluate(self, event: CandidateEvent) -> EligibilityDecision: ...


class NoEntryEligibilityFilter:
    """Frozen baseline; future OU/regime work plugs in at this boundary."""

    def evaluate(self, event: CandidateEvent) -> EligibilityDecision:
        return EligibilityDecision()


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    candidate_event_id: str
    signal: FrozenSignal

    def as_dict(self) -> dict[str, object]:
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
    gross_pips_adverse_first: float | None = None
    gross_pips_favorable_first: float | None = None
    mae_price: float | None = None
    mfe_price: float | None = None
    time_to_mae: int | None = None
    time_to_mfe: int | None = None


def _key(signal: FrozenSignal) -> tuple[object, ...]:
    return (
        signal.instrument,
        signal.benchmark_family,
        str(signal.signal_timeframe),
        signal.session,
        signal.direction,
        signal.lookback,
    )


def _event_id(signal: FrozenSignal) -> str:
    identity = (
        *_key(signal),
        signal.signal_timestamp.isoformat(),
        signal.p0,
        signal.e0,
        signal.normalized_deviation,
        SIGNAL_THRESHOLD,
    )
    encoded = json.dumps(identity, default=str, separators=(",", ":"))
    return "stage4b-" + sha256(encoded.encode()).hexdigest()[:24]


def deduplicate_signals(
    signals: list[FrozenSignal] | tuple[FrozenSignal, ...],
) -> tuple[CandidateEvent, ...]:
    """Emit the first qualifying bar of each excursion, rearming below 2.0."""
    ordered = sorted(signals, key=lambda s: (_key(s), s.signal_timestamp))
    armed: dict[tuple[object, ...], bool] = {}
    events: list[CandidateEvent] = []
    for signal in ordered:
        if signal.threshold != SIGNAL_THRESHOLD:
            continue
        key = _key(signal)
        qualifying = abs(signal.normalized_deviation) >= SIGNAL_THRESHOLD
        if not qualifying:
            armed[key] = True
        elif armed.get(key, True):
            events.append(CandidateEvent(_event_id(signal), signal))
            armed[key] = False
    return tuple(
        sorted(events, key=lambda e: (e.signal.signal_timestamp, e.candidate_event_id))
    )


def reversion_fraction(signal: FrozenSignal, price: float) -> float:
    return int(signal.direction) * (price - signal.p0) / abs(signal.d0)


def _extremes(signal: FrozenSignal, bar: Bar) -> tuple[float, float]:
    values = (reversion_fraction(signal, bar.high), reversion_fraction(signal, bar.low))
    return max(values), min(values)


def _future_bars(
    signal: FrozenSignal, bars: tuple[Bar, ...] | list[Bar], minutes: int
) -> list[Bar]:
    deadline = signal.signal_timestamp + timedelta(minutes=minutes)
    return [b for b in bars if signal.signal_timestamp < b.available_at <= deadline]


def construct_entry(
    event: CandidateEvent, bars: tuple[Bar, ...] | list[Bar], mode: str
) -> Entry:
    """Sequentially construct one entry using completed closes only for reclaim."""
    if mode not in ENTRY_MODES:
        raise Stage4BError("unsupported entry mode")
    s = event.signal
    if mode == "immediate":
        return Entry(mode, True, s.signal_timestamp, s.p0, 0, 0.0, 0.0, 0.0)
    path = _future_bars(s, bars, 15 if mode == "m1-reclaim-p0" else 30)
    favorable = adverse = 0.0
    extension_at: datetime | None = None
    extension_wait: int | None = None
    for bar in path:
        fav, adv = _extremes(s, bar)
        favorable, adverse = max(favorable, fav), min(adverse, adv)
        wait = int((bar.available_at - s.signal_timestamp).total_seconds() / 60)
        if mode.startswith("extension") and extension_at is None and adv <= -0.25:
            extension_at, extension_wait = bar.available_at, wait
            # Reclaim must be subsequent; ordering inside this bar is unknowable.
            continue
        reclaim = int(s.direction) * (bar.close - s.p0) > 0
        if reclaim and (mode == "m1-reclaim-p0" or extension_at is not None):
            return Entry(
                mode,
                True,
                bar.available_at,
                bar.close,
                wait,
                reversion_fraction(s, bar.close),
                favorable,
                adverse,
                extension_wait,
                None
                if extension_at is None
                else int((bar.available_at - extension_at).total_seconds() / 60),
            )
    return Entry(
        mode,
        False,
        pre_entry_max_favorable=favorable,
        pre_entry_max_adverse=adverse,
        time_to_extension25=extension_wait,
    )


def path_diagnostic(
    event: CandidateEvent, bars: tuple[Bar, ...] | list[Bar]
) -> tuple[dict[str, object], ...]:
    rows = []
    for horizon in DIAGNOSTIC_HORIZONS:
        hits: dict[str, int | None] = {
            k: None
            for k in ("reversion25", "extension25", "reversion50", "extension50")
        }
        ambiguity: dict[str, bool] = {"25": False, "50": False}
        for bar in _future_bars(event.signal, bars, horizon):
            fav, adv = _extremes(event.signal, bar)
            minute = int(
                (bar.available_at - event.signal.signal_timestamp).total_seconds() / 60
            )
            for level, suffix in ((0.25, "25"), (0.5, "50")):
                rk, ek = "reversion" + suffix, "extension" + suffix
                if hits[rk] is None and fav >= level:
                    hits[rk] = minute
                if hits[ek] is None and adv <= -level:
                    hits[ek] = minute
                ambiguity[suffix] |= fav >= level and adv <= -level
        row: dict[str, object] = {"horizon_minutes": horizon}
        for level in ("25", "50"):
            r, e = hits["reversion" + level], hits["extension" + level]
            order = (
                "none"
                if r is None and e is None
                else "reversion"
                if e is None or (r is not None and r < e)
                else "extension"
                if r is None or e < r
                else "ambiguous_same_minute"
            )
            row.update(
                {
                    f"reversion{level}_hit": r is not None,
                    f"extension{level}_hit": e is not None,
                    f"time_to_reversion{level}": r,
                    f"time_to_extension{level}": e,
                    f"ordering_{level}": order,
                    f"same_minute_ambiguity_{level}": ambiguity[level] and r == e,
                }
            )
        rows.append(row)
    return tuple(rows)


def pip_size(instrument: str) -> float:
    spec = get_instrument_spec(instrument)
    return 10 ** -(spec.price_precision - 1)


def simulate_exit(
    event: CandidateEvent,
    entry: Entry,
    bars: tuple[Bar, ...] | list[Bar],
    *,
    tp_fraction: float,
    sl_fraction: float | None,
    time_stop_minutes: int,
) -> ExitResult:
    """Evaluate joint first exit and path extrema only through actual exit."""
    if not entry.executed or entry.timestamp is None or entry.price is None:
        raise Stage4BError("exit requires an executed entry")
    if entry.r_at_entry is not None and entry.r_at_entry >= tp_fraction:
        raise Stage4BError("target_already_passed_at_entry")
    deadline = entry.timestamp + timedelta(minutes=time_stop_minutes)
    path = [b for b in bars if entry.timestamp < b.available_at <= deadline]
    mae = mfe = 0.0
    tmae = tmfe = 0
    direction = int(event.signal.direction)
    for bar in path:
        fav_r, adv_r = _extremes(event.signal, bar)
        favorable = max(
            0.0,
            direction * (bar.high - entry.price),
            direction * (bar.low - entry.price),
        )
        adverse = max(
            0.0,
            -direction * (bar.high - entry.price),
            -direction * (bar.low - entry.price),
        )
        minute = int((bar.available_at - entry.timestamp).total_seconds() / 60)
        if favorable > mfe:
            mfe, tmfe = favorable, minute
        if adverse > mae:
            mae, tmae = adverse, minute
        tp = fav_r >= tp_fraction
        sl = sl_fraction is not None and adv_r <= -sl_fraction
        if tp or sl:
            tp_price = event.signal.p0 + direction * tp_fraction * abs(event.signal.d0)
            sl_price = (
                None
                if sl_fraction is None
                else event.signal.p0 - direction * sl_fraction * abs(event.signal.d0)
            )
            ambiguous = tp and sl
            adverse_price = sl_price if sl else tp_price
            favorable_price = tp_price
            scale = pip_size(event.signal.instrument)

            return ExitResult(
                True,
                bar.available_at,
                "ambiguous" if ambiguous else ("tp" if tp else "sl"),
                "ambiguous_same_minute" if ambiguous else "unambiguous",
                adverse_price,
                favorable_price,
                direction * (adverse_price - entry.price) / scale,
                direction * (favorable_price - entry.price) / scale,
                mae,
                mfe,
                tmae,
                tmfe,
            )
    exact = next((b for b in path if b.available_at == deadline), None)
    if exact is None:
        return ExitResult(
            False, mae_price=mae, mfe_price=mfe, time_to_mae=tmae, time_to_mfe=tmfe
        )
    pnl = direction * (exact.close - entry.price) / pip_size(event.signal.instrument)
    return ExitResult(
        True,
        deadline,
        "time_stop",
        "unambiguous",
        exact.close,
        exact.close,
        pnl,
        pnl,
        mae,
        mfe,
        tmae,
        tmfe,
    )


def target_already_passed(entry: Entry, tp_fraction: float) -> bool:
    return (
        entry.executed
        and entry.r_at_entry is not None
        and entry.r_at_entry >= tp_fraction
    )


def methodology_id() -> str:
    frozen = {
        "threshold": SIGNAL_THRESHOLD,
        "entry_modes": ENTRY_MODES,
        "tp": TP_FRACTIONS,
        "sl": SL_FRACTIONS,
        "time_stops": TIME_STOPS_MINUTES,
        "schema": STAGE4B_REPORT_SCHEMA_VERSION,
    }
    return "sha256:" + sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


STAGE4B_METHODOLOGY_ID = methodology_id()
