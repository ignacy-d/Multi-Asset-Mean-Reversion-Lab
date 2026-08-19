"""Stage 4A descriptive event-path diagnostics against a frozen equilibrium.

This module deliberately consumes already-qualified signal observations.  It does
not construct trades, suppress overlapping observations, or make signals from
future data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import pairwise

from mr_lab.data import Bar, Timeframe
from mr_lab.providers.instruments import get_instrument_spec
from mr_lab.research import Direction

STAGE4A_SCHEMA_VERSION = "stage-4a-event-path-v1"
FIXED_HORIZONS_MINUTES = (5, 15, 30, 60, 120)
FIRST_PASSAGE_LEVELS = (0.25, 0.50, 0.75, 1.00)
PRE_SIGNAL_HORIZONS_MINUTES = (5, 15, 30)
PATH_MINUTES = 120


class EventPathError(ValueError):
    """Raised when Stage 4A inputs violate frozen event/path semantics."""


@dataclass(frozen=True, slots=True)
class FrozenSignal:
    """One qualifying Stage 3B observation, frozen before outcomes are examined."""

    instrument: str
    signal_timestamp: datetime
    benchmark_family: str
    signal_timeframe: Timeframe
    session: str | None
    lookback: int
    threshold: float
    direction: Direction
    p0: float
    e0: float
    d0: float
    normalized_deviation: float
    source_corpus_id: str
    assembled_dataset_id: str | None
    strategy_spec_id: str

    def __post_init__(self) -> None:
        get_instrument_spec(self.instrument)
        if (
            self.signal_timestamp.tzinfo is None
            or self.signal_timestamp.utcoffset() != timedelta(0)
        ):
            raise EventPathError("signal_timestamp must be timezone-aware UTC")
        if self.signal_timeframe not in (
            Timeframe("5m"),
            Timeframe("15m"),
            Timeframe("1h"),
        ):
            raise EventPathError("signal_timeframe must be M5, M15, or H1")
        for name in ("benchmark_family", "source_corpus_id", "strategy_spec_id"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise EventPathError(f"{name} must be a non-empty string")
        if type(self.lookback) is not int or self.lookback not in (20, 40):
            raise EventPathError("lookback must be one of the frozen values 20 or 40")
        if self.threshold not in (1.0, 1.5, 2.0, 2.5):
            raise EventPathError("threshold must be one of the frozen Stage 3B values")
        if not isinstance(self.direction, Direction):
            raise EventPathError("direction must be LONG or SHORT")
        if any(
            not math.isfinite(value)
            for value in (self.p0, self.e0, self.d0, self.normalized_deviation)
        ):
            raise EventPathError(
                "signal prices and normalized deviation must be finite"
            )
        if not math.isclose(self.d0, self.p0 - self.e0, rel_tol=1e-12, abs_tol=1e-15):
            raise EventPathError("d0 must equal p0 - e0")
        if self.d0 == 0 or int(self.direction) * self.d0 >= 0:
            raise EventPathError("direction must point toward equilibrium")


@dataclass(frozen=True, slots=True)
class HorizonDiagnostic:
    horizon_minutes: int
    signed_price_return: float
    signed_return_bps: float
    signed_return_pips: float
    reversion_fraction: float


@dataclass(frozen=True, slots=True)
class FirstPassageDiagnostic:
    level: float
    hit: bool
    time_to_hit_minutes: int | None


@dataclass(frozen=True, slots=True)
class PresignalDiagnostic:
    horizon_minutes: int
    raw_price_movement: float | None
    signed_price_movement: float | None
    impulse_share: float | None


@dataclass(frozen=True, slots=True)
class Stage4AEvent:
    """Deterministic event record; path metrics are absent if M1 is incomplete."""

    signal: FrozenSignal
    stage4a_methodology_id: str
    future_path_complete: bool
    missing_future_minutes: tuple[int, ...]
    horizons: tuple[HorizonDiagnostic, ...]
    first_passage: tuple[FirstPassageDiagnostic, ...]
    max_reversion_fraction: float | None
    mae_price: float | None
    mae_pips: float | None
    mae_bps: float | None
    mae_fraction_d0: float | None
    mfe_price: float | None
    mfe_pips: float | None
    mfe_bps: float | None
    mfe_fraction_d0: float | None
    time_to_mae_minutes: int | None
    time_to_mfe_minutes: int | None
    presignal: tuple[PresignalDiagnostic, ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["signal"]["signal_timestamp"] = self.signal.signal_timestamp.isoformat()
        value["signal"]["signal_timeframe"] = str(self.signal.signal_timeframe)
        value["signal"]["direction"] = self.signal.direction.name.lower()
        return value

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )


def _methodology_id() -> str:
    semantics = {
        "first_passage_levels": FIRST_PASSAGE_LEVELS,
        "fixed_horizons_minutes": FIXED_HORIZONS_MINUTES,
        "path_minutes": PATH_MINUTES,
        "presignal_horizons_minutes": PRE_SIGNAL_HORIZONS_MINUTES,
        "schema_version": STAGE4A_SCHEMA_VERSION,
    }
    encoded = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(encoded.encode()).hexdigest()}"


STAGE4A_METHODOLOGY_ID = _methodology_id()


def reversion_fraction(signal: FrozenSignal, price: float) -> float:
    """Return direction-normalized recovery of the frozen displacement."""
    if (
        not isinstance(price, int | float)
        or isinstance(price, bool)
        or not math.isfinite(price)
    ):
        raise EventPathError("path price must be finite")
    return int(signal.direction) * (price - signal.p0) / abs(signal.d0)


def frozen_signal_from_vwap(
    feature,
    *,
    threshold: float,
    benchmark_family: str,
    source_corpus_id: str,
    assembled_dataset_id: str | None,
    strategy_spec_id: str,
) -> FrozenSignal | None:
    """Adapt one native or canonical-M1 frozen VWAP feature without recomputing it."""
    from mr_lab.vwap_benchmark import signal_direction

    direction = signal_direction(feature, threshold)
    if direction is None:
        return None
    assert feature.vwap is not None and feature.vwap_deviation_z is not None
    observation = feature.observation
    return FrozenSignal(
        observation.bar.instrument,
        observation.available_at,
        benchmark_family,
        observation.bar.timeframe,
        feature.anchor_session,
        feature.volatility_lookback,
        threshold,
        direction,
        feature.price,
        feature.vwap,
        feature.price - feature.vwap,
        feature.vwap_deviation_z,
        source_corpus_id,
        assembled_dataset_id,
        strategy_spec_id,
    )


def frozen_signal_from_bollinger(
    feature,
    *,
    threshold: float,
    session: str | None,
    source_corpus_id: str,
    assembled_dataset_id: str | None,
    strategy_spec_id: str,
) -> FrozenSignal | None:
    """Adapt one qualifying frozen Bollinger feature without changing eligibility."""
    from mr_lab.bollinger_benchmark import signal_direction

    direction = signal_direction(feature, threshold)
    if direction is None:
        return None
    assert feature.middle is not None and feature.bollinger_z is not None
    observation = feature.observation
    price = observation.bar.close
    return FrozenSignal(
        observation.bar.instrument,
        observation.available_at,
        "bollinger",
        observation.bar.timeframe,
        session,
        feature.rolling_lookback,
        threshold,
        direction,
        price,
        feature.middle,
        price - feature.middle,
        feature.bollinger_z,
        source_corpus_id,
        assembled_dataset_id,
        strategy_spec_id,
    )


def _validate_m1(bars: Iterable[Bar], signal: FrozenSignal) -> dict[datetime, Bar]:
    items = tuple(bars)
    if any(
        bar.instrument != signal.instrument or bar.timeframe != Timeframe("1m")
        for bar in items
    ):
        raise EventPathError(
            "path must contain canonical M1 bars for the signal instrument"
        )
    if any(left.available_at >= right.available_at for left, right in pairwise(items)):
        raise EventPathError("M1 bars must have unique increasing availability times")
    return {bar.available_at: bar for bar in items}


def diagnose_event(signal: FrozenSignal, m1_bars: Iterable[Bar]) -> Stage4AEvent:
    """Measure one frozen signal using exact causal M1 observations around T."""
    by_time = _validate_m1(m1_bars, signal)
    pip_size = 10 ** -(get_instrument_spec(signal.instrument).price_precision - 1)
    future = {
        minute: by_time.get(signal.signal_timestamp + timedelta(minutes=minute))
        for minute in range(1, PATH_MINUTES + 1)
    }
    missing = tuple(minute for minute, bar in future.items() if bar is None)
    complete = not missing

    horizons = []
    for minute in FIXED_HORIZONS_MINUTES:
        bar = future[minute]
        if bar is None:
            continue
        movement = int(signal.direction) * (bar.close - signal.p0)
        horizons.append(
            HorizonDiagnostic(
                minute,
                movement,
                movement / signal.p0 * 10_000,
                movement / pip_size,
                movement / abs(signal.d0),
            )
        )

    first_passage: tuple[FirstPassageDiagnostic, ...] = ()
    max_reversion = mae = mfe = None
    mae_time = mfe_time = None
    if complete:
        path = tuple(
            (minute, reversion_fraction(signal, future[minute].close))
            for minute in future
        )  # type: ignore[union-attr]
        first_passage = tuple(
            FirstPassageDiagnostic(
                level,
                any(value >= level for _, value in path),
                next((minute for minute, value in path if value >= level), None),
            )
            for level in FIRST_PASSAGE_LEVELS
        )
        max_reversion = max(value for _, value in path)
        signed_moves = tuple((minute, value * abs(signal.d0)) for minute, value in path)
        mfe = max(0.0, max(value for _, value in signed_moves))
        mae = max(0.0, max(-value for _, value in signed_moves))
        mfe_time = next((minute for minute, value in signed_moves if value == mfe), 0)
        mae_time = next((minute for minute, value in signed_moves if -value == mae), 0)

    presignal = []
    for minute in PRE_SIGNAL_HORIZONS_MINUTES:
        prior = by_time.get(signal.signal_timestamp - timedelta(minutes=minute))
        raw = signal.p0 - prior.close if prior else None
        presignal.append(
            PresignalDiagnostic(
                minute,
                raw,
                int(signal.direction) * raw if raw is not None else None,
                abs(raw) / abs(signal.d0) if raw is not None else None,
            )
        )

    def scale(value: float | None, divisor: float) -> float | None:
        return value / divisor if value is not None else None

    return Stage4AEvent(
        signal,
        STAGE4A_METHODOLOGY_ID,
        complete,
        missing,
        tuple(horizons),
        first_passage,
        max_reversion,
        mae,
        scale(mae, pip_size),
        scale(mae, signal.p0) * 10_000 if mae is not None else None,
        scale(mae, abs(signal.d0)),
        mfe,
        scale(mfe, pip_size),
        scale(mfe, signal.p0) * 10_000 if mfe is not None else None,
        scale(mfe, abs(signal.d0)),
        mae_time,
        mfe_time,
        tuple(presignal),
    )


def diagnose_events(
    signals: Iterable[FrozenSignal], m1_bars: Iterable[Bar]
) -> tuple[Stage4AEvent, ...]:
    """Diagnose observations in stable methodology-identity/timestamp order."""
    bars = tuple(m1_bars)
    ordered = sorted(
        signals,
        key=lambda item: (
            item.instrument,
            item.signal_timestamp,
            item.benchmark_family,
            item.signal_timeframe.value,
            item.session or "",
            item.lookback,
            item.threshold,
            item.strategy_spec_id,
        ),
    )
    return tuple(diagnose_event(signal, bars) for signal in ordered)


def events_to_jsonl(events: Iterable[Stage4AEvent]) -> str:
    """Serialize in deterministic event order without runtime metadata."""
    return "".join(f"{event.to_json()}\n" for event in events)


__all__ = [
    "FIRST_PASSAGE_LEVELS",
    "FIXED_HORIZONS_MINUTES",
    "STAGE4A_METHODOLOGY_ID",
    "STAGE4A_SCHEMA_VERSION",
    "FrozenSignal",
    "HorizonDiagnostic",
    "PresignalDiagnostic",
    "Stage4AEvent",
    "diagnose_event",
    "diagnose_events",
    "events_to_jsonl",
    "frozen_signal_from_bollinger",
    "frozen_signal_from_vwap",
    "reversion_fraction",
]
