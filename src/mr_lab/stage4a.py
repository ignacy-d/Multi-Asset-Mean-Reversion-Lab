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
FROZEN_EQUILIBRIUM_RULE = "e0_fixed_at_signal_time-v1"
REVERSION_FRACTION_DEFINITION = (
    "direction_times_price_minus_p0_divided_by_absolute_d0-v1"
)
FIXED_HORIZON_PRICE_RULE = "exact_clock_canonical_m1_close-v1"
HORIZON_DIAGNOSTIC_DEFINITION = (
    "directional_price_movement_arithmetic_return_bps_pips_and_reversion-v1"
)
PIP_SIZE_RULE = "verified_instrument_precision_minus_one_decimal_place-v1"
PATH_EXTREME_RULE = "long_high_low_short_low_high-v1"
FIRST_PASSAGE_DEFINITION = (
    "first_m1_minute_favorable_extreme_reversion_at_or_above_level-v1"
)
MAE_DEFINITION = "maximum_nonnegative_adverse_intraminute_price_excursion-v1"
MFE_DEFINITION = "maximum_nonnegative_favorable_intraminute_price_excursion-v1"
PATH_TIME_RESOLUTION = "first_qualifying_m1_minute_no_intrabar_order_inference-v1"
MISSING_FUTURE_PATH_POLICY = (
    "retain_available_fixed_snapshots_pathwise_metrics_unavailable-v1"
)
PRESIGNAL_MOVEMENT_DEFINITION = "p0_minus_exact_clock_prior_m1_close-v1"
IMPULSE_SHARE_DEFINITION = "absolute_p0_minus_prior_close_divided_by_absolute_d0-v1"


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
    signed_price_movement: float
    signed_arithmetic_return: float
    signed_return_bps: float
    signed_return_pips: float
    reversion_fraction: float


@dataclass(frozen=True, slots=True)
class ForwardOutcome:
    """Direction-signed exact-clock outcome independent of an equilibrium."""

    horizon_minutes: int
    signed_price_movement: float
    signed_arithmetic_return: float
    signed_return_bps: float
    signed_return_pips: float


@dataclass(frozen=True, slots=True)
class DirectionalPathDiagnostic:
    """Common exact-clock discovery outcomes without equilibrium assumptions."""

    instrument: str
    signal_timestamp: datetime
    direction: Direction
    p0: float
    future_path_complete: bool
    missing_future_minutes: tuple[int, ...]
    horizons: tuple[ForwardOutcome, ...]
    mae_price: float | None
    mae_pips: float | None
    mae_bps: float | None
    mfe_price: float | None
    mfe_pips: float | None
    mfe_bps: float | None
    time_to_mae_minutes: int | None
    time_to_mfe_minutes: int | None


@dataclass(frozen=True, slots=True)
class DirectionalPathRequest:
    instrument: str
    signal_timestamp: datetime
    direction: Direction
    p0: float


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
        "first_passage_definition": FIRST_PASSAGE_DEFINITION,
        "first_passage_levels": FIRST_PASSAGE_LEVELS,
        "fixed_horizon_price_rule": FIXED_HORIZON_PRICE_RULE,
        "fixed_horizons_minutes": FIXED_HORIZONS_MINUTES,
        "frozen_equilibrium_rule": FROZEN_EQUILIBRIUM_RULE,
        "horizon_diagnostic_definition": HORIZON_DIAGNOSTIC_DEFINITION,
        "impulse_share_definition": IMPULSE_SHARE_DEFINITION,
        "mae_definition": MAE_DEFINITION,
        "mfe_definition": MFE_DEFINITION,
        "missing_future_path_policy": MISSING_FUTURE_PATH_POLICY,
        "path_minutes": PATH_MINUTES,
        "path_extreme_rule": PATH_EXTREME_RULE,
        "path_time_resolution": PATH_TIME_RESOLUTION,
        "pip_size_rule": PIP_SIZE_RULE,
        "presignal_movement_definition": PRESIGNAL_MOVEMENT_DEFINITION,
        "presignal_horizons_minutes": PRE_SIGNAL_HORIZONS_MINUTES,
        "reversion_fraction_definition": REVERSION_FRACTION_DEFINITION,
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


def _prepare_m1_index(bars: Iterable[Bar], instrument: str) -> dict[datetime, Bar]:
    """Materialize, validate, and index one canonical M1 corpus exactly once."""
    items = tuple(bars)
    if any(
        bar.instrument != instrument or bar.timeframe != Timeframe("1m")
        for bar in items
    ):
        raise EventPathError(
            "path must contain canonical M1 bars for the signal instrument"
        )
    if any(left.available_at >= right.available_at for left, right in pairwise(items)):
        raise EventPathError("M1 bars must have unique increasing availability times")
    return {bar.available_at: bar for bar in items}


def diagnose_directional_path(
    *,
    instrument: str,
    signal_timestamp: datetime,
    direction: Direction,
    p0: float,
    m1_bars: Iterable[Bar],
    horizons: tuple[int, ...] = (15, 30, 60, 120),
    path_minutes: int = 120,
) -> DirectionalPathDiagnostic:
    """Apply Stage 4A's exact-clock M1 close/extreme semantics generically."""
    request = DirectionalPathRequest(instrument, signal_timestamp, direction, p0)
    return diagnose_directional_paths((request,), m1_bars, horizons, path_minutes)[0]


def diagnose_directional_paths(
    requests: Iterable[DirectionalPathRequest],
    m1_bars: Iterable[Bar],
    horizons: tuple[int, ...] = (15, 30, 60, 120),
    path_minutes: int = 120,
) -> tuple[DirectionalPathDiagnostic, ...]:
    """Diagnose a same-instrument batch while constructing the M1 index once."""
    requests = tuple(requests)
    if not requests:
        return ()
    instrument = requests[0].instrument
    if any(request.instrument != instrument for request in requests):
        raise EventPathError("directional path requests must share an instrument")
    get_instrument_spec(instrument)
    by_time = _prepare_m1_index(m1_bars, instrument)
    return tuple(
        _diagnose_directional_request(request, by_time, horizons, path_minutes)
        for request in requests
    )


def _diagnose_directional_request(request, by_time, horizons, path_minutes):
    instrument = request.instrument
    signal_timestamp = request.signal_timestamp
    direction = request.direction
    p0 = request.p0
    if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() != timedelta(0):
        raise EventPathError("signal_timestamp must be timezone-aware UTC")
    if (
        not isinstance(direction, Direction)
        or isinstance(p0, bool)
        or not isinstance(p0, int | float)
        or not math.isfinite(p0)
        or p0 <= 0
    ):
        raise EventPathError("direction and p0 must be valid")
    if (
        type(path_minutes) is not int
        or path_minutes < 1
        or not horizons
        or any(
            type(value) is not int or not 1 <= value <= path_minutes
            for value in horizons
        )
        or len(set(horizons)) != len(horizons)
    ):
        raise EventPathError("horizons must be unique positive minutes within the path")
    future = tuple(
        (minute, by_time.get(signal_timestamp + timedelta(minutes=minute)))
        for minute in range(1, path_minutes + 1)
    )
    missing = tuple(minute for minute, bar in future if bar is None)
    pip_size = 10 ** -(get_instrument_spec(instrument).price_precision - 1)
    snapshots = []
    for minute in horizons:
        bar = by_time.get(signal_timestamp + timedelta(minutes=minute))
        if bar is None:
            continue
        movement = int(direction) * (bar.close - p0)
        arithmetic = int(direction) * (bar.close / p0 - 1.0)
        snapshots.append(
            ForwardOutcome(
                minute, movement, arithmetic, arithmetic * 10_000, movement / pip_size
            )
        )
    mae = mfe = None
    mae_time = mfe_time = None
    if not missing:
        favorable = tuple(
            (
                minute,
                max(
                    0.0,
                    int(direction) * (bar.high - p0),
                    int(direction) * (bar.low - p0),
                ),
            )
            for minute, bar in future
            if bar is not None
        )
        adverse = tuple(
            (
                minute,
                max(
                    0.0,
                    -int(direction) * (bar.high - p0),
                    -int(direction) * (bar.low - p0),
                ),
            )
            for minute, bar in future
            if bar is not None
        )
        mfe = max(value for _, value in favorable)
        mae = max(value for _, value in adverse)
        mfe_time = next(minute for minute, value in favorable if value == mfe)
        mae_time = next(minute for minute, value in adverse if value == mae)
    return DirectionalPathDiagnostic(
        instrument,
        signal_timestamp,
        direction,
        p0,
        not missing,
        missing,
        tuple(snapshots),
        mae,
        None if mae is None else mae / pip_size,
        None if mae is None else mae / p0 * 10_000,
        mfe,
        None if mfe is None else mfe / pip_size,
        None if mfe is None else mfe / p0 * 10_000,
        mae_time,
        mfe_time,
    )


def _diagnose_event_from_index(
    signal: FrozenSignal, by_time: dict[datetime, Bar]
) -> Stage4AEvent:
    """Measure one frozen signal from an already validated canonical M1 index."""
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
        arithmetic_return = int(signal.direction) * (bar.close / signal.p0 - 1.0)
        horizons.append(
            HorizonDiagnostic(
                minute,
                movement,
                arithmetic_return,
                arithmetic_return * 10_000,
                movement / pip_size,
                movement / abs(signal.d0),
            )
        )

    first_passage: tuple[FirstPassageDiagnostic, ...] = ()
    max_reversion = mae = mfe = None
    mae_time = mfe_time = None
    if complete:
        extremes = tuple(
            (
                minute,
                bar.high if signal.direction is Direction.LONG else bar.low,
                bar.low if signal.direction is Direction.LONG else bar.high,
            )
            for minute, bar in future.items()
            if bar is not None
        )
        favorable_path = tuple(
            (minute, reversion_fraction(signal, favorable))
            for minute, favorable, _ in extremes
        )
        first_passage = tuple(
            FirstPassageDiagnostic(
                level,
                any(value >= level for _, value in favorable_path),
                next(
                    (minute for minute, value in favorable_path if value >= level),
                    None,
                ),
            )
            for level in FIRST_PASSAGE_LEVELS
        )
        max_reversion = max(value for _, value in favorable_path)
        favorable_moves = tuple(
            (minute, int(signal.direction) * (price - signal.p0))
            for minute, price, _ in extremes
        )
        adverse_moves = tuple(
            (minute, -int(signal.direction) * (price - signal.p0))
            for minute, _, price in extremes
        )
        mfe = max(0.0, max(value for _, value in favorable_moves))
        mae = max(0.0, max(value for _, value in adverse_moves))
        mfe_time = next(
            (minute for minute, value in favorable_moves if value == mfe), 0
        )
        mae_time = next((minute for minute, value in adverse_moves if value == mae), 0)

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
        mae / signal.p0 * 10_000 if mae is not None else None,
        scale(mae, abs(signal.d0)),
        mfe,
        scale(mfe, pip_size),
        mfe / signal.p0 * 10_000 if mfe is not None else None,
        scale(mfe, abs(signal.d0)),
        mae_time,
        mfe_time,
        tuple(presignal),
    )


def diagnose_event(signal: FrozenSignal, m1_bars: Iterable[Bar]) -> Stage4AEvent:
    """Measure one frozen signal using exact causal M1 observations around T."""
    by_time = _prepare_m1_index(m1_bars, signal.instrument)
    return _diagnose_event_from_index(signal, by_time)


def diagnose_events(
    signals: Iterable[FrozenSignal], m1_bars: Iterable[Bar]
) -> tuple[Stage4AEvent, ...]:
    """Diagnose observations in stable methodology-identity/timestamp order."""
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
    if not ordered:
        return ()
    instrument = ordered[0].instrument
    if any(signal.instrument != instrument for signal in ordered):
        raise EventPathError("signals must share the canonical M1 corpus instrument")
    by_time = _prepare_m1_index(m1_bars, instrument)
    return tuple(_diagnose_event_from_index(signal, by_time) for signal in ordered)


def events_to_jsonl(events: Iterable[Stage4AEvent]) -> str:
    """Serialize in deterministic event order without runtime metadata."""
    return "".join(f"{event.to_json()}\n" for event in events)


__all__ = [
    "FIRST_PASSAGE_LEVELS",
    "FIXED_HORIZONS_MINUTES",
    "STAGE4A_METHODOLOGY_ID",
    "STAGE4A_SCHEMA_VERSION",
    "DirectionalPathDiagnostic",
    "DirectionalPathRequest",
    "ForwardOutcome",
    "FrozenSignal",
    "HorizonDiagnostic",
    "PresignalDiagnostic",
    "Stage4AEvent",
    "diagnose_directional_path",
    "diagnose_directional_paths",
    "diagnose_event",
    "diagnose_events",
    "events_to_jsonl",
    "frozen_signal_from_bollinger",
    "frozen_signal_from_vwap",
    "reversion_fraction",
]
