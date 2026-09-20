"""Point-in-time signal core for the RV-LEADLAG-2024-v1 discovery study.

The frozen primary hypothesis is *laggard catch-up*, not leader fade.  This
module deliberately contains no data acquisition, empirical I/O, execution,
cost, entry, or outcome logic.  Scientific status is
``2024_DISCOVERY_NOT_CONFIRMATION``.

More complex models are follow-up families only if this simple effect survives
empirical screening: VAR/distributed-lag tests directional lag transmission;
ECM/VECM tests adjustment toward a shared long-run equilibrium; a
Kalman/state-space beta allows a time-varying relative-value relationship; and
OU/AR(1) is appropriate only for a demonstrably stationary pair residual, not
automatically for raw pair prices.  None of those models is implemented here.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Literal

from mr_lab.data.models import Bar, DataContractError, Timeframe

STUDY_ID = "RV-LEADLAG-2024-v1"
SCIENTIFIC_STATUS = "2024_DISCOVERY_NOT_CONFIRMATION"
M5 = Timeframe("5m")


@dataclass(frozen=True, slots=True)
class Relationship:
    """One frozen, unordered structural relationship."""

    relationship_id: str
    instrument_a: str
    instrument_b: str


RELATIONSHIPS = (
    Relationship("EURUSD_GBPUSD", "EURUSD", "GBPUSD"),
    Relationship("AUDUSD_NZDUSD", "AUDUSD", "NZDUSD"),
    Relationship("USDJPY_USDCHF", "USDJPY", "USDCHF"),
)


@dataclass(frozen=True, slots=True)
class LeadLagParameters:
    """Frozen v1 signal parameters, exposed explicitly for provenance.

    This public specification is intentionally not a tuning surface. A changed
    value describes a different study and must not emit events carrying the
    ``RV-LEADLAG-2024-v1`` identity.
    """

    lookback_returns: int = 288
    shock_threshold: float = 2.0
    max_laggard_ratio: float = 0.50
    cooldown: timedelta = timedelta(minutes=15)

    def __post_init__(self) -> None:
        expected = (288, 2.0, 0.50, timedelta(minutes=15))
        actual = (
            self.lookback_returns,
            self.shock_threshold,
            self.max_laggard_ratio,
            self.cooldown,
        )
        if actual != expected:
            raise ValueError(
                "RV-LEADLAG-2024-v1 parameters are frozen at lookback_returns=288, "
                "shock_threshold=2.0, max_laggard_ratio=0.50, cooldown=15 minutes"
            )


DEFAULT_PARAMETERS = LeadLagParameters()


@dataclass(frozen=True, slots=True)
class LeadLagEvent:
    """Economic-integration-neutral event emitted by the signal core."""

    study_id: str
    event_id: str
    timestamp: datetime
    relationship_id: str
    leader: str
    laggard: str
    direction: Literal[-1, 1]
    leader_z: float
    laggard_z: float
    leader_abs_z: float
    laggard_abs_z: float
    z_ratio: float
    leader_return: float
    laggard_return: float
    leader_z_bin: str


@dataclass(frozen=True, slots=True)
class _ReturnObservation:
    timestamp: datetime
    returns: Mapping[str, float]


def leader_z_bin(abs_z: float) -> str:
    """Return the frozen diagnostic label for an event leader's absolute z."""
    if not math.isfinite(abs_z) or abs_z < 2.0:
        raise ValueError("leader abs(z) must be finite and at least 2.0")
    if abs_z < 2.5:
        return "2.00 <= |z| < 2.50"
    if abs_z < 3.0:
        return "2.50 <= |z| < 3.00"
    return "|z| >= 3.00"


def _validated_bars(bars: Iterable[Bar]) -> dict[str, dict[datetime, Bar]]:
    instruments = {
        instrument
        for relationship in RELATIONSHIPS
        for instrument in (relationship.instrument_a, relationship.instrument_b)
    }
    by_instrument: dict[str, dict[datetime, Bar]] = {
        instrument: {} for instrument in instruments
    }
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    for bar in bars:
        if not isinstance(bar, Bar):
            raise DataContractError("lead-lag input must contain only canonical Bars")
        if bar.instrument not in instruments:
            raise DataContractError(
                f"instrument is outside the frozen universe: {bar.instrument}"
            )
        if bar.timeframe != M5 or bar.close_time - bar.open_time != M5.duration:
            raise DataContractError("lead-lag input requires complete M5 bars")
        if (bar.open_time - epoch) % M5.duration != timedelta(0):
            raise DataContractError(
                "M5 bars must use non-overlapping epoch-aligned windows"
            )
        indexed = by_instrument[bar.instrument]
        if bar.close_time in indexed:
            raise DataContractError("duplicate instrument observation")
        indexed[bar.close_time] = bar
    return by_instrument


def _synchronized_returns(
    relationship: Relationship, by_instrument: Mapping[str, Mapping[datetime, Bar]]
) -> tuple[_ReturnObservation, ...]:
    a_bars = by_instrument[relationship.instrument_a]
    b_bars = by_instrument[relationship.instrument_b]
    timestamps = sorted(a_bars.keys() & b_bars.keys())
    output: list[_ReturnObservation] = []
    for timestamp in timestamps:
        previous = timestamp - M5.duration
        if previous not in a_bars or previous not in b_bars:
            continue
        current_a, current_b = a_bars[timestamp], b_bars[timestamp]
        previous_a, previous_b = a_bars[previous], b_bars[previous]
        if (
            max(
                current_a.available_at,
                current_b.available_at,
                previous_a.available_at,
                previous_b.available_at,
            )
            > timestamp
        ):
            raise DataContractError(
                "synchronized bars must be available by their close time"
            )
        closes = (current_a.close, current_b.close, previous_a.close, previous_b.close)
        if any(close <= 0 for close in closes):
            raise DataContractError("log returns require positive closes")
        output.append(
            _ReturnObservation(
                timestamp,
                MappingProxyType(
                    {
                        relationship.instrument_a: math.log(
                            current_a.close / previous_a.close
                        ),
                        relationship.instrument_b: math.log(
                            current_b.close / previous_b.close
                        ),
                    }
                ),
            )
        )
    return tuple(output)


def _event_id(
    relationship_id: str, timestamp: datetime, leader: str, laggard: str
) -> str:
    identity = "|".join(
        (STUDY_ID, relationship_id, timestamp.isoformat(), leader, laggard)
    )
    return f"sha256:{hashlib.sha256(identity.encode('ascii')).hexdigest()}"


def detect_leadlag_events(
    bars: Iterable[Bar], parameters: LeadLagParameters = DEFAULT_PARAMETERS
) -> tuple[LeadLagEvent, ...]:
    """Detect frozen v1 events without mutating input or performing any I/O.

    Each current return is normalized by the prior ``lookback_returns`` valid
    synchronized returns using sample standard deviation (``ddof=1``). Missing
    intervals are never filled, and a return spanning a missing bar is invalid.
    Cooldown is maintained independently for each relationship.
    """
    if parameters != DEFAULT_PARAMETERS:
        raise ValueError("detect_leadlag_events is bound to the frozen v1 parameters")
    by_instrument = _validated_bars(bars)
    events: list[LeadLagEvent] = []
    for relationship in RELATIONSHIPS:
        observations = _synchronized_returns(relationship, by_instrument)
        last_emitted: datetime | None = None
        for index in range(parameters.lookback_returns, len(observations)):
            current = observations[index]
            prior = observations[index - parameters.lookback_returns : index]
            instruments = (relationship.instrument_a, relationship.instrument_b)
            z: dict[str, float] = {}
            for instrument in instruments:
                history = [item.returns[instrument] for item in prior]
                deviation = statistics.stdev(history)
                if not math.isfinite(deviation) or deviation <= 0:
                    z = {}
                    break
                mean = statistics.fmean(history)
                value = (current.returns[instrument] - mean) / deviation
                if not math.isfinite(value):
                    z = {}
                    break
                z[instrument] = value
            if len(z) != 2 or (z[instruments[0]] > 0) != (z[instruments[1]] > 0):
                continue
            if z[instruments[0]] == 0 or z[instruments[1]] == 0:
                continue
            absolute = {instrument: abs(z[instrument]) for instrument in instruments}
            if absolute[instruments[0]] == absolute[instruments[1]]:
                continue
            leader = max(instruments, key=absolute.__getitem__)
            laggard = instruments[1] if leader == instruments[0] else instruments[0]
            if absolute[leader] < parameters.shock_threshold:
                continue
            ratio = absolute[laggard] / absolute[leader]
            if ratio > parameters.max_laggard_ratio:
                continue
            if (
                last_emitted is not None
                and current.timestamp < last_emitted + parameters.cooldown
            ):
                continue
            direction: Literal[-1, 1] = 1 if z[leader] > 0 else -1
            events.append(
                LeadLagEvent(
                    study_id=STUDY_ID,
                    event_id=_event_id(
                        relationship.relationship_id, current.timestamp, leader, laggard
                    ),
                    timestamp=current.timestamp,
                    relationship_id=relationship.relationship_id,
                    leader=leader,
                    laggard=laggard,
                    direction=direction,
                    leader_z=z[leader],
                    laggard_z=z[laggard],
                    leader_abs_z=absolute[leader],
                    laggard_abs_z=absolute[laggard],
                    z_ratio=ratio,
                    leader_return=current.returns[leader],
                    laggard_return=current.returns[laggard],
                    leader_z_bin=leader_z_bin(absolute[leader]),
                )
            )
            last_emitted = current.timestamp
    return tuple(
        sorted(events, key=lambda event: (event.timestamp, event.relationship_id))
    )
