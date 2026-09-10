"""Causal rolling Ornstein--Uhlenbeck estimates for benchmark deviations."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256

from mr_lab.data import Timeframe
from mr_lab.stage4b import CandidateEvent, SignalState

RESIDUAL_DEFINITION = "completed-price-minus-causal-benchmark-equilibrium-v1"
ADJACENCY_SEMANTICS = "exactly-one-signal-timeframe-valid-transitions-v1"
INCLUSION_SEMANTICS = "include-valid-transition-ending-at-current-observation-v1"
ESTIMATOR_VERSION = "ols-ar1-with-intercept-v1"


class OrnsteinUhlenbeckError(ValueError):
    """Raised when process inputs violate causal or identity contracts."""


@dataclass(frozen=True, slots=True)
class OrnsteinUhlenbeckProcessSpec:
    """Versioned rolling estimator semantics; the window counts valid pairs."""

    window_transitions: int
    residual_definition: str = RESIDUAL_DEFINITION
    adjacency_semantics: str = ADJACENCY_SEMANTICS
    estimator_version: str = ESTIMATOR_VERSION
    inclusion_semantics: str = INCLUSION_SEMANTICS

    def __post_init__(self) -> None:
        if type(self.window_transitions) is not int or self.window_transitions < 3:
            raise OrnsteinUhlenbeckError("window_transitions must be at least 3")
        for field, expected in (
            ("residual_definition", RESIDUAL_DEFINITION),
            ("adjacency_semantics", ADJACENCY_SEMANTICS),
            ("estimator_version", ESTIMATOR_VERSION),
            ("inclusion_semantics", INCLUSION_SEMANTICS),
        ):
            if getattr(self, field) != expected:
                raise OrnsteinUhlenbeckError(f"unsupported {field}")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def process_spec_id(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ResidualObservation:
    """One direction-independent, completed benchmark deviation."""

    instrument: str
    benchmark_family: str
    signal_timeframe: Timeframe
    session: str | None
    lookback: int
    strategy_spec_id: str
    available_at: datetime
    p0: float
    e0: float

    @property
    def current_deviation(self) -> float:
        return self.p0 - self.e0

    @property
    def process_key(self) -> tuple[object, ...]:
        return (
            self.instrument,
            self.benchmark_family,
            str(self.signal_timeframe),
            self.session,
            self.lookback,
            self.strategy_spec_id,
        )

    @property
    def process_id(self) -> str:
        encoded = json.dumps(self.process_key, separators=(",", ":"))
        return "ou-process-" + sha256(encoded.encode()).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class OrnsteinUhlenbeckProcessState:
    process_id: str
    process_spec_id: str
    available_at: datetime
    current_deviation: float
    transition_count: int
    delta_minutes: float
    intercept: float | None = None
    phi: float | None = None
    phi_standard_error: float | None = None
    kappa_per_minute: float | None = None
    long_run_mean: float | None = None
    half_life_minutes: float | None = None
    innovation_sigma: float | None = None
    stationary_sigma: float | None = None
    ornstein_uhlenbeck_score: float | None = None
    fit_r_squared: float | None = None
    is_structurally_valid: bool = False
    status: str = "unavailable"
    invalid_reason: str | None = "insufficient_history"


def residual_observations(states) -> tuple[ResidualObservation, ...]:
    """Collapse LONG/SHORT representations, failing closed on disagreement."""
    unique: dict[tuple[object, ...], ResidualObservation] = {}
    for state in states:
        if not isinstance(state, SignalState):
            raise OrnsteinUhlenbeckError("all inputs must be SignalState instances")
        if not all(math.isfinite(value) for value in (state.p0, state.e0)):
            raise OrnsteinUhlenbeckError("p0 and e0 must be finite")
        observation = ResidualObservation(
            state.instrument,
            state.benchmark_family,
            state.signal_timeframe,
            state.session,
            state.lookback,
            state.strategy_spec_id,
            state.timestamp,
            state.p0,
            state.e0,
        )
        key = (*observation.process_key, observation.available_at)
        existing = unique.get(key)
        if existing is not None and (
            existing.p0 != observation.p0
            or existing.e0 != observation.e0
            or existing.current_deviation != observation.current_deviation
        ):
            raise OrnsteinUhlenbeckError(
                "conflicting duplicate residual process observation"
            )
        unique[key] = observation
    return tuple(
        sorted(
            unique.values(), key=lambda item: (str(item.process_key), item.available_at)
        )
    )


def _unavailable(observation, spec, transition_count, reason):
    return OrnsteinUhlenbeckProcessState(
        observation.process_id,
        spec.process_spec_id,
        observation.available_at,
        observation.current_deviation,
        transition_count,
        observation.signal_timeframe.duration.total_seconds() / 60,
        invalid_reason=reason,
    )


def fit_ou_state(observation, transitions, spec):
    """Fit OLS over supplied valid pairs, including the pair ending now."""
    count = len(transitions)
    if count < spec.window_transitions:
        return _unavailable(observation, spec, count, "insufficient_history")
    pairs = tuple(transitions)[-spec.window_transitions :]
    previous = [pair[0] for pair in pairs]
    following = [pair[1] for pair in pairs]
    if not all(math.isfinite(value) for value in (*previous, *following)):
        return _unavailable(observation, spec, count, "non_finite_input")
    x_mean = math.fsum(previous) / len(previous)
    y_mean = math.fsum(following) / len(following)
    sxx = math.fsum((value - x_mean) ** 2 for value in previous)
    if not math.isfinite(sxx) or sxx <= 0:
        return _unavailable(observation, spec, count, "degenerate_regression")
    phi = (
        math.fsum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(previous, following, strict=True)
        )
        / sxx
    )
    intercept = y_mean - phi * x_mean
    predictions = [intercept + phi * value for value in previous]
    sse = math.fsum(
        (actual - predicted) ** 2
        for actual, predicted in zip(following, predictions, strict=True)
    )
    syy = math.fsum((value - y_mean) ** 2 for value in following)
    variance = sse / (len(pairs) - 2)
    values = (phi, intercept, sse, variance)
    if not all(math.isfinite(value) for value in values):
        return _unavailable(observation, spec, count, "non_finite_estimate")
    r_squared = 1 - sse / syy if syy > 0 else None
    innovation_sigma = math.sqrt(variance) if variance > 0 else None
    phi_se = math.sqrt(variance / sxx) if variance > 0 else None
    common = dict(
        process_id=observation.process_id,
        process_spec_id=spec.process_spec_id,
        available_at=observation.available_at,
        current_deviation=observation.current_deviation,
        transition_count=count,
        delta_minutes=observation.signal_timeframe.duration.total_seconds() / 60,
        intercept=intercept,
        phi=phi,
        phi_standard_error=phi_se,
        innovation_sigma=innovation_sigma,
        fit_r_squared=r_squared,
    )
    if phi <= 0:
        return OrnsteinUhlenbeckProcessState(
            **common, status="invalid", invalid_reason="phi_non_positive"
        )
    if phi >= 1:
        return OrnsteinUhlenbeckProcessState(
            **common, status="invalid", invalid_reason="phi_not_below_one"
        )
    if innovation_sigma is None:
        return OrnsteinUhlenbeckProcessState(
            **common,
            status="invalid",
            invalid_reason="non_positive_innovation_variance",
        )
    delta = common["delta_minutes"]
    kappa = -math.log(phi) / delta
    mean = intercept / (1 - phi)
    half_life = math.log(2) / kappa
    stationary_sigma = innovation_sigma / math.sqrt(1 - phi**2)
    mapped = (kappa, mean, half_life, stationary_sigma)
    if not all(math.isfinite(value) for value in mapped):
        return OrnsteinUhlenbeckProcessState(
            **common, status="invalid", invalid_reason="undefined_ou_mapping"
        )
    if stationary_sigma <= 0:
        return OrnsteinUhlenbeckProcessState(
            **common, status="invalid", invalid_reason="undefined_stationary_variance"
        )
    return OrnsteinUhlenbeckProcessState(
        **common,
        kappa_per_minute=kappa,
        long_run_mean=mean,
        half_life_minutes=half_life,
        stationary_sigma=stationary_sigma,
        ornstein_uhlenbeck_score=(observation.current_deviation - mean)
        / stationary_sigma,
        is_structurally_valid=True,
        status="valid",
        invalid_reason=None,
    )


def build_ou_states(states, spec) -> tuple[OrnsteinUhlenbeckProcessState, ...]:
    """Build causal states with bounded windows of exact-adjacent transitions."""
    observations = residual_observations(states)
    grouped = defaultdict(list)
    for observation in observations:
        grouped[observation.process_key].append(observation)
    output = []
    for key in sorted(grouped, key=str):
        previous = None
        transitions = deque(maxlen=spec.window_transitions)
        for observation in grouped[key]:
            if previous is not None and (
                observation.available_at - previous.available_at
                == observation.signal_timeframe.duration
            ):
                transitions.append(
                    (previous.current_deviation, observation.current_deviation)
                )
            output.append(fit_ou_state(observation, transitions, spec))
            previous = observation
    return tuple(sorted(output, key=lambda item: (item.available_at, item.process_id)))


def align_candidate_states(events, states):
    """Align only the state available at exactly the completed signal timestamp."""
    index = {
        (state.process_id, state.available_at, state.process_spec_id): state
        for state in states
    }
    rows = []
    for event in events:
        if not isinstance(event, CandidateEvent):
            raise OrnsteinUhlenbeckError(
                "all candidates must be CandidateEvent instances"
            )
        signal = event.signal
        observation = ResidualObservation(
            signal.instrument,
            signal.benchmark_family,
            signal.signal_timeframe,
            signal.session,
            signal.lookback,
            signal.strategy_spec_id,
            signal.signal_timestamp,
            signal.p0,
            signal.e0,
        )
        matches = [
            state
            for (process_id, timestamp, _), state in index.items()
            if process_id == observation.process_id
            and timestamp == signal.signal_timestamp
        ]
        if not matches:
            raise OrnsteinUhlenbeckError("candidate has no exact-time process state")
        for state in sorted(matches, key=lambda item: item.process_spec_id):
            rows.append(
                {
                    "candidate_event_id": event.candidate_event_id,
                    "instrument": signal.instrument,
                    "signal_timestamp": signal.signal_timestamp.isoformat(),
                    "benchmark_family": signal.benchmark_family,
                    "signal_timeframe": str(signal.signal_timeframe),
                    "session": signal.session,
                    "direction": signal.direction.name,
                    "lookback": signal.lookback,
                    "strategy_spec_id": signal.strategy_spec_id,
                    "z": signal.normalized_deviation,
                    "p0": signal.p0,
                    "e0": signal.e0,
                    "d0": signal.d0,
                    **{
                        key: value.isoformat() if isinstance(value, datetime) else value
                        for key, value in asdict(state).items()
                    },
                }
            )
    return tuple(rows)
