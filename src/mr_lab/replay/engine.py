"""Causal M1 replay using the frozen Stage4B and OU primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from mr_lab.data import Bar, Timeframe
from mr_lab.data.resampling import IncompleteWindow, _aligned_open, resample_bars
from mr_lab.ornstein_uhlenbeck import (
    FrozenOuEligibilityFilter,
    IncrementalOuStateBuilder,
    ResidualObservation,
    build_candidate_ou_states,
    candidate_process_keys,
    frozen_ou_eligibility_spec,
)
from mr_lab.replay.parity import FrozenOuDecision, FrozenOuDecisionDetails
from mr_lab.research import build_research_observations
from mr_lab.sessions import DEFAULT_SESSION_SPEC
from mr_lab.signals import AlphaSignal, SignalProvenance, SignalReference
from mr_lab.stage4b import SignalState, SignalStateDeduplicator, deduplicate_states
from mr_lab.stage4b_runner import IncrementalVwapSignalStateBuilder


class ReplayError(ValueError):
    """Raised when a stream violates deterministic replay semantics."""


M15 = Timeframe("15m")


class IncrementalResampler:
    """Incrementally reproduce :func:`resample_bars` epoch-window semantics."""

    def __init__(self, target_timeframe: Timeframe = M15) -> None:
        self.target_timeframe = target_timeframe
        self._window: list[Bar] = []
        self._window_open: datetime | None = None
        self._last: Bar | None = None
        self.incomplete_windows: list[IncompleteWindow] = []

    def push(self, bar: Bar) -> Bar | None:
        if bar.timeframe != Timeframe("1m"):
            raise ReplayError("replay input must be completed canonical M1 bars")
        if self._last is not None:
            if bar.open_time == self._last.open_time:
                raise ReplayError("duplicate M1 input")
            if bar.open_time < self._last.open_time:
                raise ReplayError("M1 input must be chronological")
            if (
                bar.instrument != self._last.instrument
                or bar.price_basis != self._last.price_basis
                or bar.volume_semantics != self._last.volume_semantics
            ):
                raise ReplayError("M1 stream semantics changed")
        window_open = _aligned_open(bar.open_time, self.target_timeframe.duration)
        emitted = None
        if self._window_open is not None and window_open != self._window_open:
            emitted = self._finish_window()
        self._window_open = window_open
        self._window.append(bar)
        self._last = bar
        expected = self.target_timeframe.duration // bar.timeframe.duration
        if len(self._window) == expected:
            emitted = self._finish_window()
        elif len(self._window) > expected:
            raise ReplayError("too many M1 components in aligned window")
        return emitted

    def _finish_window(self) -> Bar | None:
        result = resample_bars(tuple(self._window), self.target_timeframe)
        self.incomplete_windows.extend(result.incomplete_windows)
        self._window = []
        self._window_open = None
        return result.bars[0] if result.bars else None

    def finish(self) -> tuple[IncompleteWindow, ...]:
        """Fail closed on a trailing partial window and return all gaps."""
        if self._window:
            self._finish_window()
        return tuple(self.incomplete_windows)


@dataclass(frozen=True, slots=True)
class ReplayPrefix:
    """A causal stream update containing only newly completed M1/M15 bars."""

    completed_m1: tuple[Bar, ...]
    completed_m15: tuple[Bar, ...]
    available_at: datetime


class FrozenOuResearchBuilder:
    """Bounded production adapter over shared incremental VWAP primitives."""

    def __init__(self, manifest: dict[str, str | None]):
        self._features = IncrementalVwapSignalStateBuilder(manifest)

    def push_m1(self, bar: Bar) -> None:
        observation = build_research_observations((bar,), DEFAULT_SESSION_SPEC)[0]
        self._features.push_m1(observation)

    def push_m15(self, bar: Bar) -> tuple[SignalState, ...]:
        observation = build_research_observations((bar,), DEFAULT_SESSION_SPEC)[0]
        return tuple(
            state
            for state in self._features.push_target(observation)
            if state.session == "london" and state.direction.name == "SHORT"
        )

    def advance(self, prefix: ReplayPrefix) -> tuple[SignalState, ...]:
        """Consume every newly completed bar once through the typed boundary."""
        for bar in prefix.completed_m1:
            if bar.available_at > prefix.available_at:
                raise ReplayError("replay prefix contains future M1 data")
            self.push_m1(bar)
        output = []
        for bar in prefix.completed_m15:
            if bar.available_at > prefix.available_at:
                raise ReplayError("replay prefix contains future M15 data")
            output.extend(self.push_m15(bar))
        return tuple(output)

    @property
    def retained_rolling_values(self):
        return self._features.retained_rolling_values


def _decision(candidate, state, spec):
    gate = FrozenOuEligibilityFilter(spec, (state,))
    decision = gate.evaluate(candidate)
    metadata = dict(decision.metadata)
    signal = candidate.signal
    return FrozenOuDecision(
        AlphaSignal(
            module_id="frozen-ou-crossasset-v1",
            source_event_id=candidate.candidate_event_id,
            instrument=signal.instrument,
            timestamp=signal.signal_timestamp,
            direction=signal.direction.name,
            confidence=None,
            reference=SignalReference(signal.p0, "completed-signal-close"),
            invalidation=None,
            provenance=SignalProvenance(
                signal.strategy_spec_id,
                signal.source_corpus_id,
                signal.assembled_dataset_id,
            ),
        ),
        FrozenOuDecisionDetails(
            signal.benchmark_family,
            signal.lookback,
            signal.p0,
            signal.e0,
            signal.normalized_deviation,
            state.process_id,
            state.process_spec_id,
            state.status,
            state.is_structurally_valid,
            state.invalid_reason,
            state.ornstein_uhlenbeck_score,
            state.half_life_minutes,
            decision.eligible,
            metadata["eligibility_reason"],
            decision.filter_spec_id,
        ),
    )


def build_frozen_ou_decisions(
    states: Iterable[SignalState], filter_name="frozen-ou-crossasset-v1"
) -> tuple[FrozenOuDecision, ...]:
    """Apply the research Stage4B identity, OU process, and frozen gate exactly."""
    states = tuple(states)
    spec = frozen_ou_eligibility_spec(filter_name)
    candidates = deduplicate_states(states)
    candidate_states = build_candidate_ou_states(
        states, spec.process_spec, candidate_process_keys(candidates)
    )
    state_index = {
        (state.process_id, state.available_at): state for state in candidate_states
    }
    output = []
    for candidate in candidates:
        signal = candidate.signal
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
        key = (observation.process_id, signal.signal_timestamp)
        state = state_index.get(key)
        if state is None:
            raise ReplayError(
                "candidate is missing its exact-time OU process state: "
                f"{candidate.candidate_event_id}"
            )
        output.append(_decision(candidate, state, spec))
    return tuple(output)


class FrozenOuReplayEngine:
    """Replay completed M1 candles through shared Stage4B/OU research code.

    Every M1 is admitted once to the causal builder. Every complete M15 advances
    four bounded VWAP/OU processes once; no complete input prefix is rebuilt.
    """

    def __init__(
        self,
        state_builder: FrozenOuResearchBuilder,
        filter_name="frozen-ou-crossasset-v1",
    ):
        self._resampler = IncrementalResampler()
        self._builder = state_builder
        self._spec = frozen_ou_eligibility_spec(filter_name)
        self._deduplicator = SignalStateDeduplicator()
        self._ou = IncrementalOuStateBuilder(self._spec.process_spec)
        self.completed_m15_count = 0

    def push(self, bar: Bar) -> tuple[FrozenOuDecision, ...]:
        completed = self._resampler.push(bar)
        completed_m15 = () if completed is None else (completed,)
        states = self._builder.advance(
            ReplayPrefix((bar,), completed_m15, bar.available_at)
        )
        if completed is None:
            if states:
                raise ReplayError("state builder emitted before an M15 completion")
            return ()
        self.completed_m15_count += 1
        if any(state.timestamp > completed.available_at for state in states):
            raise ReplayError("state builder returned future information")
        output = []
        for state in states:
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
            process_state = self._ou.push(observation)
            candidate = self._deduplicator.push(state)
            if candidate is not None:
                if candidate.signal.signal_timestamp != completed.available_at:
                    raise ReplayError("state builder produced a retroactive candidate")
                output.append(_decision(candidate, process_state, self._spec))
        return tuple(sorted(output, key=lambda item: item.signal.source_event_id))

    def run(self, bars: Iterable[Bar]) -> tuple[FrozenOuDecision, ...]:
        output = []
        for bar in bars:
            output.extend(self.push(bar))
        self._resampler.finish()
        return tuple(output)

    @property
    def incomplete_windows(self) -> tuple[IncompleteWindow, ...]:
        return tuple(self._resampler.incomplete_windows)

    @property
    def retained_transition_count(self):
        return self._ou.retained_transition_count
