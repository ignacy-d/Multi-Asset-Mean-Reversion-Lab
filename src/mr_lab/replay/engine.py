"""Causal M1 replay using the frozen Stage4B and OU primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from mr_lab.data import Bar, Timeframe
from mr_lab.data.resampling import IncompleteWindow, _aligned_open, resample_bars
from mr_lab.ornstein_uhlenbeck import (
    FrozenOuEligibilityFilter,
    ResidualObservation,
    build_candidate_ou_states,
    candidate_process_keys,
    frozen_ou_eligibility_spec,
)
from mr_lab.replay.parity import FrozenOuDecision, FrozenOuDecisionDetails
from mr_lab.research import build_research_observations
from mr_lab.sessions import DEFAULT_SESSION_SPEC
from mr_lab.signals import AlphaSignal, SignalProvenance, SignalReference
from mr_lab.stage4b import SignalState, deduplicate_states
from mr_lab.stage4b_runner import assemble_vwap_signal_states


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
    """All canonical information legally available at an M15 close."""

    completed_m1: tuple[Bar, ...]
    completed_m15: tuple[Bar, ...]
    available_at: datetime


StateBuilder = Callable[[ReplayPrefix], Iterable[SignalState]]


class FrozenOuResearchBuilder:
    """Production adapter over the same VWAP assembly used by Stage4B research."""

    def __init__(self, manifest: dict[str, str | None]):
        self._manifest = manifest

    def __call__(self, prefix: ReplayPrefix) -> tuple[SignalState, ...]:
        m1 = build_research_observations(prefix.completed_m1, DEFAULT_SESSION_SPEC)
        m15 = build_research_observations(prefix.completed_m15, DEFAULT_SESSION_SPEC)
        return tuple(
            state
            for state in assemble_vwap_signal_states(m1, m15, self._manifest)
            if state.session == "london" and state.direction.name == "SHORT"
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
    gate = FrozenOuEligibilityFilter(spec, candidate_states)
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
        decision = gate.evaluate(candidate)
        metadata = dict(decision.metadata)
        output.append(
            FrozenOuDecision(
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
        )
    return tuple(output)


class FrozenOuReplayEngine:
    """Replay completed M1 candles through shared Stage4B/OU research code.

    ``state_builder`` is the production boundary for an alpha module.  It is
    called with only completed M1 and M15 prefixes, never future bars.  The
    builder should use the existing research feature construction and return all
    valid Stage4B states in that prefix. Rebuilding from the prefix is
    intentionally simple for M0 and makes point-in-time parity auditable.
    """

    def __init__(
        self, state_builder: StateBuilder, filter_name="frozen-ou-crossasset-v1"
    ):
        self._resampler = IncrementalResampler()
        self._m1: list[Bar] = []
        self._bars: list[Bar] = []
        self._builder = state_builder
        self._spec = frozen_ou_eligibility_spec(filter_name)
        self._emitted: set[str] = set()

    @property
    def completed_bars(self) -> tuple[Bar, ...]:
        return tuple(self._bars)

    def push(self, bar: Bar) -> tuple[FrozenOuDecision, ...]:
        completed = self._resampler.push(bar)
        self._m1.append(bar)
        if completed is None:
            return ()
        self._bars.append(completed)
        prefix = ReplayPrefix(
            tuple(self._m1), tuple(self._bars), completed.available_at
        )
        states = tuple(self._builder(prefix))
        if any(state.timestamp > completed.available_at for state in states):
            raise ReplayError("state builder returned future information")
        decisions = build_frozen_ou_decisions(states, self._spec.name)
        output = []
        for decision in decisions:
            event_id = decision.signal.source_event_id
            if event_id in self._emitted:
                continue
            if decision.signal.timestamp != completed.available_at:
                raise ReplayError("state history produced a retroactive candidate")
            output.append(decision)
            self._emitted.add(event_id)
        return tuple(output)

    def run(self, bars: Iterable[Bar]) -> tuple[FrozenOuDecision, ...]:
        output = []
        for bar in bars:
            output.extend(self.push(bar))
        self._resampler.finish()
        return tuple(output)

    @property
    def incomplete_windows(self) -> tuple[IncompleteWindow, ...]:
        return tuple(self._resampler.incomplete_windows)
