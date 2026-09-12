"""Causal M1 replay using the frozen Stage4B and OU primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
from mr_lab.signals import AlphaSignal
from mr_lab.stage4b import SignalState, deduplicate_states


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


StateBuilder = Callable[[tuple[Bar, ...]], Iterable[SignalState]]


class FrozenOuReplayEngine:
    """Replay completed M1 candles through shared Stage4B/OU research code.

    ``state_builder`` is the production boundary for an alpha module.  It is
    called with only the completed M15 prefix, never future bars.  The builder
    should use the existing research feature construction and return all valid
    Stage4B states in that prefix.  Rebuilding from the prefix is intentionally
    simple for M0 and makes point-in-time parity auditable.
    """

    def __init__(
        self, state_builder: StateBuilder, filter_name="frozen-ou-crossasset-v1"
    ):
        self._resampler = IncrementalResampler()
        self._bars: list[Bar] = []
        self._builder = state_builder
        self._spec = frozen_ou_eligibility_spec(filter_name)
        self._emitted: set[str] = set()

    @property
    def completed_bars(self) -> tuple[Bar, ...]:
        return tuple(self._bars)

    def push(self, bar: Bar) -> tuple[AlphaSignal, ...]:
        completed = self._resampler.push(bar)
        if completed is None:
            return ()
        self._bars.append(completed)
        states = tuple(self._builder(tuple(self._bars)))
        if any(state.timestamp > completed.available_at for state in states):
            raise ReplayError("state builder returned future information")
        candidates = deduplicate_states(states)
        candidate_states = build_candidate_ou_states(
            states, self._spec.process_spec, candidate_process_keys(candidates)
        )
        state_index = {(s.process_id, s.available_at): s for s in candidate_states}
        gate = FrozenOuEligibilityFilter(self._spec, candidate_states)
        output = []
        for candidate in candidates:
            if candidate.candidate_event_id in self._emitted:
                continue
            if candidate.signal.signal_timestamp != completed.available_at:
                continue
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
            state = state_index[(observation.process_id, signal.signal_timestamp)]
            decision = gate.evaluate(candidate)
            metadata = dict(decision.metadata)
            output.append(
                AlphaSignal(
                    candidate.candidate_event_id,
                    signal.signal_timestamp,
                    signal.instrument,
                    signal.direction.name,
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
                )
            )
            self._emitted.add(candidate.candidate_event_id)
        return tuple(output)

    def run(self, bars: Iterable[Bar]) -> tuple[AlphaSignal, ...]:
        output = []
        for bar in bars:
            output.extend(self.push(bar))
        self._resampler.finish()
        return tuple(output)

    @property
    def incomplete_windows(self) -> tuple[IncompleteWindow, ...]:
        return tuple(self._resampler.incomplete_windows)
