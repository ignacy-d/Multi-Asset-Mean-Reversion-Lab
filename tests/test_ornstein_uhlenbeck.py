import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from mr_lab.data import Timeframe
from mr_lab.ornstein_uhlenbeck import (
    OrnsteinUhlenbeckError,
    OrnsteinUhlenbeckProcessSpec,
    align_candidate_states,
    build_ou_states,
    fit_ou_state,
    residual_observations,
)
from mr_lab.research import Direction
from mr_lab.stage4b import SignalState, deduplicate_states

T = datetime(2024, 1, 2, 12, tzinfo=UTC)


def signal(minute, deviation, direction=Direction.LONG, *, p0=None):
    p0 = 1.0 + deviation if p0 is None else p0
    return SignalState(
        "EURUSD",
        T + timedelta(minutes=5 * minute),
        "vwap",
        Timeframe("5m"),
        "london",
        direction,
        20,
        p0,
        1.0,
        -2.1 if direction is Direction.LONG else 2.1,
        "corpus",
        "dataset",
        "strategy",
        True,
    )


def observation(value=0.0):
    item = signal(0, value)
    return residual_observations((item,))[0]


def test_exact_ar1_to_ou_mapping_and_standard_errors():
    spec = OrnsteinUhlenbeckProcessSpec(4)
    xs = (-3.0, -1.0, 1.0, 3.0)
    errors = (0.01, -0.03, 0.03, -0.01)
    pairs = tuple(
        (x, 0.2 + 0.8 * x + error) for x, error in zip(xs, errors, strict=True)
    )
    state = fit_ou_state(observation(2.0), pairs, spec)
    assert state.phi == pytest.approx(0.8)
    assert state.intercept == pytest.approx(0.2)
    assert state.kappa_per_minute == pytest.approx(-math.log(0.8) / 5)
    assert state.long_run_mean == pytest.approx(1.0)
    assert state.half_life_minutes == pytest.approx(math.log(2) * 5 / -math.log(0.8))
    expected_innovation = math.sqrt(sum(value**2 for value in errors) / 2)
    assert state.innovation_sigma == pytest.approx(expected_innovation)
    assert state.stationary_sigma == pytest.approx(expected_innovation / 0.6)
    assert state.phi_standard_error is not None
    assert state.is_structurally_valid


def test_deterministic_synthetic_process_recovery():
    values = [0.7]
    seed = 17
    for _ in range(400):
        seed = (1103515245 * seed + 12345) % (2**31)
        noise = ((seed / 2**31) - 0.5) * 0.02
        values.append(0.01 + 0.72 * values[-1] + noise)
    states = build_ou_states(
        tuple(signal(index, value) for index, value in enumerate(values)),
        OrnsteinUhlenbeckProcessSpec(256),
    )
    assert states[-1].phi == pytest.approx(0.72, abs=0.04)
    assert states[-1].long_run_mean == pytest.approx(0.01 / 0.28, abs=0.01)


def test_prefix_invariance_and_future_outlier():
    base = tuple(signal(index, math.sin(index) / 100) for index in range(12))
    spec = OrnsteinUhlenbeckProcessSpec(5)
    prefix = build_ou_states(base, spec)
    extended = build_ou_states((*base, signal(12, 1_000_000)), spec)
    assert extended[: len(prefix)] == prefix


def test_exact_adjacency_gap_rejection_and_retained_valid_pairs():
    states = tuple(signal(index, value) for index, value in enumerate((0, 1, 2)))
    after_gap = replace(signal(4, 3), timestamp=T + timedelta(minutes=30))
    resumed = replace(signal(5, 4), timestamp=T + timedelta(minutes=35))
    output = build_ou_states(
        (*states, after_gap, resumed), OrnsteinUhlenbeckProcessSpec(3)
    )
    assert [item.transition_count for item in output] == [0, 1, 2, 2, 3]


def test_rolling_window_is_last_n_valid_transitions():
    values = (0.0, 0.1, -0.03, 0.08, -0.01, 0.04)
    spec = OrnsteinUhlenbeckProcessSpec(3)
    output = build_ou_states(tuple(signal(i, x) for i, x in enumerate(values)), spec)
    expected = fit_ou_state(observation(values[-1]), tuple(pairwise(values))[-3:], spec)
    actual = output[-1]
    assert (actual.intercept, actual.phi, actual.innovation_sigma) == pytest.approx(
        (expected.intercept, expected.phi, expected.innovation_sigma)
    )
    assert actual.transition_count == 3


def test_direction_duplicates_are_collapsed_and_conflicts_fail_closed():
    long = signal(0, -0.1)
    short = replace(long, direction=Direction.SHORT)
    assert len(residual_observations((long, short))) == 1
    conflict = replace(short, p0=2.0)
    with pytest.raises(OrnsteinUhlenbeckError, match="conflicting duplicate"):
        residual_observations((long, conflict))


def test_optional_session_processes_have_a_deterministic_total_order():
    contextual = signal(0, -0.1)
    global_context = replace(contextual, session=None)
    observations = residual_observations((contextual, global_context))
    assert len(observations) == 2


def test_insufficient_and_degenerate_states_are_explicit():
    spec = OrnsteinUhlenbeckProcessSpec(3)
    assert (
        build_ou_states((signal(0, 1),), spec)[0].invalid_reason
        == "insufficient_history"
    )
    state = fit_ou_state(observation(), ((1, 2), (1, 3), (1, 4)), spec)
    assert state.invalid_reason == "degenerate_regression"


@pytest.mark.parametrize(
    ("pairs", "reason"),
    [
        (((-1, 1), (0, 0), (1, -1)), "phi_non_positive"),
        (((-1, -2), (0, 0), (1, 2)), "phi_not_below_one"),
    ],
)
def test_structurally_invalid_phi_is_reported(pairs, reason):
    state = fit_ou_state(observation(), pairs, OrnsteinUhlenbeckProcessSpec(3))
    assert state.status == "invalid"
    assert state.invalid_reason == reason
    assert not state.is_structurally_valid


def test_zero_innovation_variance_is_invalid():
    pairs = ((-1, -0.5), (0, 0.0), (1, 0.5))
    state = fit_ou_state(observation(), pairs, OrnsteinUhlenbeckProcessSpec(3))
    assert state.invalid_reason == "non_positive_innovation_variance"


def test_spec_and_process_identity_are_deterministic_and_direction_independent():
    assert (
        OrnsteinUhlenbeckProcessSpec(32).process_spec_id
        == OrnsteinUhlenbeckProcessSpec(32).process_spec_id
    )
    long = residual_observations((signal(0, -0.1),))[0]
    short = residual_observations((signal(0, -0.1, Direction.SHORT),))[0]
    assert long.process_id == short.process_id


def test_candidate_alignment_requires_exact_time_and_never_uses_later_state():
    states = tuple(
        signal(i, value) for i, value in enumerate((0.01, -0.02, -0.03, -0.04))
    )
    event = deduplicate_states((states[2],))[0]
    fitted = build_ou_states(states, OrnsteinUhlenbeckProcessSpec(3))
    row = align_candidate_states((event,), fitted)[0]
    assert row["available_at"] == states[2].timestamp.isoformat()
    assert row["current_deviation"] == states[2].p0 - states[2].e0
    with pytest.raises(OrnsteinUhlenbeckError, match="no exact-time"):
        align_candidate_states((event,), fitted[3:])


def test_non_finite_residual_input_fails_closed():
    with pytest.raises(OrnsteinUhlenbeckError, match="finite"):
        residual_observations((replace(signal(0, 0), p0=math.inf),))
