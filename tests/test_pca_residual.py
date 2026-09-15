from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from mr_lab.pca_residual import (
    Direction,
    PCAResidualConfig,
    PCAResidualEngine,
    PCAResidualError,
    ShockStateDetector,
    SynchronizedClosePanel,
    reconstruct_from_components,
)


def panel_from_returns(returns, instruments=("A", "B", "C", "D")):
    returns = np.asarray(returns, dtype=float)
    closes = np.vstack(
        [np.full(returns.shape[1], 100.0), 100 * np.exp(np.cumsum(returns, axis=0))]
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    stamps = tuple(start + timedelta(minutes=i) for i in range(len(closes)))
    return SynchronizedClosePanel(stamps, tuple(instruments), closes)


def process(n=100, shock_index=None, shock=0.0):
    t = np.arange(n, dtype=float)
    factors = np.column_stack((0.001 * np.sin(t / 4), 0.0013 * np.cos(t / 7)))
    loadings = np.array([[1.0, 0.2], [-0.4, 1.2], [0.8, -0.7], [-1.1, -0.3]])
    values = factors @ loadings.T
    if shock_index is not None:
        values[shock_index, 0] += shock
    return values


def config(**changes):
    values = dict(
        pca_training_window=24,
        components=2,
        residual_normalization_window=12,
        variance_epsilon=1e-14,
    )
    values.update(changes)
    return PCAResidualConfig(**values)


def test_pure_two_factor_process_has_numerically_tiny_residuals():
    out = PCAResidualEngine(config()).run(panel_from_returns(process()))
    assert max(abs(row.residual) for row in out) < 2e-11


def test_idiosyncratic_shock_has_expected_identity_sign_and_fade():
    shock_at = 70
    panel = panel_from_returns(process(shock_index=shock_at, shock=0.03))
    out = PCAResidualEngine(config()).run(panel)
    stamp = panel.timestamps[shock_at + 1]
    row = next(row for row in out if row.timestamp == stamp and row.instrument == "A")
    assert row.residual > 5
    assert row.residual_zscore > 2
    assert row.event_emitted
    assert row.residual_shock_sign is Direction.POSITIVE
    assert row.fade_direction is Direction.NEGATIVE


def test_prefix_and_future_mutation_invariance():
    base = process(80)
    engine = PCAResidualEngine(config())
    prefix = engine.run(panel_from_returns(base[:65]))
    extended = engine.run(panel_from_returns(base))
    assert prefix == extended[: len(prefix)]
    changed = base.copy()
    changed[66:] += np.random.default_rng(9).normal(0, 0.01, changed[66:].shape)
    mutation = engine.run(panel_from_returns(changed))
    cutoff = panel_from_returns(base).timestamps[66]
    assert [x for x in extended if x.timestamp <= cutoff] == [
        x for x in mutation if x.timestamp <= cutoff
    ]


def test_current_observation_is_excluded_from_all_current_statistics():
    base = process(75)
    changed = base.copy()
    changed[60, 0] += 0.05
    original = PCAResidualEngine(config()).run(panel_from_returns(base))
    shocked = PCAResidualEngine(config()).run(panel_from_returns(changed))
    stamp = panel_from_returns(base).timestamps[61]
    before_a = next(x for x in original if x.timestamp == stamp and x.instrument == "A")
    after_a = next(x for x in shocked if x.timestamp == stamp and x.instrument == "A")
    # Its raw return changes, while the strictly-prior fitted state does not.
    prior_scale = base[60 - 24 : 60, 0].std(ddof=1)
    assert after_a.standardized_return - before_a.standardized_return == pytest.approx(
        0.05 / prior_scale
    )
    assert abs(after_a.residual_zscore) > 10


def test_component_sign_and_instrument_order_invariance():
    value = np.array([1.0, 2.0, -1.0, 0.5])
    mean = np.array([0.1, -0.2, 0.3, 0.4])
    components = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5]])
    assert np.array_equal(
        reconstruct_from_components(value, mean, components),
        reconstruct_from_components(value, mean, -components),
    )
    values = process()
    normal = PCAResidualEngine(config()).run(panel_from_returns(values))
    permutation = [2, 0, 3, 1]
    names = tuple(np.array(("A", "B", "C", "D"))[permutation])
    permuted = PCAResidualEngine(config()).run(
        panel_from_returns(values[:, permutation], names)
    )

    def key(x):
        return (x.timestamp, x.instrument)

    a = {key(x): x for x in normal}
    b = {key(x): x for x in permuted}
    for k in a:
        assert a[k].residual == pytest.approx(b[k].residual, abs=2e-12)
        assert a[k].factor_reconstruction == pytest.approx(
            b[k].factor_reconstruction, abs=2e-12
        )


def test_degenerate_and_invalid_panels_fail_closed():
    with pytest.raises(PCAResidualError, match="variance"):
        PCAResidualEngine(config()).run(panel_from_returns(np.zeros((50, 4))))
    valid = panel_from_returns(process(50))
    bad_cases = [
        SynchronizedClosePanel(
            (*valid.timestamps[:-1], valid.timestamps[-2]),
            valid.instruments,
            valid.closes,
        ),
        SynchronizedClosePanel(valid.timestamps, ("A", "A", "C", "D"), valid.closes),
        SynchronizedClosePanel(
            valid.timestamps, valid.instruments, valid.closes[:, :3]
        ),
    ]
    for bad in bad_cases:
        with pytest.raises(PCAResidualError):
            PCAResidualEngine(config()).run(bad)
    for value in (np.nan, np.inf, 0.0, -1.0):
        closes = valid.closes.copy()
        closes[2, 1] = value
        with pytest.raises(PCAResidualError):
            PCAResidualEngine(config()).run(
                SynchronizedClosePanel(valid.timestamps, valid.instruments, closes)
            )


def test_configuration_and_component_compatibility_fail_closed():
    for kwargs in (
        {"components": 0},
        {"rearm_threshold": 2.0},
        {"shock_threshold": float("nan")},
    ):
        with pytest.raises(PCAResidualError):
            config(**kwargs)
    with pytest.raises(PCAResidualError, match="residual dimension"):
        PCAResidualEngine(config(components=4)).run(panel_from_returns(process()))


def test_shock_state_machine_rearms_strictly_below_threshold():
    detector = ShockStateDetector(2, 1)
    states = [detector.update(z) for z in (2.1, 3, 1, 0.9, -2.2)]
    assert [s.event_emitted for s in states] == [True, False, False, False, True]
    assert states[0].fade_direction is Direction.NEGATIVE
    assert states[-1].fade_direction is Direction.POSITIVE


def test_active_episode_keeps_entry_sign_across_opposite_threshold():
    detector = ShockStateDetector(2, 1)
    entered, opposite = (detector.update(z) for z in (2.5, -2.5))

    assert entered.event_emitted
    assert entered.shock_sign is Direction.POSITIVE
    assert entered.fade_direction is Direction.NEGATIVE
    assert opposite.active
    assert not opposite.event_emitted
    assert opposite.shock_sign is Direction.POSITIVE
    assert opposite.fade_direction is Direction.NEGATIVE


def test_rearmed_episode_may_establish_opposite_sign():
    detector = ShockStateDetector(2, 1)
    entered, rearmed, opposite = (detector.update(z) for z in (2.5, 0.5, -2.5))

    assert entered.shock_sign is Direction.POSITIVE
    assert not rearmed.active
    assert rearmed.shock_sign is Direction.NONE
    assert rearmed.fade_direction is Direction.NONE
    assert opposite.event_emitted
    assert opposite.shock_sign is Direction.NEGATIVE
    assert opposite.fade_direction is Direction.POSITIVE


def test_known_reversal_direction_is_representable_without_outcome_logic():
    state = ShockStateDetector().update(3.0)
    programmed_next_residual_move = -0.5
    assert state.fade_direction is Direction.NEGATIVE
    assert state.fade_direction * programmed_next_residual_move > 0


def test_seeded_noise_does_not_manufacture_directional_edge():
    rng = np.random.default_rng(8128)
    values = rng.normal(0, 0.001, size=(500, 4))
    out = PCAResidualEngine(
        config(pca_training_window=48, residual_normalization_window=24)
    ).run(panel_from_returns(values))
    by_time = {}
    for row in out:
        if row.event_emitted:
            by_time[(row.timestamp, row.instrument)] = int(row.fade_direction)
    returns_by_stamp = {
        panel_from_returns(values).timestamps[i + 1]: values[i]
        for i in range(len(values))
    }
    outcomes = []
    names = ("A", "B", "C", "D")
    for (stamp, instrument), direction in by_time.items():
        next_stamp = stamp + timedelta(minutes=1)
        if next_stamp in returns_by_stamp:
            outcomes.append(
                direction * returns_by_stamp[next_stamp][names.index(instrument)]
            )
    assert outcomes
    assert abs(float(np.mean(outcomes))) < 0.0005
