import copy
import hashlib
from math import exp, log

import numpy as np
import pytest
from test_pca_residual import config, panel_from_returns, process
from test_pca_stage0_runner import long_panel_series

import mr_lab.pca_residual as pca_residual
import mr_lab.pca_rv_quality_2024 as quality
from mr_lab.pca_residual import PCAResidualConfig, PCAResidualEngine
from mr_lab.pca_stage0_runner import (
    INSTRUMENTS,
    _sha,
    build_panel,
    event_outcome,
    panel_identity_at,
)


def ar_series(phi, alpha=0.25, count=256):
    values = [1.75]
    for _ in range(count - 1):
        values.append(alpha + phi * values[-1])
    return values


def test_closed_form_ar1_and_half_life_arithmetic():
    result = quality.estimate_ar1(ar_series(0.8), 1e-12)
    assert result.residual_ar1_phi == pytest.approx(0.8)
    assert result.residual_ar1_alpha == pytest.approx(0.25)
    assert result.residual_ou_kappa == pytest.approx(-log(0.8))
    assert result.residual_half_life_rows == pytest.approx(log(2) / -log(0.8))
    assert result.residual_ou_eligible


@pytest.mark.parametrize("phi", [-0.2, 0.0, 1.0, 1.01])
def test_non_mean_reverting_phi_is_explicitly_invalid(phi):
    result = quality.estimate_ar1(ar_series(phi), 1e-12)
    assert result.residual_ar1_phi == pytest.approx(phi)
    assert result.residual_ou_kappa is None
    assert result.residual_half_life_rows is None
    assert not result.residual_ou_eligible
    assert result.residual_half_life_bin == "NON_MEAN_REVERTING_OR_INVALID"


def test_zero_variance_lagged_regressor_fails_closed():
    result = quality.estimate_ar1([2.0] * 256, 1e-12)
    assert result.residual_ar1_phi is None
    assert not result.residual_ou_eligible


def test_exact_half_life_boundary_is_inclusive():
    at_limit = quality.estimate_ar1(ar_series(exp(-log(2) / 60)), 1e-12)
    beyond = quality.estimate_ar1(ar_series(exp(-log(2) / 60.0001)), 1e-12)
    assert at_limit.residual_half_life_rows == pytest.approx(60)
    assert at_limit.residual_ou_eligible
    assert at_limit.residual_half_life_bin == ">30_to_60"
    assert not beyond.residual_ou_eligible
    assert beyond.residual_half_life_bin == ">60"


def test_current_and_future_residuals_cannot_affect_prior_ar_estimate():
    prior = ar_series(0.7)
    expected = quality.estimate_ar1(prior, 1e-12)
    assert quality.estimate_ar1(prior, 1e-12) == expected
    mutated_current_and_future = [*prior, 1e20, -1e20]
    assert quality.estimate_ar1(mutated_current_and_future[:256], 1e-12) == expected


def test_idio_ratio_arithmetic_and_closed_denominator():
    assert quality.idio_ratio(-3, 1, 1e-12) == pytest.approx(0.75)
    assert quality.idio_ratio(0, 0, 1e-12) is None


def test_projection_distance_is_sign_invariant_zero_and_detects_difference():
    base = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    different = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    assert quality.projection_subspace_distance(base, -base) == pytest.approx(0)
    assert quality.projection_subspace_distance(base, base) == pytest.approx(0)
    assert quality.projection_subspace_distance(base, different) > 0


def test_quality_instrumentation_is_exactly_legacy_equivalent():
    panel = panel_from_returns(process(100, shock_index=70, shock=0.03))
    engine = PCAResidualEngine(config())
    legacy = engine.run(panel)
    instrumented = tuple(row.observation for row in engine.run_with_quality(panel))
    assert instrumented == legacy


def test_legacy_fast_path_never_constructs_quality_payloads(monkeypatch):
    panel = panel_from_returns(process(80))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy run constructed a quality payload")

    monkeypatch.setattr(pca_residual, "PCAQualityObservation", forbidden)
    assert PCAResidualEngine(config()).run(panel)


def test_prior_residual_payload_exists_only_for_emitted_events():
    panel = panel_from_returns(process(100, shock_index=70, shock=0.03))
    rows = PCAResidualEngine(config()).run_with_quality(panel)
    assert any(row.observation.event_emitted for row in rows)
    assert all(
        (row.prior_residuals is not None) == row.observation.event_emitted
        for row in rows
    )


def test_ar_calculation_is_invoked_only_for_emitted_events(monkeypatch):
    panel = panel_from_returns(process(100, shock_index=70, shock=0.03))
    rows = PCAResidualEngine(config()).run_with_quality(panel)
    event = next(row for row in rows if row.observation.event_emitted)
    non_event = next(row for row in rows if not row.observation.event_emitted)
    calls = 0

    def counted(_prior, _epsilon):
        nonlocal calls
        calls += 1
        return quality._invalid_ar()

    monkeypatch.setattr(quality, "estimate_ar1", counted)
    with pytest.raises(quality.PCAQualityError, match="only for emitted"):
        quality.enrich_event({}, non_event)
    assert calls == 0
    quality.enrich_event({}, event)
    assert calls == 1


def test_current_and_future_mutation_do_not_change_event_prior_ar_input():
    original = process(100, shock_index=70, shock=0.03)
    changed = process(100, shock_index=70, shock=0.06)
    changed[71:] += np.random.default_rng(31).normal(0, 0.02, changed[71:].shape)
    engine = PCAResidualEngine(config())
    first = next(
        row
        for row in engine.run_with_quality(panel_from_returns(original))
        if row.observation.instrument == "A" and row.observation.event_emitted
    )
    second = next(
        row
        for row in engine.run_with_quality(panel_from_returns(changed))
        if row.observation.timestamp == first.observation.timestamp
        and row.observation.instrument == "A"
    )
    assert second.observation.event_emitted
    assert second.prior_residuals == first.prior_residuals


def test_source_sha_rejection(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    with pytest.raises(quality.PCAQualityError, match="SHA256"):
        quality.verify_source_hash(source)


def test_regenerated_identity_and_float_mismatch_rejected():
    event = {field: field for field in quality.IDENTITY_FIELDS}
    event.update({field: 1.0 for field in quality.FLOAT_FIELDS})
    changed = dict(event)
    changed["event_id"] = "different"
    with pytest.raises(quality.PCAQualityError, match="identity mismatch"):
        quality.verify_regenerated_event(event, changed)
    changed = dict(event)
    changed["residual"] = 2.0
    with pytest.raises(quality.PCAQualityError, match="value mismatch"):
        quality.verify_regenerated_event(event, changed)


def test_quality_identity_is_canonical_and_deterministic():
    encoded = quality._identity(quality.QUALITY_SPEC)
    assert encoded == quality.QUALITY_STUDY_ID
    assert (
        encoded
        == "sha256:"
        + hashlib.sha256(
            quality.json.dumps(
                quality.QUALITY_SPEC, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )


def test_quality_identity_binds_every_frozen_feature_definition():
    for field in (
        "base_process_identity",
        "residual_ar1",
        "residual_ou",
        "idio_ratio",
        "k2_explained_variance_ratio",
        "k2_eigengap_ratio",
        "subspace_distance",
        "cross_sectional_standardized_return_dispersion",
        "bootstrap",
    ):
        changed = copy.deepcopy(quality.QUALITY_SPEC)
        changed[field] = "changed"
        assert quality._identity(changed) != quality.QUALITY_STUDY_ID


def test_incremental_panel_identity_is_stage0_equivalent():
    incremental = quality._PrefixPanelIdentities("panel-source")
    stamps = ["2024-01-01T00:00:00+00:00", "2024-01-01T00:01:00+00:00"]
    for index, stamp in enumerate(stamps, 1):
        assert incremental.add(stamp) == _sha(
            {
                "source_identity": "panel-source",
                "synchronized_timestamps": stamps[:index],
                "activity_policy": quality.ACTIVITY_POLICY,
            }
        )


def test_multi_instrument_timestamp_reuses_exact_frozen_panel_identity():
    built = build_panel(
        long_panel_series(90), {name: "synthetic" for name in INSTRUMENTS}
    )
    engine = PCAResidualEngine(
        PCAResidualConfig(
            pca_training_window=24,
            components=2,
            residual_normalization_window=12,
            shock_threshold=0.5,
            rearm_threshold=0.1,
            variance_epsilon=1e-14,
        )
    )
    rows = list(quality.iter_quality_with_panel_identities(built, engine))
    by_timestamp = {}
    for row, identity in rows:
        by_timestamp.setdefault(row.observation.timestamp, []).append((row, identity))
    assert by_timestamp
    for timestamp, timestamp_rows in by_timestamp.items():
        identities = {identity for _, identity in timestamp_rows}
        assert len(timestamp_rows) == len(INSTRUMENTS)
        assert identities == {panel_identity_at(built, timestamp)}
    emitted, identity = next(
        (row, identity) for row, identity in rows if row.observation.event_emitted
    )
    regenerated = event_outcome(
        built, emitted.observation, causal_panel_identity=identity
    )
    frozen = event_outcome(built, emitted.observation)
    assert regenerated["event_id"] == frozen["event_id"]


def test_runner_refuses_existing_output_before_data_access(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(quality.PCAQualityError, match="overwrite"):
        quality.run(tmp_path / "not-read.jsonl", output)


def test_streaming_writer_flushes_completed_lines_and_refuses_overwrite(tmp_path):
    path = tmp_path / "events.jsonl"
    with quality.StreamingJSONLWriter(path) as writer:
        writer.write({"event_id": "first"})
        assert path.read_text() == '{"event_id": "first"}\n'
    with pytest.raises(FileExistsError):
        quality.StreamingJSONLWriter(path)


def test_bootstrap_contract_is_frozen():
    assert quality.BOOTSTRAP_REPLICATES == 10_000
    assert quality.BOOTSTRAP_SEED == 20240918
