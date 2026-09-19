import hashlib
from math import exp, log

import numpy as np
import pytest
from test_pca_residual import config, panel_from_returns, process

import mr_lab.pca_rv_quality_2024 as quality
from mr_lab.pca_residual import PCAResidualEngine
from mr_lab.pca_stage0_runner import _sha


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
