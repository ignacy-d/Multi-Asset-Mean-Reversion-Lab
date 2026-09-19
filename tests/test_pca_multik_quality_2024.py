import copy
import gzip
import json
from math import exp, log

import numpy as np
import pytest
from test_pca_residual import config, panel_from_returns, process

import mr_lab.pca_multik_quality_2024 as multik
import mr_lab.pca_residual as residual
from mr_lab.pca_residual import MultiKPCAResidualEngine, PCAResidualEngine


def _rows(shock=0.03):
    values = np.random.default_rng(44).normal(0, 0.002, (100, 4))
    values[70, 0] += shock
    panel = panel_from_returns(values)
    return panel, tuple(MultiKPCAResidualEngine(config()).iter_with_quality(panel))


def test_one_eigh_per_fitted_row_feeds_all_k(monkeypatch):
    panel = panel_from_returns(process(80))
    calls = 0
    original = residual.np.linalg.eigh

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(residual.np.linalg, "eigh", counted)
    rows = tuple(MultiKPCAResidualEngine(config()).iter_with_quality(panel))
    timestamps = {row.observation.timestamp for row in rows}
    assert calls == len(process(80)) - config().pca_training_window
    assert {row.pca_k for row in rows} == {1, 2, 3}
    assert timestamps


def test_k2_is_exactly_legacy_equivalent():
    panel, rows = _rows()
    legacy = PCAResidualEngine(config()).run(panel)
    assert tuple(row.observation for row in rows if row.pca_k == 2) == legacy


def test_k_residuals_differ_and_event_payloads_are_event_only():
    _, rows = _rows()
    by_key = {}
    for row in rows:
        by_key.setdefault((row.observation.timestamp, row.observation.instrument), {})[
            row.pca_k
        ] = row
        assert (row.prior_residuals is not None) == row.observation.event_emitted
    assert any(
        len({values[k].observation.residual for k in (1, 2, 3)}) > 1
        for values in by_key.values()
    )


def test_future_mutation_preserves_prefix_and_prior_ar_payload():
    original = np.random.default_rng(44).normal(0, 0.002, (100, 4))
    original[70, 0] += 0.03
    changed = original.copy()
    changed[72:] += np.random.default_rng(12).normal(0, 0.2, changed[72:].shape)
    first = tuple(
        MultiKPCAResidualEngine(config()).iter_with_quality(
            panel_from_returns(original)
        )
    )
    second = tuple(
        MultiKPCAResidualEngine(config()).iter_with_quality(panel_from_returns(changed))
    )
    cutoff = first[0].observation.timestamp
    left = [r for r in first if r.observation.timestamp <= cutoff]
    right = [r for r in second if r.observation.timestamp <= cutoff]
    assert [r.observation for r in left] == [r.observation for r in right]
    for one, two in zip(left, right, strict=True):
        np.testing.assert_array_equal(one.prior_residuals, two.prior_residuals)


def test_k_specific_histories_and_shock_states_are_independent():
    engine = MultiKPCAResidualEngine(config())
    assert engine.components == (1, 2, 3)
    _, rows = _rows()
    states = {
        k: [r.observation.event_emitted for r in rows if r.pca_k == k]
        for k in (1, 2, 3)
    }
    assert all(states.values())
    assert len({tuple(value) for value in states.values()}) > 1


def test_explained_variance_and_eigengap_are_k_specific():
    _, rows = _rows()
    stamp = rows[0].observation.timestamp
    selected = {
        row.pca_k: row
        for row in rows
        if row.observation.timestamp == stamp and row.observation.instrument == "A"
    }
    assert selected[1].explained_variance_ratio < selected[2].explained_variance_ratio
    assert selected[2].explained_variance_ratio < selected[3].explained_variance_ratio
    assert all(selected[k].eigengap_ratio is not None for k in (1, 2, 3))


def test_half_life_bins_and_inclusive_gate_are_frozen():
    assert [multik.HALF_LIFE_BINS[i] for i in range(6)] == [
        "<=5",
        ">5_to_15",
        ">15_to_30",
        ">30_to_60",
        ">60",
        "NON_MEAN_REVERTING_OR_INVALID",
    ]
    values = [1.75]
    phi = exp(-log(2) / 60)
    for _ in range(255):
        values.append(0.25 + phi * values[-1])
    result = multik.estimate_ar1(values, 1e-12)
    assert result.residual_half_life_rows == pytest.approx(60)
    assert result.residual_ou_eligible


def test_identity_changes_with_half_life_bins():
    changed = copy.deepcopy(multik.COMMON_SPEC)
    changed["half_life_bins"] = ("changed",)
    assert multik._identity(changed) != multik.STUDY_IDENTITIES["overall"]


def test_exact_six_groups():
    assert multik.GROUPS == (
        "K1_RAW",
        "K1_OU_HL_LE_60",
        "K2_RAW",
        "K2_OU_HL_LE_60",
        "K3_RAW",
        "K3_OU_HL_LE_60",
    )


def test_atomic_gzip_shard_is_readable_hashed_and_manifested(tmp_path):
    root = tmp_path / "run"
    events = root / "events"
    events.mkdir(parents=True)
    manifest = {"shards": [], "finalized_output_bytes": 0, "run_complete": False}
    path = root / "manifest.json"
    multik._atomic_json(path, manifest)
    writer = multik.AtomicGzipShardWriter(events, manifest, path, event_limit=2)
    writer.write({"timestamp": "2024-01-01T00:00:00+00:00", "x": 1})
    assert not (events / "part-00000.jsonl.gz").exists()
    writer.write({"timestamp": "2024-01-01T00:01:00+00:00", "x": 2})
    shard = events / "part-00000.jsonl.gz"
    assert json.loads(path.read_text())["run_complete"] is False
    with gzip.open(shard, "rt") as source:
        assert len(source.readlines()) == 2
    assert manifest["shards"][0]["sha256"] == multik._sha_file(shard)
    assert not tuple(events.glob("*.tmp"))


def test_output_cap_preserves_final_shard_and_incomplete_manifest(tmp_path):
    root = tmp_path / "run"
    events = root / "events"
    events.mkdir(parents=True)
    manifest = {"shards": [], "finalized_output_bytes": 0, "run_complete": False}
    path = root / "manifest.json"
    multik._atomic_json(path, manifest)
    writer = multik.AtomicGzipShardWriter(
        events, manifest, path, event_limit=1, max_output_bytes=1
    )
    with pytest.raises(multik.MultiKStudyError, match="output bytes"):
        writer.write({"timestamp": "2024-01-01T00:00:00+00:00"})
    assert (events / "part-00000.jsonl.gz").exists()
    assert not json.loads(path.read_text())["run_complete"]


def test_postprocessor_rejects_incomplete_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text('{"run_complete": false}\n')
    with pytest.raises(multik.MultiKStudyError, match="incomplete"):
        multik.postprocess(tmp_path)


def test_overwrite_refusal_occurs_before_source_access(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(multik.MultiKStudyError, match="overwrite"):
        multik.run(tmp_path / "missing", output)


def test_source_hash_fails_closed(tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text("{}\n")
    with pytest.raises(ValueError, match="SHA256"):
        multik.verify_source_hash(source)
