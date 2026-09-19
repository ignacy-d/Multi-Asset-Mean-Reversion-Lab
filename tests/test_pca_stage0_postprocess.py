import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import mr_lab.pca_stage0_postprocess as post


def event(month, instrument="EURUSD", value=1.0, *, complete=True, process="p"):
    row = {
        "timestamp": f"2024-{month}-02T00:00:00+00:00",
        "instrument": instrument,
        "entry_complete": True,
        "process_identity": process,
        "activity_policy": "policy",
    }
    for horizon in post.HORIZONS:
        row[f"h{horizon}_complete"] = complete if horizon == 15 else True
        row[f"h{horizon}_signed_bps_return"] = (
            value if horizon != 15 or complete else None
        )
    return row


def write_events(path: Path, events) -> bytes:
    raw = b"".join(json.dumps(item, sort_keys=True).encode() + b"\n" for item in events)
    path.write_bytes(raw)
    return raw


def test_normal_processing_is_deterministic_and_records_provenance(tmp_path):
    source = tmp_path / "events.jsonl"
    raw = write_events(
        source,
        [event("01", "EURUSD", 2), event("02", "GBPUSD", -1)],
    )
    first = post.build_summary(*post.read_events(source))
    second = post.build_summary(*post.read_events(source))
    assert first == second
    assert first["event_count"] == 2
    assert first["source_events_sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert first["process_identity"] == "p"
    assert first["activity_policy"] == "policy"
    assert first["primary_promotion_status"] == post.BLOCKER
    assert first["executable_entry_fraction"] == 1.0
    assert first["horizon_statistics"]["h15"]["n"] == 2


def test_malformed_json_is_rejected(tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text("{bad}\n")
    with pytest.raises(post.PostprocessError, match="malformed JSON"):
        post.read_events(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("process_identity", "different", "inconsistent process_identity"),
        ("activity_policy", "different", "inconsistent activity_policy"),
    ],
)
def test_inconsistent_artifact_identity_is_rejected(tmp_path, field, value, message):
    rows = [event("01"), event("02")]
    rows[1][field] = value
    source = tmp_path / "events.jsonl"
    write_events(source, rows)
    with pytest.raises(post.PostprocessError, match=message):
        post.read_events(source)


def test_missing_required_field_is_rejected(tmp_path):
    row = event("01")
    del row["h30_complete"]
    source = tmp_path / "events.jsonl"
    write_events(source, [row])
    with pytest.raises(post.PostprocessError, match="missing required fields"):
        post.read_events(source)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "1"])
def test_non_finite_or_non_numeric_complete_outcome_is_rejected(tmp_path, value):
    row = event("01")
    row["h15_signed_bps_return"] = value
    source = tmp_path / "events.jsonl"
    write_events(source, [row])
    with pytest.raises(post.PostprocessError, match="must be finite"):
        post.read_events(source)


def test_incomplete_h15_is_excluded_from_distribution_and_bootstrap(tmp_path):
    source = tmp_path / "events.jsonl"
    write_events(source, [event("01", value=3), event("02", value=999, complete=False)])
    summary = post.build_summary(*post.read_events(source))
    assert summary["horizon_statistics"]["h15"]["n"] == 1
    bootstrap = summary["h15_calendar_month_block_bootstrap"]
    assert bootstrap["observed_calendar_month_blocks"] == 1
    assert bootstrap["bootstrap_mean_bps"] == 3


def test_bootstrap_is_seeded_and_uses_whole_month_blocks(tmp_path):
    source = tmp_path / "events.jsonl"
    rows = [
        event("01", value=0),
        event("01", value=2),
        event("02", value=10),
    ]
    write_events(source, rows)
    events, digest = post.read_events(source)
    first = post.build_summary(events, digest)["h15_calendar_month_block_bootstrap"]
    second = post.build_summary(events, digest)["h15_calendar_month_block_bootstrap"]
    assert first == second
    rng = np.random.default_rng(post.BOOTSTRAP_SEED)
    expected = []
    blocks = ([0.0, 2.0], [10.0])
    for _ in range(post.BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, 2, size=2)
        expected.append(np.mean([x for index in chosen for x in blocks[index]]))
    assert first["bootstrap_mean_bps"] == pytest.approx(float(np.mean(expected)))
    assert first["observed_calendar_month_blocks"] == 2
    assert first["replicates"] == 10_000
    assert first["seed"] == 20240918


def test_concentration_and_signed_contribution_arithmetic(tmp_path):
    source = tmp_path / "events.jsonl"
    rows = [
        event("01", "A", 3),
        event("02", "A", -1),
        event("04", "B", -2),
        event("07", "C", 0),
    ]
    write_events(source, rows)
    summary = post.build_summary(*post.read_events(source))
    assert summary["event_count_concentration"] == {
        "maximum_instrument_fraction": 0.5,
        "maximum_calendar_quarter_fraction": 0.5,
    }
    assert summary["signed_h15_contribution_bps_by_instrument"] == {
        "A": 2.0,
        "B": -2.0,
        "C": 0.0,
    }
    assert summary["absolute_signed_contribution_concentration"]["value"] == 0.5
    assert summary["instruments_with_positive_mean_complete_h15"] == 1


def test_sha_changes_with_exact_source_bytes(tmp_path):
    source = tmp_path / "events.jsonl"
    write_events(source, [event("01")])
    first = post.read_events(source)[1]
    source.write_bytes(source.read_bytes().rstrip(b"\n"))
    second = post.read_events(source)[1]
    assert first != second


def test_output_refuses_overwrite_and_never_mutates_source(tmp_path):
    source = tmp_path / "events.jsonl"
    original = write_events(source, [event("01")])
    output = tmp_path / "recovered"
    result = post.postprocess(source, output)
    assert result == output / "summary.json"
    assert source.read_bytes() == original
    with pytest.raises(post.PostprocessError, match="refusing to overwrite"):
        post.postprocess(source, output)
    assert source.read_bytes() == original


def test_processing_requires_only_explicit_events_file(tmp_path, monkeypatch):
    source = tmp_path / "chosen.jsonl"
    write_events(source, [event("01")])

    def forbidden_discovery(*_args, **_kwargs):
        raise AssertionError("filesystem discovery is forbidden")

    monkeypatch.setattr(Path, "glob", forbidden_discovery)
    monkeypatch.setattr(Path, "rglob", forbidden_discovery)
    post.postprocess(source, tmp_path / "out")
