import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import mr_lab.stage4a_runner as runner
from mr_lab.bollinger_benchmark import BollingerStrategySpec
from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.research import Direction
from mr_lab.stage4a import frozen_signal_from_bollinger, frozen_signal_from_vwap
from mr_lab.stage4a_reporting import STAGE4A_REPORT_SCHEMA_VERSION
from mr_lab.stage4a_runner import (
    REGISTRY_SCHEMA_VERSION,
    Stage4ARunnerError,
    load_corpus_registry,
    read_and_validate_manifest,
    select_verified_registry_entries,
    validate_registry_entry,
    write_execution_audit,
)
from mr_lab.vwap_benchmark import VwapStrategySpec
from mr_lab.vwap_m1_robustness import CANONICAL_M1, VwapRobustnessStrategySpec


def manifest(tmp_path: Path, **changes) -> Path:
    value = {
        "instrument": "EURUSD",
        "requested_start_date": "2024-01-01",
        "requested_end_date": "2024-12-31",
        "successful_component_dates": ["2024-01-02"],
        "confirmed_absent_dates": ["2024-01-01"],
        "components": [{"requested_day": "2024-01-02"}],
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
    }
    value.update(changes)
    (tmp_path / "corpus-manifest.json").write_text(json.dumps(value))
    return tmp_path


def verified_entry(instrument="EURUSD"):
    return {
        "instrument": instrument,
        "source_workflow_run_id": 123,
        "source_artifact_id": 456,
        "source_artifact_name": f"dukascopy-{instrument}-m1-bid-2024-full-year",
        "corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "requested_start_date": "2024-01-01",
        "requested_end_date": "2024-12-31",
        "verification_status": "verified",
    }


def test_manifest_identity_is_preserved(tmp_path):
    result = read_and_validate_manifest(manifest(tmp_path), "EURUSD")
    assert (result["corpus_id"], result["assembled_dataset_id"]) == (
        "corpus",
        "dataset",
    )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("requested_start_date", "2025-01-01"),
        ("requested_end_date", "2025-01-01"),
        ("successful_component_dates", ["2025-01-01"]),
        ("confirmed_absent_dates", ["2025-01-01"]),
        ("components", [{"requested_day": "2025-01-01"}]),
        ("components", ["2024-01-01"]),
        ("components", [{}]),
        ("components", [{"requested_day": "bad-date"}]),
        ("successful_component_dates", [12]),
        ("successful_component_dates", ["bad-date"]),
        ("confirmed_absent_dates", [12]),
        ("confirmed_absent_dates", ["bad-date"]),
        ("instrument", "GBPUSD"),
    ],
)
def test_manifest_guard_rejects_every_malformed_or_out_of_scope_date(
    tmp_path, change, value
):
    with pytest.raises(Stage4ARunnerError):
        read_and_validate_manifest(manifest(tmp_path, **{change: value}), "EURUSD")


def test_registry_exact_pins_and_status_model():
    registry = load_corpus_registry(Path("configs/stage4a-2024-corpus-registry.json"))
    entries = registry["instruments"]
    assert entries["EURUSD"]["verification_status"] == "pending-reviewer-verification"
    assert entries["GBPUSD"]["verification_status"] == "pending-acquisition"
    expected = {
        "USDJPY": (
            32122032975,
            9326494647,
            "sha256:aa43bffa74ca61a15455cbd2bd74c4cb45993129a057f2f781a30647b597a4e9",
            "sha256:d6b11d469eff71d81af5c31d7d29638287d864fd9221451ffb713204b9109571",
        ),
        "AUDUSD": (
            32150777156,
            9354746762,
            "sha256:5e12385b0c952b24c3c4e3ee5023ed6069b42adb361ab57f4f39579f8b450d9d",
            "sha256:e0c9d7381404548df170215293ad6e63866c55a37d47197f21e852c36aa40015",
        ),
        "AUDJPY": (
            32150801479,
            9360642306,
            "sha256:209edd9658147145cb5021a7132e499ecfdb530f786766e347216c50730b45fd",
            "sha256:76a6dc298749fdf9c3b605218ad8b17bdbd540e5a065566b76dc6d8c6563dad8",
        ),
    }
    for name, values in expected.items():
        entry = entries[name]
        assert entry["verification_status"] == "verified"
        assert (
            entry["source_workflow_run_id"],
            entry["source_artifact_id"],
            entry["corpus_id"],
            entry["assembled_dataset_id"],
        ) == values


@pytest.mark.parametrize("schema", [None, "stage-4a-2024-corpus-registry-v2"])
def test_registry_rejects_missing_or_unsupported_schema(tmp_path, schema):
    value = json.loads(Path("configs/stage4a-2024-corpus-registry.json").read_text())
    if schema is None:
        value.pop("registry_schema_version")
    else:
        value["registry_schema_version"] = schema
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(value))
    with pytest.raises(Stage4ARunnerError):
        load_corpus_registry(path)


def test_verified_entry_is_strict_and_manifest_bound():
    entry = verified_entry()
    validated = validate_registry_entry("EURUSD", entry, manifest={**entry})
    assert validated == entry
    for field, bad in (
        ("source_artifact_id", 0),
        ("source_workflow_run_id", True),
        ("corpus_id", ""),
        ("instrument", "GBPUSD"),
    ):
        changed = {**entry, field: bad}
        with pytest.raises(Stage4ARunnerError):
            validate_registry_entry("EURUSD", changed)
    with pytest.raises(Stage4ARunnerError):
        validate_registry_entry(
            "EURUSD", entry, manifest={**entry, "corpus_id": "other"}
        )


def test_partial_registry_fails_closed_for_all_but_verified_single_proceeds():
    registry = load_corpus_registry(Path("configs/stage4a-2024-corpus-registry.json"))
    assert (
        select_verified_registry_entries(registry, "USDJPY")[0]["instrument"]
        == "USDJPY"
    )
    with pytest.raises(Stage4ARunnerError):
        select_verified_registry_entries(registry, "ALL")
    with pytest.raises(Stage4ARunnerError):
        select_verified_registry_entries(registry, "GBPUSD")


def test_signal_grid_uses_family_constants_identities_and_contexts(monkeypatch):
    calls = []
    overlap = SimpleNamespace(
        observation=SimpleNamespace(
            sessions=SimpleNamespace(active_sessions=("london", "new_york"))
        )
    )
    monkeypatch.setattr(
        runner, "build_research_observations", lambda *args: (SimpleNamespace(),)
    )
    monkeypatch.setattr(runner, "resample_bars", lambda *args: SimpleNamespace(bars=()))
    monkeypatch.setattr(runner, "build_vwap_features", lambda *args: (object(),))
    monkeypatch.setattr(
        runner, "build_canonical_m1_vwap_features", lambda *args: (object(),)
    )
    monkeypatch.setattr(runner, "build_bollinger_features", lambda *args: (overlap,))

    def vwap(_feature, **kwargs):
        calls.append(
            (
                kwargs["benchmark_family"],
                kwargs["threshold"],
                kwargs["strategy_spec_id"],
                None,
            )
        )
        return object()

    def bollinger(_feature, **kwargs):
        calls.append(
            (
                "bollinger",
                kwargs["threshold"],
                kwargs["strategy_spec_id"],
                kwargs["session"],
            )
        )
        return object()

    monkeypatch.setattr(runner, "frozen_signal_from_vwap", vwap)
    monkeypatch.setattr(runner, "frozen_signal_from_bollinger", bollinger)
    signals = runner.assemble_frozen_signals(
        SimpleNamespace(bars=()), {"corpus_id": "c", "assembled_dataset_id": "d"}
    )
    assert len(signals) == len(calls)
    assert {family for family, *_ in calls} == {
        "vwap",
        "vwap-canonical-m1",
        "bollinger",
    }
    assert {threshold for _, threshold, *_ in calls} == {1.0, 1.5, 2.0, 2.5}
    assert len(calls) == 3 * (2 * 4 * 2 + 2 * 4 * 3)
    for family, threshold, strategy_id, context in calls:
        lookbacks = (20, 40)
        expected = {
            "vwap": {
                VwapStrategySpec(x, threshold).strategy_spec_id for x in lookbacks
            },
            "vwap-canonical-m1": {
                VwapRobustnessStrategySpec(x, threshold, CANONICAL_M1).strategy_spec_id
                for x in lookbacks
            },
            "bollinger": {
                BollingerStrategySpec(x, threshold).strategy_spec_id for x in lookbacks
            },
        }
        assert strategy_id in expected[family]
        if family == "bollinger":
            assert context in (None, "london", "new_york")


def test_adapters_preserve_strict_stage3b_eligibility_and_direction_parity():
    opened = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bar = Bar(
        "EURUSD",
        Timeframe("5m"),
        opened,
        opened + timedelta(minutes=5),
        opened + timedelta(minutes=5),
        1.0,
        1.01,
        0.99,
        1.0,
        PriceBasis.BID,
        1.0,
        VolumeSemantics.QUOTE_ACTIVITY,
    )
    observation = SimpleNamespace(
        bar=bar, available_at=bar.available_at, is_research_active=True
    )

    def vwap(z):
        return SimpleNamespace(
            observation=observation,
            anchor_session="london",
            volatility_lookback=20,
            price=1.0,
            vwap=1.1 if z < 0 else 0.9,
            vwap_deviation_z=z,
        )

    common = {
        "threshold": 1.0,
        "benchmark_family": "vwap",
        "source_corpus_id": "corpus",
        "assembled_dataset_id": "dataset",
        "strategy_spec_id": VwapStrategySpec(20, 1.0).strategy_spec_id,
    }
    assert frozen_signal_from_vwap(vwap(1.0), **common) is None
    assert frozen_signal_from_vwap(vwap(-1.0), **common) is None
    assert frozen_signal_from_vwap(vwap(-1.01), **common).direction is Direction.LONG
    assert frozen_signal_from_vwap(vwap(1.01), **common).direction is Direction.SHORT

    bollinger = SimpleNamespace(
        observation=observation,
        rolling_lookback=20,
        middle=1.1,
        bollinger_z=-1.01,
    )
    signal = frozen_signal_from_bollinger(
        bollinger,
        threshold=1.0,
        session="london",
        source_corpus_id="corpus",
        assembled_dataset_id="dataset",
        strategy_spec_id=BollingerStrategySpec(20, 1.0).strategy_spec_id,
    )
    assert signal.direction is Direction.LONG and signal.session == "london"


def test_runner_batches_diagnostics_and_reporting_once(monkeypatch, tmp_path):
    calls = {"load": 0, "diagnose": 0, "report": 0}
    data = SimpleNamespace(
        metadata=SimpleNamespace(instrument="EURUSD", dataset_id="dataset"),
        bars=(object(),),
    )
    monkeypatch.setattr(
        runner,
        "read_and_validate_manifest",
        lambda *_: {
            "instrument": "EURUSD",
            "corpus_id": "corpus",
            "assembled_dataset_id": "dataset",
        },
    )
    monkeypatch.setattr(
        runner,
        "load_offline_corpus",
        lambda *_: calls.__setitem__("load", calls["load"] + 1) or data,
    )
    monkeypatch.setattr(runner, "assemble_frozen_signals", lambda *_: (1, 2, 3))
    monkeypatch.setattr(
        runner,
        "diagnose_events",
        lambda signals, bars: calls.__setitem__("diagnose", calls["diagnose"] + 1)
        or tuple(signals),
    )
    monkeypatch.setattr(
        runner,
        "write_stage4a_outputs",
        lambda events, output: calls.__setitem__("report", calls["report"] + 1) or {},
    )
    runner.run_stage4a_2024(tmp_path, tmp_path, "EURUSD")
    assert calls == {"load": 1, "diagnose": 1, "report": 1}


def test_execution_audit_is_bound_hashed_deterministic_and_exact_inventory(tmp_path):
    payloads = {"events.jsonl": b"e\n", "matrix.csv": b"m\n", "report.md": b"r\n"}
    for name, content in payloads.items():
        (tmp_path / name).write_bytes(content)
    hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()
    }
    summary = {
        "hashes": hashes,
        "complete_path_total": 2,
        "incomplete_path_total": 1,
        "event_total": 3,
        "matrix_row_count": 4,
        "stage4a_methodology_ids": ["method"],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    entry = verified_entry()
    manifest_value = {**entry}
    first = write_execution_audit(
        tmp_path,
        entry,
        "abc",
        registry_schema_version=REGISTRY_SCHEMA_VERSION,
        manifest=manifest_value,
    ).read_bytes()
    second = write_execution_audit(
        tmp_path,
        entry,
        "abc",
        registry_schema_version=REGISTRY_SCHEMA_VERSION,
        manifest=manifest_value,
    ).read_bytes()
    audit = json.loads(first)
    assert first == second
    assert set(path.name for path in tmp_path.iterdir()) == {
        *runner.RESEARCH_FILES,
        "execution-audit.json",
    }
    assert audit["source_workflow_run_id"] == 123 and audit["source_artifact_id"] == 456
    assert audit["reporting_schema_version"] == STAGE4A_REPORT_SCHEMA_VERSION
    assert audit["registry_schema_version"] == REGISTRY_SCHEMA_VERSION
    assert audit["total_event_count"] == 3 and audit["matrix_row_count"] == 4
    assert set(audit["research_output_sha256"]) == set(runner.RESEARCH_FILES)
    assert "timestamp" not in json.dumps(audit).lower()
    with pytest.raises(Stage4ARunnerError):
        write_execution_audit(
            tmp_path,
            entry,
            "abc",
            registry_schema_version=REGISTRY_SCHEMA_VERSION,
            manifest={**manifest_value, "corpus_id": "wrong"},
        )
    (tmp_path / "events.jsonl").write_text("tampered")
    with pytest.raises(Stage4ARunnerError):
        write_execution_audit(
            tmp_path,
            entry,
            "abc",
            registry_schema_version=REGISTRY_SCHEMA_VERSION,
            manifest=manifest_value,
        )


def test_workflow_preflight_metadata_outputs_and_env_step_regression():
    workflow = Path(".github/workflows/run-stage-4a-real-2024.yml").read_text()
    assert "options: [ALL, EURUSD, GBPUSD, USDJPY, AUDUSD, AUDJPY]" in workflow
    assert workflow.count("inputs:") == 1 and "source_workflow_run_id:" not in workflow
    assert (
        "registry-preflight:" in workflow
        and "select_verified_registry_entries" in workflow
    )
    assert (
        "refs/heads/main" in workflow
        and "fail-fast: false" in workflow
        and "max-parallel: 2" in workflow
    )
    for check in (
        "r.get('id')",
        "r.get('conclusion')",
        "a.get('id')",
        "a.get('name')",
        "a.get('expired')",
        "workflow_run",
    ):
        assert check in workflow
    assert "uv run mr-lab-stage4a" in workflow
    for name in (*runner.RESEARCH_FILES, "execution-audit.json"):
        assert f"result/{name}" in workflow
    resolve = workflow.index("name: Resolve the selected validated registry entry")
    consume = workflow.index("name: Reverify and download the pinned immutable corpus")
    assert resolve < consume
    assert workflow.index('f.write(f"RUN_ID=', resolve, consume) < consume
    assert (
        workflow.index('gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_ID"')
        > consume
    )
