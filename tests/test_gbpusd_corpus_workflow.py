from pathlib import Path


def test_gbpusd_uses_exact_bounded_generic_acquisition_and_publication() -> None:
    workflow = Path(".github/workflows/acquire-historical-sample.yml").read_text()

    assert "options: [EURUSD, GBPUSD, USDJPY, AUDUSD, AUDJPY]" in workflow
    assert "--year 2024 --month ${{ matrix.month }}" in workflow
    assert '--instrument "${{ inputs.instrument }}"' in workflow
    assert "--year 2024 --instrument \"${{ inputs.instrument }}\"" in workflow
    assert "verify-year" in workflow
    assert "dukascopy-${{ inputs.instrument }}-m1-bid-2024-full-year" in workflow
    assert "if-no-files-found: error" in workflow


def test_gbpusd_registry_stays_unpinned_until_real_artifact_exists() -> None:
    import json

    registry = json.loads(
        Path("configs/stage4a-2024-corpus-registry.json").read_text()
    )
    entry = registry["instruments"]["GBPUSD"]

    assert entry == {
        "assembled_dataset_id": None,
        "corpus_id": None,
        "instrument": "GBPUSD",
        "requested_end_date": "2024-12-31",
        "requested_start_date": "2024-01-01",
        "source_artifact_id": None,
        "source_artifact_name": "dukascopy-GBPUSD-m1-bid-2024-full-year",
        "source_workflow_run_id": None,
        "verification_status": "pending-acquisition",
    }
