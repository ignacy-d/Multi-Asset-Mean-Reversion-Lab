from pathlib import Path


def test_gbpusd_uses_exact_bounded_generic_acquisition_and_publication() -> None:
    workflow = Path(".github/workflows/acquire-historical-sample.yml").read_text()

    assert (
        "options: [EURUSD, GBPUSD, USDJPY, AUDUSD, AUDJPY, "
        "USDCAD, USDCHF, NZDUSD, EURGBP]" in workflow
    )
    assert "--year 2024 --month ${{ matrix.month }}" in workflow
    assert '--instrument "${{ inputs.instrument }}"' in workflow
    assert '--year 2024 --instrument "${{ inputs.instrument }}"' in workflow
    assert "verify-year" in workflow
    assert "dukascopy-${{ inputs.instrument }}-m1-bid-2024-full-year" in workflow
    assert "if-no-files-found: error" in workflow


def test_gbpusd_registry_records_verified_local_checkpointed_corpus() -> None:
    import json

    registry = json.loads(Path("configs/stage4a-2024-corpus-registry.json").read_text())
    entry = registry["instruments"]["GBPUSD"]

    assert entry == {
        "assembled_dataset_id": (
            "sha256:1333e8d1b378b2b08f88a857a07f017a61a744b1787c8db68f629e6c3c0fe70a"
        ),
        "corpus_id": (
            "sha256:16d58bfa2de1a5a771f7a0f9034f4e05a0c1b900002276e4c6e28d85ac4628cf"
        ),
        "instrument": "GBPUSD",
        "requested_end_date": "2024-12-31",
        "requested_start_date": "2024-01-01",
        "source_acquisition_commit_sha": "94a43c36e1a5fa1b3b9e98890dc2d62a156592c4",
        "source_artifact_id": None,
        "source_artifact_name": "dukascopy-GBPUSD-m1-bid-2024-full-year",
        "source_mode": "local-checkpointed",
        "source_workflow_run_id": None,
        "verification_status": "verified",
    }
