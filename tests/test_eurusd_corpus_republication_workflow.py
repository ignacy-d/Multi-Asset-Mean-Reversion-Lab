from pathlib import Path

WORKFLOW = Path(".github/workflows/republish-eurusd-2024-corpus.yml")


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_is_manual_only_and_main_only() -> None:
    text = workflow_text()
    trigger = text.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    events = ("push:", "pull_request:", "schedule:")
    assert not any(event in trigger for event in events)
    assert 'test "$GITHUB_REF" = refs/heads/main' in text


def test_exact_legacy_source_and_metadata_guards_are_pinned() -> None:
    text = workflow_text()
    assert 'LEGACY_RUN_ID: "31746932310"' in text
    assert 'LEGACY_ARTIFACT_ID: "9202006993"' in text
    assert "dukascopy-eurusd-m1-bid-2024-01-01-2024-12-31" in text
    assert 'run.get("conclusion") != "success"' in text
    assert 'artifact.get("expired") is not False' in text
    assert 'artifact.get("workflow_run") or {}).get("id") != run["id"]' in text


def test_exact_frozen_manifest_contract_and_offline_verification_are_required() -> None:
    text = workflow_text()
    assert '"requested_start_date": "2024-01-01"' in text
    assert '"requested_end_date": "2024-12-31"' in text
    corpus_id = (
        "sha256:ce14fc8557b11afd11f81c928370452eca4f023b1d9d72c296ff69d7cb073009"
    )
    dataset_id = (
        "sha256:af4c1e166e94c4a10e40586ef33319aa134531d93f84c59922b746aca559c916"
    )
    assert corpus_id in text
    assert dataset_id in text
    assert "load_offline_corpus(corpus_dir)" in text
    assert "CorpusManifest(" in text
    assert 'reconstructed.corpus_id != expected["corpus_id"]' in text


def test_republication_name_retention_and_no_acquisition_path() -> None:
    text = workflow_text()
    canonical_block = text.split(
        "name: dukascopy-EURUSD-m1-bid-2024-full-year", 1
    )[1].split("- name: Upload deterministic migration audit", 1)[0]
    assert "if-no-files-found: error" in canonical_block
    assert "retention-days: 90" in canonical_block
    assert "acquire(" not in text
    assert "dukascopy.com" not in text
