from pathlib import Path


WORKFLOW = Path('.github/workflows/republish-eurusd-2024-corpus.yml')


def test_republication_workflow_is_manual_main_only_and_pins_legacy_source():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'workflow_dispatch:' in workflow
    assert 'pull_request:' not in workflow
    assert 'push:' not in workflow
    assert 'refs/heads/main' in workflow
    assert '31746932310' in workflow
    assert '9202006993' in workflow
    assert 'dukascopy-eurusd-m1-bid-2024-01-01-2024-12-31' in workflow


def test_republication_workflow_fails_closed_on_metadata_and_identity_guards():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    for required in (
        "run.get('id')",
        "run.get('conclusion')",
        "artifact.get('id')",
        "artifact.get('name')",
        "artifact.get('expired')",
        "artifact.get('workflow_run')",
        'sha256:ce14fc8557b11afd11f81c928370452eca4f023b1d9d72c296ff69d7cb073009',
        'sha256:af4c1e166e94c4a10e40586ef33319aa134531d93f84c59922b746aca559c916',
        "'requested_start_date': '2024-01-01'",
        "'requested_end_date': '2024-12-31'",
        "'instrument': 'EURUSD'",
    ):
        assert required in workflow


def test_republication_uses_current_offline_reconstruction_and_rebuilds_identity():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'load_offline_corpus' in workflow
    assert 'build_corpus_manifest' in workflow
    assert 'DailyPayload' in workflow
    assert 'rebuilt.corpus_id' in workflow
    assert 'offline reconstructed dataset identity mismatch' in workflow
    assert 'rebuilt corpus identity mismatch' in workflow


def test_republication_is_packaging_only_and_uses_current_canonical_name():
    workflow = WORKFLOW.read_text(encoding='utf-8')
    assert 'dukascopy-EURUSD-m1-bid-2024-full-year' in workflow
    assert 'if-no-files-found: error' in workflow
    assert 'gh api' in workflow
    assert 'actions/artifacts/$LEGACY_ARTIFACT_ID/zip' in workflow
    assert 'dukascopy.com' not in workflow
    assert 'acquire_range(' not in workflow
    assert 'acquire(' not in workflow
