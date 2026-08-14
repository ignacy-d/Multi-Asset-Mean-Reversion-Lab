import hashlib
import json
import lzma
import struct
from datetime import date
from pathlib import Path

import pytest

from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import (
    RangeAcquisitionError,
    build_corpus_manifest,
    load_offline_corpus,
)

RECORD = struct.Struct(">5if")


def payload(offset: int) -> bytes:
    return lzma.compress(
        RECORD.pack(0, 110000 + offset, 110001 + offset, 109999, 110002 + offset, 1.0)
    )


def corpus(path: Path) -> tuple[dict[str, object], tuple[date, date]]:
    days = date(2024, 1, 2), date(2024, 1, 3)
    raws = payload(0), payload(10)
    manifest = build_corpus_manifest(
        days[0],
        days[1],
        [DailyPayload(days[0], raws[0]), DailyPayload(days[1], raws[1])],
        [],
    )
    path.mkdir()
    for day, raw in zip(days, raws, strict=True):
        stem = path / f"EURUSD-{day.isoformat()}-M1-BID"
        stem.with_suffix(".bi5").write_bytes(raw)
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "requested_date": day.isoformat(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        )
    data = manifest.as_dict()
    (path / "corpus-manifest.json").write_text(json.dumps(data))
    return data, days


def test_offline_reconstruction_uses_manifest_dates_not_directory_order(
    tmp_path: Path,
) -> None:
    data, _ = corpus(tmp_path / "corpus")
    (tmp_path / "corpus" / "EURUSD-1999-01-01-M1-BID.bi5").write_bytes(b"junk")

    result = load_offline_corpus(tmp_path / "corpus")

    assert result.metadata.dataset_id == data["assembled_dataset_id"]
    assert [
        component.requested_day.isoformat() for component in result.manifest.components
    ] == data["successful_component_dates"]


def test_offline_missing_or_corrupt_component_fails(tmp_path: Path) -> None:
    _, days = corpus(tmp_path / "corpus")
    (tmp_path / "corpus" / f"EURUSD-{days[0].isoformat()}-M1-BID.bi5").unlink()
    with pytest.raises(RangeAcquisitionError, match="missing or corrupt"):
        load_offline_corpus(tmp_path / "corpus")


def test_offline_reconstructed_identity_must_match(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    data, _ = corpus(corpus_dir)
    data["assembled_dataset_id"] = "sha256:not-the-dataset"
    (corpus_dir / "corpus-manifest.json").write_text(json.dumps(data))
    with pytest.raises(RangeAcquisitionError, match="identity"):
        load_offline_corpus(corpus_dir)
