import hashlib
import json
import lzma
import struct
from datetime import date
from pathlib import Path

import pytest

import mr_lab.providers.dukascopy_range as range_module
from mr_lab.data import VolumeSemantics
from mr_lab.providers.dukascopy import AcquisitionError, ProviderNoData
from mr_lab.providers.dukascopy_bi5 import Bi5ParseError
from mr_lab.providers.dukascopy_multiday import DailyPayload, assemble_daily_payloads
from mr_lab.providers.dukascopy_range import (
    DISCOVERY_END,
    DISCOVERY_START,
    HOLDOUT_END,
    HOLDOUT_START,
    MAX_CALENDAR_DAYS,
    RangeAcquisitionError,
    acquire_range,
    build_corpus_manifest,
    enumerate_dates,
)

RECORD = struct.Struct(">5if")


def payload(day_offset: int = 0, *, volume: float = 1.0) -> bytes:
    decoded = b"".join(
        RECORD.pack(
            minute * 60,
            110_000 + day_offset + minute,
            110_001 + day_offset + minute,
            109_999 + day_offset + minute,
            110_002 + day_offset + minute,
            0.0 if minute == 0 else volume,
        )
        for minute in range(5)
    )
    return lzma.compress(decoded, format=lzma.FORMAT_ALONE)


def writer(responses: dict[date, bytes | Exception], calls: list[date]):
    def acquire_day(
        output_dir: Path, day: date, *, timeout: float, retries: int
    ) -> tuple[Path, Path]:
        del timeout, retries
        calls.append(day)
        response = responses[day]
        if isinstance(response, Exception):
            raise response
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / f"EURUSD-{day.isoformat()}-M1-BID.bi5"
        provenance = output_dir / f"EURUSD-{day.isoformat()}-M1-BID.json"
        raw.write_bytes(response)
        provenance.write_text(
            json.dumps(
                {
                    "requested_date": day.isoformat(),
                    "sha256": hashlib.sha256(response).hexdigest(),
                }
            )
        )
        return raw, provenance

    return acquire_day


def test_research_split_is_explicit_and_disjoint() -> None:
    assert (
        date(2024, 1, 1),
        date(2024, 12, 31),
    ) == (DISCOVERY_START, DISCOVERY_END)
    assert (
        date(2025, 1, 1),
        date(2025, 12, 31),
    ) == (HOLDOUT_START, HOLDOUT_END)
    assert DISCOVERY_END < HOLDOUT_START


def test_inclusive_enumeration_is_ascending_and_deterministic() -> None:
    expected = (date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1))

    assert enumerate_dates(expected[0], expected[-1]) == expected
    assert enumerate_dates(expected[0], expected[-1]) == expected


def test_reverse_range_and_safety_limit_are_rejected() -> None:
    with pytest.raises(RangeAcquisitionError, match="on or before"):
        enumerate_dates(date(2024, 1, 2), date(2024, 1, 1))
    with pytest.raises(RangeAcquisitionError, match="safety limit"):
        enumerate_dates(
            date(2024, 1, 1),
            date(2024, 1, 1) + range_module.timedelta(days=MAX_CALENDAR_DAYS),
        )


def test_multiple_days_and_confirmed_absence_are_recorded_without_filling(
    tmp_path: Path,
) -> None:
    start = date(2024, 1, 5)
    absent = date(2024, 1, 6)
    end = date(2024, 1, 7)
    first_payload, final_payload = payload(), payload(100)
    calls: list[date] = []
    responses = {
        start: first_payload,
        absent: ProviderNoData("explicit 404"),
        end: final_payload,
    }

    result = acquire_range(
        tmp_path,
        start,
        end,
        acquire_day=writer(responses, calls),
        delay_seconds=0,
    )
    audit = result.manifest.as_dict()

    assert calls == [start, absent, end]
    assert audit["successful_component_dates"] == ["2024-01-05", "2024-01-07"]
    assert audit["confirmed_absent_dates"] == ["2024-01-06"]
    assert audit["successful_component_count"] == 2
    assert audit["absent_component_count"] == 1
    assert len(result.manifest.dataset.bars) == 10
    assert {bar.open_time.date() for bar in result.manifest.dataset.bars} == {
        start,
        end,
    }
    assert result.manifest.dataset.manifest.gap_count == 1
    assert result.manifest.dataset.bars[0].volume == 0.0
    assert all(
        bar.volume_semantics is VolumeSemantics.QUOTE_ACTIVITY
        for bar in result.manifest.dataset.bars
    )
    assert json.loads(result.manifest_path.read_text()) == audit
    for component, raw_payload in zip(
        audit["components"], (first_payload, final_payload), strict=True
    ):
        assert component["raw_sha256"] == hashlib.sha256(raw_payload).hexdigest()
        assert component["daily_dataset_id"].startswith("sha256:")


def test_range_progress_preserves_absence_and_uses_default_polite_delay(
    tmp_path: Path,
) -> None:
    start, absent, end = date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7)
    messages: list[str] = []
    sleeps: list[float] = []

    result = acquire_range(
        tmp_path,
        start,
        end,
        acquire_day=writer(
            {start: payload(), absent: ProviderNoData("404"), end: payload(100)}, []
        ),
        sleeper=sleeps.append,
        logger=messages.append,
    )

    assert result.manifest.confirmed_absent_dates == (absent,)
    assert sleeps == [1.0, 1.0]
    assert messages == [
        "[1/3] acquiring 2024-01-05",
        "[2/3] acquiring 2024-01-06",
        "[2/3] absent 2024-01-06",
        "[3/3] acquiring 2024-01-07",
    ]


def test_retry_history_does_not_change_daily_dataset_or_corpus_identity(
    tmp_path: Path,
) -> None:
    day = date(2024, 1, 2)
    raw_payload = payload()
    direct_dir, retried_dir = tmp_path / "direct", tmp_path / "retried"
    direct_raw, _ = range_module.acquire(
        direct_dir,
        day,
        getter=lambda _url, _timeout: (200, raw_payload),
        sleeper=lambda _seconds: None,
        logger=lambda _message: None,
    )
    responses = iter([(503, b""), (200, raw_payload)])
    retried_raw, _ = range_module.acquire(
        retried_dir,
        day,
        getter=lambda _url, _timeout: next(responses),
        sleeper=lambda _seconds: None,
        logger=lambda _message: None,
    )

    direct = build_corpus_manifest(
        day, day, [DailyPayload(day, direct_raw.read_bytes())], []
    )
    retried = build_corpus_manifest(
        day, day, [DailyPayload(day, retried_raw.read_bytes())], []
    )

    assert direct.dataset.metadata.dataset_id == retried.dataset.metadata.dataset_id
    assert direct.corpus_id == retried.corpus_id


def test_network_or_server_failure_is_not_converted_to_absence(tmp_path: Path) -> None:
    day = date(2024, 1, 2)
    failure = AcquisitionError("HTTP 503")

    with pytest.raises(AcquisitionError, match="503"):
        acquire_range(
            tmp_path,
            day,
            day,
            acquire_day=writer({day: failure}, []),
            delay_seconds=0,
        )


def test_malformed_payload_fails_canonicalization(tmp_path: Path) -> None:
    day = date(2024, 1, 2)

    with pytest.raises(Bi5ParseError, match="not valid LZMA"):
        acquire_range(
            tmp_path,
            day,
            day,
            acquire_day=writer({day: b"malformed"}, []),
            delay_seconds=0,
        )


def test_manifest_reuses_stage_1b_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    day = date(2024, 1, 2)
    daily = DailyPayload(day, payload())
    expected = assemble_daily_payloads([daily])
    seen = []

    def assemble(inputs):
        seen.append(tuple(inputs))
        return expected

    monkeypatch.setattr(range_module, "assemble_daily_payloads", assemble)

    manifest = build_corpus_manifest(day, day, [daily], [])

    assert manifest.dataset is expected
    assert seen == [(daily,)]


def test_manifest_json_identity_and_input_order_are_deterministic() -> None:
    first_day, second_day = date(2024, 1, 2), date(2024, 1, 3)
    first = DailyPayload(first_day, payload())
    second = DailyPayload(second_day, payload(100))

    forward = build_corpus_manifest(first_day, second_day, [first, second], [])
    reverse = build_corpus_manifest(first_day, second_day, [second, first], [])

    assert forward.to_json() == reverse.to_json()
    assert forward.corpus_id == reverse.corpus_id
    assert (
        json.dumps(forward.as_dict(), sort_keys=True, separators=(",", ":"))
        == forward.to_json()
    )
    assert "retrieved_at" not in forward.to_json()
    assert "session_spec" not in forward.to_json()


def test_payload_or_requested_range_change_changes_corpus_identity() -> None:
    first_day, second_day = date(2024, 1, 2), date(2024, 1, 3)
    original = DailyPayload(first_day, payload())
    changed = DailyPayload(first_day, payload(1))
    first = build_corpus_manifest(first_day, first_day, [original], [])
    modified = build_corpus_manifest(first_day, first_day, [changed], [])
    extended = build_corpus_manifest(first_day, second_day, [original], [second_day])

    assert first.corpus_id != modified.corpus_id
    assert first.corpus_id != extended.corpus_id
    assert first.dataset.metadata.dataset_id == extended.dataset.metadata.dataset_id
