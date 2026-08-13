import lzma
import struct
from datetime import UTC, date, datetime

from mr_lab.providers.dukascopy_multiday import DailyPayload, assemble_daily_payloads
from mr_lab.sessions import DEFAULT_SESSION_SPEC, classify_bars

RECORD = struct.Struct(">5if")


def payload(offsets: tuple[int, ...]) -> bytes:
    decoded = b"".join(
        RECORD.pack(offset, 110_000, 110_001, 109_999, 110_002, 1.0)
        for offset in offsets
    )
    return lzma.compress(decoded, format=lzma.FORMAT_ALONE)


def test_classifies_stage_1b_bars_across_utc_day_and_local_date_boundary() -> None:
    dataset = assemble_daily_payloads(
        [
            DailyPayload(date(2024, 1, 2), payload((23 * 3600 + 59 * 60,))),
            DailyPayload(date(2024, 1, 3), payload((3600, 5 * 3600))),
        ]
    )

    labels = classify_bars(dataset.bars, DEFAULT_SESSION_SPEC)

    assert [label.timestamp for label in labels] == [
        datetime(2024, 1, 2, 23, 59, tzinfo=UTC),
        datetime(2024, 1, 3, 1, tzinfo=UTC),
        datetime(2024, 1, 3, 5, tzinfo=UTC),
    ]
    assert "asian_kz_20_00_et" not in labels[0].active_named_windows
    assert "asian_kz_20_00_et" in labels[1].active_named_windows
    assert "asian_kz_20_00_et" not in labels[2].active_named_windows
    assert dataset.metadata.dataset_id != DEFAULT_SESSION_SPEC.session_spec_id
