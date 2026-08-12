from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from mr_lab.data import (
    Bar,
    DatasetValidationError,
    PriceBasis,
    Timeframe,
    VolumeSemantics,
    available_bars,
    observation_identity,
    validate_dataset,
)

START = datetime(2025, 1, 2, 10, 0, tzinfo=UTC)


def make_bar(index: int = 0, **overrides: object) -> Bar:
    open_time = START + index * timedelta(minutes=5)
    values = {
        "instrument": "EURUSD",
        "timeframe": Timeframe("5m"),
        "open_time": open_time,
        "close_time": open_time + timedelta(minutes=5),
        "available_at": open_time + timedelta(minutes=5),
        "open": 10.0 + index,
        "high": 11.0 + index,
        "low": 9.0 + index,
        "close": 10.5 + index,
        "price_basis": PriceBasis.MID,
        "volume": 2.0,
        "volume_semantics": VolumeSemantics.TICK,
    }
    values.update(overrides)
    return Bar(**values)  # type: ignore[arg-type]


def test_valid_dataset_is_chronological_and_gaps_are_informational() -> None:
    report = validate_dataset([make_bar(0), make_bar(1), make_bar(3)])

    assert report.bars == (make_bar(0), make_bar(1), make_bar(3))
    assert [(gap.start, gap.end) for gap in report.gaps] == [
        (START + timedelta(minutes=10), START + timedelta(minutes=15))
    ]


def test_non_bars_and_inconsistent_semantics_are_rejected() -> None:
    with pytest.raises(DatasetValidationError, match="canonical Bar"):
        validate_dataset([make_bar(), "not a bar"])  # type: ignore[list-item]
    with pytest.raises(DatasetValidationError, match="homogeneous"):
        validate_dataset([make_bar(), make_bar(1, instrument="GBPUSD")])
    with pytest.raises(DatasetValidationError, match="homogeneous"):
        validate_dataset(
            [
                make_bar(),
                make_bar(1, volume=3.0, volume_semantics=VolumeSemantics.TRADED),
            ]
        )


def test_duplicate_observation_and_duplicate_interval_are_distinguished() -> None:
    bar = make_bar()
    with pytest.raises(DatasetValidationError, match="duplicate observation"):
        validate_dataset([bar, bar])
    with pytest.raises(DatasetValidationError, match="duplicate interval"):
        validate_dataset([bar, make_bar(close=10.75)])


def test_overlap_and_out_of_order_are_structural_errors() -> None:
    with pytest.raises(DatasetValidationError, match="overlap"):
        validate_dataset(
            [
                make_bar(),
                make_bar(
                    1,
                    open_time=START + timedelta(minutes=4),
                    close_time=START + timedelta(minutes=9),
                ),
            ]
        )
    with pytest.raises(DatasetValidationError, match="chronological"):
        validate_dataset([make_bar(1), make_bar(0)])


def test_observation_identity_includes_logical_dataset_dimensions() -> None:
    identity = observation_identity("dataset-a", make_bar())

    assert identity.dataset_id == "dataset-a"
    assert identity.instrument == "EURUSD"
    assert identity.timeframe == Timeframe("5m")
    assert identity.open_time == START
    assert identity.price_basis is PriceBasis.MID
    assert identity != observation_identity("dataset-b", make_bar())


def test_available_view_uses_available_at_and_accepts_equivalent_utc() -> None:
    delayed = make_bar(0, available_at=START + timedelta(minutes=6))
    prompt = make_bar(1)
    research_time = datetime(2025, 1, 2, 10, 5, tzinfo=ZoneInfo("UTC"))

    assert available_bars([delayed, prompt], research_time) == ()
    assert available_bars([delayed, prompt], START + timedelta(minutes=6)) == (delayed,)
    with pytest.raises(Exception, match="timezone-aware UTC"):
        available_bars([delayed], datetime(2025, 1, 2, 10, 6))
