from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import (
    Bar,
    DataContractError,
    DatasetMetadata,
    PriceBasis,
    Timeframe,
    VolumeType,
    available_bars,
    observation_identity,
    validate_dataset,
)

ONE_MINUTE = Timeframe.parse("1m")
START = datetime(2025, 1, 1, 10, tzinfo=UTC)


def bar(index: int, **changes: object) -> Bar:
    open_time = START + index * ONE_MINUTE.duration
    values = {
        "instrument": "EURUSD",
        "timeframe": ONE_MINUTE,
        "open_time": open_time,
        "close_time": open_time + ONE_MINUTE.duration,
        "available_at": open_time + ONE_MINUTE.duration,
        "open": 100.0 + index,
        "high": 101.0 + index,
        "low": 99.0 + index,
        "close": 100.5 + index,
        "price_basis": PriceBasis.MID,
        "volume": 10.0,
        "volume_type": VolumeType.TICK,
    }
    values.update(changes)
    return Bar(**values)


def metadata(**changes: object) -> DatasetMetadata:
    values = {
        "source": "synthetic",
        "instrument": "EURUSD",
        "price_basis": PriceBasis.MID,
        "volume_type": VolumeType.TICK,
        "schema_version": "1",
        "dataset_id": "synthetic-a",
        "native_timeframe": ONE_MINUTE,
    }
    values.update(changes)
    return DatasetMetadata(**values)


def test_valid_dataset_is_chronological_and_gap_is_informational() -> None:
    report = validate_dataset([bar(0), bar(1), bar(3)], metadata())

    assert report.observation_count == 3
    assert len(report.gaps) == 1
    assert report.gaps[0].previous_close == bar(1).close_time
    assert report.gaps[0].next_open == bar(3).open_time


def test_duplicate_identity_is_rejected() -> None:
    with pytest.raises(DataContractError, match="duplicate observation identity"):
        validate_dataset([bar(0), bar(0)], metadata())


def test_overlap_and_non_chronological_order_are_rejected() -> None:
    overlapping = replace(
        bar(1),
        open_time=bar(0).close_time - timedelta(seconds=1),
    )
    with pytest.raises(DataContractError, match="overlapping"):
        validate_dataset([bar(0), overlapping], metadata())
    with pytest.raises(DataContractError, match="chronological"):
        validate_dataset([bar(1), bar(0)], metadata())


@pytest.mark.parametrize(
    "changed",
    [
        bar(0, instrument="USDJPY"),
        bar(0, price_basis=PriceBasis.BID),
        bar(0, volume_type=VolumeType.TRADED),
        bar(0, timeframe=Timeframe.parse("5m")),
    ],
)
def test_dataset_semantics_must_match_metadata(changed: Bar) -> None:
    with pytest.raises(DataContractError, match="inconsistent"):
        validate_dataset([changed], metadata())


def test_identity_includes_dataset_and_price_basis() -> None:
    first = observation_identity(bar(0), "source-a")
    other_source = observation_identity(bar(0), "source-b")
    other_basis = observation_identity(bar(0, price_basis=PriceBasis.BID), "source-a")

    assert len({first, other_source, other_basis}) == 3


def test_point_in_time_view_uses_true_availability() -> None:
    delayed = bar(0, available_at=bar(0).close_time + timedelta(minutes=2))

    assert available_bars([delayed], delayed.close_time) == ()
    assert available_bars([delayed], delayed.available_at) == (delayed,)
    with pytest.raises(DataContractError, match="UTC"):
        available_bars([delayed], datetime(2025, 1, 1, 10))
