from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from math import inf, nan

import pytest

from mr_lab.data import (
    Bar,
    DataContractError,
    DatasetMetadata,
    PriceBasis,
    Timeframe,
    VolumeType,
)


def valid_bar(**changes: object) -> Bar:
    values = {
        "instrument": "EURUSD",
        "timeframe": Timeframe.parse("15m"),
        "open_time": datetime(2025, 1, 2, 10, 0, tzinfo=UTC),
        "close_time": datetime(2025, 1, 2, 10, 15, tzinfo=UTC),
        "available_at": datetime(2025, 1, 2, 10, 15, 1, tzinfo=UTC),
        "open": 1.10,
        "high": 1.12,
        "low": 1.09,
        "close": 1.11,
        "price_basis": PriceBasis.MID,
        "volume": 200.0,
        "volume_type": VolumeType.TICK,
    }
    values.update(changes)
    return Bar(**values)


def test_completed_bar_is_usable_only_from_available_at() -> None:
    bar = valid_bar()

    assert not bar.is_available_at(bar.close_time)
    assert bar.is_available_at(bar.available_at)


@pytest.mark.parametrize("field", ["open_time", "close_time", "available_at"])
def test_timestamps_must_be_utc(field: str) -> None:
    with pytest.raises(DataContractError, match="UTC"):
        valid_bar(**{field: datetime(2025, 1, 2, 10, 0)})

    non_utc = timezone(timedelta(hours=1))
    with pytest.raises(DataContractError, match="UTC"):
        valid_bar(**{field: datetime(2025, 1, 2, 10, 0, tzinfo=non_utc)})


def test_bar_time_order_is_validated() -> None:
    bar = valid_bar()

    with pytest.raises(DataContractError, match="open_time"):
        replace(bar, open_time=bar.close_time)
    with pytest.raises(DataContractError, match="available_at"):
        replace(bar, available_at=bar.close_time - timedelta(microseconds=1))


@pytest.mark.parametrize(
    ("field", "value"),
    [("open", nan), ("high", inf), ("low", -inf), ("close", nan)],
)
def test_ohlc_values_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(DataContractError, match="finite"):
        valid_bar(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"high": 1.105},
        {"low": 1.115},
    ],
)
def test_ohlc_range_must_be_consistent(changes: dict[str, float]) -> None:
    with pytest.raises(DataContractError):
        valid_bar(**changes)


def test_volume_is_optional_but_its_semantics_must_match() -> None:
    bar = valid_bar(volume=None, volume_type=VolumeType.NONE)
    assert bar.volume is None

    with pytest.raises(DataContractError, match="negative"):
        valid_bar(volume=-1)
    with pytest.raises(DataContractError, match="absent volume"):
        valid_bar(volume=None, volume_type=VolumeType.UNKNOWN)
    with pytest.raises(DataContractError, match="present volume"):
        valid_bar(volume=1, volume_type=VolumeType.NONE)


def test_semantics_require_enum_members() -> None:
    with pytest.raises(DataContractError, match="price_basis"):
        valid_bar(price_basis="mid")
    with pytest.raises(DataContractError, match="volume_type"):
        valid_bar(volume_type="tick")


@pytest.mark.parametrize("value", ["M15", "15M", "0m", "m15", "1.5h", ""])
def test_ambiguous_or_invalid_timeframes_are_rejected(value: str) -> None:
    with pytest.raises(DataContractError, match="timeframe"):
        Timeframe.parse(value)


def test_metadata_preserves_source_semantics() -> None:
    metadata = DatasetMetadata(
        source="source-a",
        instrument="EURUSD",
        native_timeframe=Timeframe.parse("1m"),
        price_basis=PriceBasis.BID,
        volume_type=VolumeType.QUOTE_ACTIVITY,
        source_timezone="Europe/London",
        schema_version="1",
        dataset_id="fingerprint-placeholder",
    )

    assert metadata.source == "source-a"
    assert metadata.price_basis is PriceBasis.BID
