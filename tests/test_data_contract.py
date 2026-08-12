from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from mr_lab.data import (
    Bar,
    DataContractError,
    DatasetMetadata,
    PriceBasis,
    Timeframe,
    VolumeSemantics,
)


def make_bar(**overrides: object) -> Bar:
    values = {
        "instrument": "EURUSD",
        "timeframe": Timeframe("15m"),
        "open_time": datetime(2025, 1, 2, 10, 0, tzinfo=UTC),
        "close_time": datetime(2025, 1, 2, 10, 15, tzinfo=UTC),
        "available_at": datetime(2025, 1, 2, 10, 15, 1, tzinfo=UTC),
        "open": 1.1,
        "high": 1.2,
        "low": 1.0,
        "close": 1.15,
        "price_basis": PriceBasis.BID,
        "volume": 42.0,
        "volume_semantics": VolumeSemantics.TICK,
    }
    values.update(overrides)
    return Bar(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["open_time", "close_time", "available_at"])
def test_bar_requires_timezone_aware_timestamps(field: str) -> None:
    with pytest.raises(DataContractError, match="timezone-aware UTC"):
        make_bar(**{field: datetime(2025, 1, 2, 10, 0)})


def test_bar_rejects_non_utc_timestamp() -> None:
    non_utc = timezone(timedelta(hours=1))
    with pytest.raises(DataContractError, match="canonical UTC"):
        make_bar(open_time=datetime(2025, 1, 2, 11, 0, tzinfo=non_utc))


def test_bar_accepts_utc_equivalent_zoneinfo_timestamp() -> None:
    utc_zoneinfo = ZoneInfo("UTC")

    bar = make_bar(open_time=datetime(2025, 1, 2, 10, 0, tzinfo=utc_zoneinfo))

    assert bar.open_time.tzinfo is utc_zoneinfo


def test_bar_time_order_and_completion_are_validated() -> None:
    close_time = datetime(2025, 1, 2, 10, 15, tzinfo=UTC)
    with pytest.raises(DataContractError, match="open_time must be before"):
        make_bar(open_time=close_time)
    with pytest.raises(DataContractError, match="available_at must be at or after"):
        make_bar(available_at=close_time - timedelta(microseconds=1))


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_ohlc_values_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(DataContractError, match="finite number"):
        make_bar(**{field: value})


def test_ohlc_extrema_are_validated_without_requiring_positive_prices() -> None:
    make_bar(open=-2.0, high=0.0, low=-3.0, close=-1.0)
    with pytest.raises(DataContractError, match="high must be at least"):
        make_bar(high=1.14)
    with pytest.raises(DataContractError, match="low must be at most"):
        make_bar(low=1.11)


@pytest.mark.parametrize("volume", [-1.0, float("nan"), float("inf")])
def test_numeric_volume_must_be_finite_and_non_negative(volume: float) -> None:
    with pytest.raises(DataContractError):
        make_bar(volume=volume)


def test_volume_presence_and_semantics_must_agree() -> None:
    make_bar(volume=None, volume_semantics=VolumeSemantics.NONE)
    with pytest.raises(DataContractError, match="volume=None requires NONE"):
        make_bar(volume=None, volume_semantics=VolumeSemantics.TRADED)
    with pytest.raises(DataContractError, match="numeric volume cannot use NONE"):
        make_bar(volume=0.0, volume_semantics=VolumeSemantics.NONE)


def test_semantic_fields_require_enum_members() -> None:
    with pytest.raises(DataContractError, match="price_basis must be a PriceBasis"):
        make_bar(price_basis="bid")
    with pytest.raises(DataContractError, match="VolumeSemantics"):
        make_bar(volume_semantics="tick")


@pytest.mark.parametrize("notation", ["1m", "5m", "15m", "1h", "1d"])
def test_timeframe_parses_canonical_fixed_duration_notation(notation: str) -> None:
    assert str(Timeframe.parse(notation)) == notation
    assert Timeframe.parse(notation).duration > timedelta(0)


@pytest.mark.parametrize("notation", ["M15", "15M", "m15", "0m", "01m", "1mo", ""])
def test_timeframe_rejects_noncanonical_or_ambiguous_notation(notation: str) -> None:
    with pytest.raises(DataContractError, match="lowercase fixed-duration"):
        Timeframe.parse(notation)


def test_bar_availability_is_point_in_time_and_requires_utc() -> None:
    bar = make_bar()
    assert not bar.is_available_at(bar.available_at - timedelta(microseconds=1))
    assert bar.is_available_at(bar.available_at)
    assert bar.is_available_at(bar.available_at + timedelta(days=1))
    with pytest.raises(DataContractError, match="timezone-aware UTC"):
        bar.is_available_at(datetime(2025, 1, 2, 10, 15, 1))


def test_bar_is_immutable() -> None:
    bar = make_bar()
    with pytest.raises(FrozenInstanceError):
        bar.close = 2.0


def test_dataset_metadata_records_provenance_and_semantics() -> None:
    metadata = DatasetMetadata(
        source="example-source",
        instrument="EURUSD",
        price_basis=PriceBasis.MID,
        volume_semantics=VolumeSemantics.NONE,
        schema_version="0B-v1",
        dataset_id="fingerprint-pending",
        native_timeframe=Timeframe("1m"),
        source_timezone="America/New_York",
    )
    assert metadata.native_timeframe == Timeframe("1m")
    with pytest.raises(FrozenInstanceError):
        metadata.source = "changed"


@pytest.mark.parametrize(
    "field", ["source", "instrument", "schema_version", "dataset_id"]
)
def test_dataset_metadata_requires_identifiers(field: str) -> None:
    values = {
        "source": "source",
        "instrument": "EURUSD",
        "price_basis": PriceBasis.TRADE,
        "volume_semantics": VolumeSemantics.TRADED,
        "schema_version": "1",
        "dataset_id": "pending",
    }
    values[field] = " "
    with pytest.raises(DataContractError, match=f"{field} must be a non-empty"):
        DatasetMetadata(**values)  # type: ignore[arg-type]


def test_dataset_metadata_validates_typed_fields() -> None:
    common = {
        "source": "source",
        "instrument": "EURUSD",
        "price_basis": PriceBasis.TRADE,
        "volume_semantics": VolumeSemantics.TRADED,
        "schema_version": "1",
        "dataset_id": "pending",
    }
    with pytest.raises(DataContractError, match="price_basis"):
        DatasetMetadata(**(common | {"price_basis": "trade"}))  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="volume_semantics"):
        DatasetMetadata(**(common | {"volume_semantics": "traded"}))  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="native_timeframe"):
        DatasetMetadata(**(common | {"native_timeframe": "1m"}))  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="source_timezone"):
        DatasetMetadata(**(common | {"source_timezone": ""}))  # type: ignore[arg-type]
