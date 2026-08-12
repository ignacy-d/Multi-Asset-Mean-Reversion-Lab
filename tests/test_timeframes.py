import pytest

from mr_lab.config import normalize_timeframe
from mr_lab.data import DataContractError, Timeframe


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("M5", "5m"),
        ("M15", "15m"),
        ("H1", "1h"),
        ("5m", "5m"),
        ("15m", "15m"),
        ("1h", "1h"),
    ],
)
def test_experiment_timeframes_normalize_at_explicit_boundary(
    value: str, canonical: str
) -> None:
    assert normalize_timeframe(value) == Timeframe(canonical)


@pytest.mark.parametrize("value", ["M1", "H2", "15M", "monthly", ""])
def test_unsupported_timeframes_are_rejected(value: str) -> None:
    with pytest.raises(DataContractError):
        normalize_timeframe(value)
