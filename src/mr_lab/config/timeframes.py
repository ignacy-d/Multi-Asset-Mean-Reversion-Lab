"""Boundary conversion from experiment identifiers to canonical timeframes."""

from mr_lab.data.models import DataContractError, Timeframe

_EXPERIMENT_ALIASES = {"M5": "5m", "M15": "15m", "H1": "1h"}


def normalize_timeframe(value: str) -> Timeframe:
    """Convert a supported experiment identifier or canonical value."""
    if not isinstance(value, str):
        raise DataContractError("timeframe identifier must be a string")
    return Timeframe.parse(_EXPERIMENT_ALIASES.get(value, value))
