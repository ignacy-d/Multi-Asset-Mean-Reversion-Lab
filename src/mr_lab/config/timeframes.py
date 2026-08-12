"""Boundary between experiment timeframe identifiers and canonical durations."""

import re

from mr_lab.data import DataContractError, Timeframe

_EXPERIMENT_TIMEFRAME = re.compile(r"(?P<unit>[MHD])(?P<count>[1-9][0-9]*)")
_UNIT = {"M": "m", "H": "h", "D": "d"}


def normalize_timeframe(value: str) -> Timeframe:
    """Translate documented experiment or canonical syntax to a Timeframe."""
    try:
        return Timeframe.parse(value)
    except DataContractError as canonical_error:
        if (
            not isinstance(value, str)
            or (match := _EXPERIMENT_TIMEFRAME.fullmatch(value)) is None
        ):
            raise DataContractError(
                f"invalid experiment timeframe: {value!r}"
            ) from canonical_error
        return Timeframe.parse(f"{match['count']}{_UNIT[match['unit']]}")
