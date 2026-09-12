"""Generic runtime signal contracts.

The contract deliberately contains no order, position, or broker concepts.  The
OU fields are metadata produced by the frozen alpha module; consumers can route
signals without importing an OU implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class AlphaSignal:
    """A point-in-time alpha decision emitted after a completed observation."""

    source_event_id: str
    signal_timestamp: datetime
    instrument: str
    direction: str
    benchmark_family: str
    lookback: int
    p0: float
    e0: float
    z: float
    process_id: str
    process_spec_id: str
    process_status: str
    process_is_structurally_valid: bool
    process_invalid_reason: str | None
    ou_score: float | None
    half_life_minutes: float | None
    eligible: bool
    eligibility_reason: str
    filter_spec_id: str

    def __post_init__(self) -> None:
        if (
            self.signal_timestamp.tzinfo is None
            or self.signal_timestamp.utcoffset() != timedelta(0)
        ):
            raise ValueError("signal_timestamp must be UTC")
        if not all(math.isfinite(value) for value in (self.p0, self.e0, self.z)):
            raise ValueError("signal price and z fields must be finite")
