"""Small broker-independent contracts shared by all alpha modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


def _non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SignalReference:
    """Optional module-neutral price reference, not an execution instruction."""

    price: float
    kind: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.price):
            raise ValueError("reference price must be finite")
        _non_empty("reference kind", self.kind)


@dataclass(frozen=True, slots=True)
class SignalProvenance:
    """Stable source identities needed to reproduce a signal."""

    strategy_spec_id: str
    source_corpus_id: str
    assembled_dataset_id: str | None = None

    def __post_init__(self) -> None:
        _non_empty("strategy_spec_id", self.strategy_spec_id)
        _non_empty("source_corpus_id", self.source_corpus_id)
        if self.assembled_dataset_id is not None:
            _non_empty("assembled_dataset_id", self.assembled_dataset_id)


@dataclass(frozen=True, slots=True)
class AlphaSignal:
    """A generic alpha envelope with no OU, broker, order, or position fields."""

    module_id: str
    source_event_id: str
    instrument: str
    timestamp: datetime
    direction: str
    confidence: float | None = None
    reference: SignalReference | None = None
    invalidation: str | None = None
    provenance: SignalProvenance | None = None

    def __post_init__(self) -> None:
        for name in ("module_id", "source_event_id", "instrument", "direction"):
            _non_empty(name, getattr(self, name))
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be UTC")
        if self.confidence is not None and not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite when supplied")
        if self.invalidation is not None:
            _non_empty("invalidation", self.invalidation)
