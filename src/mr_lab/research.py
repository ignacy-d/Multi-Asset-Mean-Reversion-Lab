"""Minimal point-in-time research observations and fixed-clock outcome labels."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from hashlib import sha256

from mr_lab.data import Bar
from mr_lab.sessions import SessionClassification, SessionSpec, classify_bar

RESEARCH_SCHEMA_VERSION = "stage-2a-research-v1"
ACTIVITY_RULE_VERSION = "zero-volume-flat-ohlc-v1"
RETURN_DEFINITION = "close_to_close_arithmetic"


class ResearchError(ValueError):
    """Raised when a Stage 2A research input is invalid."""


class Direction(IntEnum):
    """A direction supplied by a later strategy, not a Stage 2A signal."""

    LONG = 1
    SHORT = -1


def is_no_activity_flat_bar(bar: Bar) -> bool:
    """Classify the provider observation using only this bar's fields.

    This is deliberately not a claim that the global market was officially closed.
    """
    if not isinstance(bar, Bar):
        raise ResearchError("bar must be a canonical Bar")
    return bar.volume == 0 and bar.open == bar.high == bar.low == bar.close


@dataclass(frozen=True, slots=True)
class ResearchObservation:
    """Immutable current-bar activity and reused Stage 1C session semantics."""

    bar: Bar
    is_no_activity_flat: bool
    sessions: SessionClassification

    @property
    def available_at(self):
        return self.bar.available_at

    @property
    def is_research_active(self) -> bool:
        return not self.is_no_activity_flat

    @property
    def session_spec_id(self) -> str:
        return self.sessions.session_spec_id


def build_research_observations(
    bars: Iterable[Bar], session_spec: SessionSpec
) -> tuple[ResearchObservation, ...]:
    """Label every supplied bar independently without deleting or reindexing it."""
    return tuple(
        ResearchObservation(
            bar, is_no_activity_flat_bar(bar), classify_bar(bar, session_spec)
        )
        for bar in bars
    )


@dataclass(frozen=True, slots=True)
class ResearchSpec:
    """Deterministic identity for only the methodology introduced in Stage 2A."""

    horizons: tuple[timedelta, ...]
    schema_version: str = RESEARCH_SCHEMA_VERSION
    activity_rule_version: str = ACTIVITY_RULE_VERSION
    return_definition: str = RETURN_DEFINITION

    def __post_init__(self) -> None:
        if not isinstance(self.horizons, tuple) or not self.horizons:
            raise ResearchError("horizons must be a non-empty tuple")
        if any(
            not isinstance(item, timedelta) or item <= timedelta(0)
            for item in self.horizons
        ):
            raise ResearchError("horizons must contain positive timedeltas")
        if any(item.microseconds for item in self.horizons):
            raise ResearchError("horizons must use whole elapsed seconds")
        for name in ("schema_version", "activity_rule_version", "return_definition"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ResearchError(f"{name} must be a non-empty string")
        if self.schema_version != RESEARCH_SCHEMA_VERSION:
            raise ResearchError(
                f"Stage 2A supports only {RESEARCH_SCHEMA_VERSION!r} schemas"
            )
        if self.activity_rule_version != ACTIVITY_RULE_VERSION:
            raise ResearchError(
                f"Stage 2A supports only {ACTIVITY_RULE_VERSION!r} activity rules"
            )
        if self.return_definition != RETURN_DEFINITION:
            raise ResearchError(f"Stage 2A supports only {RETURN_DEFINITION!r} returns")
        object.__setattr__(self, "horizons", tuple(sorted(set(self.horizons))))

    def as_dict(self) -> dict[str, object]:
        return {
            "activity_rule_version": self.activity_rule_version,
            "horizons_seconds": [int(item.total_seconds()) for item in self.horizons],
            "research_schema_version": self.schema_version,
            "return_definition": self.return_definition,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def research_spec_id(self) -> str:
        return f"sha256:{sha256(self.to_json().encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ForwardOutcome:
    """One exact-clock statistical label, including explicit unavailability."""

    source: ResearchObservation
    horizon: timedelta
    target_available_at: datetime
    target: ResearchObservation | None
    forward_return: float | None

    @property
    def is_available(self) -> bool:
        return self.forward_return is not None

    def signed_forward_return(self, direction: Direction) -> float | None:
        if not isinstance(direction, Direction):
            raise ResearchError("direction must be LONG or SHORT")
        if self.forward_return is None:
            return None
        return int(direction) * self.forward_return

    def as_dict(self) -> dict[str, object]:
        bar, sessions = self.source.bar, self.source.sessions
        return {
            "source_open_time": bar.open_time.isoformat(),
            "source_available_at": bar.available_at.isoformat(),
            "timeframe": str(bar.timeframe),
            "source_close": bar.close,
            "is_research_active": self.source.is_research_active,
            "active_sessions": list(sessions.active_sessions),
            "regime": sessions.regime,
            "active_named_windows": list(sessions.active_named_windows),
            "horizon_seconds": int(self.horizon.total_seconds()),
            "target_available_at": self.target_available_at.isoformat(),
            "target_close": self.target.bar.close if self.target else None,
            "forward_return": self.forward_return,
        }


def build_forward_outcomes(
    observations: Iterable[ResearchObservation], spec: ResearchSpec
) -> tuple[ForwardOutcome, ...]:
    """Match targets only at source ``available_at + horizon``; never skip gaps."""
    items = tuple(observations)
    by_availability = {item.available_at: item for item in items}
    if len(by_availability) != len(items):
        raise ResearchError("observation available_at timestamps must be unique")
    outcomes = []
    for source in items:
        for horizon in spec.horizons:
            required = source.available_at + horizon
            target = by_availability.get(required)
            valid_target = target if target and target.is_research_active else None
            value = None
            if source.is_research_active and valid_target is not None:
                if source.bar.close == 0:
                    raise ResearchError(
                        "arithmetic return is undefined for zero source close"
                    )
                value = valid_target.bar.close / source.bar.close - 1.0
                if not math.isfinite(value):
                    raise ResearchError("forward return must be finite")
            outcomes.append(
                ForwardOutcome(source, horizon, required, valid_target, value)
            )
    return tuple(outcomes)


@dataclass(frozen=True, slots=True)
class ResearchSummary:
    """Compact deterministic audit without persisting observation-level records."""

    source_bar_count: int
    active_observation_count: int
    no_activity_flat_count: int
    valid_outcomes_by_horizon: tuple[tuple[int, int], ...]
    unavailable_outcomes_by_horizon: tuple[tuple[int, int], ...]
    session_spec_id: str
    research_spec_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_bar_count": self.source_bar_count,
            "active_observation_count": self.active_observation_count,
            "no_activity_flat_count": self.no_activity_flat_count,
            "valid_outcomes_by_horizon_seconds": dict(self.valid_outcomes_by_horizon),
            "unavailable_outcomes_by_horizon_seconds": dict(
                self.unavailable_outcomes_by_horizon
            ),
            "session_spec_id": self.session_spec_id,
            "research_spec_id": self.research_spec_id,
        }


def summarize_research(
    observations: Iterable[ResearchObservation],
    outcomes: Iterable[ForwardOutcome],
    spec: ResearchSpec,
) -> ResearchSummary:
    """Summarize already-built observations and outcomes deterministically."""
    items, labels = tuple(observations), tuple(outcomes)
    session_ids = {item.session_spec_id for item in items}
    if len(session_ids) != 1:
        raise ResearchError("observations must share exactly one session_spec_id")
    valid = []
    unavailable = []
    for horizon in spec.horizons:
        seconds = int(horizon.total_seconds())
        matching = tuple(item for item in labels if item.horizon == horizon)
        valid_count = sum(item.is_available for item in matching)
        valid.append((seconds, valid_count))
        unavailable.append((seconds, len(matching) - valid_count))
    active = sum(item.is_research_active for item in items)
    return ResearchSummary(
        len(items),
        active,
        len(items) - active,
        tuple(valid),
        tuple(unavailable),
        session_ids.pop(),
        spec.research_spec_id,
    )


__all__ = [
    "ACTIVITY_RULE_VERSION",
    "RESEARCH_SCHEMA_VERSION",
    "RETURN_DEFINITION",
    "Direction",
    "ForwardOutcome",
    "ResearchError",
    "ResearchObservation",
    "ResearchSpec",
    "ResearchSummary",
    "build_forward_outcomes",
    "build_research_observations",
    "is_no_activity_flat_bar",
    "summarize_research",
]
