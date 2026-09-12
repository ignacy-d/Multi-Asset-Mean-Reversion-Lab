"""Broker-neutral contracts for opportunities, proposals, and portfolio decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from mr_lab.signals import AlphaSignal


def require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")


class OpportunityStatus(StrEnum):
    ACTIONABLE = "ACTIONABLE"
    CONFLICT = "CONFLICT"
    DISABLED = "DISABLED"


@dataclass(frozen=True, slots=True)
class OpportunitySet:
    opportunity_id: str
    opportunity_family_id: str
    sleeve_id: str
    instrument: str
    timestamp: datetime
    direction: str | None
    signals: tuple[AlphaSignal, ...]
    status: OpportunityStatus
    conflict_directions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "opportunity_id",
            "opportunity_family_id",
            "sleeve_id",
            "instrument",
        ):
            require_text(name, getattr(self, name))
        require_utc("timestamp", self.timestamp)
        if not self.signals:
            raise ValueError("an opportunity must retain at least one signal")
        if self.status is OpportunityStatus.ACTIONABLE and self.direction is None:
            raise ValueError("an actionable opportunity requires a direction")
        if (
            self.status is OpportunityStatus.CONFLICT
            and len(self.conflict_directions) < 2
        ):
            raise ValueError("a conflict requires at least two directions")

    @property
    def constituent_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (signal.module_id, signal.source_event_id) for signal in self.signals
        )


@dataclass(frozen=True, slots=True)
class PlanReference:
    """A strategy-owned plan identity, without broker order semantics."""

    kind: str
    specification_id: str

    def __post_init__(self) -> None:
        require_text("kind", self.kind)
        require_text("specification_id", self.specification_id)


@dataclass(frozen=True, slots=True)
class ProposalProvenance:
    planner_id: str
    planner_version: str

    def __post_init__(self) -> None:
        require_text("planner_id", self.planner_id)
        require_text("planner_version", self.planner_version)


@dataclass(frozen=True, slots=True)
class TradeProposal:
    proposal_id: str
    opportunity_id: str
    sleeve_id: str
    instrument: str
    direction: str
    timestamp: datetime
    strategy_policy_id: str
    entry_plan: PlanReference
    protective_plan: PlanReference
    provenance: ProposalProvenance

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "opportunity_id",
            "sleeve_id",
            "instrument",
            "direction",
            "strategy_policy_id",
        ):
            require_text(name, getattr(self, name))
        require_utc("timestamp", self.timestamp)


class PortfolioDecisionState(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    DEFER = "DEFER"


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    portfolio_decision_id: str
    proposal_id: str
    opportunity_id: str
    sleeve_id: str
    decision: PortfolioDecisionState
    reason: str
    policy_id: str
    policy_version: str

    def __post_init__(self) -> None:
        for name in (
            "portfolio_decision_id",
            "proposal_id",
            "opportunity_id",
            "sleeve_id",
            "reason",
            "policy_id",
            "policy_version",
        ):
            require_text(name, getattr(self, name))
