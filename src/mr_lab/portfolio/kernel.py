"""Policy boundary for deterministic portfolio decisions (not risk sizing)."""

from dataclasses import dataclass
from typing import Protocol

from mr_lab.portfolio.contracts import (
    PortfolioDecision,
    PortfolioDecisionState,
    TradeProposal,
    require_text,
)
from mr_lab.portfolio.identity import stable_id


class MissingPortfolioPolicyError(RuntimeError):
    """Fail-closed error raised when no explicit portfolio policy is installed."""


@dataclass(frozen=True, slots=True)
class PolicyResult:
    decision: PortfolioDecisionState
    reason: str

    def __post_init__(self) -> None:
        require_text("reason", self.reason)


class PortfolioPolicy(Protocol):
    policy_id: str
    policy_version: str

    def evaluate(self, proposal: TradeProposal) -> PolicyResult: ...


class PortfolioKernel:
    def __init__(self, policy: PortfolioPolicy | None = None) -> None:
        self._policy = policy

    def evaluate(
        self, proposals: tuple[TradeProposal, ...]
    ) -> tuple[PortfolioDecision, ...]:
        if self._policy is None:
            raise MissingPortfolioPolicyError(
                "an explicit portfolio policy is required"
            )
        require_text("policy_id", self._policy.policy_id)
        require_text("policy_version", self._policy.policy_version)
        decisions = []
        for proposal in sorted(proposals, key=lambda item: item.proposal_id):
            result = self._policy.evaluate(proposal)
            decision_id = stable_id(
                "portfolio-decision",
                proposal.proposal_id,
                self._policy.policy_id,
                self._policy.policy_version,
                result.decision,
                result.reason,
            )
            decisions.append(
                PortfolioDecision(
                    decision_id,
                    proposal.proposal_id,
                    proposal.opportunity_id,
                    proposal.sleeve_id,
                    result.decision,
                    result.reason,
                    self._policy.policy_id,
                    self._policy.policy_version,
                )
            )
        return tuple(decisions)
