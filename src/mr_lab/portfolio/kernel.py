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


class PortfolioPolicyInvariantError(ValueError):
    """Raised when a policy does not decide the complete candidate set exactly once."""


@dataclass(frozen=True, slots=True)
class PolicyResult:
    proposal_id: str
    decision: PortfolioDecisionState
    reason: str

    def __post_init__(self) -> None:
        require_text("proposal_id", self.proposal_id)
        require_text("reason", self.reason)


class PortfolioPolicy(Protocol):
    policy_id: str
    policy_version: str

    def evaluate(
        self, proposals: tuple[TradeProposal, ...]
    ) -> tuple[PolicyResult, ...]: ...


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
        candidates = tuple(sorted(proposals, key=lambda item: item.proposal_id))
        expected_ids = {proposal.proposal_id for proposal in candidates}
        if len(expected_ids) != len(candidates):
            raise PortfolioPolicyInvariantError("candidate proposal IDs must be unique")

        results = tuple(self._policy.evaluate(candidates))
        result_ids = [result.proposal_id for result in results]
        if len(result_ids) != len(set(result_ids)):
            raise PortfolioPolicyInvariantError(
                "portfolio policy returned duplicate proposal IDs"
            )
        if set(result_ids) != expected_ids:
            missing = sorted(expected_ids - set(result_ids))
            unexpected = sorted(set(result_ids) - expected_ids)
            raise PortfolioPolicyInvariantError(
                "portfolio policy must return exactly one result per proposal; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        results_by_id = {result.proposal_id: result for result in results}

        decisions = []
        for proposal in candidates:
            result = results_by_id[proposal.proposal_id]
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
