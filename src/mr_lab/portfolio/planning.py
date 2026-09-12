"""Minimal generic proposal construction boundary for strategy-owned planners."""

from mr_lab.portfolio.contracts import (
    OpportunitySet,
    OpportunityStatus,
    PlanReference,
    ProposalProvenance,
    TradeProposal,
)
from mr_lab.portfolio.identity import stable_id


def build_trade_proposal(
    opportunity: OpportunitySet,
    *,
    strategy_policy_id: str,
    entry_plan: PlanReference,
    protective_plan: PlanReference,
    provenance: ProposalProvenance,
) -> TradeProposal:
    """Build a deterministic proposal; non-actionable evidence fails closed."""
    if opportunity.status is not OpportunityStatus.ACTIONABLE:
        raise ValueError("only actionable opportunities can become trade proposals")
    assert opportunity.direction is not None
    proposal_id = stable_id(
        "proposal",
        opportunity.opportunity_id,
        opportunity.timestamp,
        strategy_policy_id,
        entry_plan.kind,
        entry_plan.specification_id,
        protective_plan.kind,
        protective_plan.specification_id,
        provenance.planner_id,
        provenance.planner_version,
    )
    return TradeProposal(
        proposal_id,
        opportunity.opportunity_id,
        opportunity.sleeve_id,
        opportunity.instrument,
        opportunity.direction,
        opportunity.timestamp,
        strategy_policy_id,
        entry_plan,
        protective_plan,
        provenance,
    )
