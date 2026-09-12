"""Multi-alpha opportunity and broker-neutral portfolio kernel."""

from mr_lab.portfolio.contracts import (
    OpportunitySet,
    OpportunityStatus,
    PlanReference,
    PortfolioDecision,
    PortfolioDecisionState,
    ProposalProvenance,
    TradeProposal,
)
from mr_lab.portfolio.kernel import (
    MissingPortfolioPolicyError,
    PolicyResult,
    PortfolioKernel,
    PortfolioPolicy,
    PortfolioPolicyInvariantError,
)
from mr_lab.portfolio.opportunity import OpportunityEngine, OpportunityInvariantError
from mr_lab.portfolio.planning import build_trade_proposal
from mr_lab.portfolio.registry import (
    AlphaModuleRegistry,
    AlphaModuleSpec,
    RegistryInvariantError,
)

__all__ = [
    "AlphaModuleRegistry",
    "AlphaModuleSpec",
    "MissingPortfolioPolicyError",
    "OpportunityEngine",
    "OpportunityInvariantError",
    "OpportunitySet",
    "OpportunityStatus",
    "PlanReference",
    "PolicyResult",
    "PortfolioDecision",
    "PortfolioDecisionState",
    "PortfolioKernel",
    "PortfolioPolicy",
    "PortfolioPolicyInvariantError",
    "ProposalProvenance",
    "RegistryInvariantError",
    "TradeProposal",
    "build_trade_proposal",
]
