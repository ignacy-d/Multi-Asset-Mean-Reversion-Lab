from dataclasses import replace
from datetime import UTC, datetime

import pytest

from mr_lab.portfolio import (
    AlphaModuleRegistry,
    AlphaModuleSpec,
    MissingPortfolioPolicyError,
    OpportunityEngine,
    OpportunityInvariantError,
    OpportunityStatus,
    PlanReference,
    PolicyResult,
    PortfolioDecisionState,
    PortfolioKernel,
    PortfolioPolicyInvariantError,
    ProposalProvenance,
    RegistryInvariantError,
    build_trade_proposal,
)
from mr_lab.signals import AlphaSignal, SignalProvenance, SignalReference

NOW = datetime(2024, 6, 3, 10, tzinfo=UTC)


def registry(*extra: AlphaModuleSpec) -> AlphaModuleRegistry:
    return AlphaModuleRegistry(
        (
            AlphaModuleSpec("mr-ou-v1", "mean-reversion", "mr-primary"),
            AlphaModuleSpec("mr-vwap-v1", "mean-reversion", "mr-primary"),
            AlphaModuleSpec("trend-test-v1", "trend", "trend-breakout"),
            AlphaModuleSpec("macro-test-v1", "macro", "macro-drift"),
            *extra,
        )
    )


def signal(
    module: str,
    event: str,
    direction: str = "LONG",
    confidence: float | None = None,
) -> AlphaSignal:
    return AlphaSignal(
        module,
        event,
        "EURUSD",
        NOW,
        direction,
        confidence,
        SignalReference(1.08, "module-observation"),
        "module-owned-invalidation",
        SignalProvenance(f"{module}-spec", "synthetic-corpus", "synthetic-set"),
    )


class AcceptFixturePolicy:
    policy_id = "fixture-accept"
    policy_version = "1"

    def evaluate(self, proposals):
        return tuple(
            PolicyResult(
                item.proposal_id,
                PortfolioDecisionState.ACCEPT,
                "accepted by fixture",
            )
            for item in proposals
        )


class CrossSleeveConflictPolicy:
    policy_id = "fixture-cross-sleeve-conflict"
    policy_version = "1"

    def evaluate(self, proposals):
        has_mean_reversion = any(
            item.sleeve_id == "mean-reversion" for item in proposals
        )
        return tuple(
            PolicyResult(
                item.proposal_id,
                (
                    PortfolioDecisionState.DEFER
                    if item.sleeve_id == "trend" and has_mean_reversion
                    else PortfolioDecisionState.ACCEPT
                ),
                (
                    "deferred for simultaneous mean-reversion candidate"
                    if item.sleeve_id == "trend" and has_mean_reversion
                    else "accepted by cross-sleeve fixture"
                ),
            )
            for item in proposals
        )


def proposal(opportunity):
    return build_trade_proposal(
        opportunity,
        strategy_policy_id=f"{opportunity.sleeve_id}-construction-v1",
        entry_plan=PlanReference("strategy-entry", "entry-v1"),
        protective_plan=PlanReference("strategy-invalidation", "protect-v1"),
        provenance=ProposalProvenance("synthetic-planner", "1"),
    )


def test_duplicate_is_idempotent_and_identity_collision_fails_closed():
    original = signal("mr-ou-v1", "event-1", confidence=0.7)
    engine = OpportunityEngine(registry())
    opportunities = engine.build((original, original))
    assert len(opportunities) == 1
    assert opportunities[0].constituent_identities == (("mr-ou-v1", "event-1"),)

    with pytest.raises(OpportunityInvariantError, match="identity collision"):
        engine.build((original, replace(original, confidence=0.8)))


def test_explicit_same_family_aggregates_and_retains_lossless_evidence():
    ou = signal("mr-ou-v1", "ou-1", confidence=0.01)
    vwap = signal("mr-vwap-v1", "vwap-1", confidence=0.99)
    opportunity = OpportunityEngine(registry()).build((vwap, ou))[0]

    assert opportunity.status is OpportunityStatus.ACTIONABLE
    assert opportunity.signals == (ou, vwap)
    assert opportunity.constituent_identities == (
        ("mr-ou-v1", "ou-1"),
        ("mr-vwap-v1", "vwap-1"),
    )
    assert opportunity.signals[0].provenance == ou.provenance
    # Confidence remains attached to module-local evidence; it controls no ordering.
    assert [item.confidence for item in opportunity.signals] == [0.01, 0.99]


def test_same_family_opposite_directions_are_explicit_non_actionable_conflict():
    evidence = (
        signal("mr-ou-v1", "long", "LONG", 0.1),
        signal("mr-vwap-v1", "short", "SHORT", 0.99),
    )
    opportunity = OpportunityEngine(registry()).build(evidence)[0]
    assert opportunity.status is OpportunityStatus.CONFLICT
    assert opportunity.direction is None
    assert opportunity.conflict_directions == ("LONG", "SHORT")
    assert opportunity.signals == evidence
    with pytest.raises(ValueError, match="only actionable"):
        proposal(opportunity)


def test_different_sleeves_and_opposite_views_remain_independent():
    mr = signal("mr-ou-v1", "mr-short", "SHORT", 0.99)
    trend = signal("trend-test-v1", "trend-long", "LONG", 0.01)
    opportunities = OpportunityEngine(registry()).build((mr, trend))

    assert len(opportunities) == 2
    assert {item.sleeve_id for item in opportunities} == {"mean-reversion", "trend"}
    assert {item.direction for item in opportunities} == {"LONG", "SHORT"}
    assert all(item.status is OpportunityStatus.ACTIONABLE for item in opportunities)
    assert all(len(item.signals) == 1 for item in opportunities)


def test_unregistered_and_disabled_modules_fail_closed():
    with pytest.raises(RegistryInvariantError, match="unregistered"):
        OpportunityEngine(registry()).build((signal("weather-unknown", "wx"),))

    disabled = AlphaModuleSpec("weather-off-v1", "weather", "weather-edge", False)
    opportunity = OpportunityEngine(registry(disabled)).build(
        (signal("weather-off-v1", "wx"),)
    )[0]
    assert opportunity.status is OpportunityStatus.DISABLED
    with pytest.raises(ValueError, match="only actionable"):
        proposal(opportunity)


def test_registry_prevents_one_opportunity_family_spanning_sleeves():
    with pytest.raises(RegistryInvariantError, match="cannot span sleeves"):
        AlphaModuleRegistry(
            (
                AlphaModuleSpec("one", "sleeve-a", "shared"),
                AlphaModuleSpec("two", "sleeve-b", "shared"),
            )
        )


def test_input_order_restart_and_all_downstream_ids_are_deterministic():
    evidence = (
        signal("mr-ou-v1", "ou", confidence=0.8),
        signal("mr-vwap-v1", "vwap", confidence=0.2),
        signal("trend-test-v1", "trend"),
    )
    first = OpportunityEngine(registry()).build(evidence)
    restarted = OpportunityEngine(registry()).build(tuple(reversed(evidence)))
    assert first == restarted

    first_proposals = tuple(proposal(item) for item in first)
    restarted_proposals = tuple(proposal(item) for item in restarted)
    assert first_proposals == restarted_proposals
    kernel = PortfolioKernel(AcceptFixturePolicy())
    assert kernel.evaluate(first_proposals) == kernel.evaluate(restarted_proposals)


def test_batch_policy_observes_siblings_and_is_input_order_invariant():
    opportunities = OpportunityEngine(registry()).build(
        (
            signal("mr-ou-v1", "mr-short", "SHORT"),
            signal("trend-test-v1", "trend-long", "LONG"),
        )
    )
    proposals = tuple(proposal(item) for item in opportunities)

    forward = PortfolioKernel(CrossSleeveConflictPolicy()).evaluate(proposals)
    reverse = PortfolioKernel(CrossSleeveConflictPolicy()).evaluate(
        tuple(reversed(proposals))
    )

    assert forward == reverse
    by_sleeve = {decision.sleeve_id: decision.decision for decision in forward}
    assert by_sleeve == {
        "mean-reversion": PortfolioDecisionState.ACCEPT,
        "trend": PortfolioDecisionState.DEFER,
    }
    trend_only = next(item for item in proposals if item.sleeve_id == "trend")
    assert (
        PortfolioKernel(CrossSleeveConflictPolicy()).evaluate((trend_only,))[0].decision
        is PortfolioDecisionState.ACCEPT
    )


def test_policy_must_return_one_result_for_every_candidate():
    candidate = proposal(
        OpportunityEngine(registry()).build((signal("macro-test-v1", "event"),))[0]
    )

    class MissingResultPolicy:
        policy_id = "invalid-missing"
        policy_version = "1"

        def evaluate(self, proposals):
            return ()

    with pytest.raises(PortfolioPolicyInvariantError, match="exactly one"):
        PortfolioKernel(MissingResultPolicy()).evaluate((candidate,))

    class DuplicateResultPolicy:
        policy_id = "invalid-duplicate"
        policy_version = "1"

        def evaluate(self, proposals):
            result = PolicyResult(
                proposals[0].proposal_id,
                PortfolioDecisionState.ACCEPT,
                "duplicate",
            )
            return (result, result)

    with pytest.raises(PortfolioPolicyInvariantError, match="duplicate"):
        PortfolioKernel(DuplicateResultPolicy()).evaluate((candidate,))


def test_proposal_identity_includes_economic_direction():
    opportunity = OpportunityEngine(registry()).build(
        (signal("trend-test-v1", "direction-event", "LONG"),)
    )[0]
    long_proposal = proposal(opportunity)
    short_proposal = proposal(replace(opportunity, direction="SHORT"))

    assert long_proposal.opportunity_id == short_proposal.opportunity_id
    assert long_proposal.proposal_id != short_proposal.proposal_id
    assert long_proposal.proposal_id == proposal(opportunity).proposal_id


@pytest.mark.parametrize("module", ["trend-test-v1", "macro-test-v1"])
def test_non_ou_alpha_flows_through_generic_pipeline(module):
    opportunity = OpportunityEngine(registry()).build((signal(module, "event"),))[0]
    trade_proposal = proposal(opportunity)
    decision = PortfolioKernel(AcceptFixturePolicy()).evaluate((trade_proposal,))[0]

    assert decision.decision is PortfolioDecisionState.ACCEPT
    assert decision.proposal_id == trade_proposal.proposal_id
    assert decision.opportunity_id == opportunity.opportunity_id
    assert not hasattr(opportunity, "ou_score")
    assert not hasattr(trade_proposal, "half_life")


def test_portfolio_kernel_requires_explicit_policy():
    opportunity = OpportunityEngine(registry()).build(
        (signal("macro-test-v1", "event"),)
    )[0]
    with pytest.raises(MissingPortfolioPolicyError, match="explicit"):
        PortfolioKernel().evaluate((proposal(opportunity),))
