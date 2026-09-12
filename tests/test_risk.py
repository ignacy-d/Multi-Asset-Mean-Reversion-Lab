from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from mr_lab.portfolio import (
    PlanReference,
    PortfolioDecision,
    PortfolioDecisionState,
    ProposalProvenance,
    TradeProposal,
)
from mr_lab.risk import (
    AccountSnapshot,
    InstrumentSizingContext,
    OpenExposure,
    ProtectiveBoundary,
    RiskDecisionState,
    RiskInvariantError,
    RiskKernel,
    RiskPolicy,
    SleeveRiskLimit,
    build_execution_intent,
    build_risk_request,
)

NOW = datetime(2024, 6, 3, 10, tzinfo=UTC)
D = Decimal


def proposal(
    sleeve="mean-reversion", instrument="EURUSD", direction="LONG", suffix="1"
):
    return TradeProposal(
        f"proposal-{suffix}",
        f"opportunity-{suffix}",
        sleeve,
        instrument,
        direction,
        NOW,
        f"{sleeve}-strategy-v1",
        PlanReference("reference", "entry-v1"),
        PlanReference("invalidation", "protect-v1"),
        ProposalProvenance("fixture", "1"),
    )


def portfolio_decision(item, state=PortfolioDecisionState.ACCEPT):
    return PortfolioDecision(
        f"decision-{item.proposal_id}",
        item.proposal_id,
        item.opportunity_id,
        item.sleeve_id,
        state,
        "fixture",
        "portfolio-fixture",
        "1",
    )


def request(item=None, *, entry="1.1000", protective="1.0900", tags=()):
    item = item or proposal()
    return build_risk_request(
        portfolio_decision(item),
        item,
        protective_boundary=ProtectiveBoundary(
            D(entry), D(protective), "strategy-stop-v1"
        ),
        factor_tags=tags,
    )


def account(*exposures, equity="10000"):
    return AccountSnapshot(
        NOW,
        "USD",
        D("10000"),
        D(equity),
        D("10000"),
        D("0"),
        D("0"),
        D("10000"),
        exposures,
    )


def context(
    item=None,
    *,
    loss="0.01",
    minimum="1",
    maximum="1000000",
    step="100",
    entry="1.1000",
    protective="1.0900",
):
    item = item or proposal()
    return InstrumentSizingContext(
        f"sizing-{item.instrument}",
        "1",
        item.instrument,
        "USD",
        NOW,
        D(entry),
        D(protective),
        D(loss),
        D(minimum),
        D(maximum),
        D(step),
    )


def policy(
    *,
    version="1",
    aggregate="0.03",
    mr_trade="0.01",
    mr_cap="0.02",
    floor=None,
    maximum_new=None,
):
    return RiskPolicy(
        "synthetic-risk",
        version,
        D(aggregate),
        (
            SleeveRiskLimit("mean-reversion", D(mr_trade), D(mr_cap), 20),
            SleeveRiskLimit("trend", D("0.01"), D("0.02"), 10),
            SleeveRiskLimit("macro", D("0.01"), D("0.01"), 30),
        ),
        timedelta(seconds=1),
        D(floor) if floor else None,
        maximum_new_positions=maximum_new,
    )


def test_only_accepted_matching_portfolio_decision_builds_request():
    item = proposal()
    built = request(item)
    assert built.proposal_id == item.proposal_id
    with pytest.raises(RiskInvariantError, match="only an accepted"):
        build_risk_request(
            portfolio_decision(item, PortfolioDecisionState.DEFER),
            item,
            protective_boundary=built.protective_boundary,
        )
    with pytest.raises(RiskInvariantError, match="identity mismatch"):
        build_risk_request(
            replace(portfolio_decision(item), proposal_id="missing"),
            item,
            protective_boundary=built.protective_boundary,
        )


def test_protective_boundary_is_required_and_directionally_consistent():
    with pytest.raises(ValueError, match="LONG protective"):
        request(protective="1.1100")
    short = proposal(direction="SHORT")
    with pytest.raises(ValueError, match="SHORT protective"):
        request(short, protective="1.0900")
    with pytest.raises(ValueError, match="finite Decimal"):
        request(protective="NaN")


def test_sizing_is_exact_conservative_and_never_raises_to_minimum_or_above_maximum():
    req = request()
    decision = RiskKernel(policy(mr_trade="0.01005")).evaluate(
        (req,), account(), (context(loss="0.03", step="100"),)
    )[0]
    assert decision.decision is RiskDecisionState.ACCEPT
    assert decision.approved_quantity == D("3300")
    assert decision.approved_monetary_risk == D("99.00")

    below = RiskKernel(policy(mr_trade="0.00001")).evaluate(
        (req,), account(), (context(minimum="100", step="100"),)
    )[0]
    above = RiskKernel(policy()).evaluate(
        (req,), account(), (context(maximum="9000"),)
    )[0]
    assert below.decision is RiskDecisionState.REJECT
    assert above.decision is RiskDecisionState.REJECT


def test_account_floor_and_existing_aggregate_and_sleeve_risk_fail_closed():
    req = request()
    floor_result = RiskKernel(policy(floor="9000")).evaluate(
        (req,), account(equity="9000"), (context(),)
    )[0]
    assert floor_result.reason == "account equity at or below hard floor"
    existing = OpenExposure(
        "position-1", "USDJPY", "LONG", D("1"), D("150"), D("150"), D("250"), "trend"
    )
    aggregate_result = RiskKernel(policy(aggregate="0.03")).evaluate(
        (req,), account(existing), (context(),)
    )[0]
    assert aggregate_result.reason == "aggregate open-risk limit exceeded"


def test_complete_batch_is_order_invariant_and_joint_limit_is_enforced():
    mr = request(proposal(suffix="mr"))
    trend_item = proposal("trend", "GBPUSD", "SHORT", "trend")
    trend = request(trend_item, protective="1.1100")
    contexts = (context(), context(trend_item, entry="1.1000", protective="1.1100"))
    kernel = RiskKernel(policy(aggregate="0.015"))
    forward = kernel.evaluate((mr, trend), account(), contexts)
    reverse = kernel.evaluate((trend, mr), account(), tuple(reversed(contexts)))
    assert forward == reverse
    assert [item.decision for item in forward].count(RiskDecisionState.ACCEPT) == 1
    accepted = next(
        item for item in forward if item.decision is RiskDecisionState.ACCEPT
    )
    assert (
        accepted.proposal_id == trend.proposal_id
    )  # explicit lower priority value wins


def test_sleeve_limit_is_independent_and_unknown_sleeve_rejects():
    macro_item = proposal("macro", "AUDUSD", suffix="macro")
    macro = request(macro_item)
    existing = OpenExposure(
        "macro-open", "NZDUSD", "LONG", D("1"), D("1"), D("1"), D("50"), "macro"
    )
    macro_result = RiskKernel(policy()).evaluate(
        (macro,), account(existing), (context(macro_item),)
    )[0]
    assert macro_result.reason == "sleeve open-risk limit exceeded"
    weather_item = proposal("weather", suffix="weather")
    weather = request(weather_item)
    assert (
        RiskKernel(policy())
        .evaluate((weather,), account(), (context(weather_item),))[0]
        .reason
        == "unknown sleeve"
    )


def test_duplicate_restart_processing_and_ids_are_deterministic():
    req = request(tags=("USD",))
    kernel = RiskKernel(policy())
    first = kernel.evaluate((req, req), account(), (context(),))
    restarted = RiskKernel(policy()).evaluate((req,), account(), (context(),))
    assert first == restarted
    assert build_execution_intent(req, first[0]) == build_execution_intent(
        req, restarted[0]
    )
    with pytest.raises(RiskInvariantError, match="identity collision"):
        kernel.evaluate(
            (req, replace(req, strategy_policy_id="changed-payload")),
            account(),
            (context(),),
        )


def test_economic_changes_propagate_through_stable_identities():
    long = request()
    changed_stop = request(protective="1.0800")
    short_item = proposal(direction="SHORT")
    short = request(short_item, protective="1.1100")
    assert (
        len({long.risk_request_id, changed_stop.risk_request_id, short.risk_request_id})
        == 3
    )
    first = RiskKernel(policy()).evaluate((long,), account(), (context(),))[0]
    changed_policy = RiskKernel(policy(version="2")).evaluate(
        (long,), account(), (context(),)
    )[0]
    assert first.risk_decision_id != changed_policy.risk_decision_id
    intent = build_execution_intent(long, first)
    changed_quantity = build_execution_intent(
        long, replace(first, approved_quantity=first.approved_quantity - D("100"))
    )
    assert intent.execution_intent_id != changed_quantity.execution_intent_id


@pytest.mark.parametrize(
    "sleeve,instrument,direction,protective",
    [
        ("mean-reversion", "EURUSD", "LONG", "1.0900"),
        ("trend", "GBPUSD", "SHORT", "1.1100"),
        ("macro", "AUDUSD", "LONG", "1.0900"),
    ],
)
def test_alpha_family_details_and_confidence_are_not_risk_inputs(
    sleeve, instrument, direction, protective
):
    item = proposal(sleeve, instrument, direction, sleeve)
    req = request(item, protective=protective)
    result = RiskKernel(policy()).evaluate(
        (req,), account(), (context(item, protective=protective),)
    )[0]
    assert result.decision is RiskDecisionState.ACCEPT


def test_stale_state_boundary_and_required_daily_anchor_fail_closed():
    req = request()
    stale_account = replace(account(), timestamp=NOW + timedelta(seconds=2))
    assert (
        "stale"
        in RiskKernel(policy()).evaluate((req,), stale_account, (context(),))[0].reason
    )
    no_anchor = replace(account(), daily_loss_anchor=None)
    strict = replace(policy(), require_daily_loss_anchor=True)
    assert (
        "unavailable"
        in RiskKernel(strict).evaluate((req,), no_anchor, (context(),))[0].reason
    )
