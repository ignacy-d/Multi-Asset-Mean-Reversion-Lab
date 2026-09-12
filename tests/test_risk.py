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
    LossLimitState,
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


def account(*exposures, equity="10000", loss_limits=None):
    return AccountSnapshot(
        NOW,
        "USD",
        D("10000"),
        D(equity),
        D("10000"),
        D("0"),
        D("0"),
        loss_limits,
        exposures,
    )


def context(
    req=None,
    *,
    loss="0.01",
    minimum="1",
    maximum="1000000",
    step="100",
    entry="1.1000",
    protective="1.0900",
):
    req = req or request()
    return InstrumentSizingContext(
        f"sizing-{req.risk_request_id}",
        "1",
        req.risk_request_id,
        req.instrument,
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
    maximum_new=None,
    require_loss_limits=False,
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
        require_loss_limits,
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
        (req,), account(), (context(req, loss="0.03", step="100"),)
    )[0]
    assert decision.decision is RiskDecisionState.ACCEPT
    assert decision.approved_quantity == D("3300")
    assert decision.approved_monetary_risk == D("99.00")

    below = RiskKernel(policy(mr_trade="0.00001")).evaluate(
        (req,), account(), (context(req, minimum="100", step="100"),)
    )[0]
    above = RiskKernel(policy()).evaluate(
        (req,), account(), (context(req, maximum="9000"),)
    )[0]
    assert below.decision is RiskDecisionState.REJECT
    assert above.decision is RiskDecisionState.REJECT


def test_existing_aggregate_and_sleeve_risk_fail_closed():
    req = request()
    existing = OpenExposure(
        "position-1", "USDJPY", "LONG", D("1"), D("150"), D("150"), D("250"), "trend"
    )
    aggregate_result = RiskKernel(policy(aggregate="0.03")).evaluate(
        (req,), account(existing), (context(req),)
    )[0]
    assert aggregate_result.reason == "aggregate open-risk limit exceeded"


def test_complete_batch_is_order_invariant_and_joint_limit_is_enforced():
    mr = request(proposal(suffix="mr"))
    trend_item = proposal("trend", "GBPUSD", "SHORT", "trend")
    trend = request(trend_item, protective="1.1100")
    contexts = (context(mr), context(trend, entry="1.1000", protective="1.1100"))
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


def test_equal_priority_group_is_atomic_when_capacity_cannot_fit_every_request():
    first = request(proposal(suffix="first"))
    second = request(proposal(instrument="GBPUSD", suffix="second"))
    first_context = context(first)
    second_context = context(second)
    kernel = RiskKernel(policy(aggregate="0.015"))

    forward = kernel.evaluate(
        (first, second), account(), (first_context, second_context)
    )
    reverse = kernel.evaluate(
        (second, first), account(), (second_context, first_context)
    )
    assert forward == reverse
    assert {item.decision for item in forward} == {RiskDecisionState.DEFER}
    assert {item.reason for item in forward} == {
        "equal-priority candidates contend for insufficient risk capacity"
    }

    renamed_first = replace(first, risk_request_id="zzz")
    renamed_second = replace(second, risk_request_id="aaa")
    renamed = kernel.evaluate(
        (renamed_first, renamed_second),
        account(),
        (
            replace(first_context, risk_request_id="zzz"),
            replace(second_context, risk_request_id="aaa"),
        ),
    )
    assert {item.decision for item in renamed} == {RiskDecisionState.DEFER}


def test_equal_priority_group_accepts_all_when_capacity_fits():
    first = request(proposal(suffix="first"))
    second = request(proposal(instrument="GBPUSD", suffix="second"))
    decisions = RiskKernel(policy(aggregate="0.03")).evaluate(
        (second, first), account(), (context(second), context(first))
    )
    assert all(item.decision is RiskDecisionState.ACCEPT for item in decisions)


def test_same_instrument_requests_use_independent_request_specific_contexts():
    mr = request(proposal(direction="SHORT", suffix="mr"), protective="1.1050")
    trend = request(proposal("trend", "EURUSD", "SHORT", "trend"), protective="1.1120")
    mr_context = context(mr, protective="1.1050", loss="0.005")
    trend_context = context(trend, protective="1.1120", loss="0.012")
    assert mr_context.loss_per_quantity != trend_context.loss_per_quantity

    kernel = RiskKernel(policy(aggregate="0.03"))
    forward = kernel.evaluate((mr, trend), account(), (mr_context, trend_context))
    reverse = kernel.evaluate((trend, mr), account(), (trend_context, mr_context))
    assert forward == reverse
    assert all(item.decision is RiskDecisionState.ACCEPT for item in forward)
    assert {item.approved_quantity for item in forward} == {D("20000"), D("8300")}


def test_conflicting_or_wrong_request_sizing_context_fails_closed():
    req = request()
    valid = context(req)
    conflicting = replace(
        valid, sizing_context_id="different", loss_per_quantity=D("1")
    )
    with pytest.raises(RiskInvariantError, match="conflicting sizing contexts"):
        RiskKernel(policy()).evaluate((req,), account(), (valid, conflicting))

    other = request(proposal(suffix="other"))
    wrongly_linked = replace(valid, risk_request_id=other.risk_request_id)
    result = RiskKernel(policy()).evaluate((req,), account(), (wrongly_linked,))[0]
    assert result.decision is RiskDecisionState.REJECT
    assert result.reason == "matching sizing context unavailable"


def test_resolved_total_and_daily_loss_floors_are_enforced_independently():
    req = request()
    total = LossLimitState("resolved-rules", "1", total_equity_floor=D("9950"))
    daily = LossLimitState("resolved-rules", "1", daily_equity_floor=D("9950"))
    total_result = RiskKernel(policy()).evaluate(
        (req,), account(loss_limits=total), (context(req),)
    )[0]
    daily_result = RiskKernel(policy()).evaluate(
        (req,), account(loss_limits=daily), (context(req),)
    )[0]
    assert total_result.reason == "new risk could breach resolved total equity floor"
    assert daily_result.reason == "new risk could breach resolved daily equity floor"

    inside = LossLimitState(
        "resolved-rules",
        "1",
        total_equity_floor=D("9800"),
        daily_equity_floor=D("9850"),
    )
    accepted = RiskKernel(policy()).evaluate(
        (req,), account(loss_limits=inside), (context(req),)
    )[0]
    assert accepted.decision is RiskDecisionState.ACCEPT


def test_resolved_loss_budget_includes_open_and_cumulative_batch_risk():
    existing = OpenExposure(
        "open", "USDJPY", "LONG", D("1"), D("150"), D("150"), D("75"), "trend"
    )
    exposure_floor = LossLimitState("resolved-rules", "1", total_equity_floor=D("9850"))
    req = request()
    assert (
        RiskKernel(policy())
        .evaluate(
            (req,), account(existing, loss_limits=exposure_floor), (context(req),)
        )[0]
        .reason
        == "new risk could breach resolved total equity floor"
    )

    trend = request(proposal("trend", "GBPUSD", "SHORT", "trend"), protective="1.1100")
    floor = replace(exposure_floor, total_equity_floor=D("9800"))
    decisions = RiskKernel(policy()).evaluate(
        (req, trend),
        account(loss_limits=floor),
        (context(req), context(trend, protective="1.1100")),
    )
    assert [item.decision for item in decisions].count(RiskDecisionState.ACCEPT) == 2

    tighter = replace(floor, total_equity_floor=D("9850"))
    constrained = RiskKernel(policy()).evaluate(
        (req, trend),
        account(loss_limits=tighter),
        (context(req), context(trend, protective="1.1100")),
    )
    assert [item.decision for item in constrained].count(RiskDecisionState.ACCEPT) == 1


def test_loss_limit_state_is_required_and_part_of_decision_identity():
    req = request()
    strict = policy(require_loss_limits=True)
    missing = RiskKernel(strict).evaluate((req,), account(), (context(req),))[0]
    assert missing.reason == "required loss-limit state unavailable"

    first_state = LossLimitState("rules", "1", total_equity_floor=D("9000"))
    second_state = replace(first_state, total_equity_floor=D("8900"))
    first = RiskKernel(strict).evaluate(
        (req,), account(loss_limits=first_state), (context(req),)
    )[0]
    second = RiskKernel(strict).evaluate(
        (req,), account(loss_limits=second_state), (context(req),)
    )[0]
    assert first.decision is second.decision is RiskDecisionState.ACCEPT
    assert first.risk_decision_id != second.risk_decision_id


def test_sleeve_limit_is_independent_and_unknown_sleeve_rejects():
    macro_item = proposal("macro", "AUDUSD", suffix="macro")
    macro = request(macro_item)
    existing = OpenExposure(
        "macro-open", "NZDUSD", "LONG", D("1"), D("1"), D("1"), D("50"), "macro"
    )
    macro_result = RiskKernel(policy()).evaluate(
        (macro,), account(existing), (context(macro),)
    )[0]
    assert macro_result.reason == "sleeve open-risk limit exceeded"
    weather_item = proposal("weather", suffix="weather")
    weather = request(weather_item)
    assert (
        RiskKernel(policy())
        .evaluate((weather,), account(), (context(weather),))[0]
        .reason
        == "unknown sleeve"
    )


def test_duplicate_restart_processing_and_ids_are_deterministic():
    req = request(tags=("USD",))
    kernel = RiskKernel(policy())
    first = kernel.evaluate((req, req), account(), (context(req),))
    restarted = RiskKernel(policy()).evaluate((req,), account(), (context(req),))
    assert first == restarted
    assert build_execution_intent(req, first[0]) == build_execution_intent(
        req, restarted[0]
    )
    with pytest.raises(RiskInvariantError, match="identity collision"):
        kernel.evaluate(
            (req, replace(req, strategy_policy_id="changed-payload")),
            account(),
            (context(req),),
        )


def test_strategy_plans_survive_into_execution_intent_and_affect_identity():
    item = proposal()
    req = request(item)
    decision = RiskKernel(policy()).evaluate((req,), account(), (context(req),))[0]
    intent = build_execution_intent(req, decision)
    assert req.entry_plan == item.entry_plan == intent.entry_plan
    assert req.protective_plan == item.protective_plan == intent.protective_plan
    assert intent.proposal_provenance == item.provenance

    changed_entry = request(
        replace(item, entry_plan=PlanReference("reference", "entry-v2"))
    )
    changed_protection = request(
        replace(
            item,
            protective_plan=PlanReference("invalidation", "protect-v2"),
        )
    )
    entry_decision = RiskKernel(policy()).evaluate(
        (changed_entry,), account(), (context(changed_entry),)
    )[0]
    protection_decision = RiskKernel(policy()).evaluate(
        (changed_protection,), account(), (context(changed_protection),)
    )[0]
    assert (
        intent.execution_intent_id
        != build_execution_intent(changed_entry, entry_decision).execution_intent_id
    )
    assert (
        intent.execution_intent_id
        != build_execution_intent(
            changed_protection, protection_decision
        ).execution_intent_id
    )


def test_risk_approval_time_is_snapshot_time_and_changes_downstream_identity():
    req = request()
    kernel = RiskKernel(policy())
    snapshot = account()
    first = kernel.evaluate((req,), snapshot, (context(req),))[0]
    restarted = kernel.evaluate((req,), snapshot, (context(req),))[0]
    first_intent = build_execution_intent(req, first)
    assert first == restarted
    assert first.decided_at == snapshot.timestamp
    assert first_intent.proposal_timestamp == req.timestamp
    assert first_intent.approved_at == snapshot.timestamp

    later_snapshot = replace(snapshot, timestamp=NOW + timedelta(seconds=1))
    later = kernel.evaluate((req,), later_snapshot, (context(req),))[0]
    later_intent = build_execution_intent(req, later)
    assert later.decided_at == later_snapshot.timestamp
    assert later.risk_decision_id != first.risk_decision_id
    assert later_intent.execution_intent_id != first_intent.execution_intent_id


def test_economic_changes_propagate_through_stable_identities():
    long = request()
    changed_stop = request(protective="1.0800")
    short_item = proposal(direction="SHORT")
    short = request(short_item, protective="1.1100")
    assert (
        len({long.risk_request_id, changed_stop.risk_request_id, short.risk_request_id})
        == 3
    )
    first = RiskKernel(policy()).evaluate((long,), account(), (context(long),))[0]
    changed_policy = RiskKernel(policy(version="2")).evaluate(
        (long,), account(), (context(long),)
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
        (req,), account(), (context(req, protective=protective),)
    )[0]
    assert result.decision is RiskDecisionState.ACCEPT


def test_stale_state_boundary_and_required_daily_anchor_fail_closed():
    req = request()
    stale_account = replace(account(), timestamp=NOW + timedelta(seconds=2))
    assert (
        "stale"
        in RiskKernel(policy())
        .evaluate((req,), stale_account, (context(req),))[0]
        .reason
    )
    no_anchor = replace(account(), loss_limit_state=None)
    strict = replace(policy(), require_loss_limit_state=True)
    assert (
        "unavailable"
        in RiskKernel(strict).evaluate((req,), no_anchor, (context(req),))[0].reason
    )
