"""Synchronous deterministic batch risk evaluation."""

from decimal import Decimal

from mr_lab.portfolio import PortfolioDecision, PortfolioDecisionState, TradeProposal
from mr_lab.portfolio.identity import stable_id

from .contracts import (
    AccountSnapshot,
    InstrumentSizingContext,
    ProtectiveBoundary,
    RiskDecision,
    RiskDecisionState,
    RiskRequest,
    SizedExecutionIntent,
)
from .policy import RiskPolicy
from .sizing import conservative_quantity


class RiskInvariantError(ValueError):
    """Critical input identity or linkage is inconsistent."""


def build_risk_request(
    decision: PortfolioDecision,
    proposal: TradeProposal,
    *,
    protective_boundary: ProtectiveBoundary,
    factor_tags: tuple[str, ...] = (),
) -> RiskRequest:
    if decision.decision is not PortfolioDecisionState.ACCEPT:
        raise RiskInvariantError(
            "only an accepted portfolio decision can become a risk request"
        )
    if (decision.proposal_id, decision.opportunity_id, decision.sleeve_id) != (
        proposal.proposal_id,
        proposal.opportunity_id,
        proposal.sleeve_id,
    ):
        raise RiskInvariantError("portfolio decision and proposal identity mismatch")
    protective_boundary.validate_for(proposal.direction)
    request_id = stable_id(
        "risk-request",
        decision.portfolio_decision_id,
        proposal.proposal_id,
        proposal.opportunity_id,
        proposal.sleeve_id,
        proposal.instrument,
        proposal.direction,
        proposal.timestamp,
        proposal.strategy_policy_id,
        protective_boundary.entry_price,
        protective_boundary.protective_price,
        protective_boundary.risk_specification_id,
        tuple(sorted(factor_tags)),
    )
    return RiskRequest(
        request_id,
        decision.portfolio_decision_id,
        proposal.proposal_id,
        proposal.opportunity_id,
        proposal.sleeve_id,
        proposal.instrument,
        proposal.direction,
        proposal.timestamp,
        proposal.strategy_policy_id,
        protective_boundary,
        tuple(sorted(factor_tags)),
    )


class RiskKernel:
    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy

    def evaluate(
        self,
        requests: tuple[RiskRequest, ...],
        account: AccountSnapshot,
        sizing_contexts: tuple[InstrumentSizingContext, ...],
    ) -> tuple[RiskDecision, ...]:
        candidates = self._deduplicate(requests)
        contexts = self._context_map(sizing_contexts)
        existing_total = sum(
            (item.money_at_risk for item in account.open_exposures), Decimal(0)
        )
        sleeve_used: dict[str, Decimal] = {}
        for exposure in account.open_exposures:
            sleeve_used[exposure.sleeve_id] = (
                sleeve_used.get(exposure.sleeve_id, Decimal(0)) + exposure.money_at_risk
            )
        aggregate_cap = (
            account.equity * self.policy.maximum_aggregate_open_risk_fraction
        )
        accepted_total = Decimal(0)
        accepted_count = 0
        ordered = sorted(
            candidates,
            key=lambda item: (
                (
                    self.policy.sleeve(item.sleeve_id).priority
                    if self.policy.sleeve(item.sleeve_id)
                    else 2**31
                ),
                item.risk_request_id,
            ),
        )
        decisions: list[RiskDecision] = []
        for request in ordered:
            context = contexts.get(request.risk_request_id)
            state, reason, monetary, fraction, quantity = self._assess(
                request,
                context,
                account,
                existing_total + accepted_total,
                sleeve_used,
                accepted_count,
                aggregate_cap,
            )
            context_id = context.sizing_context_id if context else "unavailable"
            context_version = context.version if context else "unavailable"
            decision_id = stable_id(
                "risk-decision",
                request.risk_request_id,
                self.policy.policy_id,
                self.policy.policy_version,
                account.timestamp,
                account.account_currency,
                account.balance,
                account.equity,
                account.reference_balance,
                account.realized_pnl,
                account.unrealized_pnl,
                account.loss_limit_state,
                tuple(
                    sorted(account.open_exposures, key=lambda item: item.exposure_id)
                ),
                context,
                state,
                reason,
                monetary,
                fraction,
                quantity,
            )
            decisions.append(
                RiskDecision(
                    decision_id,
                    request.risk_request_id,
                    request.portfolio_decision_id,
                    request.proposal_id,
                    state,
                    monetary,
                    fraction,
                    quantity,
                    reason,
                    self.policy.policy_id,
                    self.policy.policy_version,
                    context_id,
                    context_version,
                )
            )
            if state is RiskDecisionState.ACCEPT:
                accepted_total += monetary
                sleeve_used[request.sleeve_id] = (
                    sleeve_used.get(request.sleeve_id, Decimal(0)) + monetary
                )
                accepted_count += 1
        return tuple(sorted(decisions, key=lambda item: item.risk_request_id))

    def _assess(
        self,
        request,
        context,
        account,
        used_total,
        sleeve_used,
        accepted_count,
        aggregate_cap,
    ):
        zero = Decimal(0)
        limit = self.policy.sleeve(request.sleeve_id)
        if limit is None:
            return RiskDecisionState.REJECT, "unknown sleeve", zero, zero, zero
        if self.policy.require_loss_limit_state and account.loss_limit_state is None:
            return (
                RiskDecisionState.REJECT,
                "required loss-limit state unavailable",
                zero,
                zero,
                zero,
            )
        if (
            context is None
            or context.risk_request_id != request.risk_request_id
            or context.instrument != request.instrument
            or context.account_currency != account.account_currency
        ):
            return (
                RiskDecisionState.REJECT,
                "matching sizing context unavailable",
                zero,
                zero,
                zero,
            )
        if (
            abs(account.timestamp - request.timestamp) > self.policy.maximum_state_age
            or abs(context.timestamp - request.timestamp)
            > self.policy.maximum_state_age
        ):
            return (
                RiskDecisionState.REJECT,
                "account or sizing context is stale",
                zero,
                zero,
                zero,
            )
        boundary = request.protective_boundary
        if (
            context.entry_price != boundary.entry_price
            or context.protective_price != boundary.protective_price
        ):
            return (
                RiskDecisionState.REJECT,
                "sizing context boundary mismatch",
                zero,
                zero,
                zero,
            )
        if (
            self.policy.maximum_new_positions is not None
            and accepted_count >= self.policy.maximum_new_positions
        ):
            return (
                RiskDecisionState.REJECT,
                "simultaneous position limit exceeded",
                zero,
                zero,
                zero,
            )
        desired = account.equity * limit.risk_fraction_per_trade
        loss_limits = account.loss_limit_state
        if loss_limits is not None:
            worst_case_equity = account.equity - used_total - desired
            if (
                loss_limits.total_equity_floor is not None
                and worst_case_equity < loss_limits.total_equity_floor
            ):
                return (
                    RiskDecisionState.REJECT,
                    "new risk could breach resolved total equity floor",
                    zero,
                    zero,
                    zero,
                )
            if (
                loss_limits.daily_equity_floor is not None
                and worst_case_equity < loss_limits.daily_equity_floor
            ):
                return (
                    RiskDecisionState.REJECT,
                    "new risk could breach resolved daily equity floor",
                    zero,
                    zero,
                    zero,
                )
        sleeve_cap = account.equity * limit.maximum_open_risk_fraction
        if used_total + desired > aggregate_cap:
            return (
                RiskDecisionState.REJECT,
                "aggregate open-risk limit exceeded",
                zero,
                zero,
                zero,
            )
        if sleeve_used.get(request.sleeve_id, zero) + desired > sleeve_cap:
            return (
                RiskDecisionState.REJECT,
                "sleeve open-risk limit exceeded",
                zero,
                zero,
                zero,
            )
        quantity = conservative_quantity(desired, context)
        if quantity is None:
            return (
                RiskDecisionState.REJECT,
                "desired risk has no executable quantity",
                zero,
                zero,
                zero,
            )
        actual = quantity * context.loss_per_quantity
        return (
            RiskDecisionState.ACCEPT,
            "accepted within configured risk limits",
            actual,
            actual / account.equity,
            quantity,
        )

    @staticmethod
    def _deduplicate(requests):
        seen = {}
        for request in requests:
            prior = seen.get(request.risk_request_id)
            if prior is not None and prior != request:
                raise RiskInvariantError("risk request identity collision")
            seen[request.risk_request_id] = request
        return tuple(seen.values())

    @staticmethod
    def _context_map(contexts):
        result = {}
        for context in contexts:
            prior = result.get(context.risk_request_id)
            if prior is not None and prior != context:
                raise RiskInvariantError("conflicting sizing contexts for risk request")
            result[context.risk_request_id] = context
        return result


def build_execution_intent(
    request: RiskRequest, decision: RiskDecision
) -> SizedExecutionIntent:
    if (
        decision.decision is not RiskDecisionState.ACCEPT
        or decision.approved_quantity <= 0
    ):
        raise RiskInvariantError(
            "only an accepted, positive-size risk decision can become an "
            "execution intent"
        )
    if (
        decision.risk_request_id,
        decision.portfolio_decision_id,
        decision.proposal_id,
    ) != (request.risk_request_id, request.portfolio_decision_id, request.proposal_id):
        raise RiskInvariantError("risk decision and request identity mismatch")
    intent_id = stable_id(
        "execution-intent",
        decision.risk_decision_id,
        request.proposal_id,
        request.opportunity_id,
        request.sleeve_id,
        request.instrument,
        request.direction,
        decision.approved_quantity,
        request.strategy_policy_id,
        request.protective_boundary,
        request.timestamp,
    )
    return SizedExecutionIntent(
        intent_id,
        decision.risk_decision_id,
        request.proposal_id,
        request.opportunity_id,
        request.sleeve_id,
        request.instrument,
        request.direction,
        decision.approved_quantity,
        request.strategy_policy_id,
        request.protective_boundary.risk_specification_id,
        request.protective_boundary,
        request.timestamp,
        request.factor_tags,
    )
