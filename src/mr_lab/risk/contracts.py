"""Immutable, broker-neutral contracts at the account/risk boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from mr_lab.portfolio.contracts import (
    PlanReference,
    ProposalProvenance,
    require_text,
    require_utc,
)


def require_finite(name: str, value: Decimal, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def require_direction(direction: str) -> None:
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be LONG or SHORT")


@dataclass(frozen=True, slots=True)
class ProtectiveBoundary:
    entry_price: Decimal
    protective_price: Decimal
    risk_specification_id: str

    def __post_init__(self) -> None:
        require_finite("entry_price", self.entry_price, positive=True)
        require_finite("protective_price", self.protective_price, positive=True)
        require_text("risk_specification_id", self.risk_specification_id)

    def validate_for(self, direction: str) -> None:
        require_direction(direction)
        if direction == "LONG" and self.protective_price >= self.entry_price:
            raise ValueError("LONG protective price must be below entry price")
        if direction == "SHORT" and self.protective_price <= self.entry_price:
            raise ValueError("SHORT protective price must be above entry price")


@dataclass(frozen=True, slots=True)
class OpenExposure:
    exposure_id: str
    instrument: str
    direction: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    money_at_risk: Decimal
    sleeve_id: str
    proposal_id: str | None = None
    opportunity_id: str | None = None
    protective_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    factor_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("exposure_id", "instrument", "sleeve_id"):
            require_text(name, getattr(self, name))
        require_direction(self.direction)
        for name in ("quantity", "entry_price", "current_price"):
            require_finite(name, getattr(self, name), positive=True)
        require_finite("money_at_risk", self.money_at_risk)
        if self.money_at_risk < 0:
            raise ValueError("money_at_risk cannot be negative")
        for name in ("protective_price", "unrealized_pnl"):
            value = getattr(self, name)
            if value is not None:
                require_finite(name, value, positive=name == "protective_price")
        if len(set(self.factor_tags)) != len(self.factor_tags):
            raise ValueError("factor_tags must be unique")
        for tag in self.factor_tags:
            require_text("factor_tag", tag)


@dataclass(frozen=True, slots=True)
class LossLimitState:
    """Resolved account-equity floors supplied by an external rules layer."""

    specification_id: str
    version: str
    total_equity_floor: Decimal | None = None
    daily_equity_floor: Decimal | None = None

    def __post_init__(self) -> None:
        require_text("specification_id", self.specification_id)
        require_text("version", self.version)
        if self.total_equity_floor is None and self.daily_equity_floor is None:
            raise ValueError("loss-limit state must provide at least one equity floor")
        for name in ("total_equity_floor", "daily_equity_floor"):
            value = getattr(self, name)
            if value is not None:
                require_finite(name, value, positive=True)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    timestamp: datetime
    account_currency: str
    balance: Decimal
    equity: Decimal
    reference_balance: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    loss_limit_state: LossLimitState | None
    open_exposures: tuple[OpenExposure, ...] = ()

    def __post_init__(self) -> None:
        require_utc("timestamp", self.timestamp)
        require_text("account_currency", self.account_currency)
        for name in ("balance", "equity", "reference_balance"):
            require_finite(name, getattr(self, name), positive=True)
        for name in ("realized_pnl", "unrealized_pnl"):
            require_finite(name, getattr(self, name))
        if self.loss_limit_state is not None and not isinstance(
            self.loss_limit_state, LossLimitState
        ):
            raise ValueError("loss_limit_state must be a LossLimitState")
        seen: dict[str, OpenExposure] = {}
        for exposure in self.open_exposures:
            if exposure.exposure_id in seen:
                raise ValueError("exposure IDs must be unique")
            seen[exposure.exposure_id] = exposure


@dataclass(frozen=True, slots=True)
class RiskRequest:
    risk_request_id: str
    portfolio_decision_id: str
    proposal_id: str
    opportunity_id: str
    sleeve_id: str
    instrument: str
    direction: str
    timestamp: datetime
    strategy_policy_id: str
    entry_plan: PlanReference
    protective_plan: PlanReference
    proposal_provenance: ProposalProvenance
    protective_boundary: ProtectiveBoundary
    factor_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "risk_request_id",
            "portfolio_decision_id",
            "proposal_id",
            "opportunity_id",
            "sleeve_id",
            "instrument",
            "strategy_policy_id",
        ):
            require_text(name, getattr(self, name))
        require_utc("timestamp", self.timestamp)
        self.protective_boundary.validate_for(self.direction)
        if len(set(self.factor_tags)) != len(self.factor_tags):
            raise ValueError("factor_tags must be unique")


@dataclass(frozen=True, slots=True)
class InstrumentSizingContext:
    sizing_context_id: str
    version: str
    risk_request_id: str
    instrument: str
    account_currency: str
    timestamp: datetime
    entry_price: Decimal
    protective_price: Decimal
    loss_per_quantity: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal
    quantity_step: Decimal

    def __post_init__(self) -> None:
        for name in (
            "sizing_context_id",
            "version",
            "risk_request_id",
            "instrument",
            "account_currency",
        ):
            require_text(name, getattr(self, name))
        require_utc("timestamp", self.timestamp)
        for name in (
            "entry_price",
            "protective_price",
            "loss_per_quantity",
            "minimum_quantity",
            "maximum_quantity",
            "quantity_step",
        ):
            require_finite(name, getattr(self, name), positive=True)
        if self.minimum_quantity > self.maximum_quantity:
            raise ValueError("minimum_quantity cannot exceed maximum_quantity")


class RiskDecisionState(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    DEFER = "DEFER"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    risk_decision_id: str
    risk_request_id: str
    portfolio_decision_id: str
    proposal_id: str
    decision: RiskDecisionState
    approved_monetary_risk: Decimal
    approved_risk_fraction: Decimal
    approved_quantity: Decimal
    reason: str
    policy_id: str
    policy_version: str
    sizing_context_id: str
    sizing_context_version: str
    decided_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "risk_decision_id",
            "risk_request_id",
            "portfolio_decision_id",
            "proposal_id",
            "reason",
            "policy_id",
            "policy_version",
            "sizing_context_id",
            "sizing_context_version",
        ):
            require_text(name, getattr(self, name))
        for name in (
            "approved_monetary_risk",
            "approved_risk_fraction",
            "approved_quantity",
        ):
            value = getattr(self, name)
            require_finite(name, value)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        require_utc("decided_at", self.decided_at)
        if self.decision is RiskDecisionState.ACCEPT and (
            self.approved_monetary_risk <= 0 or self.approved_quantity <= 0
        ):
            raise ValueError(
                "accepted risk decisions require positive risk and quantity"
            )
        if self.decision is not RiskDecisionState.ACCEPT and any(
            value != 0
            for value in (
                self.approved_monetary_risk,
                self.approved_risk_fraction,
                self.approved_quantity,
            )
        ):
            raise ValueError("non-accepted risk decisions cannot approve risk")


@dataclass(frozen=True, slots=True)
class SizedExecutionIntent:
    execution_intent_id: str
    risk_decision_id: str
    proposal_id: str
    opportunity_id: str
    sleeve_id: str
    instrument: str
    direction: str
    quantity: Decimal
    strategy_policy_id: str
    entry_plan: PlanReference
    protective_plan: PlanReference
    proposal_provenance: ProposalProvenance
    risk_specification_id: str
    protective_boundary: ProtectiveBoundary
    proposal_timestamp: datetime
    approved_at: datetime
    factor_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "execution_intent_id",
            "risk_decision_id",
            "proposal_id",
            "opportunity_id",
            "sleeve_id",
            "instrument",
            "strategy_policy_id",
            "risk_specification_id",
        ):
            require_text(name, getattr(self, name))
        require_direction(self.direction)
        require_finite("quantity", self.quantity, positive=True)
        require_utc("proposal_timestamp", self.proposal_timestamp)
        require_utc("approved_at", self.approved_at)
        self.protective_boundary.validate_for(self.direction)
