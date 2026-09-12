"""Explicit configuration for generic account risk limits."""

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from mr_lab.portfolio.contracts import require_text

from .contracts import require_finite


@dataclass(frozen=True, slots=True)
class SleeveRiskLimit:
    sleeve_id: str
    risk_fraction_per_trade: Decimal
    maximum_open_risk_fraction: Decimal
    priority: int

    def __post_init__(self) -> None:
        require_text("sleeve_id", self.sleeve_id)
        for name in ("risk_fraction_per_trade", "maximum_open_risk_fraction"):
            value = getattr(self, name)
            require_finite(name, value, positive=True)
            if value > 1:
                raise ValueError(f"{name} cannot exceed one")


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    policy_id: str
    policy_version: str
    maximum_aggregate_open_risk_fraction: Decimal
    sleeve_limits: tuple[SleeveRiskLimit, ...]
    maximum_state_age: timedelta
    require_loss_limit_state: bool = False
    maximum_new_positions: int | None = None

    def __post_init__(self) -> None:
        require_text("policy_id", self.policy_id)
        require_text("policy_version", self.policy_version)
        require_finite(
            "maximum_aggregate_open_risk_fraction",
            self.maximum_aggregate_open_risk_fraction,
            positive=True,
        )
        if self.maximum_aggregate_open_risk_fraction > 1:
            raise ValueError("maximum aggregate risk fraction cannot exceed one")
        if self.maximum_state_age < timedelta(0):
            raise ValueError("maximum_state_age cannot be negative")
        ids = [limit.sleeve_id for limit in self.sleeve_limits]
        if len(ids) != len(set(ids)):
            raise ValueError("sleeve limits must be unique")
        if self.maximum_new_positions is not None and self.maximum_new_positions < 1:
            raise ValueError("maximum_new_positions must be positive")

    def sleeve(self, sleeve_id: str) -> SleeveRiskLimit | None:
        return next(
            (item for item in self.sleeve_limits if item.sleeve_id == sleeve_id), None
        )
