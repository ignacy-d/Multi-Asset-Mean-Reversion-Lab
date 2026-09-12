"""Exact and conservative quantity sizing."""

from decimal import ROUND_FLOOR, Decimal

from .contracts import InstrumentSizingContext


def conservative_quantity(
    money_risk: Decimal, context: InstrumentSizingContext
) -> Decimal | None:
    """Return step-rounded quantity, or ``None`` when it is not executable."""
    if not money_risk.is_finite() or money_risk <= 0:
        return None
    raw = money_risk / context.loss_per_quantity
    quantity = (raw / context.quantity_step).to_integral_value(
        rounding=ROUND_FLOOR
    ) * context.quantity_step
    if quantity < context.minimum_quantity or quantity > context.maximum_quantity:
        return None
    return quantity
