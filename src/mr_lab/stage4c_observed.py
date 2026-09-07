"""Stage 4C-B exact-time historical BID/ASK cost proxy.

This remains independent of Stage 4C-A.  It does not rewrite BID-derived TP/SL
paths: without tick-order and full two-sided paths that question is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from mr_lab.data import QuoteStatus, SynchronizedQuote
from mr_lab.identity import methodology_id


class ObservedExecutionUnavailable(ValueError):
    """The exact quote required for an execution observation is unavailable."""


@dataclass(frozen=True, slots=True)
class ObservedExecutionSpec:
    commission_round_turn_pips: float = 0.0
    version: str = "stage-4c-b-observed-v1"

    @property
    def identity(self) -> str:
        return methodology_id("execution-cost-spec-v1", asdict(self))


def observed_execution_result(
    *,
    direction: str,
    entry_at: datetime,
    exit_at: datetime,
    quotes: tuple[SynchronizedQuote, ...],
    spec: ObservedExecutionSpec,
) -> dict[str, object]:
    """Price entry/exit at exact observed sides, with no quote filling."""
    by_time = {quote.observed_at: quote for quote in quotes}
    try:
        entry, exit_ = by_time[entry_at], by_time[exit_at]
    except KeyError as error:
        raise ObservedExecutionUnavailable(
            "exact entry/exit quote is absent"
        ) from error
    if (
        entry.status is not QuoteStatus.SYNCHRONIZED
        or exit_.status is not QuoteStatus.SYNCHRONIZED
    ):
        raise ObservedExecutionUnavailable("entry/exit quote is malformed or one-sided")
    assert entry.bid is not None and entry.ask is not None
    assert exit_.bid is not None and exit_.ask is not None
    if direction == "LONG":
        entry_price, exit_price = entry.ask, exit_.bid
        gross_price = exit_price - entry_price
    elif direction == "SHORT":
        entry_price, exit_price = entry.bid, exit_.ask
        gross_price = entry_price - exit_price
    else:
        raise ValueError("direction must be LONG or SHORT")
    if spec.commission_round_turn_pips < 0:
        raise ValueError("commission must be non-negative")
    return {
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "observed_return_pips": gross_price / entry.pip_size
        - spec.commission_round_turn_pips,
        "entry_spread_pips": entry.spread_pips,
        "exit_spread_pips": exit_.spread_pips,
        "execution_cost_spec_id": spec.identity,
        "pathwise_tp_sl_status": "unavailable_not_resimulated",
        "venue_semantics": "dukascopy_historical_proxy_not_broker_fill",
    }
