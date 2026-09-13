"""Read-only cTrader observation adapter (M5A)."""

from .account import LiveAccountRefused, normalize_account, runtime_identity
from .broker import BrokerOwnershipError, observe_broker
from .contracts import (
    AccountObservation,
    ClosedBar,
    MarketQuote,
    ObservationLifecycle,
    WouldExecuteObservation,
)
from .host import CTraderObservationHost, NoTradeSignalPipeline
from .market import normalize_closed_bar, normalize_quote
from .storage import CTraderLocalStorageStore, encode_storage_key

__all__ = [
    "AccountObservation",
    "BrokerOwnershipError",
    "CTraderLocalStorageStore",
    "CTraderObservationHost",
    "ClosedBar",
    "LiveAccountRefused",
    "MarketQuote",
    "NoTradeSignalPipeline",
    "ObservationLifecycle",
    "WouldExecuteObservation",
    "encode_storage_key",
    "normalize_account",
    "normalize_closed_bar",
    "normalize_quote",
    "observe_broker",
    "runtime_identity",
]
