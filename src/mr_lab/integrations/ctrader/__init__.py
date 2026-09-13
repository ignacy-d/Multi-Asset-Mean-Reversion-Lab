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
from .native import (
    BarOpenedRouter,
    NativeConfigurationError,
    configured_symbol_names,
    quote_provider,
    resolve_symbols,
)
from .storage import CTraderLocalStorageStore, encode_storage_key
from .time import to_python_utc

__all__ = [
    "AccountObservation",
    "BarOpenedRouter",
    "BrokerOwnershipError",
    "CTraderLocalStorageStore",
    "CTraderObservationHost",
    "ClosedBar",
    "LiveAccountRefused",
    "MarketQuote",
    "NativeConfigurationError",
    "NoTradeSignalPipeline",
    "ObservationLifecycle",
    "WouldExecuteObservation",
    "configured_symbol_names",
    "encode_storage_key",
    "normalize_account",
    "normalize_closed_bar",
    "normalize_quote",
    "observe_broker",
    "quote_provider",
    "resolve_symbols",
    "runtime_identity",
    "to_python_utc",
]
