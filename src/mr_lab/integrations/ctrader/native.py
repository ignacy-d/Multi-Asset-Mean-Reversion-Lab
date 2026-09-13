"""Small duck-typed bridge for native cTrader values and market collections."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from .contracts import ClosedBar, MarketQuote
from .market import normalize_closed_bar, normalize_quote


class NativeConfigurationError(RuntimeError):
    """A configured platform resource cannot be resolved safely."""


def configured_symbol_names(symbols_csv: str) -> tuple[str, ...]:
    names = tuple(
        dict.fromkeys(
            item.strip().upper() for item in symbols_csv.split(",") if item.strip()
        )
    )
    if not names:
        raise NativeConfigurationError("no observation symbols configured")
    return names


def resolve_symbols(symbols_api: object, names: tuple[str, ...]) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for configured_name in names:
        symbol = symbols_api.GetSymbol(configured_name)
        if symbol is None:
            raise NativeConfigurationError(
                f"unknown configured symbol: {configured_name}"
            )
        native_name = str(symbol.Name).upper()
        if native_name != configured_name:
            raise NativeConfigurationError(
                f"configured symbol resolved ambiguously: {configured_name}"
            )
        resolved[configured_name] = symbol
    return resolved


def quote_provider(
    symbols: dict[str, object],
) -> Callable[[datetime], tuple[MarketQuote, ...]]:
    def observe(now: datetime) -> tuple[MarketQuote, ...]:
        observations: list[MarketQuote] = []
        for symbol in symbols.values():
            try:
                observations.append(normalize_quote(symbol, now))
            except (AttributeError, TypeError, ValueError):
                # A missing quote leaves its required watermark absent, making M4 STALE.
                continue
        return tuple(observations)

    return observe


class BarOpenedRouter:
    """Route exact multi-symbol boundaries from native ``Bars.BarOpened`` events."""

    def __init__(self, host: object, timeframe: str) -> None:
        self._host = host
        self._timeframe = timeframe
        self._seen: set[tuple[str, datetime, datetime]] = set()
        self._handlers: list[Callable[..., None]] = []

    def subscribe(self, market_data: object, symbols: dict[str, object]) -> None:
        for symbol in symbols.values():
            bars = market_data.GetBars(self._timeframe, symbol.Name)
            if bars is None:
                raise NativeConfigurationError(f"bars unavailable: {symbol.Name}")

            def handler(_args: object = None, *, bars=bars, symbol=symbol) -> None:
                self.on_bar_opened(symbol, bars)

            bars.BarOpened += handler
            self._handlers.append(handler)

    def on_bar_opened(self, symbol: object, bars: object) -> ClosedBar | None:
        # In BarOpened, Last(0) is new and Last(1) is exactly the just-closed bar.
        new_bar = bars.Last(0)
        closed_bar = bars.Last(1)
        normalized = normalize_closed_bar(
            str(symbol.Name), self._timeframe, closed_bar, new_bar.OpenTime
        )
        identity = (
            normalized.source_id,
            normalized.open_time,
            normalized.close_time,
        )
        if identity in self._seen:
            return None
        self._seen.add(identity)
        self._host.on_bar_closed(normalized)
        return normalized
