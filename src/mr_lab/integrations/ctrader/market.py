"""Immediate copying of duck-typed cTrader market values."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from .contracts import ClosedBar, MarketQuote


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("platform timestamp must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("platform timestamp must be timezone-aware")
    return value.astimezone(UTC)


def source_id(symbol: str, timeframe: str, kind: str) -> str:
    return f"ctrader/{symbol.upper()}/{timeframe}/{kind}"


def normalize_quote(symbol: object, server_time: object) -> MarketQuote:
    name = str(symbol.Name)
    return MarketQuote(
        source_id("" + name, "TICK", "quote"),
        name,
        _utc(server_time),
        Decimal(str(symbol.Bid)),
        Decimal(str(symbol.Ask)),
    )


def normalize_closed_bar(
    symbol_name: str,
    timeframe: str,
    bar: object,
    close_time: object,
) -> ClosedBar:
    return ClosedBar(
        source_id(symbol_name, timeframe, "closed-bar"),
        symbol_name,
        timeframe,
        _utc(bar.OpenTime),
        _utc(close_time),
        Decimal(str(bar.Open)),
        Decimal(str(bar.High)),
        Decimal(str(bar.Low)),
        Decimal(str(bar.Close)),
    )
