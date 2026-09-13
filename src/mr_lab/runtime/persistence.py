"""Narrow persistence port suitable for a future cTrader LocalStorage adapter."""

from __future__ import annotations

from typing import Protocol


class DurableStore(Protocol):
    def read_text(self, key: str) -> str | None: ...
    def write_text(self, key: str, value: str) -> None: ...
    def flush(self) -> None: ...


class InMemoryDurableStore:
    """Deterministic test adapter with explicit failure injection."""

    def __init__(self, *, fail_write: bool = False, fail_flush: bool = False) -> None:
        self.values: dict[str, str] = {}
        self.fail_write = fail_write
        self.fail_flush = fail_flush

    def read_text(self, key: str) -> str | None:
        return self.values.get(key)

    def write_text(self, key: str, value: str) -> None:
        if self.fail_write:
            raise OSError("injected durable write failure")
        self.values[key] = value

    def flush(self) -> None:
        if self.fail_flush:
            raise OSError("injected durable flush failure")
