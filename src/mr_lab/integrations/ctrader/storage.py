"""cTrader Device LocalStorage implementation of the M4 durable-store port."""

from __future__ import annotations

import hashlib
import re

_VALID_KEY = re.compile(r"^[A-Za-z0-9 ]{1,50}$")


def encode_storage_key(source_key: str) -> str:
    if not source_key:
        raise ValueError("storage key cannot be empty")
    encoded = "MRLAB " + hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:40]
    assert _VALID_KEY.fullmatch(encoded)
    return encoded


class CTraderLocalStorageStore:
    """Use an explicit Device scope and flush every M4 critical write."""

    def __init__(self, local_storage: object, device_scope: object) -> None:
        self._storage = local_storage
        self._scope = device_scope

    def read_text(self, key: str) -> str | None:
        value = self._storage.GetString(encode_storage_key(key), self._scope)
        return None if value is None else str(value)

    def write_text(self, key: str, value: str) -> None:
        self._storage.SetString(encode_storage_key(key), value, self._scope)

    def flush(self) -> None:
        self._storage.Flush(self._scope)
