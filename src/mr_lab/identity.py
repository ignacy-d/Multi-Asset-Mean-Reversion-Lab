"""Deterministic identities for versioned V2 research specifications."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum


def canonical_json(value: object) -> str:
    """Serialize semantic inputs without process- or insertion-order dependence."""

    def default(item: object) -> object:
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, datetime):
            if item.tzinfo is None:
                raise TypeError("identity datetime must be timezone-aware")
            return item.isoformat()
        if isinstance(item, date):
            return item.isoformat()
        raise TypeError(f"unsupported identity input: {type(item).__name__}")

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=default
    )


def methodology_id(namespace: str, inputs: Mapping[str, object]) -> str:
    """Hash a named specification using the repository's SHA-256 convention."""
    if not namespace.strip():
        raise ValueError("identity namespace must be non-empty")
    payload = canonical_json({"namespace": namespace, "inputs": dict(inputs)}).encode()
    return f"{namespace}:sha256:{hashlib.sha256(payload).hexdigest()}"
