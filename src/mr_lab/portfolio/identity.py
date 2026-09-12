"""Canonical, restart-stable identities used by the portfolio runtime."""

import hashlib
import json
from datetime import datetime
from enum import Enum


def stable_id(namespace: str, *parts: object) -> str:
    def primitive(value: object) -> object:
        if isinstance(value, datetime):
            return value.isoformat(timespec="microseconds")
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [primitive(item) for item in value]
        return value

    payload = json.dumps(
        [primitive(part) for part in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{namespace}-{digest}"
