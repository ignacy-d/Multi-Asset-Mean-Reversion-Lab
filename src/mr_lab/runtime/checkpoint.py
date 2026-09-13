"""Canonical checkpoint representation and strict JSON codec."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from mr_lab.execution import ExecutionState
from mr_lab.portfolio.contracts import require_text, require_utc

from .contracts import DataWatermark, RuntimeLifecycle, RuntimeState

SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    pass


class UnsupportedCheckpointSchema(CheckpointError):
    pass


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return {
            "$enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, datetime):
        require_utc("datetime", value)
        return {"$datetime": value.isoformat().replace("+00:00", "Z")}
    if isinstance(value, timedelta):
        return {"$timedelta_us": int(value.total_seconds() * 1_000_000)}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value)!r}")


def _resolve(path: str) -> type[Any]:
    if not path.startswith(
        ("mr_lab.execution.", "mr_lab.portfolio.", "mr_lab.risk.", "mr_lab.runtime.")
    ):
        raise CheckpointError("checkpoint contains an unapproved type")
    module_name, name = path.rsplit(".", 1)
    try:
        value = getattr(importlib.import_module(module_name), name)
    except (ImportError, AttributeError) as exc:
        raise CheckpointError(f"unknown checkpoint type: {path}") from exc
    if not isinstance(value, type):
        raise CheckpointError(f"invalid checkpoint type: {path}")
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    if set(value) == {"$decimal"}:
        return Decimal(value["$decimal"])
    if set(value) == {"$datetime"}:
        parsed = datetime.fromisoformat(value["$datetime"].replace("Z", "+00:00"))
        return parsed.astimezone(UTC)
    if set(value) == {"$timedelta_us"}:
        return timedelta(microseconds=value["$timedelta_us"])
    if set(value) == {"$enum", "value"}:
        return _resolve(value["$enum"])(value["value"])
    if set(value) == {"$type", "fields"} and isinstance(value["fields"], dict):
        cls = _resolve(value["$type"])
        allowed = {field.name for field in fields(cls)}
        if set(value["fields"]) != allowed:
            raise CheckpointError("checkpoint dataclass fields are incompatible")
        return cls(**{key: _decode(item) for key, item in value["fields"].items()})
    raise CheckpointError("unknown checkpoint JSON object")


@dataclass(frozen=True, slots=True)
class RuntimeCheckpoint:
    schema_version: int
    runtime_key: str
    checkpoint_sequence: int
    saved_at: datetime
    previous_lifecycle: RuntimeLifecycle
    executions: tuple[ExecutionState, ...]
    watermarks: tuple[DataWatermark, ...]
    checkpoint_identity: str

    def __post_init__(self) -> None:
        require_text("runtime_key", self.runtime_key)
        require_utc("saved_at", self.saved_at)
        if self.schema_version != SCHEMA_VERSION:
            raise UnsupportedCheckpointSchema(str(self.schema_version))
        if self.checkpoint_sequence < 1:
            raise ValueError("checkpoint sequence must be positive")
        object.__setattr__(
            self,
            "executions",
            tuple(sorted(self.executions, key=lambda x: x.execution_intent_id)),
        )
        object.__setattr__(
            self,
            "watermarks",
            tuple(sorted(self.watermarks, key=lambda x: x.source_id)),
        )


def _identity_payload(checkpoint: RuntimeCheckpoint) -> dict[str, Any]:
    return {
        key: value
        for key, value in _encode(checkpoint)["fields"].items()
        if key != "checkpoint_identity"
    }


def checkpoint_identity(checkpoint: RuntimeCheckpoint) -> str:
    payload = json.dumps(
        _identity_payload(checkpoint), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_checkpoint(state: RuntimeState, saved_at: datetime) -> RuntimeCheckpoint:
    checkpoint = RuntimeCheckpoint(
        SCHEMA_VERSION,
        state.runtime_key,
        state.checkpoint_sequence + 1,
        saved_at,
        state.lifecycle,
        state.executions,
        state.watermarks,
        "pending",
    )
    return replace(checkpoint, checkpoint_identity=checkpoint_identity(checkpoint))


def dumps_checkpoint(checkpoint: RuntimeCheckpoint) -> str:
    if checkpoint.checkpoint_identity != checkpoint_identity(checkpoint):
        raise CheckpointError("checkpoint identity mismatch")
    return json.dumps(
        _encode(checkpoint), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def loads_checkpoint(payload: str) -> RuntimeCheckpoint:
    try:
        raw = json.loads(payload)
        if (
            not isinstance(raw, dict)
            or raw.get("fields", {}).get("schema_version") != SCHEMA_VERSION
        ):
            version = (
                raw.get("fields", {}).get("schema_version")
                if isinstance(raw, dict)
                else None
            )
            raise UnsupportedCheckpointSchema(str(version))
        checkpoint = _decode(raw)
    except UnsupportedCheckpointSchema:
        raise
    except Exception as exc:
        raise CheckpointError("corrupt checkpoint") from exc
    if not isinstance(
        checkpoint, RuntimeCheckpoint
    ) or checkpoint.checkpoint_identity != checkpoint_identity(checkpoint):
        raise CheckpointError("checkpoint identity mismatch")
    return checkpoint
