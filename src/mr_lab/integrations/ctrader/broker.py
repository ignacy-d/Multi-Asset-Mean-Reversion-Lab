"""Conservative read-only ownership scan for startup reconciliation."""

from __future__ import annotations

from datetime import datetime

from mr_lab.runtime import BrokerRuntimeSnapshot

OWNERSHIP_PREFIX = "MRLAB-"


class BrokerOwnershipError(RuntimeError):
    pass


def _scan(objects: object) -> None:
    for item in objects:
        label = str(getattr(item, "Label", ""))
        if label.startswith(OWNERSHIP_PREFIX):
            raise BrokerOwnershipError(
                "apparent MR Lab broker object cannot be mapped in M5A; HALT"
            )


def observe_broker(
    positions: object, pending_orders: object, observed_at: datetime, runtime_id: str
) -> BrokerRuntimeSnapshot:
    _scan(positions)
    _scan(pending_orders)
    return BrokerRuntimeSnapshot(
        observed_at=observed_at, executions=(), account_reference=runtime_id
    )
