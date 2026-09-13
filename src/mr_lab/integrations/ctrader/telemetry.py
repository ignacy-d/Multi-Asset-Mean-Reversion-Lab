"""Compact, credential-free observation telemetry."""

from __future__ import annotations

from .contracts import AccountObservation, ClosedBar, WouldExecuteObservation


def start_lines(account: AccountObservation) -> tuple[str, ...]:
    return (
        "MRLAB START",
        "mode=OBSERVATION",
        "account=DEMO",
        f"broker={account.broker_name}",
        f"runtime={account.runtime_id}",
        "lifecycle=BOOTSTRAP",
    )


def bar_line(bar: ClosedBar) -> str:
    return f"MRLAB BAR {bar.instrument} {bar.timeframe} close={bar.close}"


def signal_line(item: WouldExecuteObservation) -> str:
    return (
        f"MRLAB WOULD_EXECUTE {item.instrument} {item.direction} "
        f"qty={item.quantity} sleeve={item.sleeve_id}"
    )
