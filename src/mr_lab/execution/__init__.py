"""Broker-agnostic execution commands, events, FSM, and deterministic test double."""

from .contracts import (
    BrokerEvent,
    CancelAcknowledged,
    CancelEntryCommand,
    EnsureProtectionCommand,
    EntryAccepted,
    EntryFill,
    EntryRejected,
    ExecutionClosed,
    ExecutionCommand,
    ExecutionLifecycle,
    ExecutionState,
    ProcessedEvent,
    ProtectionAcknowledged,
    ProtectionRejected,
    ProtectionStatus,
    SubmitEntryCommand,
)
from .fake_broker import FakeBroker, FakeBrokerScenario, FakeFill
from .fsm import ExecutionEngine, ExecutionInvariantError

__all__ = [
    "BrokerEvent",
    "CancelAcknowledged",
    "CancelEntryCommand",
    "EnsureProtectionCommand",
    "EntryAccepted",
    "EntryFill",
    "EntryRejected",
    "ExecutionClosed",
    "ExecutionCommand",
    "ExecutionEngine",
    "ExecutionInvariantError",
    "ExecutionLifecycle",
    "ExecutionState",
    "FakeBroker",
    "FakeBrokerScenario",
    "FakeFill",
    "ProcessedEvent",
    "ProtectionAcknowledged",
    "ProtectionRejected",
    "ProtectionStatus",
    "SubmitEntryCommand",
]
