"""Deterministic synchronous broker-adapter test double."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from mr_lab.portfolio.identity import stable_id

from .contracts import (
    BrokerEvent,
    CancelAcknowledged,
    CancelEntryCommand,
    EnsureProtectionCommand,
    EntryAccepted,
    EntryFill,
    EntryRejected,
    ExecutionCommand,
    ProtectionAcknowledged,
    ProtectionRejected,
    SubmitEntryCommand,
)


@dataclass(frozen=True, slots=True)
class FakeFill:
    quantity: Decimal
    price: Decimal
    delayed: bool = False


@dataclass(frozen=True, slots=True)
class FakeBrokerScenario:
    accept_entry: bool = True
    rejection_reason: str = "configured entry rejection"
    fills: tuple[FakeFill, ...] = ()
    atomic_protection: bool = False
    protection_succeeds: bool = True
    protection_rejection_reason: str = "configured protection rejection"


@dataclass(slots=True)
class FakeBroker:
    """Maps generic commands to configured events without strategy decisions."""

    scenario: FakeBrokerScenario
    _delayed: list[BrokerEvent] = field(default_factory=list, init=False)

    def handle(self, command: ExecutionCommand) -> tuple[BrokerEvent, ...]:
        if isinstance(command, SubmitEntryCommand):
            return self._submit(command)
        if isinstance(command, EnsureProtectionCommand):
            return (self._protect(command),)
        if isinstance(command, CancelEntryCommand):
            return (
                CancelAcknowledged(
                    stable_id("fake-event", command.command_id, "cancelled"),
                    command.execution_intent_id,
                    command.command_id,
                    command.broker_order_ref,
                    command.created_at,
                ),
            )
        raise TypeError("unsupported execution command")

    def release_delayed(self) -> tuple[BrokerEvent, ...]:
        events = tuple(self._delayed)
        self._delayed.clear()
        return events

    def _submit(self, command: SubmitEntryCommand) -> tuple[BrokerEvent, ...]:
        if not self.scenario.accept_entry:
            return (
                EntryRejected(
                    stable_id("fake-event", command.command_id, "rejected"),
                    command.execution_intent_id,
                    command.command_id,
                    self.scenario.rejection_reason,
                    command.created_at,
                ),
            )
        order_ref = stable_id("fake-order", command.client_idempotency_key)
        protection_ref = (
            stable_id("fake-protection", command.execution_intent_id)
            if self.scenario.atomic_protection
            else None
        )
        events: list[BrokerEvent] = [
            EntryAccepted(
                stable_id("fake-event", command.command_id, "accepted"),
                command.execution_intent_id,
                command.command_id,
                order_ref,
                command.created_at,
                self.scenario.atomic_protection,
                protection_ref,
            )
        ]
        for index, fill in enumerate(self.scenario.fills, start=1):
            event = EntryFill(
                stable_id("fake-event", command.command_id, "fill", index),
                command.execution_intent_id,
                order_ref,
                fill.quantity,
                fill.price,
                command.created_at + timedelta(microseconds=index),
            )
            (self._delayed if fill.delayed else events).append(event)
        return tuple(events)

    def _protect(self, command: EnsureProtectionCommand) -> BrokerEvent:
        event_time: datetime = command.created_at
        event_id = stable_id("fake-event", command.command_id, "protection")
        if self.scenario.protection_succeeds:
            return ProtectionAcknowledged(
                event_id,
                command.execution_intent_id,
                command.command_id,
                command.broker_order_ref,
                stable_id("fake-protection", command.command_id),
                event_time,
            )
        return ProtectionRejected(
            event_id,
            command.execution_intent_id,
            command.command_id,
            command.broker_order_ref,
            self.scenario.protection_rejection_reason,
            event_time,
        )
