"""Pure command/event finite-state machine for one economic execution intent."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from mr_lab.portfolio.contracts import require_utc
from mr_lab.portfolio.identity import stable_id
from mr_lab.risk import SizedExecutionIntent

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


class ExecutionInvariantError(RuntimeError):
    """An impossible transition, corrupt replay, or identity conflict."""


def _fingerprint(value: object) -> str:
    return stable_id("execution-fingerprint", value)


class ExecutionEngine:
    """Stateless FSM: callers persist state and dispatch returned commands.

    ``SUBMITTING``, ``PROTECTION_PENDING``, and ``CANCEL_PENDING`` deliberately
    do not auto-resend.  A future runtime reconciliation boundary must establish
    whether the adapter received or applied each persisted deterministic command
    before deciding whether a resend is safe.
    """

    def create_state(self, intent: SizedExecutionIntent) -> ExecutionState:
        return ExecutionState(
            execution_intent_id=intent.execution_intent_id,
            intent_fingerprint=_fingerprint(intent),
            proposal_id=intent.proposal_id,
            instrument=intent.instrument,
            direction=intent.direction,
            requested_quantity=intent.quantity,
            filled_quantity=Decimal(0),
            average_fill_price=None,
            lifecycle=ExecutionLifecycle.NEW,
            entry_plan=intent.entry_plan,
            protective_plan=intent.protective_plan,
            protective_boundary=intent.protective_boundary,
            strategy_policy_id=intent.strategy_policy_id,
            risk_decision_id=intent.risk_decision_id,
            proposal_provenance=intent.proposal_provenance,
            created_at=intent.approved_at,
            updated_at=intent.approved_at,
        )

    def handle_intent(
        self,
        intent: SizedExecutionIntent,
        state: ExecutionState | None = None,
    ) -> tuple[ExecutionState, tuple[ExecutionCommand, ...]]:
        if state is None:
            state = self.create_state(intent)
        if state.execution_intent_id != intent.execution_intent_id:
            raise ExecutionInvariantError("execution intent identity mismatch")
        if state.intent_fingerprint != _fingerprint(intent):
            raise ExecutionInvariantError(
                "execution_intent_id was reused with a different payload"
            )
        if state.lifecycle is not ExecutionLifecycle.NEW:
            return state, ()

        command_id = stable_id("execution-command", intent.execution_intent_id, "entry")
        command = SubmitEntryCommand(
            command_id=command_id,
            client_idempotency_key=stable_id(
                "execution-client-key", intent.execution_intent_id
            ),
            execution_intent_id=intent.execution_intent_id,
            instrument=intent.instrument,
            direction=intent.direction,
            quantity=intent.quantity,
            entry_plan=intent.entry_plan,
            protective_plan=intent.protective_plan,
            protective_boundary=intent.protective_boundary,
            strategy_policy_id=intent.strategy_policy_id,
            risk_decision_id=intent.risk_decision_id,
            proposal_id=intent.proposal_id,
            proposal_provenance=intent.proposal_provenance,
            proposal_timestamp=intent.proposal_timestamp,
            approved_at=intent.approved_at,
            created_at=intent.approved_at,
        )
        return (
            replace(
                state,
                lifecycle=ExecutionLifecycle.SUBMITTING,
                entry_command_id=command_id,
            ),
            (command,),
        )

    def handle_event(
        self, state: ExecutionState, event: BrokerEvent
    ) -> tuple[ExecutionState, tuple[ExecutionCommand, ...]]:
        if event.execution_intent_id != state.execution_intent_id:
            raise ExecutionInvariantError("broker event belongs to another execution")
        event_fingerprint = _fingerprint(event)
        for processed in state.processed_events:
            if processed.event_id == event.event_id:
                if processed.fingerprint != event_fingerprint:
                    raise ExecutionInvariantError(
                        "broker event identity was reused with a different payload"
                    )
                return state, ()
        if event.event_time < state.updated_at:
            raise ExecutionInvariantError("broker event time moved backwards")

        new_state, commands = self._apply_event(state, event)
        return (
            replace(
                new_state,
                updated_at=event.event_time,
                processed_events=(
                    *state.processed_events,
                    ProcessedEvent(event.event_id, event_fingerprint),
                ),
            ),
            commands,
        )

    def request_cancel(
        self, state: ExecutionState, requested_at: datetime
    ) -> tuple[ExecutionState, tuple[CancelEntryCommand, ...]]:
        require_utc("requested_at", requested_at)
        if requested_at < state.updated_at:
            raise ExecutionInvariantError("cancel request time moved backwards")
        if state.lifecycle is ExecutionLifecycle.CANCEL_PENDING:
            return state, ()
        if state.lifecycle is not ExecutionLifecycle.ACKNOWLEDGED:
            raise ExecutionInvariantError("entry cannot be cancelled in this lifecycle")
        if state.broker_order_ref is None or state.remaining_quantity <= 0:
            raise ExecutionInvariantError(
                "cannot cancel an entry without remaining quantity"
            )
        command_id = stable_id(
            "execution-command", state.execution_intent_id, "cancel-entry"
        )
        command = CancelEntryCommand(
            command_id,
            state.execution_intent_id,
            state.broker_order_ref,
            requested_at,
        )
        return (
            replace(
                state,
                lifecycle=ExecutionLifecycle.CANCEL_PENDING,
                cancel_command_id=command_id,
                updated_at=requested_at,
            ),
            (command,),
        )

    def _apply_event(
        self, state: ExecutionState, event: BrokerEvent
    ) -> tuple[ExecutionState, tuple[ExecutionCommand, ...]]:
        if state.is_terminal:
            raise ExecutionInvariantError(
                "terminal execution cannot process a new event"
            )
        if isinstance(event, EntryAccepted):
            return self._accept(state, event), ()
        if isinstance(event, EntryRejected):
            return self._reject(state, event), ()
        if isinstance(event, EntryFill):
            return self._fill(state, event)
        if isinstance(event, CancelAcknowledged):
            return self._cancelled(state, event), ()
        if isinstance(event, ProtectionAcknowledged):
            return self._protection_acknowledged(state, event)
        if isinstance(event, ProtectionRejected):
            return self._protection_rejected(state, event), ()
        if isinstance(event, ExecutionClosed):
            return self._closed(state, event), ()
        raise ExecutionInvariantError("unsupported broker event")

    @staticmethod
    def _require_entry_command(state: ExecutionState, command_id: str) -> None:
        if state.entry_command_id != command_id:
            raise ExecutionInvariantError("event references an unrelated entry command")

    def _accept(self, state: ExecutionState, event: EntryAccepted) -> ExecutionState:
        self._require_entry_command(state, event.command_id)
        if state.lifecycle is not ExecutionLifecycle.SUBMITTING:
            raise ExecutionInvariantError("entry acceptance is out of order")
        return replace(
            state,
            lifecycle=ExecutionLifecycle.ACKNOWLEDGED,
            broker_order_ref=event.broker_order_ref,
            protection_status=(
                ProtectionStatus.ACTIVE
                if event.protection_active
                else ProtectionStatus.NOT_REQUESTED
            ),
            protection_ref=event.protection_ref,
            protected_quantity=(
                state.requested_quantity if event.protection_active else Decimal(0)
            ),
        )

    def _reject(self, state: ExecutionState, event: EntryRejected) -> ExecutionState:
        self._require_entry_command(state, event.command_id)
        if state.lifecycle is not ExecutionLifecycle.SUBMITTING:
            raise ExecutionInvariantError("entry rejection is out of order")
        return replace(
            state,
            lifecycle=ExecutionLifecycle.REJECTED,
            last_failure_reason=event.reason,
        )

    def _fill(
        self, state: ExecutionState, event: EntryFill
    ) -> tuple[ExecutionState, tuple[ExecutionCommand, ...]]:
        if state.lifecycle not in {
            ExecutionLifecycle.ACKNOWLEDGED,
            ExecutionLifecycle.PARTIALLY_FILLED,
            ExecutionLifecycle.PROTECTION_PENDING,
            ExecutionLifecycle.UNSAFE,
        }:
            raise ExecutionInvariantError("entry fill is out of order")
        if state.broker_order_ref != event.broker_order_ref:
            raise ExecutionInvariantError("fill references an unrelated broker order")
        cumulative = state.filled_quantity + event.quantity
        if cumulative > state.requested_quantity:
            raise ExecutionInvariantError("fill would exceed requested quantity")
        previous_value = (
            state.average_fill_price or Decimal(0)
        ) * state.filled_quantity
        average = (previous_value + event.price * event.quantity) / cumulative
        fully_filled = cumulative == state.requested_quantity

        if state.lifecycle is ExecutionLifecycle.UNSAFE:
            return replace(
                state,
                filled_quantity=cumulative,
                average_fill_price=average,
            ), ()

        if state.pending_protection_quantity is not None:
            return replace(
                state,
                filled_quantity=cumulative,
                average_fill_price=average,
            ), ()

        if (
            state.protection_status is ProtectionStatus.ACTIVE
            and state.protected_quantity >= cumulative
        ):
            lifecycle = (
                ExecutionLifecycle.PROTECTED
                if fully_filled
                else ExecutionLifecycle.PARTIALLY_FILLED
            )
            return replace(
                state,
                lifecycle=lifecycle,
                filled_quantity=cumulative,
                average_fill_price=average,
            ), ()

        return self._request_protection(
            replace(
                state,
                filled_quantity=cumulative,
                average_fill_price=average,
            ),
            cumulative,
            event.event_time,
        )

    @staticmethod
    def _request_protection(
        state: ExecutionState, target: Decimal, created_at: datetime
    ) -> tuple[ExecutionState, tuple[EnsureProtectionCommand, ...]]:
        if state.protection_command_id is not None:
            raise ExecutionInvariantError("protection command is already in flight")
        command_id = stable_id(
            "execution-command",
            state.execution_intent_id,
            "ensure-protection",
            target,
        )
        if state.broker_order_ref is None:
            raise ExecutionInvariantError("protection requires an acknowledged order")
        command = EnsureProtectionCommand(
            command_id,
            state.execution_intent_id,
            state.broker_order_ref,
            target,
            state.protective_plan,
            state.protective_boundary,
            created_at,
        )
        return replace(
            state,
            lifecycle=ExecutionLifecycle.PROTECTION_PENDING,
            protection_status=ProtectionStatus.PENDING,
            protection_command_id=command_id,
            pending_protection_quantity=target,
        ), (command,)

    @staticmethod
    def _cancelled(state: ExecutionState, event: CancelAcknowledged) -> ExecutionState:
        if state.lifecycle is not ExecutionLifecycle.CANCEL_PENDING:
            raise ExecutionInvariantError("cancel acknowledgement is out of order")
        if state.cancel_command_id != event.command_id:
            raise ExecutionInvariantError(
                "cancel event references an unrelated command"
            )
        if state.broker_order_ref != event.broker_order_ref:
            raise ExecutionInvariantError("cancel event references an unrelated order")
        if state.filled_quantity != 0:
            raise ExecutionInvariantError(
                "partially filled cancellation needs an explicit exposure lifecycle"
            )
        return replace(
            state,
            lifecycle=ExecutionLifecycle.CANCELLED,
            protection_status=ProtectionStatus.NOT_REQUESTED,
            protection_ref=None,
            protected_quantity=Decimal(0),
            protection_command_id=None,
            pending_protection_quantity=None,
        )

    def _protection_acknowledged(
        self, state: ExecutionState, event: ProtectionAcknowledged
    ) -> tuple[ExecutionState, tuple[ExecutionCommand, ...]]:
        if state.lifecycle is not ExecutionLifecycle.PROTECTION_PENDING:
            raise ExecutionInvariantError("protection acknowledgement is out of order")
        if state.protection_command_id != event.command_id:
            raise ExecutionInvariantError(
                "protection event references unrelated command"
            )
        if state.broker_order_ref != event.broker_order_ref:
            raise ExecutionInvariantError("protection event references unrelated order")
        target = state.pending_protection_quantity
        if target is None:
            raise ExecutionInvariantError("protection target is missing")
        acknowledged = replace(
            state,
            protection_status=ProtectionStatus.ACTIVE,
            protection_ref=event.protection_ref,
            protected_quantity=target,
            protection_command_id=None,
            pending_protection_quantity=None,
        )
        if state.filled_quantity > target:
            return self._request_protection(
                acknowledged, state.filled_quantity, event.event_time
            )
        lifecycle = (
            ExecutionLifecycle.PROTECTED
            if state.remaining_quantity == 0
            else ExecutionLifecycle.PARTIALLY_FILLED
        )
        return replace(
            acknowledged,
            lifecycle=lifecycle,
        ), ()

    @staticmethod
    def _protection_rejected(
        state: ExecutionState, event: ProtectionRejected
    ) -> ExecutionState:
        if state.lifecycle is not ExecutionLifecycle.PROTECTION_PENDING:
            raise ExecutionInvariantError("protection rejection is out of order")
        if state.protection_command_id != event.command_id:
            raise ExecutionInvariantError(
                "protection event references unrelated command"
            )
        if state.broker_order_ref != event.broker_order_ref:
            raise ExecutionInvariantError("protection event references unrelated order")
        return replace(
            state,
            lifecycle=ExecutionLifecycle.UNSAFE,
            protection_status=ProtectionStatus.FAILED,
            protection_command_id=None,
            pending_protection_quantity=None,
            last_failure_reason=event.reason,
        )

    @staticmethod
    def _closed(state: ExecutionState, event: ExecutionClosed) -> ExecutionState:
        if state.filled_quantity <= 0:
            raise ExecutionInvariantError(
                "an execution without exposure cannot be closed"
            )
        if state.remaining_quantity != 0:
            raise ExecutionInvariantError(
                "execution cannot close while entry quantity remains live"
            )
        if state.broker_order_ref != event.broker_order_ref:
            raise ExecutionInvariantError("close event references unrelated order")
        return replace(state, lifecycle=ExecutionLifecycle.CLOSED)
