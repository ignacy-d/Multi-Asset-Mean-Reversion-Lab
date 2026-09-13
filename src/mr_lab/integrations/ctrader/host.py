"""Synchronous multi-symbol cTrader observation host with no execution port."""

from __future__ import annotations

from datetime import datetime, timedelta

from mr_lab.runtime import FreshnessPolicy, RuntimeController, RuntimeLifecycle

from .account import normalize_account
from .broker import BrokerOwnershipError, observe_broker
from .contracts import ClosedBar, ObservationLifecycle, SignalOnlyPipeline
from .telemetry import bar_line, signal_line, start_lines


class NoTradeSignalPipeline:
    def on_bar_closed(self, bar: ClosedBar) -> tuple[()]:
        return ()


class CTraderObservationHost:
    """Own one central runtime for every configured symbol, without a writer."""

    def __init__(
        self,
        *,
        account: object,
        positions: object,
        pending_orders: object,
        store: object,
        symbols: tuple[str, ...],
        logger: object,
        started_at: datetime,
        pipeline: SignalOnlyPipeline | None = None,
        maximum_broker_age: timedelta = timedelta(seconds=30),
    ) -> None:
        self.account_api = account
        self.positions = positions
        self.pending_orders = pending_orders
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        self.logger = logger
        self.pipeline = pipeline or NoTradeSignalPipeline()
        self.account = normalize_account(account, started_at)
        self.lifecycle = ObservationLifecycle.BOOTSTRAP
        self.controller = RuntimeController(
            self.account.runtime_id,
            store,
            FreshnessPolicy((), timedelta(0), maximum_broker_age),
            started_at,
        )

    def start(self, now: datetime) -> None:
        for line in start_lines(self.account):
            self.logger(line)
        self.controller.restore()
        try:
            snapshot = observe_broker(
                self.positions, self.pending_orders, now, self.account.runtime_id
            )
        except BrokerOwnershipError as exc:
            self.halt(str(exc), now)
            raise
        self.controller.reconcile_startup(snapshot, (), now)
        self._sync_lifecycle()
        self.logger(f"MRLAB STATUS lifecycle={self.lifecycle}")

    def heartbeat(self, now: datetime) -> None:
        try:
            snapshot = observe_broker(
                self.positions, self.pending_orders, now, self.account.runtime_id
            )
        except BrokerOwnershipError as exc:
            self.halt(str(exc), now)
            raise
        self.controller.refresh(snapshot, (), now)
        self.account = normalize_account(self.account_api, now)
        self._sync_lifecycle()
        self.logger(
            f"MRLAB STATUS lifecycle={self.lifecycle} broker_age_ms=0 "
            f"equity={self.account.equity} balance={self.account.balance}"
        )

    def on_bar_closed(self, bar: ClosedBar) -> None:
        self.logger(bar_line(bar))
        observations = self.pipeline.on_bar_closed(bar)
        if not observations:
            self.logger("MRLAB SIGNAL none")
        for observation in observations:
            self.logger(signal_line(observation))

    def halt(self, diagnostic: str, now: datetime) -> None:
        self.controller.halt(diagnostic, now)
        self.lifecycle = ObservationLifecycle.HALTED
        self.logger(f"MRLAB HALT diagnostic={type(diagnostic).__name__}")

    def _sync_lifecycle(self) -> None:
        core = self.controller.state.lifecycle
        self.lifecycle = {
            RuntimeLifecycle.SYNCED: ObservationLifecycle.SYNCED,
            RuntimeLifecycle.STALE: ObservationLifecycle.STALE,
            RuntimeLifecycle.HALTED: ObservationLifecycle.HALTED,
        }.get(core, ObservationLifecycle.BOOTSTRAP)
