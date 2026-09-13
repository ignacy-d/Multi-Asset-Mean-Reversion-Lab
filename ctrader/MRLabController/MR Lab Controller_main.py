"""Native cTrader Python entrypoint for M5A observation mode only."""
# ruff: noqa: E402, F403, F405

import sys
from datetime import timedelta
from pathlib import Path

import clr

clr.AddReference("cAlgo.API")

from cAlgo.API import *
from robot_wrapper import *

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from mr_lab.integrations.ctrader import (
    BarOpenedRouter,
    CTraderLocalStorageStore,
    CTraderObservationHost,
    NativeConfigurationError,
    configured_symbol_names,
    quote_provider,
    resolve_symbols,
    to_python_utc,
)


class MRLabController:
    """Callback facade; the generated wrapper supplies cTrader properties."""

    def on_start(self):
        if api.Account.IsLive:
            api.Print("MR LAB M5A REFUSES LIVE ACCOUNT")
            api.Stop()
            return
        if str(api.Mode).upper() != "OBSERVATION_ONLY":
            api.Print(f"MRLAB REFUSES UNSUPPORTED MODE mode={api.Mode}")
            api.Stop()
            return
        try:
            now = to_python_utc(api.Server.TimeInUtc)
            names = configured_symbol_names(str(api.SymbolsCsv))
            self._symbols = resolve_symbols(api.Symbols, names)
            store = CTraderLocalStorageStore(
                api.LocalStorage,
                LocalStorageScope.Device,
            )
            self._host = CTraderObservationHost(
                account=api.Account,
                positions=api.Positions,
                pending_orders=api.PendingOrders,
                store=store,
                symbols=names,
                logger=api.Print,
                started_at=now,
                maximum_broker_age=timedelta(
                    seconds=max(3 * int(api.HeartbeatSeconds), 5)
                ),
                maximum_source_age=timedelta(
                    seconds=max(3 * int(api.HeartbeatSeconds), 5)
                ),
                quote_observer=quote_provider(self._symbols),
            )
            self._host.start(now)
            self._bar_router = BarOpenedRouter(
                self._host,
                native_timeframe=api.TimeFrame,
                timeframe_id=str(api.TimeFrame),
            )
            self._bar_router.subscribe(api.MarketData, self._symbols)
            api.Timer.Start(api.HeartbeatSeconds)
        except NativeConfigurationError as exception:
            api.Print(f"MRLAB HALT configuration={exception}")
            api.Stop()
            return

    def on_bar_closed(self):
        # Multi-symbol close boundaries arrive through each Bars.BarOpened event.
        pass

    def on_timer(self):
        self._host.heartbeat(to_python_utc(api.Server.TimeInUtc))

    def on_stop(self):
        api.Print("MRLAB STOP mode=OBSERVATION")

    def on_exception(self, exception):
        now = to_python_utc(api.Server.TimeInUtc)
        diagnostic = type(exception).__name__
        api.Print(f"MRLAB EXCEPTION type={diagnostic}")
        if hasattr(self, "_host"):
            self._host.halt(diagnostic, now)
