"""Native cTrader Python entrypoint for M5A observation mode only."""

import sys
from datetime import UTC
from pathlib import Path

import clr

clr.AddReference("cAlgo.API")

from cAlgo.API import *  # noqa: E402,F403
from robot_wrapper import *  # noqa: E402,F403

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from mr_lab.integrations.ctrader import (  # noqa: E402
    CTraderLocalStorageStore,
    CTraderObservationHost,
    LiveAccountRefused,
    normalize_closed_bar,
)


class MRLabController:
    """Callback facade; the generated wrapper supplies cTrader properties."""

    def on_start(self):
        try:
            now = self.Server.Time.astimezone(UTC)
            symbols = tuple(
                item.strip() for item in self.SymbolsCsv.split(",") if item.strip()
            )
            store = CTraderLocalStorageStore(
                self.LocalStorage,
                LocalStorageScope.Device,  # noqa: F405
            )
            self._host = CTraderObservationHost(
                account=self.Account,
                positions=self.Positions,
                pending_orders=self.PendingOrders,
                store=store,
                symbols=symbols,
                logger=self.Print,
                started_at=now,
            )
            self._host.start(now)
            self.Timer.Start(self.HeartbeatSeconds)
        except LiveAccountRefused as exception:
            self.Print(str(exception))
            raise

    def on_bar_closed(self):
        symbol_name = str(self.SymbolName)
        if symbol_name.upper() not in self._host.symbols:
            return
        bars = self.Bars
        bar = normalize_closed_bar(
            symbol_name,
            str(self.TimeFrame),
            bars.Last(1),
            bars.Last(0).OpenTime,
        )
        self._host.on_bar_closed(bar)

    def on_timer(self):
        self._host.heartbeat(self.Server.Time.astimezone(UTC))

    def on_stop(self):
        self.Print("MRLAB STOP mode=OBSERVATION")

    def on_exception(self, exception):
        now = self.Server.Time.astimezone(UTC)
        diagnostic = type(exception).__name__
        self.Print(f"MRLAB EXCEPTION type={diagnostic}")
        if hasattr(self, "_host"):
            self._host.halt(diagnostic, now)
