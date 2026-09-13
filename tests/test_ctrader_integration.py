from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from mr_lab.integrations.ctrader import (
    BarOpenedRouter,
    BrokerOwnershipError,
    CTraderLocalStorageStore,
    CTraderObservationHost,
    LiveAccountRefused,
    NativeConfigurationError,
    ObservationLifecycle,
    WouldExecuteObservation,
    encode_storage_key,
    normalize_account,
    normalize_closed_bar,
    normalize_quote,
    observe_broker,
    quote_provider,
    resolve_symbols,
    runtime_identity,
    to_python_utc,
)

NOW = datetime(2026, 1, 2, 12, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


class FakeStorage:
    def __init__(self) -> None:
        self.values: dict[tuple[str, object], str] = {}
        self.set_calls: list[tuple[str, str, object]] = []
        self.flush_calls: list[object] = []

    def GetString(self, key: str, scope: object) -> str | None:
        return self.values.get((key, scope))

    def SetString(self, key: str, value: str, scope: object) -> None:
        self.set_calls.append((key, value, scope))
        self.values[(key, scope)] = value

    def Flush(self, scope: object) -> None:
        self.flush_calls.append(scope)


def account(*, live: bool = False, number: int = 101) -> SimpleNamespace:
    return SimpleNamespace(
        IsLive=live,
        BrokerName="FTMO",
        Number=number,
        AccountType="Hedged",
        Asset=SimpleNamespace(Name="EUR"),
        Balance=100_000,
        Equity=99_900,
        FreeMargin=90_000,
    )


def test_live_account_is_refused_and_demo_is_normalized() -> None:
    with pytest.raises(LiveAccountRefused, match="MR LAB M5A REFUSES LIVE ACCOUNT"):
        normalize_account(account(live=True), NOW)
    result = normalize_account(account(), NOW)
    assert not result.is_live
    assert result.broker_name == "FTMO"
    assert result.currency == "EUR"
    assert not hasattr(result, "Number")


def test_runtime_identity_is_deterministic_and_account_specific() -> None:
    account_number = 123456789
    identity = runtime_identity("FTMO", account_number)
    assert identity == runtime_identity("FTMO", account_number)
    assert identity != runtime_identity("FTMO", account_number + 1)
    assert str(account_number) not in identity


def test_storage_key_codec_and_round_trip_with_explicit_flush() -> None:
    first = encode_storage_key("runtime:account/foo:checkpoint")
    second = encode_storage_key("runtime:account/bar:checkpoint")
    assert first == encode_storage_key("runtime:account/foo:checkpoint")
    assert first != second
    assert len(first) <= 50
    assert re.fullmatch(r"[A-Za-z0-9 ]+", first)
    api, scope = FakeStorage(), object()
    store = CTraderLocalStorageStore(api, scope)
    assert store.read_text("key") is None
    store.write_text("key", '{"ok":true}')
    store.flush()
    assert api.set_calls == [(encode_storage_key("key"), '{"ok":true}', scope)]
    assert api.flush_calls == [scope]
    assert store.read_text("key") == '{"ok":true}'


def test_market_values_are_copied_to_frozen_utc_contracts() -> None:
    native = SimpleNamespace(Name="EURUSD", Bid=1.1, Ask=1.2)
    quote = normalize_quote(native, NOW.astimezone(timezone(timedelta(hours=2))))
    native.Bid = 9
    assert quote.bid != native.Bid
    assert quote.timestamp.tzinfo is UTC
    with pytest.raises(FrozenInstanceError):
        quote.bid = quote.ask
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_quote(native, NOW.replace(tzinfo=None))


def test_dotnet_datetime_is_copied_to_python_utc() -> None:
    native = SimpleNamespace(
        Year=2026,
        Month=2,
        Day=3,
        Hour=4,
        Minute=5,
        Second=6,
        Millisecond=789,
    )
    assert to_python_utc(native) == datetime(2026, 2, 3, 4, 5, 6, 789000, tzinfo=UTC)


def test_closed_bar_and_multi_symbol_source_identity() -> None:
    bar = SimpleNamespace(
        OpenTime=NOW - timedelta(minutes=15),
        Open="1.10",
        High="1.15",
        Low="1.05",
        Close="1.12",
    )
    eur = normalize_closed_bar("EURUSD", "M15", bar, NOW)
    gbp = normalize_closed_bar("GBPUSD", "M15", bar, NOW)
    assert eur.close_time == NOW and eur.open < eur.close
    assert eur.source_id != gbp.source_id


def test_read_only_ownership_scan_ignores_manual_and_halts_on_reserved() -> None:
    snapshot = observe_broker([SimpleNamespace(Label="manual")], [], NOW, "runtime")
    assert snapshot.executions == ()
    with pytest.raises(BrokerOwnershipError, match="HALT"):
        observe_broker([SimpleNamespace(Label="MRLAB-unmapped")], [], NOW, "runtime")


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    def on_bar_closed(self, bar):
        self.calls += 1
        return (
            WouldExecuteObservation(
                "obs",
                bar.close_time,
                "intent",
                bar.instrument,
                "SHORT",
                bar.close,
                "mean-reversion",
                "strategy",
                "risk",
                "candidate",
            ),
        )


def make_host(storage: FakeStorage, logs: list[str], pipeline=None, observer=None):
    if observer is None:
        observer = quote_provider(
            {
                "EURUSD": SimpleNamespace(Name="EURUSD", Bid="1.1", Ask="1.2"),
                "GBPUSD": SimpleNamespace(Name="GBPUSD", Bid="1.2", Ask="1.3"),
            }
        )
    return CTraderObservationHost(
        account=account(),
        positions=[],
        pending_orders=[],
        store=CTraderLocalStorageStore(storage, "Device"),
        symbols=("EURUSD", "GBPUSD"),
        logger=logs.append,
        started_at=NOW,
        pipeline=pipeline,
        quote_observer=observer,
    )


def test_clean_start_syncs_and_restart_restores_without_execution_mutation() -> None:
    storage, logs = FakeStorage(), []
    first = make_host(storage, logs)
    first.start(NOW)
    assert first.lifecycle is ObservationLifecycle.SYNCED
    assert first.controller.state.executions == ()
    assert storage.flush_calls
    sequence = first.controller.state.checkpoint_sequence
    second = make_host(storage, logs)
    second.start(NOW + timedelta(seconds=1))
    assert second.lifecycle is ObservationLifecycle.SYNCED
    assert second.controller.state.checkpoint_sequence > sequence
    assert second.controller.state.executions == ()


def test_required_multi_symbol_quote_freshness_recovers_from_stale() -> None:
    active = {"EURUSD"}

    def observe(now):
        symbols = {
            name: SimpleNamespace(Name=name, Bid="1.1", Ask="1.2") for name in active
        }
        return quote_provider(symbols)(now)

    host = make_host(FakeStorage(), [], observer=observe)
    host.start(NOW)
    assert host.lifecycle is ObservationLifecycle.STALE
    active.add("GBPUSD")
    host.heartbeat(NOW + timedelta(seconds=1))
    assert host.lifecycle is ObservationLifecycle.SYNCED
    assert not host.controller.can_open_new_entries(NOW + timedelta(seconds=1))


def test_future_quote_timestamp_fails_freshness_closed() -> None:
    def future(_now):
        return quote_provider(
            {
                "EURUSD": SimpleNamespace(Name="EURUSD", Bid=1, Ask=2),
                "GBPUSD": SimpleNamespace(Name="GBPUSD", Bid=1, Ask=2),
            }
        )(NOW + timedelta(seconds=1))

    host = make_host(FakeStorage(), [], observer=future)
    host.start(NOW)
    assert host.lifecycle is ObservationLifecycle.STALE


def test_multi_symbol_resolution_bar_boundaries_and_deduplication() -> None:
    eur = SimpleNamespace(Name="EURUSD")
    gbp = SimpleNamespace(Name="GBPUSD")
    symbols_api = SimpleNamespace(
        GetSymbol=lambda name: {"EURUSD": eur, "GBPUSD": gbp}.get(name)
    )
    resolved = resolve_symbols(symbols_api, ("EURUSD", "GBPUSD"))
    assert resolved == {"EURUSD": eur, "GBPUSD": gbp}
    with pytest.raises(NativeConfigurationError, match="unknown configured symbol"):
        resolve_symbols(symbols_api, ("UNKNOWN",))

    class Sink:
        def __init__(self):
            self.bars = []

        def on_bar_closed(self, bar):
            self.bars.append(bar)

    class FakeEvent:
        def __init__(self):
            self.handlers = []

        def __iadd__(self, handler):
            self.handlers.append(handler)
            return self

    def stream(open_value):
        new = SimpleNamespace(OpenTime=NOW)
        closed = SimpleNamespace(
            OpenTime=NOW - timedelta(minutes=15),
            Open=open_value,
            High="1.3",
            Low="1.0",
            Close="1.2",
        )
        return SimpleNamespace(
            Last=lambda index: (new, closed)[index], BarOpened=FakeEvent()
        )

    sink = Sink()
    native_timeframe = object()
    streams = {"EURUSD": stream("1.1"), "GBPUSD": stream("1.15")}
    calls = []

    def get_bars(received_timeframe, symbol_name):
        assert received_timeframe is native_timeframe
        calls.append((received_timeframe, symbol_name))
        return streams[symbol_name]

    router = BarOpenedRouter(
        sink, native_timeframe=native_timeframe, timeframe_id="M15"
    )
    router.subscribe(SimpleNamespace(GetBars=get_bars), resolved)
    streams["EURUSD"].BarOpened.handlers[0]()
    streams["GBPUSD"].BarOpened.handlers[0]()
    eur_bar, gbp_bar = sink.bars
    assert calls == [(native_timeframe, "EURUSD"), (native_timeframe, "GBPUSD")]
    assert eur_bar.open_time == NOW - timedelta(minutes=15)
    assert eur_bar.close_time == NOW
    assert eur_bar.open == Decimal("1.1")
    assert eur_bar.timeframe == "M15"
    assert isinstance(eur_bar.source_id, str)
    assert eur_bar.source_id != gbp_bar.source_id
    assert [bar.instrument for bar in sink.bars] == ["EURUSD", "GBPUSD"]
    assert router.on_bar_opened(gbp, streams["GBPUSD"]) is None
    assert len(sink.bars) == 2


def test_signal_only_pipeline_emits_telemetry_before_m3(monkeypatch) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("M3 submission boundary crossed")

    monkeypatch.setattr("mr_lab.runtime.RuntimeController.submit_intent", forbidden)
    pipeline, logs = FakePipeline(), []
    host = make_host(FakeStorage(), logs, pipeline)
    bar = normalize_closed_bar(
        "EURUSD",
        "M15",
        SimpleNamespace(
            OpenTime=NOW - timedelta(minutes=15),
            Open="1.1",
            High="1.2",
            Low="1.0",
            Close="1.15",
        ),
        NOW,
    )
    host.on_bar_closed(bar)
    assert pipeline.calls == 1 and not called
    assert host.controller.state.executions == ()
    assert any("WOULD_EXECUTE" in line for line in logs)


def _export_module():
    path = ROOT / "tools" / "export_ctrader_bundle.py"
    spec = importlib.util.spec_from_file_location("export_ctrader_bundle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_is_deterministic_allowlisted_and_complete(tmp_path: Path) -> None:
    exporter = _export_module()
    destination = tmp_path / "MR Lab Controller"
    first = exporter.export(destination)
    content = {name: (destination / name).read_bytes() for name in first}
    second = exporter.export(destination)
    assert first == second
    assert content == {name: (destination / name).read_bytes() for name in second}
    assert "MR Lab Controller_main.py" in first
    assert "python/mr_lab/runtime/controller.py" in first
    assert "python/mr_lab/integrations/ctrader/host.py" in first
    forbidden_parts = {"tests", "sealed", ".git", "notebooks", "__pycache__"}
    assert not any(forbidden_parts.intersection(Path(name).parts) for name in first)
    assert all(
        Path(name).suffix == ".py" for name in first if "python/mr_lab/data/" in name
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        (
            "import sys; "
            f"sys.path.insert(0, {str(destination / 'python')!r}); "
            "import mr_lab.runtime, mr_lab.execution, mr_lab.integrations.ctrader"
        ),
    ]
    subprocess.run(command, cwd=tmp_path, check=True)


def test_observation_deployment_has_no_known_broker_write_api() -> None:
    forbidden = (
        "Execute" + "MarketOrder",
        "Place" + "LimitOrder",
        "Place" + "StopOrder",
        "Modify" + "Position",
        "Modify" + "PendingOrder",
        "Cancel" + "PendingOrder",
        "Close" + "Position",
    )
    paths = [
        ROOT / "ctrader/MRLabController/MR Lab Controller_main.py",
        *(ROOT / "src/mr_lab/integrations/ctrader").glob("*.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert all(name not in source for name in forbidden)


def test_native_entrypoint_uses_api_wrapper_and_robot_has_no_access_rights() -> None:
    native = (ROOT / "ctrader/MRLabController/MR Lab Controller_main.py").read_text()
    for required in (
        "api.Account",
        "api.Server.TimeInUtc",
        "api.LocalStorage",
        "api.Timer",
        "api.Print",
        "api.Symbols",
        "api.MarketData",
    ):
        assert required in native
    for forbidden in ("self.Account", "self.Server", "self.LocalStorage", "self.Timer"):
        assert forbidden not in native
    assert "native_timeframe=api.TimeFrame" in native
    assert "timeframe_id=str(api.TimeFrame)" in native
    assert "BarOpenedRouter(self._host, str(api.TimeFrame))" not in native
    companion = (ROOT / "ctrader/MRLabController/MRLabController.cs").read_text()
    assert "[Robot(" in companion
    assert "AccessRights = AccessRights.None" in companion
    assert "TimeZone = TimeZones.UTC" in companion


def test_native_live_start_stops_before_timer_or_host(monkeypatch) -> None:
    calls: list[str] = []
    api = SimpleNamespace(
        Account=SimpleNamespace(IsLive=True),
        Print=lambda message: calls.append(message),
        Stop=lambda: calls.append("STOP"),
        Timer=SimpleNamespace(Start=lambda _seconds: calls.append("TIMER")),
    )
    clr = SimpleNamespace(AddReference=lambda _name: None)
    wrapper = type(sys)("robot_wrapper")
    wrapper.api = api
    wrapper.__all__ = ["api"]
    calgo = type(sys)("cAlgo")
    calgo_api = type(sys)("cAlgo.API")
    monkeypatch.setitem(sys.modules, "clr", clr)
    monkeypatch.setitem(sys.modules, "robot_wrapper", wrapper)
    monkeypatch.setitem(sys.modules, "cAlgo", calgo)
    monkeypatch.setitem(sys.modules, "cAlgo.API", calgo_api)
    path = ROOT / "ctrader/MRLabController/MR Lab Controller_main.py"
    spec = importlib.util.spec_from_file_location("mrlab_native_live_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.MRLabController()
    instance.on_start()
    assert calls == ["MR LAB M5A REFUSES LIVE ACCOUNT", "STOP"]
    assert not hasattr(instance, "_host")


def test_core_packages_have_no_ctrader_imports() -> None:
    for package in ("portfolio", "risk", "execution", "runtime"):
        for path in (ROOT / "src/mr_lab" / package).glob("*.py"):
            assert "cAlgo" not in path.read_text(encoding="utf-8")
