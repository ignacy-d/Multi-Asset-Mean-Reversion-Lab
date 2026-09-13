from __future__ import annotations

import importlib.util
import re
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mr_lab.integrations.ctrader import (
    BrokerOwnershipError,
    CTraderLocalStorageStore,
    CTraderObservationHost,
    LiveAccountRefused,
    ObservationLifecycle,
    WouldExecuteObservation,
    encode_storage_key,
    normalize_account,
    normalize_closed_bar,
    normalize_quote,
    observe_broker,
    runtime_identity,
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
        Asset="EUR",
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


def make_host(storage: FakeStorage, logs: list[str], pipeline=None):
    return CTraderObservationHost(
        account=account(),
        positions=[],
        pending_orders=[],
        store=CTraderLocalStorageStore(storage, "Device"),
        symbols=("EURUSD", "GBPUSD"),
        logger=logs.append,
        started_at=NOW,
        pipeline=pipeline,
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
    forbidden_parts = {"tests", "data", "sealed", ".git", "notebooks", "__pycache__"}
    assert not any(forbidden_parts.intersection(Path(name).parts) for name in first)


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


def test_core_packages_have_no_ctrader_imports() -> None:
    for package in ("portfolio", "risk", "execution", "runtime"):
        for path in (ROOT / "src/mr_lab" / package).glob("*.py"):
            assert "cAlgo" not in path.read_text(encoding="utf-8")
