# MR Lab Controller — M5A deployment

M5A is a native Python cBot that runs inside the local cTrader Windows runtime.
It is not an Open API client or an external daemon. It reads platform state and
emits observations, but has no broker-writing adapter. The hard startup gate
prints `MR LAB M5A REFUSES LIVE ACCOUNT` and calls `api.Stop()` on every live
account before constructing the host or timer.

The Python entrypoint uses cTrader's global `api` wrapper, including
`api.Account`, `api.Server.TimeInUtc`, `api.Symbols`, `api.MarketData`,
`api.LocalStorage`, and `api.Timer`. Its generated-style C# companion declares
`[Robot(AccessRights = AccessRights.None, TimeZone = TimeZones.UTC)]`;
LocalStorage needs no filesystem access permission. M5A accepts only the
`OBSERVATION_ONLY` mode and stops on every other value.

## Build from the repository

The GitHub repository is authoritative. **DO NOT edit deployed files as the
long-term source of truth.** The workflow is:

```text
repo → tests → export → cTrader build
```

1. Install or open cTrader on Windows.
2. Sign in to the intended **DEMO** account (never a live account for M5A).
3. Open **Algo**.
4. Create a Python cBot named `MR Lab Controller`.
5. Select **Show in folder**. The expected directory is
   `Documents/cAlgo/Sources/Robots/MR Lab Controller/MR Lab Controller/`.
6. From a repository checkout, run:
   `python tools/export_ctrader_bundle.py --destination "<project directory>"`.
7. Build the cBot.
8. Add **ONE** instance. Every symbols-CSV entry is resolved with
   `api.Symbols.GetSymbol`, its independent Bars collection is obtained through
   `api.MarketData.GetBars`, and its `BarOpened` event is subscribed to by the
   central host. An unavailable symbol fails closed. The attached chart is only
   host context, not a separate portfolio runtime.
9. Confirm once more that the selected account says DEMO.
10. Start the instance.
11. Inspect the cBot log.
12. Expect `MRLAB START`, `mode=OBSERVATION`, `account=DEMO`, broker/runtime
    identity, and `lifecycle=BOOTSTRAP`, followed by checkpoint restore, an empty
    read-only broker reconciliation, required quote freshness validation, and
    `SYNCED`. `SYNCED` means platform health only; trading is not enabled.
13. Stop and restart the cBot, then confirm checkpoint restoration as below.

The adapter stores JSON checkpoints using `LocalStorageScope.Device`. Keys
include a deterministic hash derived from controller version, broker, account,
and DEMO environment, so a recreated instance on the same device can recover
without exposing the account number or loading another account's checkpoint.
cTrader's key receives only `MRLAB ` plus a SHA-256 prefix (valid characters,
under 50 characters). Every critical write is explicitly flushed; periodic
autosave is not used as a durability guarantee.

## Manual FTMO/cTrader restart smoke test

Run 1:

```text
start → BOOTSTRAP → no checkpoint / clean broker → SYNCED → checkpoint written
```

Stop the cBot. Run 2:

```text
start → BOOTSTRAP → checkpoint restored → broker/account observation
      → reconcile → SYNCED
```

The second run demonstrates Device LocalStorage restoration. It does not resume
hypothetical trading. Unrelated manual positions are ignored. An object using
the reserved `MRLAB-` ownership prefix cannot yet be mapped by M5A and therefore
halts observation rather than inventing ownership.

## Safety boundary

Market prices, closed bars, account values, positions, and pending orders are
copied into immutable broker-neutral values. Timer callbacks provide synchronous
heartbeats and refresh every configured symbol's required quote watermark. A
missing, stale, or future-dated source cannot report `SYNCED`. Each subscribed
`BarOpened` event uses `Last(1)` as the just-closed bar and `Last(0).OpenTime` as
its exact close boundary; duplicate boundaries are ignored. Native .NET
`DateTime` fields are copied to timezone-aware Python UTC values at the adapter
boundary. The production signal pipeline is deliberately no-trade until replay
parity is complete. A future signal-only pipeline ends at
`WouldExecuteObservation`, before M3 execution state or command generation.
