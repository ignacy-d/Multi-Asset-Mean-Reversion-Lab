# Stage 4C-A preregistration — FTMO transaction-cost viability

Status: **FROZEN BEFORE IMPLEMENTATION AND BEFORE VIEWING STAGE 4C NET RESULTS**

This stage is a deterministic cost overlay on the already-frozen Stage 4B 2024 trade construction. It must answer whether the Stage 4B gross-BID behaviour is large enough to survive current FTMO-style execution costs. It must not change the Stage 4B signal, deduplication, entry, exit, threshold, benchmark, lookback, timeframe, or session methodology.

## 1. Research boundary

- Dataset under evaluation: **2024 in-sample only**.
- Sealed 2025 remains untouched.
- Implement first for the four verified Stage 4B instruments: `EURUSD`, `USDJPY`, `AUDUSD`, `AUDJPY`.
- `GBPUSD` is appended later under this exact unchanged Stage 4C specification when its verified Stage 4B source exists.
- Stage 4B source methodology is the already-merged code at `3090682b61090de1b4a30efc18a7547e92fa262e`.
- The Stage 4B interpretation freeze is in `docs/stage4b-2024-cross-asset-freeze.md`.
- No new trading filter is allowed in Stage 4C-A. In particular: no news blackout, weekday filter, threshold change, volatility filter, OU filter, benchmark consensus rule, or pair-specific parameter tuning.

Reviewed Stage 4B runs used as the source checkpoint:

| Instrument | Stage 4B run | Combined-review artifact |
| --- | ---: | ---: |
| USDJPY | `32638668214` | `9493093398` |
| AUDUSD | `32641981092` | `9493945206` |
| AUDJPY | `32641988237` | `9493948384` |
| EURUSD | `32643840766` | `9494421086` |

The implementation may consume the raw Stage 4B shard artifacts from these runs or deterministically regenerate identical Stage 4B raw rows from the pinned corpora. Whichever route is used must be explicit in `execution-audit.json`.

## 2. Frozen cost profile

Machine-readable inputs live in `configs/stage4c-ftmo-cost-profile-v1.json`.

The raw cTrader CSV files are intentionally **not committed**. The config records their exact filenames, byte sizes, row counts and SHA-256 hashes.

The profile was measured from the user's FTMO cTrader historical tick feed over the 2026-08-09 to 2026-08-21 market window, sampled as the final observed quote in each 5-second bucket.

This profile is a **current/future FTMO execution-condition overlay**, not an attempt to reconstruct FTMO spreads in 2024.

Session membership must reuse the exact DST-aware `DEFAULT_SESSION_SPEC` from `src/mr_lab/sessions.py`. Overlapping session memberships are legitimate and are counted independently in the cost-profile statistics.

Observed negative spread rows are retained in source QA, but cost-profile statistics clamp them to zero. A transaction-cost model must never receive a negative spread rebate from a crossed/stale quote anomaly.

## 3. FTMO commission

Freeze Forex commission at:

- USD 2.50 per standard lot per side;
- USD 5.00 per standard lot round turn.

Source: `https://ftmo.com/en/blog/trading-updates/trading-update-25-sep-2025/`

For USD-quoted pairs (`EURUSD`, `AUDUSD`) this is exactly `0.5` pip round turn per standard lot.

For JPY-quoted pairs (`USDJPY`, `AUDJPY`) Stage 4C-A converts the USD 5 round-turn commission into pips with:

`commission_pips = 0.005 * reference_USDJPY`

where the reference is the same 2026 FTMO spread-sample mean USDJPY mid for the originating Stage 4B session. The exact frozen values are in the cost-profile JSON. `session = null` uses the `overall` profile.

This is intentionally a current-cost normalization, not a historical 2024 FX conversion reconstruction.

## 4. FTMO 0.7% currency-conversion adjustment

For a USD account, FTMO's cTrader currency-conversion adjustment applies when the Quote Asset differs from the Account Currency.

Source: `https://ftmo.com/en/blog/trading-updates/trading-update-28-jul-2026/`

Therefore in this Stage 4C-A scope:

- `EURUSD`, `AUDUSD`: no conversion adjustment;
- `USDJPY`, `AUDJPY`: 0.7% adjustment applies.

For affected instruments, after spread and slippage are deducted from gross price P/L but before commission:

- positive realised execution P/L is multiplied by `0.993`;
- negative realised execution P/L is multiplied by `1.007`;
- zero remains zero.

Then the round-turn commission is subtracted.

The report must label this implementation as the frozen Stage 4C-A accounting approximation and not silently alter it later.

## 5. Spread semantics relative to Stage 4B BID/BID

Stage 4B is gross BID/BID.

Stage 4C-A subtracts **one spread per completed round trip, not two**:

- LONG: Stage 4B entry used BID; executable entry pays ASK while the BID exit is already represented by the Stage 4B path.
- SHORT: Stage 4B entry BID is executable; the exit must be bought back at ASK.

V1 applies the frozen spread statistic for the **originating Stage 4B signal session**. `session = null` uses the instrument's `overall` profile.

Do not add timestamp-specific 2024 spread reconstruction in this stage.

## 6. Cost scenarios

Every complete executed Stage 4B trade must be evaluated across the full frozen scenario matrix.

Spread statistic:

- `mean`
- `p75`
- `p90`
- `p95`

Round-turn slippage sensitivity in pips:

- `0.00`
- `0.10`
- `0.25`
- `0.50`

This produces 16 cost scenarios per otherwise-identical Stage 4B trade row.

`mean spread + commission + 0 slippage` is the **cost-floor headline**. It is an optimistic lower bound and must never be described as expected live P/L.

The slippage grid is a sensitivity grid only. It is not claimed to be empirically measured FTMO slippage.

## 7. Per-trade transform

Only complete, executed Stage 4B trade rows are transformed.

For each scenario define:

1. `execution_pips_adverse_first = gross_return_pips_adverse_first - spread_pips - slippage_pips`
2. Apply the 0.7% currency-conversion adjustment when required.
3. `net_pips_adverse_first = adjusted_execution_pips_adverse_first - commission_pips`

Repeat identically for the favorable-first ambiguity bound.

Do not alter candidate identity, entry/exit timestamps, entry mode, TP/SL/time-stop, exit reason, same-minute ambiguity semantics, MAE/MFE, holding time, or Stage 4B gross fields. Preserve the original gross values alongside all net values.

## 8. Required reporting

The primary unit remains the frozen Stage 4B configuration cell, not an average-of-averages and not a newly optimized strategy.

For every original Stage 4B group plus cost-scenario dimensions report at least:

- complete trade count;
- mean and median net pips;
- net p10/p25/p75/p90;
- net win/loss/zero fractions;
- mean adverse-first net pips;
- mean favorable-first net pips;
- profit factor on net pips;
- gross mean pips;
- total modeled round-turn cost mean;
- `cost_headroom_pips = mean_gross_pips - modeled_cost_before_conversion_adjustment`;
- `break_even_extra_slippage_pips` under each spread statistic;
- fraction of gross-positive Stage 4B cells remaining net-positive.

Also produce regime-level breadth summaries for `instrument × signal_timeframe × session × direction × entry_mode`, with benchmark-family/lookback/TP/SL/time-stop cells treated as dependent robustness cells rather than independent trades.

Headline interpretation must emphasize **broad net-positive plateaus**. Do not select the single maximum-mean cell.

## 9. Required outputs

Prefer a compact review artifact containing:

- `stage4c-trade-matrix.csv`
- `stage4c-regime-breadth.csv`
- `stage4c-cost-scenarios.csv`
- `stage4c-summary.json`
- `stage4c-report.md`
- `execution-audit.json`

A raw per-trade Stage 4C JSONL is optional if it makes artifacts impractically large; if omitted, the transform must still be streaming, deterministic and covered by exact tests against a small fixture.

Schema identifier: `stage-4c-report-v1`.

## 10. Audit / determinism requirements

`execution-audit.json` must include:

- Stage 4C source commit SHA;
- Stage 4B source commit/methodology identity;
- instrument;
- Stage 4B source run/artifact identities or regenerated-source identities;
- exact cost-profile file SHA-256;
- exact scenario matrix;
- row counts before and after transformation;
- source raw-shard SHA-256 commitments where available;
- output SHA-256 commitments;
- account currency;
- conversion-adjustment rule;
- explicit statement whether Volume Bands and swaps are modeled.

Reducer must fail closed on missing shards, duplicate shard identities, inconsistent Stage 4B methodology, inconsistent cost-profile hash, inconsistent scenario matrix, duplicate original trade identity + cost scenario, unexpected instrument, or malformed/non-finite gross/net values.

## 11. Tests required before real runs

At minimum:

1. BID/BID spread accounting: exactly one spread per round trip.
2. EURUSD/AUDUSD USD 5 commission equals exactly 0.5 pip.
3. JPY commission conversion matches frozen profile values.
4. 0.7% adjustment reduces positive JPY-quote P/L and enlarges negative JPY-quote P/L; USD-quote pairs unchanged.
5. Scenario cost monotonicity: increasing spread statistic or slippage can never improve net P/L.
6. Gross fields are preserved.
7. Same-minute adverse-first/favorable-first ordering remains intact.
8. Null Stage 4B session uses `overall`.
9. Streaming transform equals a simple in-memory reference on a fixture.
10. Sharded + reduced result equals unsharded result exactly on a fixture.
11. Existing determinism / PIT invariants remain true.
12. No 2025 data path is touched.

Run the full existing test suite plus new Stage 4C tests.

## 12. Explicitly deferred to Stage 4C-B / Stage 4D

Stage 4C-A does **not** add FTMO Volume Band price tiers/account-size-dependent execution, empirical slippage calibration from filled orders, swap pricing, news filters, weekday/start/end-of-week filters, spread-at-entry live rejection rules, new z thresholds, ATR/volatility filters, OU eligibility, VWAP+Bollinger consensus, or portfolio sizing.

Volume Bands are Stage 4C-B after candidate viability and position sizing.

New selectivity filters, if needed after seeing Stage 4C-A, belong to a separately preregistered Stage 4D. Stage 4C-A results must not be used to silently alter this specification.

## 13. Decision rule

A regime is not advanced merely because one cell is net-positive.

Advance only if Stage 4C-A shows a coherent net-positive region across multiple predeclared exit configurations and reasonable benchmark/lookback robustness, with positive cost headroom under more than the optimistic cost floor.

If a regime fails even `mean spread + commission + 0 slippage`, it is rejected from the current FTMO mean-reversion candidate set unless a future separately preregistered hypothesis tests a genuinely different signal/filter family.
