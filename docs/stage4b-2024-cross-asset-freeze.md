# Stage 4B 2024 Cross-Asset Interpretation Freeze

Status: **FROZEN BEFORE STAGE 4C COSTS**

> **September 2026 correction status:** This document is retained as the historical
> pre-cost interpretation produced under VWAP normalization v1. The confirmed
> gap-compression defect invalidates every VWAP-dependent conclusion below. It must
> not be used as a Stage 4C input until the v2 recomputation and a new, separately
> versioned cross-asset interpretation freeze are complete. Bollinger-only evidence
> is mathematically unaffected; no statement below is silently rewritten.

This document freezes the interpretation of the completed Stage 4B real-2024 results before transaction-cost modelling. It is not a trading-rule optimization and must not be edited in response to Stage 4C or sealed-2025 outcomes except through an explicitly versioned later-stage decision record.

## Scope and provenance

Completed Stage 4B real-2024 runs used the same merged Stage 4B code on `main` at source commit:

- `3090682b61090de1b4a30efc18a7547e92fa262e`

Reviewed combined runs:

- USDJPY: GitHub Actions run `32638668214`
- AUDUSD: GitHub Actions run `32641981092`
- AUDJPY: GitHub Actions run `32641988237`
- EURUSD: GitHub Actions run `32643840766`

GBPUSD is not part of this freeze because its verified 2024 corpus / Stage 4B run is still pending.

## Frozen interpretation rules

1. Stage 4B is **2024 in-sample only**. Sealed 2025 remains untouched.
2. Lead timeframe is **M15**. M5 and H1 are robustness views, not alternate post-hoc lead timeframes.
3. Signal threshold remains exactly the frozen Stage 3/4B threshold semantics: strict `|z| > 2.0` qualification with equality neither signal nor re-arm.
4. VWAP native, VWAP canonical-M1, and Bollinger remain **separate benchmark / robustness families**. They are not combined into an `AND`, `OR`, ensemble, or weighted signal in Stage 4C.
5. `immediate` remains the neutral baseline entry. `m1-reclaim-p0` and `extension25-then-reclaim-p0` remain predeclared alternative entry modes, not newly selected winners.
6. No single TP / SL / time-stop configuration is selected from 2024. Stage 4C must evaluate the frozen exit grid under costs and look for broad net-positive plateaus rather than the best historical cell.
7. No pair-, direction-, session-, family-, lookback-, threshold-, or timeframe-specific tuning is permitted because of these results.
8. A regime means **instrument × session × direction**. Portfolio selection happens only after costs; there is no requirement to force one or two regimes into every session.
9. Mean-reversion rejects must not be rescued by post-hoc parameter mining. Continuation-looking regimes may be retained only for a future separately preregistered continuation module.

## Frozen regime map

### USDJPY

| Session | Direction | Frozen Stage 4B interpretation | Stage 4C status |
|---|---|---|---|
| Asia | LONG | No clean M15 MR edge | Reject MR |
| Asia | SHORT | No clean M15 MR edge | Reject MR |
| London | LONG | Secondary M15 MR candidate; less clean than NY | Advance as secondary |
| London | SHORT | Continuation / anti-MR behaviour | Reject MR; continuation watch only |
| New York | LONG | **Primary clean MR candidate**; broad positive exit-grid behaviour | **Advance primary** |
| New York | SHORT | Real MR candidate but more skewed / tail-sensitive than LONG | **Advance secondary** |

Frozen entry interpretation: `immediate` is baseline. Simple `m1-reclaim-p0` is not a convincing information filter. `extension25-then-reclaim-p0` is selective and research-interesting but not promoted to winner.

### AUDUSD

| Session | Direction | Frozen Stage 4B interpretation | Stage 4C status |
|---|---|---|---|
| Asia | LONG | No convincing M15 MR edge | Reject MR |
| Asia | SHORT | Earlier weak/cost-sensitive signal did not survive Stage 4B strongly enough | Reject MR |
| London | LONG | Weak / uneven MR evidence | Research only |
| London | SHORT | **Very strong, broad, family-robust M15 MR candidate** | **Advance primary** |
| New York | LONG | Weak / model-dependent | Reject from main MR candidate set |
| New York | SHORT | Broad positive secondary MR candidate | **Advance secondary** |

Frozen entry interpretation: `immediate` is the clean baseline. `m1-reclaim-p0` adds little. `extension25-then-reclaim-p0` is a genuinely selective and promising path filter for London SHORT, but it is not selected as the production entry before costs / OOS.

### AUDJPY

| Session | Direction | Frozen Stage 4B interpretation | Stage 4C status |
|---|---|---|---|
| Asia | LONG | No M15 MR edge | Reject MR |
| Asia | SHORT | No M15 MR edge | Reject MR |
| London | LONG | Strong M15-specific MR candidate, weaker cross-timeframe robustness | **Advance secondary** |
| London | SHORT | Anti-MR / continuation-like | Reject MR |
| New York | LONG | Positive but benchmark disagreement, especially weaker Bollinger robustness | Research only |
| New York | SHORT | **Best directional AUDJPY MR candidate; robust immediate profile** | **Advance primary** |

Frozen entry interpretation: `extension25-then-reclaim-p0` is promising for NY SHORT but remains an alternative entry mode, not a selected winner. `immediate` remains the baseline.

### EURUSD

| Session | Direction | Frozen Stage 4B interpretation | Stage 4C status |
|---|---|---|---|
| Asia | LONG | No M15 MR edge | Reject MR |
| Asia | SHORT | **Real but small gross MR effect; explicitly cost-sensitive** | **Advance to costs as Asia candidate** |
| London | LONG | Stage 4B does not support the earlier slow-MR hypothesis strongly enough | Reject MR |
| London | SHORT | **Strong, broad M15 MR candidate** | **Advance primary** |
| New York | LONG | Positive mainly in VWAP families; weaker Bollinger agreement | Research only |
| New York | SHORT | **Strong secondary MR candidate** | **Advance secondary** |

Frozen entry interpretation: `immediate` is baseline. Reclaim filters do not improve London / NY SHORT robustly enough to select them before costs.

## Cross-asset candidate set entering Stage 4C

### Primary candidates

- AUDUSD × London × SHORT
- USDJPY × New York × LONG
- EURUSD × London × SHORT
- AUDJPY × New York × SHORT

### Secondary candidates

- USDJPY × New York × SHORT
- AUDUSD × New York × SHORT
- EURUSD × New York × SHORT
- AUDJPY × London × LONG
- USDJPY × London × LONG

### Explicit cost-sensitive candidate

- EURUSD × Asia × SHORT

### Research-only / not part of primary Stage 4C selection race

- AUDUSD × London × LONG
- AUDJPY × New York × LONG
- EURUSD × New York × LONG

All other completed 2024 M15 instrument × session × direction MR regimes are frozen as rejected for the main mean-reversion candidate set.

## Stage 4C objective

Stage 4C answers one question: **which frozen Stage 4B behaviours remain economically positive after executable transaction costs?**

Stage 4C must therefore add costs without changing the Stage 4B signal or path methodology. Required cost work should include, at minimum:

- executable bid/ask treatment for LONG and SHORT entries / exits;
- commission where applicable;
- spread;
- slippage assumptions;
- base and stressed cost scenarios;
- net-return distributions and win/loss metrics;
- break-even all-in cost / cost headroom;
- survival breadth across benchmark families, lookbacks, entry modes, TP, SL, and time stops.

The purpose is **not** to select the maximum-net 2024 cell. The purpose is to identify broad, economically defensible net-positive regions and then reduce the candidate regime set before sealed 2025 OOS.

## Session portfolio rule

The eventual portfolio may contain approximately 0–2 regimes per session, but this is a capacity target rather than a requirement. In particular, Asia must be allowed to remain empty if transaction costs eliminate the currently small candidate edge. No regime may be promoted merely to fill a session slot.
