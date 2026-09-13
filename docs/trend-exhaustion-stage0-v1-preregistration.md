# Trend Exhaustion Mean Reversion — Stage-0 v1 preregistration

## Status and scope

This document freezes the method **before empirical execution**. Discovery is
restricted to the authenticated, registry-pinned 2024 M1 BID corpora for
EURUSD, GBPUSD, USDJPY, AUDUSD, and AUDJPY. The 2025 OOS is sealed and is not
used by this work. Execution status is **pending**; no substitute corpus is
permitted. This family is independent of VWAP, OU, and Failed Breakout.

Hypothesis: after a sufficiently large and efficient directional intraday
trend, failure to retain additional directional progress has positive gross
forward expectancy in the opposite direction.

## Frozen causal event

Epoch-aligned M15 bars are emitted only with all 15 canonical M1 components.
Phase A is the eight completed M15 bars immediately before Phase B. Direction is
the sign of last close minus first close; efficiency is its absolute value
divided by the sum of the seven absolute consecutive close changes and must be
at least 0.60. ATR20 is the arithmetic mean of true ranges for the 20 completed
M15 bars ending at the Phase-A terminal bar (using the preceding close). It is
therefore frozen immediately before Phase B.

Absolute displacement must strictly exceed 1.25, **1.50 PRIMARY**, or 1.75 times
that frozen ATR. Phase B is the next four completed M15 bars. Its direction-
signed maximum high (uptrend) or low (downtrend) extension from Phase A's
terminal close must be at least 0.25 ATR. Its terminal close may retain at most
0.10 ATR of direction-signed progress, and its final close-to-close move must be
strictly opposite the trend. An exhausted uptrend emits SHORT; an exhausted
downtrend emits LONG at the final Phase-B bar close. Missing M15 components or
any discontinuity fails the candidate closed.

For each threshold, the detector begins armed. A qualifying window emits and
disarms it. It re-arms only after a later completed candidate window is
nonqualifying; contiguous qualifying rolling windows belong to one episode.
The rule is causal, structural, and has no tuned cooldown.

## Outcomes, diagnostics, and orthogonality

Exact-clock canonical M1 close outcomes are direction-signed at 15, 30, 60
(**PRIMARY**), and 120 minutes. Missing clocks are excluded, never substituted.
Complete 120-minute paths supply MFE and MAE. Reports include count, mean,
median, p10/p25/p75/p90, win rate, forward-return profit factor, monthly and
quarterly counts/expectancy, instrument/threshold/family views, and instrument
and quarter concentration.

Recorded but never filtering features are terminal upper/lower wick, body,
wick/body ratio, M15 range, canonical volume and prior-bar volume change,
session label and causal session distances when available, trend displacement,
efficiency, extension, retained progress, and ATR. At the completed signal-bar
close, the frozen versioned major-session specification and historical IANA
timezone rules populate the label and whole minutes since open/to close when
exactly one major session is active. Outside-session and overlapping-session
clocks are ambiguous and keep all three values null; the session-spec identity
is still recorded. These values never affect qualification. Same-instrument
overlap with frozen Module A is
reported at exact, ±15, ±30, and ±60 minutes and never filters the family.

## Classification frozen before execution

At the primary 1.50 cell, fewer than 100 unique events or fewer than 100 exact
60-minute outcomes is **INCONCLUSIVE**. With an adequate sample, **PASS** needs:

1. at least two instruments with at least 20 events each;
2. aggregate 60-minute mean greater than zero and median at least zero;
3. positive 60-minute mean on at least two instruments;
4. no instrument above 60% of events and no quarter above 50%; and
5. plateau acceptance: both neighboring cells have non-null 60-minute means,
   each at least 50% of the positive primary mean.

An adequately sampled result missing any condition is **KILL**. There are no
pair-specific rescues, post-hoc thresholds, or changes after results.

Stage-0 contains no costs, TP/SL, trailing stop, sizing, spread/slippage or
execution optimization, portfolio weighting, ML, or LLM decision. It asks only
whether gross directional mean-reversion expectancy exists.

## Pending authenticated run

```bash
uv run mr-lab-trend-exhaustion-stage0 \
  --corpus-root /path/to/frozen-2024-corpora \
  --registry configs/stage4a-2024-corpus-registry.json \
  --output-dir results/trend-exhaustion-stage0-v1
```
