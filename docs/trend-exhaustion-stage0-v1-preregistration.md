# Trend Exhaustion Mean Reversion — Stage-0 v1 preregistration

## Status and scope

This document froze the method **before empirical execution**. Discovery was
restricted to the authenticated, registry-pinned 2024 M1 BID corpora for
EURUSD, GBPUSD, USDJPY, AUDUSD, and AUDJPY. The 2025 OOS is sealed and is not
used by this work. The authenticated operator run is now **complete**; no
substitute corpus was used. This family is independent of VWAP, OU, and Failed
Breakout.

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

## Authenticated run command

```bash
uv run mr-lab-trend-exhaustion-stage0 \
  --corpus-root /path/to/frozen-2024-corpora \
  --registry configs/stage4a-2024-corpus-registry.json \
  --output-dir results/trend-exhaustion-stage0-v1
```

## Observed Stage-0 result

The frozen five-instrument 2024 discovery run completed with the preregistered
classification **KILL**. Trend Exhaustion v1 is **NOT promoted to Stage 1**.
This is a completed falsification experiment and not evidence of deployable
alpha.

At the primary 1.50 ATR threshold, all 4,808 events had an exact 60-minute
outcome. The gross direction-signed results were:

| Metric | Result |
|---|---:|
| Event count / H60 count | 4,808 / 4,808 |
| Mean H15 | +0.0037853577370946644 pips |
| Mean H30 | +0.09319883527453801 pips |
| Mean H60 | +0.12495840266221844 pips |
| Median H60 | **-0.1999999999990898 pips** |
| H60 forward-return profit factor | 1.0297884366524672 |
| H60 win rate | 0.48211314475873546 |
| Mean H120 | -0.08221713810315644 pips |
| Mean MFE | 12.152142262895177 pips |
| Mean MAE | 12.103119800332783 pips |

The adequate-sample safeguard passed. Cross-asset replication passed because
AUDJPY, GBPUSD, and USDJPY had positive primary H60 expectancy. Instrument and
quarter concentration safeguards passed, and the threshold plateau passed.
The sole decisive failed PASS condition was the frozen requirement that the
primary aggregate H60 median be non-negative: the observed value was
-0.1999999999990898 pips (about -0.20), which is below zero. The other passing
safeguards do not override that condition, and it was not weakened, removed,
reinterpreted, or rescued after observation. The deterministic classification
therefore remains **KILL**.

### Cross-asset and quarterly diagnostics

Primary H60 instrument expectancy was:

| Instrument | Mean H60 (pips) |
|---|---:|
| AUDJPY | +0.46221294363257054 |
| AUDUSD | -0.2221871713984985 |
| EURUSD | -0.18772678762006872 |
| GBPUSD | +0.1673257023932904 |
| USDJPY | +0.3840159840159522 |

Quarterly H60 expectancy was 2024-Q1 +0.3417667238421841, 2024-Q2
+0.1306799336650394, 2024-Q3 +0.6426788685524039, and 2024-Q4
-0.5897893030794711 pips. Maximum instrument concentration was
0.2081946755407654 and maximum quarter concentration was
0.2566555740432612.

### Preregistered threshold plateau

| Displacement | N | Mean H60 (pips) | H60 profit factor |
|---|---:|---:|---:|
| 1.25 ATR | 5,101 | +0.09776514408938551 | 1.023481937695406 |
| 1.50 ATR primary | 4,808 | +0.12495840266221844 | 1.0297884366524672 |
| 1.75 ATR | 4,388 | +0.12645852324520784 | 1.0300530762564977 |

The preregistered plateau **passed**: both neighboring means were positive and
at least half the positive primary mean. Plateau acceptance does not rescue the
failed primary-median condition.

### Module A overlap

At the primary threshold, same-instrument overlap was 11/4,808 exact,
28/4,808 within ±15 minutes, 51/4,808 within ±30 minutes, and 102/4,808 within
±60 minutes. Trend Exhaustion timing was therefore highly orthogonal to Module
A, but orthogonality was diagnostic only and cannot rescue insufficient
standalone evidence.

### Post-hoc observations — not v1 evidence

After the frozen classification, session diagnostics suggested that some
conditional subsets may behave materially differently from the broad family.
Asia-session behavior and JPY-related behavior are possible future research
directions. Continuous quality features such as normalized displacement and
efficiency may also contain conditional information. These observations did
not filter, reclassify, or rescue v1. They are recorded only as possible future,
separately preregistered experiments:

- `TE-Q1` — Trend Exhaustion conditional quality/meta-model;
- `TE-ASIA-1`;
- `TE-ASIA-JPY-1`.

None was promoted, validated, optimized, or tested in this PR. No session,
instrument, or quality filter was added to Trend Exhaustion v1.
