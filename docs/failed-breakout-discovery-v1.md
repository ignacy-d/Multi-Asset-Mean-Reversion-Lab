# Failed Breakout discovery preregistration v1

Scope: 2024 discovery inputs only.

## Hypothesis

A completed close beyond a previously known structural boundary, followed by failure to maintain the new level and a completed close back inside the prior range, has positive forward expectancy in the fade direction.

This family is intentionally independent of VWAP distance and the frozen OU eligibility model. It is evaluated standalone before any combination with Module A.

## Initial anchors

The implementation contract accepts precomputed causal structural levels. The first research pass should populate only these preregistered anchor families:

- previous-day high / previous-day low,
- Asia session high / low,
- first 60 minutes of the London session opening range high / low.

Anchor values and normalization scale must be available before the breakout can qualify.
Generation consumes immutable canonical M1 `Bar` values, requires exact minute
coverage for each construction window, and never fills missing observations.

Each level records explicit `available_at` and `expires_at` UTC instants. The
previous-day pair is the prior completed 17:00-to-17:00 America/New_York FX
trading day, becomes available at its closing boundary, and expires at the next
FX-day boundary. The Asia pair becomes available at the
historical session close and expires at the next Asia open. London OR60 becomes
available only after all first-60-minute bars are complete and expires at that
London session's close. Session boundaries use the existing versioned IANA
timezone session specification, including historical DST.

## Frozen normalization

Every anchor uses the same pre-event scale: the high-low range of the most recent
completed 17:00-to-17:00 America/New_York FX trading day. Historical IANA
timezone rules determine both boundaries. The FX day must have exact M1
coverage, the range must be positive, and every contributing bar must already
be available. Otherwise the anchor fails closed. This definition is
intentionally independent from VWAP, z-scores, and OU state and is not selected
from observed results.

## Detector grid

Fixed reclaim window: 30 minutes.

Minimum breakout depth, normalized by a pre-event scale:

- 0.05,
- 0.15.

No post-hoc rescue of failed anchor/depth cells.

## Event semantics

1. A level is armed only while causally available and unexpired.
2. A breakout episode starts on the first completed M1 close outside the level whose excursion reaches the minimum normalized depth.
3. The event timestamp is the first later completed M1 close back inside the old range within 30 minutes.
4. Upper-level failures produce SHORT events; lower-level failures produce LONG events.
5. Missing M1 intervals cancel the open episode and fail closed.
6. Events contain no VWAP, OU, TP, SL, cost, or trade-execution assumptions.

## Stage-0 diagnostics

For each anchor/depth cell and instrument, measure common discovery outcomes at 15, 30, 60 and 120 minutes, MFE/MAE, sample count, unique timestamps and monthly distribution.

Primary family horizon: 60 minutes.

Anchor-specific detections remain raw structural events for the anchor/depth
matrix. Family evidence deduplicates simultaneous signals by `(instrument,
signal_timestamp, direction)`. All contributing anchor families and levels
remain in the raw event artifact; coincident anchors do not create independent
family samples. Reports label `raw_structural_event_count`,
`unique_family_opportunity_count`, and `unique_global_signal_clocks` separately.
Depth aggregates apply the same family-opportunity deduplication within each
depth before measuring the parameter plateau.

## Orthogonality test against Module A

Measure timestamp overlap with frozen Module A at:

- exact timestamp,
- +/-15 minutes,
- +/-30 minutes,
- +/-60 minutes.

The frozen Module A reference exposed to this runner is instrument and clock
based. Therefore overlap compares each direction-specific Failed Breakout family
opportunity with Module A clocks for the same instrument, without claiming that
the Module A direction also matches. Module A itself is not modified.

Operational interpretation before Stage 1:

- strong independence candidate: at least 50% of failed-breakout events remain outside +/-30 minutes of Module A, while standalone directional expectancy survives cross-asset checks;
- high overlap is not automatically a failure, but the family must then be treated as a possible timing/filter layer rather than claimed as an independent alpha.

The 50% threshold is an architecture/research triage rule, not a profitability threshold and must not be tuned to 2024 results.

## Stage-0 family pass criteria

Do not promote on one best cell. Promote only if the family shows a broad, interpretable pattern with adequate event count, at least two instruments with meaningful samples, non-negative median aggregate directional outcome, positive aggregate mean at the primary horizon, and no single instrument or quarter dominating the effect.

If sample size is inadequate, classify as inconclusive rather than lowering thresholds after observing results.

## Stage-0 execution and outputs

The deterministic runner consumes only the five authenticated 2024 registry
entries and expects one established offline-corpus directory per instrument:

```bash
uv run mr-lab-failed-breakout-stage0 \
  --corpus-root /path/to/frozen-2024-corpora \
  --output-dir results/failed-breakout-stage0 \
  --registry configs/stage4a-2024-corpus-registry.json
```

It generates structural levels and both depth cells, measures exact-clock
15/30/60/120-minute signed outcomes through the common Stage 4A M1 path
semantics, records MFE/MAE, and compares distinct signal clocks with the existing
frozen OU Module A candidates at exact, 15, 30, and 60-minute windows. The
runner emits deterministic event JSONL, the complete anchor/depth/instrument
matrix (including zero-event cells), metrics and overlap JSON, a report, and a
hash-bearing summary.

The Stage-0 triage floors are declared in code rather than inferred from output:
100 aggregate events with exact 60-minute outcomes, at least 20 events in each
of at least two instruments, positive aggregate mean, non-negative aggregate
median, positive replication in at least two instruments, positive aggregate
expectancy across all physical observations in each preregistered depth cell,
no instrument above 60% of events, and no quarter
above 50%. Samples below the event/outcome floor are `INCONCLUSIVE`; adequately
sampled evidence that misses promotion criteria is `KILL`. Independence remains
a separate label based on the preregistered 50% unique-at-30-minutes rule.

Exact coverage of the previous completed FX trading day remains a deliberate
fail-closed part of the frozen scale definition. Closed weekend intervals are
skipped rather than treated as trading days, while incomplete open-market days
cannot supply a scale. Coverage and zero-event cells must be reported rather
than silently substituting a different scale after observing results.

## Current empirical execution status

**Complete.** The operator executed the deterministic runner on the five exact
registry-pinned 2024 corpora at PR head `fb6e925`. Every local manifest matched
the registry's instrument, requested date range, assembled dataset identity, and
corpus identity. The run used the frozen methodology above without changes and
emitted all six declared output artifacts. These are operator-reported results;
the corpora were not mounted and the study was not rerun in the documentation
environment.

### Observed counts

- Raw structural events: **7,448**.
- Unique family opportunities: **6,055**.
- Unique global signal clocks: **5,692**.
- Raw contributing anchors: Asia session **2,306**, London OR60 **2,774**, and
  previous day **2,368**.
- Depth 0.05: **5,812** unique family opportunities.
- Depth 0.15: **1,075** unique family opportunities.

Instrument-level opportunity counts, events per month, and the full monthly
expectancy series remain available in the deterministic `metrics.json` and
`anchor-depth-instrument.csv`. Their individual values were not included in the
operator's supplied summary and are therefore not reconstructed here.

### Aggregate signed forward outcomes

All values below are gross signed pips, before trading or cost assumptions.

| Horizon | N | Mean | Median | Win rate | Forward-return PF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15m | 6,055 | -0.06133773740709919 | -0.0999999999995449 | 0.4898431048720066 | 0.9788647033717464 |
| 30m | 6,055 | -0.1815028901733964 | -0.10000000000065512 | 0.48819157720891826 | 0.9555539378644894 |
| 60m primary | 6,055 | -0.015309661436811294 | 0.10000000000065512 | 0.5028901734104047 | 0.9971790869005591 |
| 120m | 6,055 | -0.0996696944673666 | 0.30000000000001137 | 0.5066886870355078 | 0.9865764795900231 |

Median MFE was **9.700000000000841 pips**, median MAE was
**9.600000000000364 pips**, and their median comparison ratio was
**1.010416666666716**.

### Cross-asset and temporal stability at 60 minutes

| Instrument | Mean signed pips |
| --- | ---: |
| AUDJPY | -0.5806511627907113 |
| AUDUSD | -0.015251572327013893 |
| EURUSD | 1.0217492260062448 |
| GBPUSD | 0.0620490620490584 |
| USDJPY | -0.8302912621359023 |

Maximum instrument concentration was **0.22890173410404624**.

| Quarter | Mean signed pips |
| --- | ---: |
| 2024-Q1 | 0.09616935483873483 |
| 2024-Q2 | -0.13863031914892257 |
| 2024-Q3 | 0.1400671140940019 |
| 2024-Q4 | -0.15003178639543108 |

Maximum quarter concentration was **0.2597853014037985**. The detailed monthly
series was generated by the run but was not included in the supplied compact
summary, so it is not duplicated here.

### Depth plateau

| Depth | Unique opportunities | 60m mean pips | Forward-return PF |
| --- | ---: | ---: | ---: |
| 0.05 | 5,812 | -0.03193392980039645 | 0.9941506092064903 |
| 0.15 | 1,075 | 0.19767441860468082 | 1.0304821195472915 |

The positive 0.15 cell does not rescue the family. Depth 0.05 was negative, so
the preregistered neighboring-depth plateau criterion failed.

### Module A orthogonality

Overlap is same-instrument and clock-based as preregistered.

| Window | Overlap | Failed Breakout unique vs Module A |
| --- | ---: | ---: |
| Exact | 4 | 99.93393889347647% |
| +/-15m | 155 | 97.44013212221305% |
| +/-30m | 291 | 95.19405450041288% |
| +/-60m | 476 | 92.13872832369943% |

Per-instrument overlap rows were generated in `overlap.json`, but their values
were not present in the supplied compact result and are not inferred here.

## Stage-0 conclusion

- Family classification: **KILL**.
- Independence status: **PARTIALLY ORTHOGONAL**.
- Promotion decision: **Failed Breakout v1 is not promoted to Stage 1**.

The broad cross-asset structural-failure hypothesis was successfully falsified
at Stage 0. Its signal clocks were highly distinct from Module A, but aggregate
gross expectancy was approximately zero, three of five instruments had negative
primary expectancy, two of four quarters were negative, and the required depth
plateau failed. Timing distinctness without standalone evidence is insufficient
for promotion.

### Post-hoc observation, not a rescue

The completed matrix includes positive individual cells. In particular, the
operator reported that EURUSD / Asia session / depth 0.05 had approximately 476
observations, a 60-minute mean of +1.44937 pips, a median of +0.60 pips, a
forward-return PF of 1.3631, and positive quarterly primary expectancy in all
four quarters. This is only the post-hoc hypothesis `FB-EUR-1: EURUSD structural
rejection around Asia H/L`. It is not validated alpha, does not change the v1
KILL decision, and must receive a separate preregistration before any future
research.
