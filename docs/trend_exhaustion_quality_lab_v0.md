# Trend Exhaustion Quality Lab v0 — TE-Q1

## Status and boundary

This is **POST-HOC development research** on the authenticated, registry-pinned
2024 corpora. Trend Exhaustion v1 remains permanently **KILL**. The frozen v1
detector and its result are not changed: TE-Q1 consumes its output and ranks
only unique primary opportunities having `displacement_threshold = 1.50`.
It does not decide whether an event exists. The final OOS year remains sealed.

Hypothesis TE-Q1: among already-qualified primary events, causal information
available by signal time may rank conditional future mean-reversion quality.

## Frozen contract

Continuous predictors are `normalized_displacement`, `efficiency_ratio`,
`extension_atr`, `retained_progress_atr`, `upper_wick_atr`, `lower_wick_atr`,
`body_size_atr`, `m15_range_atr`, `wick_body_ratio`, `frozen_atr20`,
`volume_change`, `distance_to_session_open_minutes`, and
`distance_to_session_close_minutes`. The first six `_atr` fields are computed
with the signal-time frozen ATR. Categorical predictors are `instrument`,
`direction`, and `session_label`. Missing session distances and volume change
are imputed in each training fold. Absolute volume is excluded: the corpora use
instrument-local quote activity, which is not cross-instrument comparable.
`volume_change` is admitted only when already emitted as a causal relative
change by the frozen detector.

The primary target is `y60_atr = signed_h60_price_movement / frozen_atr20`.
Raw signed H60 pips and the H15/H30/H120, sign, MFE, and MAE diagnostics are
outcomes only. Future observations, outcome fields, future session state,
month, and quarter are never predictors. Month and quarter are reporting keys.

The expanding folds are frozen as Jan–Apr→May, Jan–May→Jun, Jan–Jun→Jul,
Jan–Jul→Aug, Jan–Aug→Sep, Jan–Sep→Oct, Jan–Oct→Nov, and Jan–Nov→Dec.
Imputation, robust scaling, one-hot category discovery, spline construction,
and every estimator are fit anew on a training fold. Only test predictions are
pooled.

Primary models are: (A) Huber regression (`epsilon=1.35`, L2 `alpha=1.0`);
(B) regularized quantile regressions at 0.25, 0.50, and 0.75 (`alpha=0.1`),
whose mean prediction is the family score; and (C) an additive cubic spline
model with five knots per continuous feature followed by ridge regression
(`alpha=10`). Categories enter additively. There are no interactions or tuned
challenger. Deterministic tie-breaking uses event ID.

## Preregistered classification and selection

Promotion requires at least 1,000 pooled OOS rows and 200 top-quintile rows;
top-20% mean raw H60 pips > 0, median >= 0, PF >= 1.20, normalized lift over
baseline >= 0.05 ATR, Spearman >= 0.05, at least three of four adjacent
quintile means ordered, positive top-20% raw expectancy on at least three
instruments, and maximum selected-event shares <= 40% instrument, <= 30% month,
and <= 60% quarter. At least two primary model families must independently pass.
If enough observations exist but these gates fail, classification is `PARK`;
otherwise it is `INCONCLUSIVE`. These thresholds are frozen before execution.

All three families are reported. Selection checks Huber, quantile, then additive
in that simplicity order and chooses the first within 0.02 ATR top-quintile
lift and 0.01 Spearman of the empirical leader. It never selects from one best
bucket. Promotion is development advancement, not an alpha claim.

## Execution

No substitute data may be downloaded. With the existing frozen detector export:

```bash
uv run mr-lab-te-quality \
  --events "$FROZEN_2024_CORPUS_ROOT/trend-exhaustion-v1/primary-events.jsonl" \
  --output results/te-q1-development/report.json
```

Output includes post-hoc EDA, pooled score buckets, stability/concentration,
and deterministic month-block bootstrap intervals. No execution optimization is
part of this lab.
