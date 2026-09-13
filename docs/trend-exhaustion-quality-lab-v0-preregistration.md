# Trend Exhaustion Quality Lab v0 — TE-Q1 preregistration

## Status and hypothesis

TE-Q1 is explicitly **POST-HOC development research**. It asks whether causal
information available at signal time can rank conditional future mean-reversion
quality among already-qualified primary Trend Exhaustion opportunities. Trend
Exhaustion v1 remains frozen and permanently classified `KILL`; TE-Q1 neither
rescues nor reclassifies it, and promotion below is not final alpha validation.

The sole development universe is 2024. The final OOS year remains sealed. The
runner accepts only the explicit authenticated 2024 registry, validates registry
and corpus manifests before loading offline M1 data, verifies assembled dataset
identity, canonically resamples M1 to M15, runs the unchanged frozen detector at
only `displacement_threshold = 1.50`, creates typed `DirectionalPathRequest`s,
uses machine-readable `diagnose_directional_paths`, and joins each unique event
to one causal feature row. No event-export interchange format is an input.

Before any per-instrument manifest or corpus access, the runner requires the
literal development root `/mnt/e/mr-lab/frozen-2024`, the canonical
`configs/stage4a-2024-corpus-registry.json`, its exact schema, the exact five
instruments, verified status, and the exact 2024 date range on every entry. It
does not discover or fall back to another root. Loaded bar timestamps and event
timestamps are checked again for UTC 2024 as defense in depth.

## Frozen feature and outcome contract

Continuous predictors are `normalized_displacement`, `efficiency_ratio`,
`exhaustion_extension / frozen_atr20`, `retained_progress / frozen_atr20`,
`upper_wick / frozen_atr20`, `lower_wick / frozen_atr20`,
`body_size / frozen_atr20`, `m15_range / frozen_atr20`, `wick_body_ratio`,
`minutes_from_session_open`, and `minutes_to_session_close`. Categorical
predictors are `instrument`, `direction`, and `session_label`.

Raw `frozen_atr20` and `volume_change` are not predictors. Future returns,
`y60_atr`, MFE, MAE, time-to-extreme, future volatility/bars/session state,
month, quarter, and sealed information are forbidden predictors. Month and
quarter are reporting keys. `volume_change` may only be descriptive.

The target is `y60_atr = signed H60 price movement / frozen_atr20`, obtained
from the typed H60 directional-path outcome. Raw H60 pips are retained for
economic reporting; H15/H30/H120, MFE, and MAE are outcome diagnostics only.

## Frozen analysis

The expanding walk-forward folds are Jan–Apr→May, Jan–May→Jun,
Jan–Jun→Jul, Jan–Jul→Aug, Jan–Aug→Sep, Jan–Sep→Oct, Jan–Oct→Nov, and
Jan–Nov→Dec. Imputation, scaling, category discovery, spline fitting, and the
estimator are fitted inside each training fold. Evaluation pools test
predictions only.

Primary families are (A) Huber regression (`epsilon=1.35`, `alpha=1.0`),
(B) quantile regressions at 0.25, 0.50, and 0.75 (`alpha=0.1`) whose predeclared
family score is their pointwise median, and (C) additive cubic splines with five
knots per continuous feature and Ridge `alpha=10`. There are no unrestricted
interactions and no boosted trees.

Pooled OOS reporting includes quintiles, top 20%, top 10%, Spearman rank
correlation, top-minus-bottom normalized spread, top-selection lifts, and the
number of ordered adjacent quintile pairs. Buckets report N, normalized mean
and median, raw-pip mean and median, PF, win rate, and MFE/MAE means. Top-20%
stability is reported by instrument, test month, quarter, direction, and
session. Uncertainty uses 2,000 deterministic whole-month block-bootstrap
replicates with seed `20240913` for top-20% raw and normalized means and the
top-minus-bottom normalized spread.

Regeneration must reproduce the frozen v1 primary event universe: 4,808 total,
with AUDJPY 958, AUDUSD 951, EURUSD 937, GBPUSD 961, and USDJPY 1,001. These
counts are reported as a provenance/reproducibility diagnostic and are not an
alpha-selection criterion. A mismatch fails closed without changing the
detector. Nonfinite required ranking diagnostics are represented as JSON null,
cannot pass promotion, and are handled deterministically during model selection.

POST-HOC EDA reports missingness, quantiles and expectancy buckets for each
causal predictor, plus descriptive instrument, direction, and session summaries.
It cannot directly create an optimized filter.

## Promotion and model selection

Each family independently passes only with pooled OOS N ≥ 1,000; top-20% N ≥
200; top-20% raw H60 mean > 0, median ≥ 0, and PF ≥ 1.20; normalized top-20%
lift ≥ 0.05 ATR; pooled Spearman ≥ 0.05; at least three of four adjacent
quintile mean pairs ordered; positive top-20% raw expectancy on at least three
instruments; and maximum selected instrument, month, and quarter shares of 40%,
30%, and 60%, respectively. At least two primary families must independently
pass for `PROMOTE_TO_FROZEN_CANDIDATE`. Insufficient sample size is
`INCONCLUSIVE`; otherwise the result is `PARK`.

All families are reported. The simplest family within 0.02 ATR top-quintile
lift and 0.01 Spearman of the empirical leader is selected, ordered Huber,
quantile, then additive splines. No isolated best bucket determines selection.

## Execution

With authenticated corpora already present, run:

```console
uv run mr-lab-trend-exhaustion-quality --corpus-root /mnt/e/mr-lab/frozen-2024 --registry configs/stage4a-2024-corpus-registry.json --output-dir results/trend-exhaustion-quality-v0
```

Do not download substitutes or execute against any other year.
