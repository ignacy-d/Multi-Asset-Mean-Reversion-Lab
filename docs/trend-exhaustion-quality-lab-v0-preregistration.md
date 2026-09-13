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
For loaded bars, 2024 applies to the opening period: every timestamp must be
UTC, each open must be in 2024, close must equal open plus the canonical
timeframe duration, and availability cannot precede close. Consequently, the
natural `2025-01-01T00:00:00Z` completion/availability of the M1 bar opened at
`2024-12-31T23:59:00Z` is valid; a bar opened in 2025 remains forbidden.

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

## Canonical development result

Empirical execution against the authenticated local 2024 development corpus is
**COMPLETE**. Primary-event regeneration exactly matched frozen Trend Exhaustion
v1: 4,808 events in total, comprising AUDJPY 958, AUDUSD 951, EURUSD 937,
GBPUSD 961, and USDJPY 1,001. The pooled expanding walk-forward OOS sample was
N=3,237.

The frozen development classification is **`PARK`**. Frozen Trend Exhaustion v1
remains permanently **`KILL`**. TE-Q1 does not rescue or reclassify v1, and its
model selection does not mean promotion. The preregistered requirement that at
least two primary model families independently pass was not met. This result
must not be reinterpreted as a near-pass.

### Primary model outcomes

| Primary family | Spearman | Top-20 raw H60 mean (pips) | Top-20 raw median (pips) | Top-20 PF | Top-20 normalized lift (ATR) | Ordered adjacent quintile pairs | Frozen outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Huber | 0.0017910110418523223 | +0.6983024691358163 | -0.1999999999991564 | 1.139945568132619 | 0.6637643380931977 | 3/4 | Does not independently pass |
| Quantile family | -0.0024873046534666806 | -0.5302469135802249 | -0.5499999999997174 | 0.8583560062659794 | -0.005181084960279164 | 2/4 | Does not independently pass |
| Additive splines | 0.013361783572481913 | +0.037037037037041864 | 0.0 | 1.0083168728558074 | 0.3925187665450436 | 2/4 | Does not independently pass |

Huber's top-20 month-block-bootstrap 95% interval for raw H60 mean was
approximately [-1.2803, +3.2562] pips. Huber fails promotion especially on its
negative top-20 median, PF below 1.20, and Spearman below 0.05. The quantile
family's top-20 selection was temporally concentrated in Q2 and failed the
frozen concentration gates. Additive splines also did not independently pass.

Huber is the selected model under the preregistered model-selection rule. That
selection is only the required relative choice among the reported families; it
does not override any promotion gate and does not change `classification = PARK`.

## POST-HOC BACKLOG — not part of the frozen TE-Q1 conclusion

### TE-Q2-SIMPLE

The following are **POST-HOC development observations** and provenance for a
possible future hypothesis only. They do not rescue TE-Q1, do not reclassify
Trend Exhaustion v1, and must not be converted into a trading rule in this PR.
Full-2024 descriptive EDA suggests that high `normalized_displacement` and high
`efficiency_ratio` may contain nonlinear conditional information:

- `normalized_displacement` highest quintile: N=961, H60 mean +1.42955 pips,
  median +0.10 pips, PF 1.30756.
- `normalized_displacement` highest decile: N=480, H60 mean +2.00458 pips,
  median +0.50 pips, PF 1.48199.
- `efficiency_ratio` highest quintile: N=961, H60 mean +0.78533 pips, median
  +0.20 pips, PF 1.20252.
- `efficiency_ratio` highest decile: N=480, H60 mean +1.30729 pips, median
  +0.35 pips, PF 1.35704.

### Future methodological note

For future quality-ranking studies that pool predictions from expanding
walk-forward folds, investigate preregistered fold-local percentile/rank
calibration before pooling because raw model score levels may not be comparable
across folds. This is a future design note only and does not alter TE-Q1.
