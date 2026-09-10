# Ornstein–Uhlenbeck process core v1

## Research role

This module models the completed-price deviation from the existing causal
benchmark, `x_t = p_t - e_t`, rather than raw FX price. The benchmark supplies a
point-in-time equilibrium; the residual is therefore the process for which
mean-reverting dynamics are meaningful. Native VWAP, canonical-M1 VWAP, and
Bollinger values are taken unchanged from `assemble_signal_states`. The module
does not construct a competing benchmark and does not alter Stage 4B or Stage 4C.

The conceptual precedents are Avellaneda and Lee's autoregressive modeling of
factor-model residuals, Mudchanatongsuk, Primbs, and Wong's OU spread model, and
Elliott, van der Hoek, and Malcolm's mean-reverting Gaussian Markov pairs model.
Their numerical windows, thresholds, entry levels, and annualization choices are
not transferred to this intraday FX setting.

## Model and units

The continuous process is

```text
dX_t = kappa * (mu - X_t) dt + sigma dW_t.
```

For an exactly equally spaced interval `Δ`, its discrete AR(1) form is

```text
X_(t+Δ) = intercept + phi * X_t + epsilon
phi = exp(-kappa * Δ)
kappa = -log(phi) / Δ
mu = intercept / (1 - phi).
```

The implementation reports `kappa_per_minute`; `Δ`, half-life, and all other
time quantities are explicitly in minutes. It never silently annualizes. The
mean-reversion half-life is `log(2) / kappa`. With the OLS residual standard
deviation `innovation_sigma`, stationary standard deviation is
`innovation_sigma / sqrt(1 - phi²)`. The diagnostic equilibrium score is
`(current_deviation - mu) / stationary_sigma`. This score is descriptive only.

OLS with an intercept uses the trailing configurable number of valid transition
pairs. Innovation variance uses residual sum of squares divided by `N - 2`, and
the slope standard error is derived from that variance and centered regressor
sum of squares. No heavy statistical dependency is required.

## Point-in-time and exact-gap semantics

Every input represents a completed observation and its state has
`available_at == t`. A state at `t` may include the transition ending at `t`,
because both endpoints are then available. The implementation traverses each
process in timestamp order and emits the state immediately; it never revisits an
emitted value, so observations after `t` cannot affect the state at `t`.

A pair is admitted only when its timestamps differ by exactly one signal
timeframe. For M15, 10:00 to 10:15 is valid and 10:00 to 10:30 is not. A missing
bar, inactive interval, outage, overnight break, or weekend therefore contributes
no synthetic pair. The bounded rolling deque retains earlier independently valid
pairs across a gap; the window is the last `N` valid one-step transitions, not
the last `N` arbitrary observations. This deliberately differs from clearing all
history at a gap while still preventing elapsed-time compression.

Process identity includes instrument, benchmark family, signal timeframe,
session/context, lookback, and upstream strategy specification. Direction is
excluded. Identical LONG and SHORT materializations at one process timestamp are
collapsed, while disagreement in `p0`, `e0`, or deviation fails closed.

## Structural status

Before a full window exists, status is unavailable with
`insufficient_history`. Degenerate regressors, non-finite inputs or estimates,
non-positive innovation variance, and undefined mappings are explicit reasons.
The standard continuous-time mapping is structurally valid only for
`0 < phi < 1`; non-positive and at-least-one estimates remain recorded but are
marked invalid. A `phi` close to one is mathematically valid here. This version
does **not** select a practical-speed cutoff or any trading eligibility threshold.

## Candidate diagnostics and next step

The local `mr-lab-ou-diagnostics` command accepts an explicit verified 2024
corpus, registry, instrument, output directory, and one or more
`--window-transitions` values. It reuses existing feature assembly and Stage 4B
candidate generation, then writes candidate-aligned CSV diagnostics, a summary,
and a provenance/hash audit. Alignment requires the process state at exactly the
candidate's completed signal timestamp; later states are never substituted.
Rows contain no trade return, exit, or other future outcome.

The next research step may compare frozen Stage 4B candidate outcomes
conditionally on these states using 2024 in-sample evidence, treating windows as
a hypothesis family rather than selecting an attractive result. Only after a
broad robust rule is documented, tested, and frozen may a later integration
version change eligibility. All logic and choices must be frozen before opening
the sealed 2025 holdout.
