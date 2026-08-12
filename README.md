# Multi-Asset Mean Reversion Lab

Multi-Asset Mean Reversion Lab is a Python research-engineering framework for
systematically evaluating **families** of mean-reversion hypotheses across
instruments, timeframes, sessions, models, rules, and cost assumptions. It does
not assert that an edge exists.

## Current stage: 0C — dataset and resampling semantics

The repository provides a typed experiment configuration contract and a
provider-neutral semantics for completed OHLC bars, logical datasets, and safe
fixed-duration resampling. **No trading strategy, feature, signal, session
classifier, provider integration, data download, or backtest has been
implemented.**

## Research philosophy

- Parameterize research dimensions instead of embedding them in strategy code.
- Treat a hypothesis as a family of configurations; one failed variant does not
  reject the family.
- Separate broad discovery from robustness and out-of-sample confirmation.
- Preserve strict point-in-time semantics: a value at time `t` may use only
  information available at or before `t`.
- Use UTC as the internal timestamp standard and keep market-data providers and
  execution systems outside research logic.
- Prefer objective variables and reproducible experiments over discretionary
  labels or attractive backtest results.

## Foundation and future architecture

`mr_lab.config` owns the small Stage 0A contract. A configuration selects the
instrument, timeframe, session, strategy, feature/model, entry and exit
parameters, and cost-model identifier. Future modules can therefore consume a
configuration without owning those choices.

`mr_lab.data` defines validated bar, timeframe, price-basis, volume-semantics,
and dataset-metadata values without choosing a dataframe or storage engine. A
bar begins at `open_time`, ends at `close_time`, and its complete OHLC values
may be used only when `available_at <= research_time`. `available_at` is at
least `close_time`; complete values must never be treated as known at the bar's
open. The per-row model defines semantics and does not require large datasets
to be stored as millions of Python objects.

Collection validation enforces homogeneous source semantics, deterministic
ordering, unique observations, and non-overlapping intervals while reporting
gaps as information rather than errors. Point-in-time views select solely by
`available_at <= research_time`.

Fixed-duration resampling uses Unix-epoch-anchored UTC boundaries, requires an
integer source-to-target ratio, and omits incomplete windows while reporting
them explicitly. Output availability is the later of the target close and all
constituent availability times. Experiment identifiers such as `M15` and `H1`
are normalized at the configuration boundary to canonical `15m` and `1h`
durations. Deterministic transformations must remain prefix invariant: results
available by time T must be identical whether computed from data available by T
or selected from a full-history computation.

Later stages may add provider adapters, immutable raw-data handling,
point-in-time features, session handling, strategies, backtesting, experiment
metadata, risk, and reporting. Those packages are deliberately absent until
they have real behavior and enforceable interfaces; empty directories would
create the appearance of architecture without a tested contract. Research and
broker/execution integrations will remain separate.

## Setup

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --dev
```

The checked-in `uv.lock` pins the complete Stage 0A environment.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

An example experiment contract is in `configs/example.toml`. Loading it only
parses and validates configuration; it does not execute research or trading.
