# Multi-Asset Mean Reversion Lab

Multi-Asset Mean Reversion Lab is a Python research-engineering framework for
systematically evaluating **families** of mean-reversion hypotheses across
instruments, timeframes, sessions, models, rules, and cost assumptions. It does
not assert that an edge exists.

## Current stage: 0B — canonical market data contract

The Stage 0A typed experiment configuration remains intact. Stage 0B adds a
small, immutable semantic contract for provider-neutral completed OHLC bars and
their dataset metadata. **No data acquisition, strategy, feature, signal,
session classifier, resampling, or backtest has been implemented.**

## Canonical market data

`mr_lab.data.Bar` defines one completed observation interval. `open_time` is the
interval beginning, `close_time` is its end, and `available_at` is the earliest
instant at which the complete bar may safely be used. All three timestamps must
be timezone-aware and use canonical UTC; a bar is point-in-time usable at
research time `T` only when `available_at <= T`. In particular, a completed
candle is never fully known at its open.

Each bar explicitly records whether its OHLC values are bid, ask, mid, trade,
other, or unknown prices. Volume is optional and separately identifies traded,
tick, quote-activity, unknown, or no-volume semantics. Tick volume is therefore
not implicitly treated as centralized traded volume.

`Timeframe` accepts explicit lowercase fixed-duration notation such as `15m`
and `1h`, not provider aliases such as `M15`. Stage 0A experiment timeframes
remain intentionally opaque identifiers (including `"M15"`); converting that
boundary to canonical `"15m"` belongs to Stage 0C and is not implemented here.

The per-row model establishes semantics only. It is neutral about providers and
future storage layout: large datasets need not be represented as collections of
Python `Bar` objects. Dataset metadata records source and semantic provenance,
but Stage 0B does not provide adapters or generate dataset fingerprints.

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

The checked-in `uv.lock` pins the complete Stage 0B environment. Stage 0B adds
no runtime or development dependencies.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

An example experiment contract is in `configs/example.toml`. Loading it only
parses and validates configuration; it does not execute research or trading.
