# Multi-Asset Mean Reversion Lab

Multi-Asset Mean Reversion Lab is a Python research-engineering framework for
systematically evaluating **families** of mean-reversion hypotheses across
instruments, timeframes, sessions, models, rules, and cost assumptions. It does
not assert that an edge exists.

## Current stage: 0C — dataset semantics and fixed-duration resampling

The Stage 0A typed experiment configuration and Stage 0B immutable canonical
bar contract remain intact. Stage 0C adds storage-neutral collection validation,
point-in-time views, an explicit experiment-timeframe boundary, and conservative
fixed-duration resampling. **No data acquisition, provider adapter, strategy,
feature, signal, session classifier, or backtest has been implemented.**

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
and `1h`, not provider aliases. Stage 0A experiment timeframes remain opaque
identifiers; `normalize_timeframe` is the explicit boundary that maps supported
identifiers (`M5`, `M15`, and `H1`) to canonical data-layer timeframes. Canonical
inputs are also accepted, while provider notation does not enter resampling.

The collection validator requires canonical bars with one instrument, timeframe,
price basis, and volume meaning, in deterministic chronological order. Duplicate
observations, duplicate intervals, overlaps, and malformed durations are
structural errors. Gaps are reported separately as information: they are legal,
are not forward-filled, and may represent closures, illiquidity, or missing data.
Observation identity includes the logical dataset ID, instrument, timeframe,
open time, and price basis. `available_bars` selects solely by
`available_at <= research_time`, including delayed observations.

Resampling supports only larger fixed durations that are integer multiples of
the homogeneous source duration. Target windows are aligned by flooring UTC
time to duration boundaries anchored at the Unix epoch; there is no session or
calendar anchoring. Complete windows use first open, maximum high, minimum low,
final close, and compatible-volume summation (or preserve absent volume).
Incomplete windows are skipped and explicitly reported rather than synthesized.
Output availability is the later of the target close and every component's
availability. Tests enforce prefix invariance: transforming the by-time prefix
matches the already-available outputs selected from the full transformation.

These APIs operate on small tuples/lists without prescribing future storage:
large datasets need not be represented as collections of Python `Bar` objects.
Dataset metadata continues to record source and semantic provenance; no adapter
or dataset fingerprint generator is included.

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

The checked-in `uv.lock` pins the complete environment. Stage 0C adds no runtime
or development dependencies.

## External historical-data acquisition bridge

The manually triggered **Acquire historical sample** GitHub Actions workflow
downloads the frozen Dukascopy EURUSD M1 BID sample for 2024-01-02. External
acquisition is intentionally delegated to a GitHub-hosted runner because Codex
Cloud is not the acquisition environment. The workflow uses a public HTTPS GET,
validates the LZMA/BI5 container structure, and uploads the immutable raw payload
plus JSON SHA-256 provenance as a short-lived workflow artifact.

Raw datasets remain ignored under `data/raw/` and are never committed. Trigger
`.github/workflows/acquire-historical-sample.yml` manually from GitHub's Actions
tab and download the named run artifact for inspection. Canonical research logic
remains provider-neutral; Stage 1A canonical parsing and validation are not yet
complete and must follow inspection of the real artifact.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

An example experiment contract is in `configs/example.toml`. Loading it only
parses and validates configuration; it does not execute research or trading.
