# Multi-Asset Mean Reversion Lab

Multi-Asset Mean Reversion Lab is a Python research-engineering framework for
systematically evaluating **families** of mean-reversion hypotheses across
instruments, timeframes, sessions, models, rules, and cost assumptions. It does
not assert that an edge exists.

## Current stage: 1A-P — real BI5 parsing and canonicalization

The earlier typed configuration, immutable canonical bar contract, collection
validation, and conservative resampling remain intact. Stage 1A-P adds one
narrow provider-local parser for the observed Dukascopy EURUSD M1 BID BI5
candle representation. **No strategy, feature, signal, session classifier,
backtest, generic provider protocol, or execution integration has been
implemented.**

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
Dataset metadata records source and semantic provenance. The Dukascopy parser's
dataset ID hashes canonical JSON containing stable provider, instrument,
timeframe, price side, requested day, raw SHA-256, parser version, and canonical
schema version. Retrieval time and other operational details are excluded.

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
checks its frozen SHA-256, parses and canonically validates all M1 bars, uses the
generic resampler for M5, M15, and H1, then uploads the immutable raw payload,
acquisition provenance, and a deterministic canonicalization audit as a
short-lived artifact.

The established boundary is:

```text
Dukascopy
    ↓
GitHub Actions acquisition
    ↓
immutable raw BI5 + raw SHA
    ↓
provider-local BI5 parser
    ↓
canonical Bar + DatasetMetadata
    ↓
validate_dataset()
    ↓
generic resample_bars()
```

The parser is explicitly limited to the verified EURUSD M1 BID daily candle
format: big-endian `>5if` records containing seconds from the requested UTC day,
open, close, low, high integer prices, and volume. EURUSD prices use the observed
`integer / 100000` scale. Bars represent completed intervals, use BID prices,
and become available at interval close. Gaps remain gaps; zero-volume bars are
retained.

Official JForex [`IBar.getVolume()`](https://www.dukascopy.com/client/javadoc/com/dukascopy/api/IBar.html#getVolume--)
documentation defines volume as the sum of best-price volumes for every tick in
the bar. The parser therefore uses `VolumeSemantics.QUOTE_ACTIVITY`; this is
quote-side activity, not centralized executed FX traded volume. Numeric values,
including zero, are retained.

Raw datasets remain ignored under `data/raw/` and are never committed; the tiny
four-record test fixture only freezes the observed binary schema and is not a
research dataset. Trigger
`.github/workflows/acquire-historical-sample.yml` manually from GitHub's Actions
tab and download the named run artifact for inspection. Acquisition and parsing
stay separate, canonical research logic remains provider-neutral, and no generic
data-source protocol has been introduced.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

An example experiment contract is in `configs/example.toml`. Loading it only
parses and validates configuration; it does not execute research or trading.
