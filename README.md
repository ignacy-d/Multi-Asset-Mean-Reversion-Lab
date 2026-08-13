# Multi-Asset Mean Reversion Lab

Multi-Asset Mean Reversion Lab is a Python research-engineering framework for
systematically evaluating **families** of mean-reversion hypotheses across
instruments, timeframes, sessions, models, rules, and cost assumptions. It does
not assert that an edge exists.

## Current stage: 1D — reproducible historical research corpus

The earlier typed configuration, immutable canonical bar contract, collection
validation, conservative resampling, Stage 1A daily parser, and Stage 1B
multi-day assembly remain intact, as do the separate Stage 1C historical session
semantics. Stage 1D adds a bounded, explicit-date acquisition boundary and a
deterministic corpus-request audit. **No strategy, feature, signal, backtest,
generic provider protocol, broker calendar, or execution integration has been
implemented.**

## Frozen first research split

The initial EURUSD research pipeline pre-registers these inclusive UTC calendar
periods:

| Role | Start | End | Stage 1D treatment |
| --- | --- | --- | --- |
| Discovery | 2024-01-01 | 2024-12-31 | Default acquisition range |
| Future OOS holdout | 2025-01-01 | 2025-12-31 | Untouched; do not acquire, inspect, summarize, or analyze yet |

The 2025 holdout is deliberately reserved until strategy families and research
methodology have been frozen. Stage 1D acquires only the 2024 discovery period
by default and never combines discovery and holdout automatically. This is the
first frozen discovery period for the initial EURUSD pipeline, not necessarily
the final discovery corpus for the whole project. Neither period, nor selecting
them in advance, implies that an edge exists.

## Stage 1D corpus boundary

```text
explicit historical date range
    ↓
GitHub Actions external acquisition
    ↓
immutable successful daily BI5 payloads + provenance
    ↓
Stage 1A canonicalization
    ↓
Stage 1B multi-day assembly
    ↓
deterministic corpus manifest
    ↓
future research engine
```

`enumerate_dates` requires explicit inclusive dates, returns deterministic
ascending dates, rejects reverse ranges, and limits one request to 370 calendar
days. `acquire_range` downloads sequentially with a small configurable delay;
it does not infer dates from the clock or filesystem ordering. The supported
scope remains Dukascopy EURUSD native M1 BID, canonical UTC, with
`QUOTE_ACTIVITY` volume. The verified Stage 1A decoder is reused unchanged and
is not generalized to other instruments.

An HTTP 404 for the exact Dukascopy daily-file URL is the only response treated
as confirmed provider absence. It is recorded and acquisition continues.
Timeouts, DNS/transport failures, every other unexpected HTTP response
(including server errors), empty/HTML/corrupt/malformed payloads, parsing
failures, and canonical validation failures abort the run. No exchange calendar
is consulted, and absent dates and missing intervals are never synthesized or
forward-filled. This narrow 404 rule reflects an explicit missing resource; the
repository does not claim that every weekend or holiday must return 404.

Each successful component retains its raw BI5 file and Stage 1A acquisition
provenance, raw SHA-256, requested date, and daily dataset ID. Stage 1B's
`assemble_daily_payloads` creates the logical market dataset and its existing
assembled dataset ID. The corpus manifest separately binds the requested range,
successful dates, confirmed absent dates, component identities, assembled
identity, canonical time span, M1/gap counts, and complete/incomplete M5, M15,
and H1 counts.

The corpus ID is SHA-256 over canonical sorted compact JSON. Paths, retrieval
times, workflow IDs, current time, host timezone, machine names, and randomness
are excluded. Reacquiring byte-identical successful payloads with the same
request and absent dates produces the same ID; changing a raw payload or the
requested range changes it. Daily dataset IDs, the assembled market dataset ID,
and corpus request ID remain conceptually distinct. Stage 1C's
`session_spec_id` is not included in any of them; later experiments will record
the market `dataset_id`, `session_spec_id`, and research specification
separately.

## Historical session semantics

The explicit Stage 1C boundary is:

```text
canonical UTC bars
    ↓
historical IANA timezone conversion (zoneinfo)
    ↓
independent major-session membership
    ↓
derived overlap / session-only regime
    ↓
named research windows / killzone candidates
```

`classify_bar` classifies a completed bar by its canonical UTC `open_time`.
Thus an M15 bar opening at a session's local 08:00 is in that session, while
the preceding 07:45 bar is not. Every interval is half-open `[start, end)` and
cross-midnight intervals are supported. Classification is a pure function of
the timestamp and immutable `SessionSpec`; it neither mutates bars nor examines
prices, later bars, or eventual daily statistics. `classify_bars` labels only
observations supplied by the caller and therefore does not invent missing bars.

The versioned `DEFAULT_SESSION_SPEC` contains configurable **research**
definitions, not claims about a centralized official spot-FX exchange:

| Major session | IANA timezone | Local half-open window |
| --- | --- | --- |
| `asia` | `Asia/Tokyo` | 09:00–18:00 |
| `london` | `Europe/London` | 08:00–17:00 |
| `new_york` | `America/New_York` | 08:00–17:00 |

Membership booleans are independent. Regimes such as `london_only` and
`london_new_york_overlap` are derived from the sessions simultaneously active
on the historical date; UTC overlap hours are never hardcoded. Any number of
sessions, including a future three-way overlap, is represented generically.

The same specification also ships the following initial **hypothesis windows**.
All use `America/New_York`, so their UTC placement follows historical EST/EDT:

| Candidate | Local half-open window |
| --- | --- |
| `asian_kz_20_00_et` | 20:00–00:00 |
| `london_kz_02_05_et` | 02:00–05:00 |
| `london_core_02_04_et` | 02:00–04:00 |
| `new_york_kz_07_10_et` | 07:00–10:00 |
| `new_york_kz_0830_1100_et` | 08:30–11:00 |
| `london_close_10_12_et` | 10:00–12:00 |
| `new_york_lunch_11_13_et` | 11:00–13:00 |

These names and boundaries are configurations to test empirically, not assumed
alpha or proven market truths. Multiple candidates deliberately coexist and
may overlap rather than allowing the implementation to select the most
attractive result after the fact. Optional active weekdays use the window's
local date; for a cross-midnight window, this means the date on which it starts.

`SessionSpec.to_json()` sorts semantically unordered definitions and emits
canonical compact JSON. `session_spec_id` hashes that JSON independently of
machine state, paths, current date, display timezone, and the market
`dataset_id`. Experiments can consequently record both identities without a
change of session hypotheses pretending to be a change of underlying data.

The timezone roles remain deliberately separate:

- **source timezone** belongs to provider-local canonicalization;
- **canonical UTC** is the immutable internal timestamp carried by `Bar`;
- **session timezone** is an IANA civil timezone used only to interpret a local
  research window on each historical date;
- **researcher/display timezone** is optional presentation after labeling; and
- **future broker/server timezone** will belong to a separate execution and
  symbol-hours boundary.

Stage 1C never reinterprets canonical timestamps using `source_timezone` and
never reads the machine's local offset. A researcher's current Polish local
time—or London, New York, or any other display time—has zero influence on a
historical label. Likewise, whatever timezone a future broker chart displays
cannot change market-session semantics. A later broker architecture may map
canonical UTC to broker server time and symbol trading hours, but this stage has
no broker dependency.

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

`assemble_daily_payloads` accepts immutable `DailyPayload` values that pair raw
bytes with their requested UTC dates. It reuses Stage 1A canonicalization,
rejects duplicate dates and malformed components, sorts by date, validates the
combined bars, and never fills missing intervals or calendar days. Its stable
assembled identity hashes canonical sorted compact JSON containing the fixed
semantics and the ordered component dates, raw SHA-256 values, and daily dataset
IDs. Input enumeration order, paths, retrieval timestamps, workflow state, and
machine state cannot affect that identity.

The serializable multi-day manifest exposes the covered dates and per-day raw
and canonical identities, the combined time range and M1 count, gaps, and
M5/M15/H1 complete and incomplete-window counts. It is a logical audit, not a
large-scale storage format. Generic resampling remains fixed-duration,
UTC-epoch-aligned, availability-aware, and prefix-invariant across day
boundaries.

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

The manually triggered **Acquire historical corpus** GitHub Actions workflow
accepts required `start_date` and `end_date` inputs. Their defaults are the
discovery-only range `2024-01-01` and `2024-12-31`. It installs the locked
environment, performs bounded sequential public HTTPS acquisition, validates
each successful day through Stage 1A, assembles through Stage 1B, writes the
corpus manifest, and uploads successful BI5 files, per-day provenance, and the
manifest as a 14-day artifact. Any genuine acquisition or corruption error
fails the job. Its 180-minute timeout is intended for the bounded year-long
sequential run. No credentials are required.

The established boundary is:

```text
Dukascopy
    ↓
GitHub Actions acquisition
    ↓
immutable raw BI5 + raw SHA
    ↓
Stage 1A provider-local daily BI5 canonicalization
    ↓
ordered Stage 1B multi-day assembly
    ↓
deterministic dataset manifest / identity
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
research dataset. For the frozen Stage 1A smoke path, override both workflow
inputs to `2024-01-02`. Trigger
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
