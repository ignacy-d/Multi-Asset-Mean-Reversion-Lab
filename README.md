# Multi-Asset Mean Reversion Lab

## Frozen Stage 4 discovery atlas

`mr-lab-stage4-discovery-atlas` creates a descriptive family-level atlas from
existing, authenticated Stage 4B raw outputs. It never runs signal or trading
logic, rejects inputs outside the frozen 2024 research period, verifies the
Stage 4B audit commitments, and keeps baseline and filtered evidence separate.

```bash
uv run mr-lab-stage4-discovery-atlas \
  --stage4b-dir results/stage4b/EURUSD/shard-0 \
  --stage4b-dir results/stage4b/EURUSD/shard-1 \
  --module-a-dir results/stage4b-module-a/EURUSD/shard-0 \
  --spool-dir /tmp/mr-lab-atlas-spool \
  --output-dir results/stage4-discovery-atlas
```

The atlas authenticates candidate and trade JSONL as streams and stores the
minimal join metadata plus complete trades in a temporary SQLite spool. The
spool is removed on success or failure; `--spool-dir` selects its parent when
the system temporary filesystem is too small. Progress logs report authenticated
shards, candidate and trade rows, exact-cell groups, and elapsed time.

The generated matrices retain setup families, deduplicate correlated variants
at execution time, aggregate equivalent cross-asset hypotheses, preserve the
four frozen cost scenarios, and report descriptive overlap with Module A. The
shortlist uses evidence categories rather than a score or historical-best-cell
ranking. Raw Stage 4B input is required because monthly stability, execution
overlap, and duplicate-adjusted frequency cannot be authenticated from the
compact Stage 4C matrices alone.

Every input directory must be an authenticated Stage 4B shard, and every shard
of each logical run must be supplied. Performance remains attached to the full
signal/filter/exit cell. Setup-family rows summarize the distribution of exact
cell expectancies as an exit plateau, while execution-family rows contain only
duplicate-adjusted opportunity counts and memberships—never arbitrarily chosen
P/L. Optional frozen-OU Module A shards have an explicit overlap-only role and
are not pooled with baseline performance evidence.

Multi-Asset Mean Reversion Lab is a Python research-engineering framework for
systematically evaluating **families** of mean-reversion hypotheses across
instruments, timeframes, sessions, models, rules, and cost assumptions. It does
not assert that an edge exists.

## Current stage: Stage 3A multi-asset data enablement

Completed work comprises Stage 0A–1D foundation, canonical data, validation,
sessions and frozen corpus infrastructure; the Stage 2A point-in-time research
engine; the frozen Stage 2B VWAP and Stage 2C Bollinger benchmarks; and the
canonical-M1 VWAP construction robustness path. Stage 3A now makes the provider
and data boundary explicitly multi-asset while leaving all strategy arithmetic,
sessions, thresholds, lookbacks, and horizons unchanged.

The frozen production universe is exactly EURUSD, GBPUSD, USDJPY, AUDUSD, and
AUDJPY. Immutable `ProviderInstrumentSpec` values bind the canonical and provider
symbols, integer price scale and decimal precision, native M1 timeframe,
BID basis, QUOTE_ACTIVITY volume semantics, Dukascopy identity, and an explicit
verification state. EURUSD remains production-verified at scale 100,000 (five
decimals). GBPUSD and AUDUSD retain scales of 100,000/five; USDJPY and AUDJPY
retain scales of 1,000/three. GBPUSD, USDJPY, AUDUSD, and AUDJPY were promoted to
production-verified only after successful bounded real-provider verification run
`32113934283`. Normal acquisition, canonicalization, corpus assembly, and
replication now accept all five frozen instruments and continue to fail closed
for unsupported instruments. No research module infers these values.

Stage 3A verification is deliberately bounded to one public 2024-01-02 M1 BID
file for each new instrument. The pull-request and manually dispatched
`verify-stage-3a-instruments.yml` workflow checks URL resolution, LZMA decompression, whole
24-byte records, candidate scaling, plausible positive OHLC, metadata,
nonnegative quote activity, M1 construction, and generic resampling. This is the
only path allowed to use unverified candidates, and it creates no strategy
results. Promotion remains a deliberate source change after a successful run.
The same date can be checked locally with:

```bash
uv run python -m mr_lab.providers.verify_instruments \
  --output-dir data/raw/stage-3a-verification
```

New-instrument daily, assembled, and corpus identities include the complete
instrument spec in canonical sorted compact JSON before SHA-256 hashing. A
corpus contains exactly one instrument and mixed components fail explicitly.
The legacy EURUSD identity schemas and hash inputs are retained byte-for-byte,
so historical EURUSD IDs and result artifacts do not change.

Stage 3B is next: frozen 2024 multi-asset/session replication. Given one already
frozen corpus, the reusable runner executes the unchanged VWAP and Bollinger
grids for M5, M15, and H1, optionally adds canonical-M1 VWAP robustness, and
writes instrument-labelled documents:

```bash
uv run mr-lab-replicate \
  --instrument USDJPY \
  --corpus-dir /path/to/usdjpy-2024-corpus \
  --output-dir results/stage-3b-usdjpy \
  --canonical-m1-vwap
```

The runner writes one deterministic JSON document for every construction and
timeframe (nine documents when canonical-M1 robustness is enabled), followed by
a compact Markdown inventory. The inventory deliberately displays the
predeclared USDJPY natural sessions Asia and New York first, then London and any
remaining contexts; it retains links to every unranked result document and does
not alter or interpret the frozen benchmark rows.

The manually dispatched `run-stage-3b-multi-asset-real-2024.yml` workflow takes
an instrument and source workflow run ID, then resolves exactly one matching
instrument-labelled full-year corpus artifact from that run. It executes this
exact replication interface without provider access, verifies every expected
provenance and result identity, and uploads the JSON documents, comparison
report, and execution audit as one immutable, instrument-labelled GitHub Actions
artifact. It must be dispatched from the repository's `main` branch; dispatches
from any other ref fail before artifact download or replication.

## Stage 4A event-path diagnostics

Stage 4A adds a descriptive event record for every qualifying, frozen Stage 3B
signal observation. Observations are retained as-is and can overlap heavily;
they are **not independent trades**. No breach deduplication or trade construction
is performed. At signal time `T`, the research close is frozen as `P0`, the
applicable VWAP or Bollinger center is frozen as `E0`, and `D0 = P0 - E0`.
Movement is always evaluated against that frozen `E0`; later equilibrium values
cannot move the target or affect signal eligibility.

The diagnostic path uses exact, causal canonical-M1 observations after `T` and
does not stop at session boundaries. Direction-normalized snapshots at 5, 15,
30, 60, and 120 minutes use the exact-clock M1 **close**. Pathwise first passage
of 25%, 50%, 75%, and 100% reversion, continuous maximum reversion, and MFE use
the favorable intraminute extreme (high for long MR, low for short MR); MAE uses
the adverse extreme (low for long MR, high for short MR). Hit and excursion times
resolve to the first M1 minute containing the extreme. The ordering of high and
low within that minute is unknown and is not inferred.

Minimal causal pre-signal fields record raw and signed close movement over 5,
15, and 30 minutes plus `impulse_share_h = abs(P0 - P[-h]) / abs(D0)`. These
formation-speed values are descriptive and do not classify shocks. Stage 4A is
not a trade simulator, and its path extremes do not create TP/SL rules.

A missing minute anywhere in the required 120-minute future path marks the event
incomplete: fixed snapshots that exist remain explicit, while first passage,
maximum reversion, and pathwise MAE/MFE are unavailable rather than computed on
a shortened path. Each unavailable pre-signal offset is independently `null`.

The deterministic Stage 4A reporting API writes exactly `events.jsonl`,
`matrix.csv`, `summary.json`, and `report.md`. The matrix groups by instrument,
benchmark family, signal timeframe, session, direction, lookback, threshold, and
the unchanged event methodology ID. Exact-snapshot aggregates expose their own
available counts. First-passage hit rates, MAE/MFE, and maximum-reversion
summaries use complete 120-minute paths only; hit-time summaries use hits only.
Missing observations are excluded rather than replaced with zero. Quantiles use
Python's `statistics.quantiles(method="inclusive")` linear interpolation at
`(n - 1) * p`, with a singleton returning its sole value.

The concise report starts with the predeclared `|z| >= 2.0` anchor, then shows
the 1.0/1.5/2.0/2.5 threshold response, directional asymmetry with LONG and SHORT
preserved, and descriptive benchmark robustness. Rows remain in methodology
identity order: nothing is ranked, selected, or optimized. Stage 4A rows are
conditional observations and may overlap heavily; observation counts are **not
independent trade counts**, and no test may assume event independence here.
Stage 4A adds no Stage 4B construction or deduplication, costs, spreads,
slippage, trade or execution rules, TP/SL, optimization, dynamic equilibrium,
filters, models, reporting dashboard, or 2025 analysis.

Later work remains realistic transaction costs and execution semantics,
dependence-aware/event-level trade construction, OU quality filtering only if
simple families justify it, protected OOS/walk-forward, portfolio risk, Monte
Carlo/prop-risk analysis, and an eventual live/tick execution layer. The 2025
holdout remains rejected and untouched.

## Canonical-M1 VWAP construction robustness

The alternative equilibrium applies the unchanged Stage 2B HLC3 and quote-activity
formula independently to each Stage 1C session instance, but uses canonical M1 bars.
At a research observation available at `T`, its session-instance VWAP includes only
M1 observations with `available_at <= T`; no value is carried between session
instances or between overlapping London/New York streams. Native research-bar
volatility, signal boundaries, thresholds, lookbacks, and exact-clock outcomes stay
frozen. The separate robustness identity records `vwap_construction_source` as
`canonical_m1` or `native_timeframe` without changing historical Stage 2B IDs.

The offline-only runner validates the complete manifest date declaration before
loading any component and writes separate results:

```bash
uv run mr-lab-vwap-m1-robustness \
  --corpus-dir /path/to/saved-2024-corpus \
  --timeframe M15 \
  --output results/stage-2b-vwap-m1-robustness-2024-m15.json
```

The manually dispatched
`.github/workflows/run-stage-2b-vwap-m1-robustness-real-2024.yml` workflow reuses
the frozen Stage 1D artifact without contacting Dukascopy and emits an independent
audit and artifact.

## Stage 2C Bollinger benchmark semantics

For each M5, M15, and H1 observation, the equilibrium is the arithmetic mean of
CLOSE over exactly 20 or 40 consecutive canonical observations ending at the
current completed bar. Dispersion is the sample standard deviation of those
same CLOSE values, and `bollinger_z = (close - middle) / sample_stddev`. A zero
dispersion makes the feature unavailable.

Every observation in the window must be Stage 2A research-active. An inactive
flat zero-volume filler remains in canonical time, invalidates any required
window containing it, and is never skipped or allowed to signal. A moving
zero-volume bar remains active. Consequently the implementation neither
compresses elapsed time nor reaches farther into history across inactive gaps.

The frozen discovery grid is lookbacks 20/40 and absolute z thresholds
1.0/1.5/2.0/2.5. Strict inequalities generate LONG below the negative threshold
and SHORT above the positive threshold; equality does not signal. Exact-clock
Stage 2A outcomes are 15/30/60/120 minutes for M5 and M15, and only 60/120
minutes for H1.

Unlike Stage 2B, session membership does not anchor or reset the Bollinger
equilibrium. Existing Stage 1C Asia, London, and New York memberships are only
contextual breakdowns. Because memberships are independent and overlap, a
signal can legitimately contribute to both London and New York summaries;
these counts are not a mutually exclusive partition. Named-window labels remain
attached to the underlying research observations for later work.

`BollingerStrategySpec` freezes CLOSE, arithmetic rolling mean, sample CLOSE
standard deviation, and the consecutive-active canonical-window rule. It hashes
canonical sorted compact JSON with SHA-256, independently of dataset, session,
and research identities. Output provides unranked overall and contextual rows,
all/long/short directions, counts, exact-clock availability, mean/median signed
return, win rate, sample standard deviation, standard error, descriptive
t-statistic, available months, and normalized monthly frequency. The
t-statistic alone is not a significance claim.

The offline runner checks every manifest request, successful/absent, and
component date against the frozen 2024 discovery interval before calling
`load_offline_corpus()`; it never acquires data:

```bash
uv run mr-lab-bollinger-benchmark \
  --corpus-dir /path/to/saved-2024-corpus \
  --timeframe M15 \
  --output results/stage-2c-2024-m15.json
```

The manual `.github/workflows/run-stage-2c-real-2024.yml` workflow reuses the
frozen Stage 1D artifact and emits separate Stage 2C M5/M15/H1 artifacts without
modifying Stage 2B results.

## Stage 2B VWAP benchmark semantics

For every research-active bar in each independently active Stage 1C major
session, the benchmark defines `HLC3 = (high + low + close) / 3` and computes
`VWAP_t = cumulative(HLC3 * volume) / cumulative(volume)` through the current
completed bar. Dukascopy volume has `QUOTE_ACTIVITY` semantics, so this is an
**activity-weighted bar VWAP**, not centralized executed-volume FX VWAP.
Inactive flat zero-activity filler bars stay in canonical time but neither add
weight nor generate signals. A price-moving zero-volume bar remains
research-active and adds zero weight; VWAP is unavailable while cumulative
weight is zero.

Each stream resets at its Stage 1C major session's historical local start using
the configured IANA timezone and DST rules. Asia, London, and New York are
independent anchors. Thus an overlap bar can carry distinct London and New York
VWAPs while retaining its contextual `active_sessions`, regime, and named
windows.

The benchmark price is the current close. `relative_deviation = close / VWAP -
1`, and `vwap_deviation_z = relative_deviation / rolling_volatility`, where
rolling volatility is the sample standard deviation of exactly the configured
number of consecutive close-to-close arithmetic bar returns ending at the
current bar. Both adjacent observations must be research-active for a return;
the denominator is unavailable during warm-up, across inactive filler, or when
zero. This explicit rule uses past/current completed bars only and preserves
prefix invariance.

The frozen discovery grid is M5/M15/H1, exact-clock 15/30/60/120-minute Stage
2A horizons, thresholds 1.0/1.5/2.0/2.5, and volatility lookbacks 20/40 bars.
Strict inequalities create LONG below `-threshold` and SHORT above
`+threshold`; equality creates no signal. Every configuration and direction
(`all`, `long`, `short`) remains visible and is never profit-ranked.

`VwapStrategySpec` hashes canonical sorted compact JSON with SHA-256. Its
versioned inputs are HLC3, quote-activity weighting, major-session-instance
reset, the normalized-deviation formula, lookback, and threshold.
`strategy_spec_id` remains separate from `dataset_id`, `session_spec_id`, and
`research_spec_id`.

The offline-only runner reconstructs data exclusively with
`load_offline_corpus()`, rejects any manifest declaring dates outside the
frozen 2024 discovery period before reading BI5 components, resamples without
compressing inactive elapsed time, and emits deterministic compact JSON or CSV
summaries. It never acquires data:

```bash
uv run mr-lab-vwap-benchmark \
  --corpus-dir /path/to/saved-2024-corpus \
  --timeframe M15 \
  --output results/stage-2b-m15.json
```

Summary rows retain all four independent identities, timeframe, anchor,
threshold, lookback, horizon, and direction, plus signal/LONG/SHORT counts,
valid and unavailable counts, mean/median signed forward return, win rate,
sample standard deviation, standard error, a descriptive t-statistic when
defined, and signals per available UTC research month. A t-statistic alone is
not a claim of statistical significance.

## Stage 2A research semantics

`is_no_activity_flat_bar` is true exactly when `volume == 0` and
`open == high == low == close`. It reads only the current canonical bar. This is
a provider-observation classification, not a claim that the global FX market
was officially closed. Such a bar is not initially research-active, but remains
in the canonical sequence with its original timestamps. Zero volume with price
movement and nonzero volume with flat OHLC both remain active.

`build_research_observations` preserves one observation per supplied bar and
reuses Stage 1C `classify_bar`; it does not duplicate session logic. A completed
bar becomes observable only at its existing `available_at`. M5, M15, and H1 are
handled as ordinary canonical fixed-duration bars.

Forward horizons are positive elapsed `timedelta` values. For a source available
at `t`, the target must have `available_at == t + horizon` exactly. Missing and
inactive exact targets make the label unavailable; the engine never selects the
next active bar, deletes inactive time, or compresses gaps. The sole Stage 2A
return is the same-price-basis close-to-close arithmetic label
`target_close / source_close - 1`. It is statistical BID-data research output,
not executable long/short PnL. `Direction.LONG` and `Direction.SHORT` only sign a
label supplied to them and do not generate signals.

`ResearchSpec` sorts and deduplicates its horizon set and hashes canonical
compact sorted JSON with SHA-256. Its inputs are the research schema version,
activity-rule version, horizons in whole seconds, and return definition.
`research_spec_id` deliberately excludes both market `dataset_id` and
`session_spec_id`, so all three identities can be recorded independently.

`load_offline_corpus` is the narrow no-network bridge from a saved Stage 1D
directory. It reads successful dates declared by `corpus-manifest.json`, never
discovers days by directory enumeration, verifies each conventional BI5 file
against its provenance and manifest SHA-256, reconstructs through the existing
Stage 1B assembler, and requires the resulting dataset identity to equal the
stored assembled identity. Missing, corrupt, or mismatched components fail.

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
days. `acquire_range` downloads sequentially with a configurable one-second
default delay and logs progress for every requested day;
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
Transient HTTP and transport failures use six retries after the initial request
with bounded 1, 2, 4, 8, 16, and 30 second backoffs. HTTP 404 is never retried.

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

## Roadmap

- **Stage 1D:** reproducible discovery corpus.
- **Stage 2A (completed):** point-in-time research engine.
- **Stage 2B (completed):** session-reset VWAP deviation benchmark.
- **Stage 2C (current):** rolling Bollinger deviation benchmark.
- **Next:** VWAP construction robustness: native-timeframe session VWAP versus
  canonical-M1 session VWAP sampled on research timeframes.
- **Then:** multi-asset and session replication.
- **Later:** realistic costs and execution assumptions.
- **Later:** consider OU only if simple benchmark families justify it.
- **Protected:** out-of-sample and walk-forward evaluation remains untouched.

The split remains frozen: **2024 is discovery; 2025 is the untouched future OOS
holdout.** Stage 2C neither reads, acquires, summarizes, nor analyzes 2025.

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
# Stage 4A real 2024 path diagnostics

After review and merge, `run-stage-4a-real-2024.yml` runs either `ALL` five frozen
instruments or one selected instrument. Source workflow/artifact identities come
from `configs/stage4a-2024-corpus-registry.json`; no source run ID is entered at
dispatch. USDJPY, AUDUSD, and AUDJPY are verified; EURUSD awaits reviewer
verification and GBPUSD awaits full-year acquisition. An individual verified
instrument may run, but `ALL` fails preflight until all five are verified. The
workflow runs only from `main`.

Each instrument artifact contains exactly `events.jsonl`, `matrix.csv`,
`summary.json`, `report.md`, and operational provenance in
`execution-audit.json`. The research files remain the frozen Stage 4A
conditional-observation diagnostics—not trades. Runs are bounded to 2024
discovery data; the 2025 holdout remains sealed, including for incomplete
late-December paths.

Pins are immutable and fail closed. If a pinned GitHub artifact expires, a
reviewed registry replacement or reacquisition is required; the workflow never
selects a latest or substitute artifact automatically.

## Stage 4B runtime sharding

`mr-lab-stage4b` remains the canonical unsharded local command. Its optional
`--shard-index` and `--shard-count` flags are operational execution controls,
not research parameters: the runner always assembles every signal state and
performs the frozen global deduplication/re-arm pass before selecting a shard.
Candidates are grouped by instrument, benchmark family, signal timeframe,
session, direction, and lookback. The deterministically sorted group sequence
is divided into contiguous ranges, approximately balanced by candidate count;
a group is never split.

The real-2024 workflow fixes the operational count at four and does not expose
it through `workflow_dispatch`. Each completed shard independently uploads a raw
artifact containing candidate and trade rows and a compact artifact containing
reports and its hash-bearing manifest. Raw hashes are computed by the shard with
streaming SHA-256. `mr-lab-stage4b-reduce` downloads only compact artifacts,
independently verifies their files, and requires the manifest's raw hashes and
row counts without materializing raw files. It concatenates group-complete
aggregate and distribution rows rather than averaging medians or quantiles. The
immutable raw artifacts and their manifest commitments are the canonical raw
result; the reducer creates the canonical combined review artifact without
constructing or downloading giant `trades.jsonl` files.
