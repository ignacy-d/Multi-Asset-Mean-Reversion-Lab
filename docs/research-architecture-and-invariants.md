# Research architecture and invariants

Status: **authoritative implementation map as of the repository revision that
adds this document**. This is a description of the existing system, not a new
strategy specification. The frozen interpretation and cost preregistration
remain controlling where this map and an implementation comment differ.

## 1. System overview

```text
Dukascopy M1 BID BI5 bytes + retrieval provenance
  -> LZMA/24-byte decoding -> immutable UTC Bar objects
  -> daily/multiday dataset -> immutable offline corpus + corpus-manifest.json
  -> pinned 2024 corpus registry
  -> completed-bar ResearchObservation + DST-aware session labels
  -> conservative UTC M5/M15/H1 resampling
  -> native VWAP | canonical-M1 VWAP | Bollinger features (independent families)
  -> Stage 4A frozen signals -> M1 event-path diagnostics + reports/audit
  -> Stage 4B all valid states -> global dedup/re-arm -> candidate events
  -> entry construction -> future M1 exit/path simulation (gross BID/BID)
  -> raw candidate/trade JSONL + compact reports + audit
  -> optional stable-group shards -> fail-closed Stage 4B reducer
  -> frozen 2024 cross-asset interpretation (not executable code)
  -> Stage 4C-A immutable cost overlay (16 scenarios per complete trade)
  -> compact trade matrix, regime breadth, scenarios, report, summary, audit
  -> optional hash-owned Stage 4C shards -> fail-closed Stage 4C reducer
  -> eventual, separately authorized sealed-2025 OOS gate (not implemented)
```

The provider boundary is isolated under `mr_lab.providers`; canonical data and
research code do not depend on broker execution. Console entry points are
declared in `pyproject.toml`: the benchmark runners, replication runner,
`mr-lab-stage4a`, `mr-lab-stage4b`, both reducers, and `mr-lab-stage4c`.

## 2. Stage-by-stage data flow

### 2.1 Provider acquisition, decoding, and canonical data

**Inputs and transformation.** `providers/dukascopy.py` builds explicit daily
URLs, downloads bytes, validates payloads, hashes the bytes, and writes the raw
`.bi5` plus JSON provenance. `dukascopy_bi5.py` LZMA-decompresses whole 24-byte
records, applies a production-verified per-instrument price scale, validates
ordered M1 offsets/OHLC/volume, and emits frozen `Bar` values. Bars are M1 BID,
quote-activity volume, UTC, and become available at their close. Instrument
facts live in `providers/instruments.py`; verification-only candidates are kept
separate from production-verified specifications.

**Outputs and provenance.** Daily metadata uses `bar-v1` and parser identifier
`dukascopy-bi5-eurusd-m1-bid-v1`; its `dataset_id` hashes stable canonicalization
inputs including the raw payload SHA-256. Retrieval time is audit metadata, not
dataset identity. Invalid compression, records, scales, chronology, response
status, or unverified instruments fail with explicit exceptions; no bar is
synthesized.

**State and PIT.** Raw files are intended to be immutable. `Bar` and
`DatasetMetadata` are frozen dataclasses. `Bar` requires timezone-aware UTC and
`available_at >= close_time`; `available_bars` is the legal as-of view.

### 2.2 Multiday corpus, manifest, and registry

`dukascopy_multiday.py` assembles validated daily payloads and gives the assembled
dataset a deterministic identity. `dukascopy_range.py` requires an explicit,
bounded range, classifies only provider 404 as absence, and requires every date
to be successful or explicitly absent. Stage 3 acquisition accepts only
`2024-01-01..2024-12-31`, refuses to overwrite a corpus manifest, and records
component hashes, counts, gaps, resample counts, parser/schema facts, dataset ID,
and a hash-derived `corpus_id`. `load_offline_corpus` reads only dates declared by
the manifest, verifies each raw file against both daily provenance and manifest,
then reconstructs and checks the assembled identity.

`configs/stage4a-2024-corpus-registry.json` is the source-selection registry
(`stage-4a-2024-corpus-registry-v1`). It pins instrument, exact 2024 range,
corpus/dataset IDs, workflow run, artifact ID/name, and verification status.
Stage 4A/B validate the manifest **before** corpus bytes and require exact 2024
metadata. A selected production entry must be `verified`; `ALL` fails until all
members of the frozen universe are verified. Corpus bytes and manifests are
immutable source state; registry verification status is controlled operational
state and must be reviewed rather than inferred.

Failure is closed for a bad schema/universe/status, partial pin, name/range/date
mismatch, absent/corrupt component, hash mismatch, or reconstructed identity
mismatch.

### 2.3 PIT observations, sessions, and resampling

`research.py` wraps each completed bar without deleting/reindexing it, labels a
zero-volume flat bar inactive from its own fields, and attaches session
classification. `sessions.py` classifies the bar **open time** against immutable
half-open local-clock windows using IANA historical timezone rules. Overlap is
allowed. `DEFAULT_SESSION_SPEC` (`stage-1c-research-v1`) defines Asia, London,
New York and named windows; its semantic JSON has a SHA-256 `session_spec_id`.

`data/resampling.py` validates a homogeneous ordered source, uses epoch-aligned
fixed UTC windows, emits only complete contiguous source multiples, reports
incomplete windows, and sets output availability to at least the window close
and the latest component availability. It never fills gaps. All outputs are
derived and reproducible; inputs are not mutated. Malformed timestamps,
heterogeneous/overlapping bars, invalid timeframe multiples, or boundary-crossing
components fail.

### 2.4 Benchmark features and earlier research labels

`vwap_benchmark.py` builds session-anchored native-timeframe VWAP and trailing
deviation normalization; `vwap_m1_robustness.py` constructs the separately
identified canonical-M1 VWAP view; `bollinger_benchmark.py` uses exactly N
consecutive active closes and clears its window across gaps. All are one-pass,
completed-observation calculations and have immutable, hashed strategy specs.
Strict threshold comparisons mean equality does not signal. `research.py` also
supports fixed-clock forward outcomes, but availability is explicit and adding a
future target cannot alter its source observation.

These three benchmark families are separate robustness views. They must not be
combined into consensus, AND/OR, ensemble, or weights without a new preregistered
methodology. Benchmark CLIs validate manifest-declared discovery dates before
loading corpus data and emit deterministic, unranked tables.

### 2.5 Stage 4A: descriptive event paths

**Inputs.** A verified exact-2024 corpus/registry entry, resampled observations,
the complete frozen family/lookback/threshold/timeframe grid, and canonical M1.
`stage4a_runner.py` reuses the benchmark builders; it does not recalculate their
semantics. A `FrozenSignal` captures timestamp, family, timeframe, session,
lookback, threshold, direction, `p0`, equilibrium `e0`, displacement `d0`, z,
corpus/dataset IDs, and strategy-spec ID.

**Transformation/PIT.** `stage4a.py` fixes equilibrium at signal time, then uses
exact-clock M1 observations strictly before the signal for presignal metrics and
minutes 1..120 after it for diagnostic outcomes. Future M1 is legitimate only as
an ex-post label. Missing future minutes retain available fixed snapshots but
make pathwise metrics unavailable. It diagnoses; it does not deduplicate or
construct trades.

**Outputs/provenance.** `events.jsonl`, `matrix.csv`, `summary.json`, and
`report.md` use stable order and `stage-4a-report-v1`; summary hashes the other
outputs. With `--registry`, `execution-audit.json` binds those hashes to source
commit, registry/artifact/run, corpus/dataset, range, counts, and the semantic
`STAGE4A_METHODOLOGY_ID` (a hash of named path rules). Missing M1 is reported;
invalid signal/M1 identity, order, UTC, price, methodology, registry, or hashes
fail. The local CLI only writes the audit when `--registry` is supplied and then
requires `GITHUB_SHA`; this is an operational limitation noted below.

### 2.6 Stage 4B: candidates, entries, paths, and gross artifacts

**Inputs.** Stage 4B regenerates features from the verified corpus rather than
consuming Stage 4A files. It freezes M5/M15/H1, lookbacks 20/40, z threshold 2,
three entry modes, TP fractions `(0.25,.50,.75,1)`, SL extensions
`(.25,.50,1,None)`, and 30/60/120-minute stops in `stage4b.py`. The default
eligibility filter is `none-v1`.

**Candidates.** Both directions receive every valid active state. State identity
is instrument × family × timeframe × session × direction × lookback. A strict
`|z| > 2` state qualifies; after an event that identity is disarmed until a
valid state has strict `|z| < 2`. Equality neither signals nor re-arms. Dedup is
global and sorted before any shard partition. Candidate IDs hash state identity,
timestamp, `p0/e0/z`, and threshold.

**Entries and paths.** `immediate` enters at signal timestamp/`p0`.
`m1-reclaim-p0` waits up to 15 completed M1 closes; the extension mode waits up
to 30 minutes for an adverse 0.25 displacement followed by a same-or-later
completed close beyond `p0`. Entry eligibility is evaluated once before any
future entry path. A target already passed at entry is ineligible. Exit simulation
then reads only M1 bars strictly after entry, jointly resolves TP/SL and the exact
time stop, headlines adverse-first when both barriers occur in one M1, preserves
the favorable-first bound, and treats exit-bar extrema as bounds rather than
inventing intrabar order. Incomplete paths stay explicitly incomplete.

**Outputs.** `candidate-events.jsonl` and `trades.jsonl` are raw immutable
records; compact artifacts are diagnostics/distributions, `trade-matrix.csv`,
`summary.json`, `report.md`, and `execution-audit.json`. Trade returns are gross
**BID/BID**, not executable expectancy. The audit (`stage-4b-report-v1`) records
source commit, corpus/dataset, registry artifact/run, filter identity, counts,
`STAGE4B_METHODOLOGY_ID`, and output hashes. Errors are raised on incompatible
source identity, duplicate M1 availability, invalid entry/exit, inconsistent
filter identities, and invalid shard arguments.

### 2.7 Stage 4B sharding and reduction

See §5. Shards retain both raw and compact artifacts. The compact reducer
validates raw hash/count commitments without downloading the raw files, then
concatenates already group-complete aggregates; it deliberately never merges
quantiles. The reduced audit records every shard manifest and combined hashes.

### 2.8 Frozen cross-asset interpretation

`docs/stage4b-2024-cross-asset-freeze.md` is the human decision record between
gross construction and costs. It does not filter Stage 4C rows in code. The cost
overlay runs all complete source trades, while reviewers apply the frozen regime
map and breadth objective. This separation prevents a reporting label from
silently changing signal or trade construction.

### 2.9 Stage 4C-A cost overlay and reporting

`stage4c.py` accepts only complete Stage 4B rows for EURUSD, USDJPY, AUDUSD, or
AUDJPY and pins both the live Stage 4B methodology ID and historical source
commit `3090682…`. It creates new dictionaries, preserving source gross fields.
`stage4c_runner.py` can consume one or more existing `trades.jsonl` files or
regenerate Stage 4B and capture its immutable callback stream. SQLite is only a
temporary bounded-memory grouping/spooling mechanism: a primary key rejects
duplicate trade identity, rows are read in key order, and the database is
removed. `--spool-dir` changes storage location only.

Outputs (`stage-4c-report-v1`) are `stage4c-trade-matrix.csv`,
`stage4c-regime-breadth.csv`, `stage4c-cost-scenarios.csv`, summary, report, and
execution audit. Optional debug JSONL is not a production output. The audit
binds source hashes/identity, corpus/dataset/registry, Stage 4B methodology and
source commit, Stage 4C commit, exact cost-profile file hash, scenario matrix,
counts, output hashes, account currency, conversion rule, and the explicit
absence of volume-band and swap modelling. Invalid/nonfinite/duplicate inputs or
missing provenance fail closed.

`tools/run_stage4c_local.sh` is the reviewed production-compute path: it accepts
only four instruments, uses a `corpora/2024/<instrument>` location, verifies
registry identities before compute, runs focused tests/lint, routes heavy temp
and configurable spool storage, runs regenerated Stage 4B→4C, verifies outputs
and hashes, rejects debug/spool leakage, and builds a compact review archive.

The `.github/workflows` files document earlier acquisition, verification,
republishing, Stage 2/3 runs, and Stage 4A/B artifact production/reduction. They
historically resolved and reverified pinned artifacts and produced reviewable
audits. They are not the production Stage 4C compute plane and must not be
triggered for production computation.

### 2.10 Eventual sealed OOS gate

There is currently **no implemented OOS authorization/gate**. Acquisition and
Stage 4A/B manifest validators reject non-2024 ranges, and Stage 4C has no 2025
path, but these are discovery-stage guards—not authority to locate or inspect
sealed data. Until a separately reviewed gate exists, 2025 data must not be
listed, searched, read, copied, or hashed. Any future gate must pin the frozen
candidate portfolio/methodology first, require explicit authorization, expose
only prescribed outputs, and record access and provenance.

## 3. Research invariants

`Code`, `tests`, and `docs` below distinguish actual enforcement from intent.

| Invariant | Implementation | Tests | Enforcement |
|---|---|---|---|
| A value at research time `t` may depend only on bars with `available_at <= t`; a completed bar is unavailable before close. | `data/models.py`, `data/dataset.py`; resampling propagates maximum component availability. | `test_data_contract.py`, `test_dataset.py`, `test_resampling.py` | Code + tests + docs |
| Prefix invariance: transforming an available prefix equals the corresponding full-history output prefix. | Immutable per-bar observation/session transforms and one-pass benchmark/resample builders. | `test_dukascopy_bi5.py`, `test_resampling.py`, `test_dukascopy_multiday.py`, `test_research.py`, `test_sessions.py`, `test_vwap_benchmark.py`, `test_vwap_m1_robustness.py`, `test_bollinger_benchmark.py` | Code + tests + docs |
| Internal timestamps are timezone-aware UTC; DST is converted only at the session boundary. | `Bar`, `_require_canonical_utc`, IANA `ZoneInfo`. | `test_data_contract.py`, `test_sessions.py`, multiday session tests | Code + tests + docs |
| Sealed 2025 cannot be touched before authorization. | Acquisition bounds and Stage 4A/B manifest-first exact-2024 validation; Stage 4C accepts rows, not data paths. No filesystem-wide access control exists. | Runner/benchmark tests use synthetic 2025 metadata and assert loaders are not called; Stage 4C test checks source has no `2025` path. | Partial code + tests; full prohibition is policy only |
| Raw inputs are immutable and derived datasets reproducible. | Frozen objects, manifest refusal to overwrite, component SHA checks, reconstructed dataset ID. | provider, offline-corpus, monthly/range tests | Code + tests + docs |
| z semantics remain frozen and causal. | Benchmark builders/spec IDs; strict signal functions; Stage 4B constant threshold. | benchmark prefix/boundary/spec tests; `test_stage4b.py` | Code + tests + freeze doc |
| VWAP native, VWAP canonical-M1, and Bollinger are independent robustness families. | Distinct builders, schema/spec IDs, family field; no combination code. | family-specific tests and Stage 4A/B runner tests | Code structure + tests + freeze doc |
| Candidate dedup/re-arm is per full state identity; only valid strict `|z| < 2` re-arms. | `stage4b.deduplicate_states`; global state assembly occurs before sharding. | `test_stage4b.py`, `test_stage4b_runner.py`, equivalence test | Code + tests + docs |
| Future M1 may resolve an outcome only after signal/entry; it cannot decide eligibility except in predeclared entry modes after eligibility is evaluated. | Stage 4A minute offsets; Stage 4B eligibility-before-entry, paths start at +1 minute. | `test_stage4a.py`, `test_stage4b.py` | Code + tests + docs |
| Stage 4B is gross BID/BID. | Canonical BID corpus and gross return engine; reports label it. | Stage 4B engine/runner tests | Code + tests + docs |
| Stage 4C adds costs and never changes Stage 4B gross fields or ambiguity ordering. | Copy-on-transform union; adverse/favorable fields retained. | `test_stage4c.py` preservation/order/scenario tests | Code + tests + preregistration |
| Increasing spread/slippage cannot improve net P/L. | Frozen subtractive transform plus sign-dependent JPY adjustment. | Stage 4C scenario monotonicity test | Code + tests + preregistration |
| Serialization/group ordering is deterministic. | Canonical sorted JSON, explicit event/group/row sorts, SQLite key ordering. | runner determinism and sharded-equivalence tests | Code + tests |
| Reducers accept only complete, nonoverlapping, compatible provenance. | Stage 4B manifest index/group/hash checks; Stage 4C methodology/source-commit/profile/scenario/source-commitment/ownership/count checks. | `test_stage4b_reducer.py`, `test_stage4b_equivalence.py`, extensive reducer cases in `test_stage4c.py` | Code + tests + docs |
| Hashes bind concrete source artifacts and produced outputs. | Raw/component/corpus/dataset hashes, registry artifact IDs, stage audits, shard state/output commitments. | provider, Stage 4A/B reducer, Stage 4C tests | Code + tests + docs |
| Rejected MR regimes cannot be resurrected post hoc. | No executable allow/deny gate; frozen regime document is controlling interpretation. | None. | Documentation only |
| No single 2024 maximum cell becomes production selection; seek breadth/plateaus. | Reports contain no ranking and Stage 4C computes breadth, but a consumer can still cherry-pick. | Reporting assertions cover wording/metrics, not downstream decisions. | Documentation + reporting only |
| Discovery, confirmation, and OOS remain distinct. | Exact-2024 runner boundaries and labels. No sealed-OOS workflow exists. | Boundary tests | Partial code + docs |

### Known implementation/documentation mismatch

Stage 4B's `SEMANTICS["threshold_qualification"]` string says
`abs-z-ge-2`, while both benchmark signal functions and the frozen decision
record implement **strict** `|z| > 2`; equality is neither a signal nor a re-arm.
The executable behavior and frozen document agree, but because the misleading
string is an input to `STAGE4B_METHODOLOGY_ID`, correcting it would change the
identity. Do not silently edit it in a documentation change; resolve it through
an explicit compatibility/version decision and tests.

## 4. Provenance chain

Conceptually, take one 2024 M1 observation:

1. Its raw BI5 bytes have a SHA-256 in daily provenance. Canonicalization facts
   plus that digest produce a daily dataset ID. Ordered daily component records
   produce an assembled dataset ID; the exact requested range, successes,
   absences, components, schema facts, counts, and assembled ID produce the
   `corpus_id`.
2. The registry pins that corpus/dataset pair to a verified workflow run and
   immutable artifact ID/name. Before loading data, Stage 4A/B compare exact
   2024 manifest metadata; offline loading rehashes every declared byte payload
   and reconstructs the assembled ID.
3. Resampled observation availability and `session_spec_id` define its temporal
   context. A benchmark spec's canonical JSON yields `strategy_spec_id`.
   Stage 4A's `FrozenSignal` carries strategy, corpus, and dataset identities;
   the Stage 4A event carries a hash of path semantics. Its audit hashes the raw
   events and compact reports and binds them to source commit/artifact.
4. Stage 4B independently regenerates the same causal features. The candidate
   ID commits to state identity, time, prices, z, and threshold; each trade adds
   filter/entry/exit-grid dimensions. The audit records methodology ID, commit,
   corpus/dataset/artifact, counts, and exact file hashes. A reduced result
   retains each raw shard's name, hashes, and row counts.
5. Stage 4C identifies a trade by all configuration fields plus candidate ID.
   Its source component record includes exact trade-file/callback hash(es),
   corpus/dataset and (for regeneration) registry hash, then hashes that canonical
   record into `source_input_commitment`. The audit adds pinned Stage 4B commit
   and methodology, exact cost-profile hash, scenario matrix, Stage 4C commit,
   row counts, and output hashes. Thus an auditor can identify the bytes and
   semantics behind a reported net cell without treating a workflow run alone
   as the data identity.

## 5. Sharding semantics

### Stage 4B

The stable group is exactly instrument × benchmark family × signal timeframe ×
session × direction × lookback. The runner first builds **all** states and runs
the global re-arm/dedup state machine. Only the resulting candidate events are
grouped. Group keys are JSON-sorted, retained whole, and assigned as contiguous
ranges balanced approximately by candidate count. Therefore no shard can split
a state machine, candidate group, configuration cell, or quantile population.

Each shard reruns source reconstruction and global candidate generation, then
simulates only its selected complete groups. Its manifest commits the full group
universe, owned group range, full/shard candidate counts, source identities,
filter, raw/compact file hashes and row counts. The reducer requires exactly
indices `0..N-1`, identical provenance, identical full group universe, ordered
nonoverlapping exhaustive group ranges, full candidate coverage, and valid
compact hashes plus raw commitments. Since compact statistics are already
group-complete, concatenation and deterministic sorting are semantically equal
to unsharded execution; quantiles are never averaged or merged. The equivalence
test compares raw row sets, compact CSVs, and summaries.

### Stage 4C

Stage 4C sharding already exists (the preregistration's “if added later” is now
historical). Ownership is `first_8_bytes(SHA256(canonical_config_group_json)) mod
shard_count`, where the config group includes the complete Stage 4B configuration
cell. Every independent shard reads the same complete source and records global
read/complete counts but spools only owned complete rows. State JSONL preserves
only the configuration, candidate ID, completeness, and two gross pip bounds.

The reducer requires identical source component commitment, methodology/source
commit, profile hash, scenarios, corpus/dataset, source mode and global counts;
unique complete indices and nonoverlapping declared ownership; valid state
hash/count; deterministic ownership; every declared group represented; and total
owned rows/scenarios equal the global source totals. It then reruns the canonical
cost/report transform over ordered state rows. Tests require byte equality of
the six review outputs with an unsharded run.

Any future Stage 4C sharding revision must retain full-group ownership, global
source equivalence, exact input/state hashes, exhaustive/no-overlap coverage,
16 scenarios per complete row, deterministic reduction, and byte-level
unsharded equivalence. Changing shard count may change operational ownership but
must not change research results.

## 6. Frozen research decisions

Do not reinterpret this map; the detailed controlling record is
`docs/stage4b-2024-cross-asset-freeze.md`.

| Instrument | Primary | Secondary | Cost-sensitive / research-only | Rejected MR |
|---|---|---|---|---|
| USDJPY | New York LONG | London LONG; New York SHORT | — | both Asia; London SHORT |
| AUDUSD | London SHORT | New York SHORT | London LONG research-only | both Asia; New York LONG |
| AUDJPY | New York SHORT | London LONG | New York LONG research-only | both Asia; London SHORT |
| EURUSD | London SHORT | New York SHORT | Asia SHORT cost-sensitive; New York LONG research-only | Asia LONG; London LONG |

M15 remains the lead view; M5/H1 are robustness only. `immediate` remains the
neutral baseline; reclaim entries are predeclared alternatives, not winners.
No TP/SL/time stop, family, lookback, direction, pair, session, or threshold was
selected as a 2024 winner. GBPUSD was pending and is outside this freeze.
Rejected mean-reversion regimes remain rejected; continuation-looking behavior
requires a future, separate preregistration. The eventual portfolio capacity of
roughly 0–2 regimes per session is not a quota, and Asia may be empty.

## 7. Stage 4C semantics

The machine-readable frozen profile is
`configs/stage4c-ftmo-cost-profile-v1.json`; rationale and decision rules are in
`docs/stage4c-preregistration.md`.

* Spread levels are mean, p75, p90, and p95. Round-turn slippage levels are
  0.00, 0.10, 0.25, and 0.50 pips: **16 scenarios** per complete trade.
* Stage 4B is BID/BID, so exactly one spread is subtracted per round trip: LONG
  pays ASK at entry; SHORT buys at ASK on exit. The originating signal session
  selects the profile; null session selects `overall`.
* Commission is USD 2.50/lot/side, USD 5 round turn. It is exactly 0.5 pip for
  EURUSD/AUDUSD. JPY pairs use frozen session-specific
  `0.005 × reference USDJPY` values from the same current spread sample.
* For USDJPY/AUDJPY, after spread/slippage but before commission, positive P/L
  is multiplied by 0.993, negative by 1.007, and zero is unchanged. USD-quote
  pairs receive no adjustment. This is a frozen accounting approximation and a
  current execution-condition overlay, not reconstructed 2024 broker costs.
* The mean-spread/zero-slippage case is an optimistic cost floor—not expected
  live P/L. Swaps and volume bands are explicitly not modeled.
* Break-even total slippage is the round-turn slippage at which the exact
  sign-dependent mean-net transform reaches zero, found by deterministic
  bisection; if net at zero slippage is nonpositive, it is zero. Reported
  break-even **extra** slippage subtracts the scenario's current slippage from
  that total and floors the result at zero.
* Interpretation is by regime breadth: the dependent cells span family,
  lookback, entry, TP, SL, and stop. Advance only coherent net-positive plateaus
  with cost headroom beyond the floor; never select one maximum 2024 cell.

## 8. Operational versus methodological changes

Operational changes must preserve byte/semantic equivalence and audit identity;
methodological changes require explicit versioning, preregistration, and review.

| Operational (no research meaning) | Methodological (changes the experiment) |
|---|---|
| Move SQLite spool or temp location | Change z definition, threshold, equality, or re-arm rule |
| Batch/stream writes without changing rows/order | Add a news/weekday/volatility/OU/consensus filter |
| Add progress/resource logging | Choose another TP, SL, time stop, or entry rule |
| Resume a complete deterministic shard from verified commitments | Exclude a session/regime after seeing costs or OOS |
| Change shard count with exact reduced equivalence | Combine benchmark families or promote M5/H1 to lead |
| Package the same committed outputs elsewhere | Change spread source/statistic, commission, JPY conversion, swaps, or volume bands |
| Improve fail-closed validation without excluding valid frozen inputs | Alter incomplete-path or same-minute ambiguity treatment |

If an “operational” edit changes ordering, floating-point grouping, membership,
included rows, identities, or output values, it is methodological until proven
otherwise by exact equivalence tests.

## 9. Threat model and audit findings

Future developers or agents could invalidate the research by:

* globbing, indexing, searching, hashing, previewing, or auto-discovering sealed
  2025 files before the gate—even if no model consumes them;
* trusting a directory/artifact label without verifying manifest dates and byte
  hashes, or opening bytes before validating metadata;
* using close-time information at open time, treating an incomplete resample as
  complete, forward-filling a gap, or allowing a later observation to rewrite a
  rolling feature;
* changing UTC/session-open/DST/overlap semantics or replacing IANA history with
  fixed UTC offsets;
* silently changing z, strict equality, inactivity, VWAP reset/weighting,
  Bollinger windows, or joining benchmark families;
* deduplicating inside shards, splitting stable groups, re-arming on equality or
  invalid observations, or changing candidate identity inputs;
* letting future path decide baseline eligibility, using signal-bar future
  extrema, assuming intrabar order, or discarding incomplete/ambiguous outcomes;
* treating BID/BID gross results as executable, subtracting two spreads, using
  timestamp-specific historical costs, or overwriting gross fields;
* averaging shard quantiles, accepting incomplete/duplicate shards, mixing
  commits/profiles/corpora/scenarios, or weakening hash verification;
* selecting a best cell, filling a session quota, resurrecting a rejected MR
  regime, or changing the frozen map after seeing costs/OOS;
* confusing a current FTMO overlay with 2024 execution reconstruction, or
  forgetting unmodeled swaps/volume bands;
* using GitHub Actions as production compute, relying on expiring artifacts as
  sole identity, auto-merging a result, or publishing without audit files.

### Architecture ambiguities / controls that remain incomplete

1. **No sealed-OOS access gate exists.** Current exact-2024 validators are good
   defense in depth, but the “do not enumerate/read/hash” rule is not enforceable
   by these Python functions. A future gate and environment-level permissions
   need separate design without touching the sealed data now.
2. **Frozen interpretation is not machine-enforced.** Stage 4C calculates every
   complete Stage 4B regime; rejected/primary labels and non-resurrection are
   reviewer controls. A versioned, hash-bound decision manifest could be added
   before OOS, but must not be invented from results during this documentation
   task.
3. **Anti-cherry-picking is not machine-enforced.** Reports state breadth, but no
   code can prevent a downstream reader choosing a maximum cell. Freeze the
   final selection rule/portfolio in a reviewed artifact before OOS.
4. **Stage 4A local audit ergonomics are coupled to `GITHUB_SHA`.** With a
   registry, the CLI cannot simply derive local HEAD as Stage 4B/C do. This does
   not change methodology but should be normalized in a focused operational PR.
5. **Stage 4B methodology identity contains the `ge` label mismatch** described
   in §3. Resolve only through an explicit compatibility decision.
6. **Registry artifact history is operational provenance, not permanent byte
   storage.** IDs/names may refer to expiring artifacts; corpus/dataset/component
   hashes are the durable content commitments.

No production defect was fixed in this audit; these findings are deliberately
kept separate from the documentation change.

## 10. Developer checklist

- [ ] Have I avoided even enumerating, searching, reading, copying, or hashing
      sealed 2025 data, and validated all paths/metadata before byte access?
- [ ] Is this operational-only? If methodology can change, stop and create a
      versioned preregistration/decision record before implementation or results.
- [ ] Do UTC, availability, prefix invariance, gaps, DST, and overlap semantics
      remain unchanged and tested?
- [ ] Are benchmark families, lead/robustness timeframes, strict z, re-arm,
      entries, exits, ambiguity, and gross BID/BID semantics still frozen?
- [ ] Are corpus/dataset/registry, strategy/methodology, commit, cost profile,
      scenario, input, shard, and output identities present and verified?
- [ ] Do malformed, missing, duplicate, nonfinite, out-of-range, or incompatible
      inputs fail before data/results are consumed?
- [ ] Are output order and unsharded/sharded reduction deterministic, with exact
      equivalence tests where relevant?
- [ ] Does reporting preserve gross fields, label modeled omissions, emphasize
      breadth, and avoid automatic winner selection?
- [ ] Did I run focused tests/static checks without triggering GitHub Actions or
      unrelated production computation?
- [ ] Has a human reviewed the change? Do not merge automatically.
