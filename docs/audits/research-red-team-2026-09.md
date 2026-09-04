# Research red-team audit — September 2026

Status: **adversarial code and methodology audit, not a strategy revision**

## Executive conclusion

The research is **not yet trustworthy enough to begin Stage 4C production**.
The principal blocker is a confirmed gap-compression bug in the trailing-volatility
denominator shared by both VWAP families.  The implementation treats observations
on opposite sides of a missing interval (including routine weekend closures) as
consecutive.  That contradicts the documented consecutive-return methodology and
can change z-scores, threshold crossings, re-arm state, and all downstream VWAP
trades.  Existing 2024 Stage 4B VWAP and canonical-M1-VWAP results therefore require
recomputation after a versioned fix; Bollinger results are not implicated by this
finding.

Stage 4C also accepts rows whose instrument disagrees with its declared source
instrument, and the existing-raw route hashes whatever files it is handed without
verifying those hashes against the approved Stage 4B audit.  These are production
provenance blockers even though they do not alter the mathematical cost transform.

No evidence of sealed-OOS access was produced.  This audit did not access,
enumerate, search for, open, read, or hash any actual sealed data.  Confidence is
**moderate-high for the reviewed code paths**: all 370 existing tests pass in the
project environment and focused source inspection covered every requested layer,
but no large 2024 artifact was opened and empirical effect sizes were deliberately
not re-estimated.

### Decision summary

| Question | Conclusion |
| --- | --- |
| Proceed to Stage 4C production? | **No.** Fix and recompute the VWAP Stage 4B inputs; close Stage 4C source-binding gaps first. |
| Frozen Stage 4B invalidated? | **Partly.** All conclusions relying on either VWAP family need recomputation. Bollinger-only evidence remains technically unaffected. The cross-family frozen interpretation cannot be relied on unchanged. |
| Issues that can wait | Methodology-label inconsistency, reducer raw-file trust-boundary wording, and stronger automated enforcement of the frozen regime map. |
| Sealed OOS contaminated? | **No evidence of contamination; sealed resources were not touched by this audit.** |

## Scope and method

The audit traced provider decoding and acquisition, immutable bar contracts,
multi-day assembly, manifests and registry selection, resampling, sessions, VWAP,
canonical-M1 VWAP, Bollinger, Stage 4A, Stage 4B state/candidates/entry/path/exit,
both shard reducers, Stage 4C source modes/spooling/costs/reporting, and the two
freeze documents.  It reviewed tests rather than production result artifacts.

Commands used for validation:

```text
uv run pytest -q
# 370 passed in 23.72s
```

The initial bare `pytest -q` invocation failed collection because the src-layout
package was not installed on that interpreter; this was an environment invocation
issue, not a research-test failure.  No workflow was triggered and no production
compute was run.

## Findings

### HIGH — Confirmed bug: VWAP volatility silently compresses missing time

1. **Title:** Missing intervals are treated as consecutive returns in both VWAP z-score families.
2. **Affected stage/file/function:** Stage 2B through Stage 4B;
   `src/mr_lab/vwap_benchmark.py::build_vwap_features`, reused as the native
   volatility denominator by
   `src/mr_lab/vwap_m1_robustness.py::build_canonical_m1_vwap_features`.
3. **Mechanism:** The builder advances `previous` for each supplied observation and
   calculates `current / previous - 1` whenever both are active.  It never checks
   that the two bars are exactly one timeframe apart.  Resampling intentionally
   omits incomplete and wholly absent windows, so Friday-to-Sunday/Monday and any
   feed outage become one synthetic “consecutive bar” return.  The rolling deque is
   not cleared at the gap.  Bollinger explicitly performs the missing check and
   clears its window, demonstrating the intended invariant.
4. **Bias:** A multi-hour/weekend price move is divided as though it were one M5,
   M15, or H1 return.  It can inflate the rolling standard deviation for the next
   20/40 observations, change z qualification, and change the below-threshold
   observations that re-arm candidate generation.  The sign of bias is not fixed,
   but selection and reported robustness can change.
5. **Frozen Stage 4B:** **Yes.** Native VWAP and canonical-M1 VWAP both use this
   denominator. Cross-family claims and regime labels relying on those families
   are no longer supported without recomputation.
6. **Stage 4C:** **Yes, transitively.** Stage 4C would faithfully cost-transform an
   invalidated source trade population.
7. **Sealed OOS:** Remains uncontaminated; the defect was demonstrated from code
   semantics and synthetic-test structure only.
8. **Minimal evidence:** Supply two active same-timeframe observations separated by
   more than one bar duration, followed by `lookback - 1` contiguous active bars.
   `build_vwap_features` emits a volatility value incorporating the cross-gap
   return.  The analogous Bollinger builder clears its window at the same gap.
9. **Fix:** Version the VWAP normalization methodology; before appending a return,
   require exact adjacency (preferably both interval adjacency and compatible
   availability order), otherwise clear the return deque and append an unavailable
   return. Add synthetic gap, weekend, prefix-invariance, and native/canonical
   parity regression tests.
10. **Recompute:** **Yes.** Recompute Stage 4A/4B for both VWAP families on every
    2024 instrument, then revisit the frozen cross-asset interpretation before any
    production Stage 4C run. Bollinger rows need not be recomputed for this bug.

Classification: **confirmed bug** (methodological, not performance-only).

### HIGH — Confirmed validation bug: Stage 4C can silently mix instruments

1. **Title:** Declared source instrument is not enforced against individual trade rows.
2. **Affected stage/file/function:** Stage 4C source/spool path;
   `src/mr_lab/stage4c_runner.py::run_rows`, `_spool_rows`, and
   `src/mr_lab/stage4c.py::validate_trade`.
3. **Mechanism:** `validate_trade` checks only that each row names *an* allowed
   instrument. Neither it nor `_spool_rows` compares the row instrument with
   `source_audit["instrument"]`. Because instrument is part of the configuration
   key, mixed rows are accepted as separate groups and receive their own costs.
4. **Bias:** A nominal single-instrument report can contain another pair, corrupt
   regime breadth, counts, source attribution, and any portfolio interpretation.
5. **Frozen Stage 4B:** No direct effect on Stage 4B construction.
6. **Stage 4C:** **Direct blocker.** It disproves the required claim that Stage 4C
   cannot silently mix instruments.
7. **Sealed OOS:** Remains uncontaminated.
8. **Minimal evidence:** Call `run_rows` with an audit declaring `EURUSD` and one
   otherwise-valid `AUDUSD` row. The run succeeds and produces AUDUSD output under
   an EURUSD execution audit. Existing unit fixtures even omit the audit instrument
   in several direct calls, which the API accepts.
9. **Fix:** Require a non-empty audit instrument in all source modes and reject
   every row that differs before it enters SQLite. Repeat the assertion while
   reducing shard state. Add one-row mismatch and mixed-input regression tests.
10. **Recompute:** Stage 4B does not require recomputation. Any Stage 4C artifact
    produced before the fix must be checked or regenerated.

Classification: **confirmed bug** (provenance and aggregation validation).

### MEDIUM — Confirmed validation gap: existing-raw commitment does not authenticate the approved Stage 4B artifact

1. **Title:** The source commitment binds supplied bytes, but not the approved checkpoint.
2. **Affected stage/file/function:** Stage 4C existing-raw CLI route;
   `src/mr_lab/stage4c_runner.py::main`, `_source_input_identity`, and `run_rows`.
3. **Mechanism:** The CLI reads a Stage 4B audit, then overwrites/adds
   `source_trade_sha256` with hashes of the files supplied on the command line. It
   does not compare those hashes to `execution-audit.json` output commitments or
   the raw-shard hashes recorded by the Stage 4B reduction audit. A substituted or
   stale trade file therefore receives a perfectly valid new commitment. The
   methodology ID is merely copied from the supplied JSON audit and trade rows do
   not carry corpus/methodology identity for cross-checking.
4. **Bias:** Selective, stale, or unrelated trades can masquerade as the reviewed
   Stage 4B source while retaining a cryptographically correct hash of the wrong
   bytes.
5. **Frozen Stage 4B:** Does not alter the historical computation, but weakens the
   claim that Stage 4C consumes those frozen outputs.
6. **Stage 4C:** **Yes.** Production provenance is not fail-closed.
7. **Sealed OOS:** Remains uncontaminated.
8. **Minimal evidence:** Give the CLI an internally plausible approved audit and a
   different valid trade JSONL. Its hash is computed and accepted rather than
   compared with the audit's declared raw commitment.
9. **Fix:** Define a canonical Stage 4B source bundle manifest. Require exact local
   file-name/hash matches, verify the audit itself is the approved checkpoint, and
   bind instrument, corpus, dataset, methodology, shard universe, row counts, and
   ordered file hashes into one canonical commitment. Do not accept caller-created
   hashes as evidence of approval.
10. **Recompute:** No Stage 4B recompute solely for this issue. Regenerate or
    re-attest affected Stage 4C artifacts from authenticated inputs.

Classification: **confirmed validation gap / plausible provenance-substitution risk**.

### MEDIUM — Documentation/enforcement weakness: the frozen regime map is not represented in Stage 4C output

1. **Title:** Rejected and advancing regimes are indistinguishable to machine consumers.
2. **Affected stage/file/function:** `docs/stage4b-2024-cross-asset-freeze.md` and
   `src/mr_lab/stage4c_runner.py::_regime_rows`/report generation.
3. **Mechanism:** Stage 4C correctly evaluates the complete frozen grid, but neither
   matrix nor breadth rows carry frozen status (`primary`, `secondary`,
   `cost-sensitive`, `research-only`, `rejected`). The generated report is generic
   and cannot prevent a downstream consumer from ranking/promoting a rejected
   regime.
4. **Bias:** It makes post-hoc resurrection or maximum-cell selection easy, despite
   correct warnings in prose.
5. **Frozen Stage 4B:** Interpretation is documented but not mechanically bound.
6. **Stage 4C:** Yes, at interpretation/reporting rather than arithmetic level.
7. **Sealed OOS:** Remains uncontaminated.
8. **Minimal evidence:** `_regime_rows` groups all instruments/timeframes/sessions/
   directions/entries and returns only counts/fractions; the report contains no
   frozen-map table or status field.
9. **Fix:** Encode the already-frozen map in a versioned, hashed, machine-readable
   decision record and join it to reporting without filtering or tuning. Reject
   unknown mappings and label research-only/rejected rows conspicuously.
10. **Recompute:** No trade recomputation; regenerate reporting after adding the
    frozen decision-record identity.

Classification: **documentation weakness with selection-leakage risk**.

### LOW — Documentation weakness: threshold semantics disagree inside the methodology identity

1. **Title:** Stage 4B methodology label says `abs-z-ge-2` although implemented and
   frozen qualification is strict `|z| > 2`.
2. **Affected stage/file/function:** `src/mr_lab/stage4b.py::SEMANTICS`,
   `deduplicate_states`, benchmark `signal_direction` functions, and the freeze.
3. **Mechanism:** Signal builders use strict inequalities, consistent with the
   freeze. The hashed semantics string says greater-than-or-equal, while
   `deduplicate_states` accepts a caller-marked qualifying state at equality.
4. **Bias:** Current runner construction does not qualify equality, so practical
   bias is unlikely. Alternate callers could create equality events and the
   methodology ID inaccurately describes the run.
5. **Frozen Stage 4B:** No demonstrated event change in the official runner.
6. **Stage 4C:** Only provenance/interpretability.
7. **Sealed OOS:** Remains uncontaminated.
8. **Minimal evidence:** A `SignalState(qualifying=True, z=2.0)` passes
   `deduplicate_states`, while benchmark direction functions return no signal at
   equality and the freeze says equality neither signals nor re-arms.
9. **Fix:** Fail qualifying equality states and create a versioned methodology ID
   whose canonical description says strict greater-than. Preserve old IDs in old
   audits; never silently mutate their meaning.
10. **Recompute:** Probably not, after confirming no official equality event;
    provenance should be versioned and re-attested.

Classification: **documentation weakness / defensive-validation bug**.

### LOW — Design tradeoff: Stage 4B reduction attests raw shards without reading them

1. **Title:** Compact reducer validates manifest commitments but does not locally
   verify raw shard bytes.
2. **Affected stage/file/function:** `src/mr_lab/stage4b_reducer.py::reduce_shards`.
3. **Mechanism:** Compact files are rehashed; raw files need only have manifest hash
   and row-count entries. The audit explicitly states that the reducer did not
   download or rehash raw files.
4. **Bias:** Not a semantic bug in compact reduction, but a user may overread the
   combined audit as proof that raw artifacts were present and matched. A later
   consumer must independently authenticate them.
5. **Frozen Stage 4B:** Compact equivalence remains supported by an end-to-end
   synthetic test; raw-byte availability is a separate trust boundary.
6. **Stage 4C:** Existing-raw ingestion must close the authentication gap described
   above.
7. **Sealed OOS:** Remains uncontaminated.
8. **Minimal evidence:** The reducer checks manifest entries for raw names but calls
   `_sha256_file` only for compact shard outputs.
9. **Fix:** Keep lightweight reduction if operationally necessary, but name the
   trust boundary prominently and require downstream retrieval to verify every raw
   byte/hash/row count against the reduction audit.
10. **Recompute:** No, unless an artifact fails later verification.

Classification: **design tradeoff and documentation risk**.

## Adversarial check matrix

| Check | Result |
| --- | --- |
| PIT / bar completion | Bar contract requires `available_at >= close_time`; resampling exposes a bar no earlier than its window close and latest component availability. Feature builders consume completed observations. **No future leak found.** |
| Prefix invariance | Existing tests cover resampling, VWAP, Bollinger, and Stage 4A/4B paths. The gap bug does not violate ordinary prefix invariance; it violates elapsed-time/consecutive-window semantics, showing why both invariants are needed. |
| Sessions / DST / timezone | Canonical inputs require UTC; conversion uses IANA zones and half-open local windows. **No defect found.** Session assignment at bar open is explicit and longstanding. |
| VWAP | Session instance and current completed bar are PIT. **Gap-contaminated volatility confirmed.** |
| Canonical-M1 VWAP | Cursor admits only M1 observations available by research time and matches session instance. **No future admission found**, but it inherits the native denominator gap defect. |
| Bollinger | Clears its window on missing intervals and requires active consecutive closes. **No defect found.** |
| z qualification / re-arm | State grouping is stable and below-threshold states re-arm only at strict `< 2`; equality is neutral. Runner qualification is strict `> 2`. Label/alternate-caller inconsistency noted above. |
| Duplicate candidates | Deterministic group-local armed state and event IDs; no duplicate path found. |
| Stable Stage 4B partitioning | Candidates are fully constructed/deduplicated before whole stable groups are partitioned. Synthetic end-to-end equivalence covers multiple families, contexts, directions, lookbacks, and a re-arming group. **Supported.** |
| M1 entry/path use | Entry searches completed closes strictly after signal availability; exits use post-entry completed M1 bars. Missing exact time-stop bars lead to incomplete results rather than a later substitute. **Conservative; no leak found.** |
| TP/SL/time stop / same-bar ambiguity | Joint barrier check is adverse-first headline with favorable bound. Exact close handles time stop. Exit-bar extrema are bounds, not falsely certain excursion. **No error found.** |
| Direction symmetry / pips | Direction is applied consistently to prices and barriers; pip size derives from verified instrument precision. Synthetic tests cover both directions and JPY precision. **No defect found.** |
| Stage 4B reducer provenance | Rejects incomplete/duplicate shard indexes, mismatched corpus/dataset/methodology/filter identity, unordered/overlapping groups, compact hash/count mismatch. Raw trust boundary noted above. |
| Stage 4C source SHA | Hash binds bytes supplied to the run, but not necessarily the approved bytes. **Checkpoint-authentication gap confirmed.** |
| Stage 4C instrument isolation | **Disproved; mixed instruments are accepted.** |
| Spread / slippage monotonicity | Transform subtracts each once before a monotone sign-dependent JPY adjustment. Tests order by numeric spread (correctly, since empirical mean need not precede p75) and prove non-improvement. |
| Commission / JPY ordering | One round-turn spread, one round-turn commission; JPY adjustment follows execution P/L after spread/slippage and precedes commission, matching preregistration. **No defect found.** |
| Gross immutability | Dict union returns a new row and tests assert caller row/gross/ambiguity fields are preserved. |
| Stage 4C spool | SQLite primary key rejects duplicate configuration + candidate identity; deterministic group/identity ordering removes input-order dependence. External/default spool equality is tested. |
| Stage 4C sharding/reducer | Stable hash ownership, complete-row coverage, 16-scenario count, state hash/count, group ownership, methodology/source/profile/matrix identity, and instrument checks are enforced at reduction. Synthetic byte-equivalence is tested. Upstream per-row instrument check is still missing. |
| Dependent configurations | Stage 4B caveat and Stage 4C breadth call cells dependent; no code treats cells as independent trades in reported breadth. Statistical dependence remains an interpretation obligation. |
| Winner/cherry-picking controls | Reports do not rank a maximum cell and state the plateau rule, but frozen regime status is not machine-bound. |
| Data gaps | Resampling reports partial windows and does not synthesize absent windows; forward outcomes require exact clocks; M1 exact time stops fail incomplete. **VWAP normalization is the exception and is the central finding.** |
| Sealed gate | Frozen-year manifest/registry validation rejects out-of-bound declared dates before corpus loading. This is good fail-closed behavior. No actual sealed resource was touched. |

## Required work before production Stage 4C

1. Version and fix VWAP exact-adjacency semantics; add regression tests.
2. Recompute all 2024 Stage 4A/4B native and canonical-M1 VWAP evidence.
3. Re-review (do not optimize) the frozen interpretation using the corrected full
   evidence and issue an explicit versioned amendment or confirmation.
4. Enforce row-to-audit instrument equality in Stage 4C and its shard route.
5. Authenticate existing raw files against an approved Stage 4B bundle commitment,
   rather than merely hashing caller-selected files.
6. Run the complete suite and synthetic sharded/unsharded equivalence tests before
   any production cost overlay.

## Follow-ups that can wait

- Encode the already-frozen regime map as a hashed decision record and label all
  Stage 4C regime rows.
- Correct/version the threshold semantics label and reject equality from direct
  `deduplicate_states` callers.
- Clarify the Stage 4B raw-shard trust boundary in operational documentation.
- Add property-based permutations for floating-point/input-order determinism. The
  current SQLite canonical ordering is strong, but exact floating aggregation is
  only as reproducible as the JSON numeric representation and Python runtime.

No strategy parameter, threshold, candidate regime, holding period, or cost was
changed or proposed on the basis of observed performance.
