# Frozen MR + OU nine-pair replay

Status: **FROZEN REPLICATION CONTRACT**

This study replays, without tuning, the current frozen Stage 4B engine and its
opt-in `frozen-ou-crossasset-v1` gate over the exact instruments and explicit
2024 corpus paths in `configs/fx-universe-2024-registry-v1.json`. It does not
inspect or discover any other corpus. Provider-originated closed-market flat M1
padding is retained because neither frozen method contains a market-activity or
weekend filter. A future filter would require a separate preregistration.

## Recovered frozen specifications

The MR engine uses M5, M15, and H1 completed bars; lookbacks 20 and 40; native
session VWAP, canonical-M1 VWAP, and Bollinger families; strict signal selection
at `|z| > 2.0` (equality does not signal or re-arm); immediate,
M1-reclaim-p0, and extension-25%-then-reclaim-p0 entries; TP fractions 0.25,
0.50, 0.75, and 1.00; stop extensions 0.25, 0.50, 1.00, and no price stop; and
30, 60, and 120 minute time stops. It retains the frozen completed-M1 entry,
joint first-exit, adverse-first same-minute, and certain MAE/MFE semantics.

The causal OU gate applies only to M15 London SHORT native/canonical-M1 VWAP
signals with lookback 20 or 40. It uses exactly 128 valid transitions, requires
directional score strictly greater than 1.5, and requires a finite half-life no
greater than 120 minutes. Missing, invalid, non-finite, mismatched, and
out-of-scope states fail closed. Its downstream grid is immediate entry, TP
0.75 or 1.0, SL 0.25 or 0.5, and time stop 60 or 120 minutes.

## Output and cost interpretation

`comparison.json` keeps the entire Stage 4B MR grid as the descriptive
`mr_full_grid` view. Its like-for-like comparison instead derives an
`mr_comparable_baseline` from ungated MR rows using exactly the OU signal scope
and downstream grid stated above. OU retention is therefore
`N(MR+OU) / N(mr_comparable_baseline)`, never `N(MR+OU) / N(mr_full_grid)`.

Gross/zero-cost metrics are reported for the full grid, comparable baseline,
and MR+OU. The realistic-cost headline is the already-frozen mean-spread,
zero-slippage cost floor and remains distinct from gross. Comparable-baseline
and MR+OU net results are emitted only when the authenticated Stage 4C profile
supplies all required spread, commission, and conversion inputs; otherwise the
net result is null and the status is
`BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE`.

The repository contains no machine-readable frozen numeric Stage 4B result
artifact against which to compare the historical five. The replay therefore
marks those checks `BLOCKED_MISSING_FROZEN_NUMERIC_REFERENCE` rather than
inventing a baseline. This does not prevent deterministic reproduction; it
prevents an unsupported claim of numeric equivalence.

## External execution

From the repository root on WSL:

```bash
set -o pipefail; mkdir -p results/replay-mr-ou-costs-9pair && \
  uv run mr-lab-replay-mr-ou-costs \
    --registry configs/fx-universe-2024-registry-v1.json \
    --cost-profile configs/stage4c-ftmo-cost-profile-v1.json \
    --output-dir results/replay-mr-ou-costs-9pair/output \
  2>&1 | tee results/replay-mr-ou-costs-9pair/replay.log
```

The runner resumes a completed matching instrument/variant, rejects incomplete
or identity-mismatched artifacts, validates the manifest hash and corpus IDs,
and never searches for replacement data.

## Streaming and disk behavior

Replay aggregation consumes Stage 4B's immutable `trade_row_consumer` stream.
It does not load `trades.jsonl` into Python dictionaries. Exact medians retain
only little-endian float64 return arrays (eight bytes per complete observation);
counts, sums, win/loss totals, MFE, and MAE remain scalar accumulators. New replay
variants explicitly use compact Stage 4B mode, so `trades.jsonl` is an audited
empty placeholder rather than a redundant multi-gigabyte row artifact. Default
Stage 4B callers still persist every trade row unchanged.

The already-completed `AUDJPY/mr` artifact from
`4ae3b537fa533917dc4f400d85e6d9909b81e0f7` is reusable when its registry,
corpus, dataset, Stage 4B methodology, OU specification, cost profile, variant,
filter, and complete output-audit identities match. On first resume its existing
trade file is streamed once, with progress and SHA-256 verification, to create
the compact aggregate. It is not deleted or recomputed. A wrapper-only source
revision difference is recorded but is not treated as a methodology mismatch.
Any scientific identity or output-contract mismatch fails closed.

For the observed 5,947,644-row AUDJPY full grid, its primary exact-median array
is about 47.6 MB, plus much smaller comparable/net arrays and the ordinary
Stage 4B compact reports. Subsequent variants do not add another multi-gigabyte
trade JSONL, so the replay no longer scales toward roughly 100 GB of redundant
trade-row storage. The existing legacy 9.8 GB artifact remains in place unless a
human removes it after review.
