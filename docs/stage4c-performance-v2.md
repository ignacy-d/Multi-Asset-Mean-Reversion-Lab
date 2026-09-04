# Stage 4C performance v2

This refactor changes only the disk-backed grouping data path. The cost model,
scenario expansion, aggregation, reporting, and provenance commitment format are
unchanged.

## Root causes and selected changes

The existing-raw CLI first hashed every source file and then reopened it for
JSONL ingestion. It also inserted every accepted row with a separate Python
`execute` call into a `WITHOUT ROWID` table physically ordered by trade identity.
The consumer subsequently requested a different order, by configuration group
and then identity, requiring SQLite to sort the spool.

The optimized path:

1. updates SHA-256 from each exact binary JSONL line immediately before parsing
   it, reducing the source from two complete passes to one;
2. bulk-loads 10,000 rows per `executemany` call inside an explicit transaction;
3. physically orders the `WITHOUT ROWID` table by `(group_key, identity)`, exactly
   the order consumed by reporting, so the ordered scan no longer needs a sort.

Trade identity already contains every group-key field before the candidate event
identifier. Therefore two equal trade identities necessarily have equal group
keys, and the composite primary key preserves the prior duplicate rejection
semantics. Hashes are installed into the same provenance component only after
successful exhaustion of every source stream; partial or malformed input cannot
produce an audit.

## Synthetic benchmark

`uv run python tools/benchmark_stage4c_spool.py` generates 300,000 shuffled rows
across 400 deterministic groups (113.5 MiB source) on local `/tmp`, excluding
fixture generation from the timed interval. It measures source hashing,
ingestion, and the complete ordered scan.

| implementation | wall time | throughput | spool size | source passes |
| --- | ---: | ---: | ---: | ---: |
| baseline | 6.831 s | 43,916 rows/s | 124.3 MiB | 2 |
| optimized | 5.325 s | 56,339 rows/s | 124.3 MiB | 1 |

This run improved wall time by 22.0% and throughput by 28.3% on a warm, fast
cloud filesystem. Process peak RSS was approximately 33.9 MiB (the measurement
includes fixture generation and both sequential cases). The source-pass saving
should matter more for a source on slower storage. Exact timings remain sensitive
to the filesystem and SQLite build.

A heap table followed by separate unique-identity and ordered-scan indexes was
rejected: it writes and retains multiple B-trees and increases spool size. An
append spool plus external sort was also rejected for this incremental change
because it would duplicate SQLite's sorting machinery and introduce a larger new
failure/cleanup surface. The selected clustered key removes the existing sort
without an additional index.

## Real WSL validation

Run the existing guarded local wrapper with the spool root on native WSL storage:

```bash
STAGE4C_SPOOL_ROOT="$HOME/.cache/mr-lab/stage4c-spool" \
  bash tools/run_stage4c_local.sh AUDUSD
```

Production data must remain outside CI. Compare all final artifact hashes and
the source provenance fields against an equivalent baseline run before accepting
the operational result.
