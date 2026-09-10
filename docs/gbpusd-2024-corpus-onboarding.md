# GBPUSD 2024 corpus onboarding

## Promotion record

GBPUSD is promoted in the frozen registry as a verified
`local-checkpointed` corpus. All twelve monthly checkpoints were acquired with
the repository's authenticated, resumable Dukascopy pipeline at commit
`94a43c36e1a5fa1b3b9e98890dc2d62a156592c4`. Full-year assembly and the
independent `verify-year` pass agreed on both frozen identities recorded in the
registry.

This source mode deliberately has null `source_workflow_run_id` and
`source_artifact_id` values: the corpus did not originate from a GitHub Actions
artifact, and workflow provenance must not be fabricated. Its logical source
name remains `dukascopy-GBPUSD-m1-bid-2024-full-year`.

## Reverification contract

The corpus can be independently checked with the provider-agnostic frozen-year
verifier at its operator-managed location:

```bash
uv run python -m mr_lab.providers.dukascopy_monthly verify-year \
  --year 2024 --instrument GBPUSD --corpus-dir <gbpusd-corpus-dir>
```

The verifier reconstructs the dataset from immutable raw components, checks
all component hashes, checks that successful and confirmed-absent dates exactly
partition the requested calendar, authenticates the corpus identity, and
requires the verified GBPUSD Dukascopy/M1/BID decoding contract. A provider 404
is the only acquisition outcome represented as confirmed absence; transport,
parsing, assembly, and verification failures stop the pipeline.

The operator-managed corpus itself is not repository content and must not be
committed. Downstream Stage 4 provenance carries the explicit source mode,
acquisition commit, corpus identity, and assembled dataset identity while
preserving the null GitHub identifiers.
