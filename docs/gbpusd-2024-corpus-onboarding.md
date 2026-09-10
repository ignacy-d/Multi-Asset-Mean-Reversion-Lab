# GBPUSD 2024 corpus onboarding

This change is **Phase A** only. No corpus was acquired while preparing it and
the GBPUSD registry entry remains `pending-acquisition`; its four provenance
pins must remain null until the following workflow succeeds. The generic
checkpointed acquisition workflow is the existing publication path, now with
an additional fail-closed verification immediately before upload.

## Acquire and verify the real artifact

Dispatch the workflow from the commit intended to identify the corpus:

```bash
gh workflow run acquire-historical-sample.yml \
  --ref <committed-branch-or-sha> -f instrument=GBPUSD
gh run list --workflow acquire-historical-sample.yml \
  --branch <committed-branch> --event workflow_dispatch
gh run watch <source_workflow_run_id> --exit-status
```

The successful run must contain exactly one non-expired artifact named
`dukascopy-GBPUSD-m1-bid-2024-full-year`. Resolve and download it by its real
ID rather than by guessing a pin:

```bash
REPO=ignacy-d/Multi-Asset-Mean-Reversion-Lab
RUN=<source_workflow_run_id>
NAME=dukascopy-GBPUSD-m1-bid-2024-full-year
gh api "repos/$REPO/actions/runs/$RUN/artifacts?per_page=100" > artifacts.json
jq --arg name "$NAME" \
  '[.artifacts[] | select(.name == $name and .expired == false)] | if length == 1 then .[0] else error("expected exactly one live artifact") end' \
  artifacts.json > artifact.json
ARTIFACT=$(jq -er '.id' artifact.json)
test "$(jq -r '.workflow_run.id' artifact.json)" = "$RUN"
gh api "repos/$REPO/actions/artifacts/$ARTIFACT/zip" > gbpusd-corpus.zip
rm -rf gbpusd-corpus && mkdir gbpusd-corpus
unzip -q gbpusd-corpus.zip -d gbpusd-corpus
uv run python -m mr_lab.providers.dukascopy_monthly verify-year \
  --year 2024 --instrument GBPUSD --corpus-dir gbpusd-corpus
```

The verifier reconstructs the dataset from immutable raw components, checks
all component hashes, checks that successful and confirmed-absent dates exactly
partition the requested 2024 calendar, authenticates the corpus identity, and
requires the verified GBPUSD Dukascopy/M1/BID decoding contract. A provider 404
is the only acquisition outcome represented as confirmed absence; transport,
parsing, assembly, and verification failures stop the workflow.

## Stage the Phase B registry-only change

Copy the printed `corpus_id` and `assembled_dataset_id`, and the real `RUN` and
`ARTIFACT` values above, into the GBPUSD entry in
`configs/stage4a-2024-corpus-registry.json`. Change `verification_status` to
`verified`, without changing its instrument, dates, or artifact name. Then run:

```bash
uv run pytest tests/test_dukascopy_monthly.py tests/test_stage4a_runner.py \
  tests/test_stage4b_runner.py tests/test_ornstein_uhlenbeck.py
uv run pytest
uv run ruff check .
```

That registry promotion must be a separate reviewable Phase B commit/PR. Never
promote from a failed run, an expired artifact, a differently named artifact,
or identity values that were not printed by the verifier for those downloaded
bytes.
