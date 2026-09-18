# Stage 4C zero-cost diagnostic for four authenticated 2024 pairs

This discovery-only diagnostic runs the unchanged frozen Stage 4B gross trade
engine for `EURGBP,NZDUSD,USDCAD,USDCHF`. It compares the like-for-like frozen
MR grid without eligibility filtering to `frozen-ou-bidirectional-v1`. It does
not optimize, rank instruments, select a best cell, or provide broker-realistic
expectancy or out-of-sample confirmation.

Zero cost means that both adverse-first and favorable-first returns are copied
directly from the corresponding Stage 4B gross return. Spread, commission, and
slippage are zero and currency-conversion adjustment is disabled. The frozen
Stage 4C-A transform is not called and remains unchanged.

Only the explicit entries in `configs/fx-universe-2024-registry-v1.json` are
accepted. The CLI requires the exact instrument set and validates the registry
schema, exact 2024 bounds, authentication state, identities, and instrument
orientation before passing an explicit registry corpus path to Stage 4B. It
does not search for substitute corpora. Non-2024 provenance fails before any
corpus path is inspected.

## Local WSL production run

```bash
cd ~/projects/Multi-Asset-Mean-Reversion-Lab
git fetch origin
git switch codex/stage4c-zero-cost-new4-2024
git pull --ff-only

uv run mr-lab-stage4c-zero-cost \
  --registry configs/fx-universe-2024-registry-v1.json \
  --instruments EURGBP,NZDUSD,USDCAD,USDCHF \
  --output-dir results/stage4c-zero-cost-new4-2024
```

The compact final files are `stage4c-zero-cost-cross-asset.csv`,
`stage4c-zero-cost-cross-asset.json`, `stage4c-zero-cost-report.md`, and
`execution-audit.json`. The hidden `.stage4b` directory contains the ordinary
compact Stage 4B working reports; trade JSONL persistence is disabled.

No empirical new-pair result is bundled or claimed. The authenticated local
2024 corpora must be used to produce empirical output.
