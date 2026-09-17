# Frozen-2024 Stage 4C zero-cost diagnostic

This path answers a narrow diagnostic question: whether authenticated, complete
rows from the **unchanged frozen Stage 4B methodology** contain descriptive raw
mean-reversion edge before execution costs. It is separate from Stage 4C-A and
does not claim executable or production net returns.

The identity transform is exact: spread, commission, and slippage are zero;
currency-conversion adjustment is disabled; and both zero-cost ordering values
equal their corresponding Stage 4B gross-pip values. The runner does not tune,
rank, or select instruments or configurations.

## Required local input

Each explicit corpus directory must be a completed Stage 4B output directory
containing `trades.jsonl` and `execution-audit.json`. The audit must authenticate
the trade-file hash, the frozen Stage 4B methodology, instrument, corpus and
assembled-dataset identities, and an entirely 2024 requested date range. The
path itself must have an explicit `2024` or `_2024_` component. Validation of
the requested year and path is lexical and happens before any filesystem read.

## One explicit pair

```bash
cd /home/redsq/projects/Multi-Asset-Mean-Reversion-Lab
git fetch origin
git switch codex/stage4c-zero-cost-new-pairs-2024
git pull

tools/run_stage4c_zero_cost_local.sh \
  --instrument EURCAD \
  --year 2024 \
  --corpus /PATH/TO/EURCAD_2024_CORPUS \
  --output /PATH/TO/EURCAD_2024_ZERO_COST_OUTPUT
```

Replace the example ticker and both `/PATH/TO/...` placeholders. The ticker is
an explicit uppercase six-letter FX symbol; no implicit pair allowlist or corpus
discovery occurs.

## Explicit batch manifest

Create a CSV containing only the pairs intended for this run:

```csv
instrument,corpus_path,year
EURCAD,/PATH/TO/EURCAD_2024_CORPUS,2024
GBPJPY,/PATH/TO/GBPJPY_2024_CORPUS,2024
```

Then run:

```bash
tools/run_stage4c_zero_cost_local.sh \
  --manifest /PATH/TO/EXPLICIT_NEW_PAIRS_2024.csv \
  --output /PATH/TO/NEW_PAIRS_2024_ZERO_COST_OUTPUT
```

Only manifest rows are processed, sequentially. The runner never walks a parent
directory. Per-instrument outputs include the compact trade matrix, regime
breadth, summary, report, and execution audit. Batch output additionally includes
the deterministic cross-asset CSV and Markdown report. Production evidence does
not exist until the user executes this command against the real local corpora.
