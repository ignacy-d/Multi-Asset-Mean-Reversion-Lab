#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="results/ou-bidirectional-2024-v1"
if [[ -e "$OUT/output/comparison.json" || -e "$OUT/work" ]]; then
  echo "Refusing to overwrite prior empirical artifacts in $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
exec uv run mr-lab-replay-ou-bidirectional \
  --registry configs/fx-universe-2024-registry-v1.json \
  --cost-profile configs/stage4c-ftmo-cost-profile-v1.json \
  --output-dir "$OUT" 2>&1 | tee "$OUT/replay.log"
