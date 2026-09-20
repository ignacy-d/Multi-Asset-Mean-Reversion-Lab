#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [OUTPUT_DIR] [COST_PROFILE]" >&2
  exit 2
fi

args=(
  --registry configs/fx-universe-2024-registry-v1.json
  --output-dir "${1:-results/vol-compression-breakout-2024-v1}"
)
if [[ $# -eq 2 ]]; then
  args+=(--cost-profile "$2")
fi
uv run mr-lab-vol-compression-breakout-2024 "${args[@]}"
