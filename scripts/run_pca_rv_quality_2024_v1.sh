#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 SOURCE_EVENTS_JSONL [OUTPUT_DIR]" >&2
  exit 2
fi
uv run mr-lab-pca-rv-quality-2024 \
  --source-events "$1" \
  --output-dir "${2:-results/pca-rv-quality-2024-v1}"
