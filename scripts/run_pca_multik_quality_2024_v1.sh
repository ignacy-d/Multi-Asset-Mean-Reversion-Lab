#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 SOURCE_EVENTS_JSONL [OUTPUT_DIR] [MAX_OUTPUT_BYTES]" >&2
  exit 2
fi
uv run mr-lab-pca-multik-quality-2024 \
  --source-events "$1" \
  --output-dir "${2:-results/pca-multik-quality-2024-v1}" \
  --max-output-bytes "${3:-2147483648}"
