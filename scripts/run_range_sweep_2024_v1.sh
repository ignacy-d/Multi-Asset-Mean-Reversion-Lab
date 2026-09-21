#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run mr-lab-range-sweep-2024-v1 --output-dir "${1:-results/range-sweep-2024-v1}"
