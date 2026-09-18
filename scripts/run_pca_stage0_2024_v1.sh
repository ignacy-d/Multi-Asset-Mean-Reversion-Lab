#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run mr-lab-pca-stage0-2024 --output-dir "${1:-results/rv-pca-shock-2024-stage0-v1}"
