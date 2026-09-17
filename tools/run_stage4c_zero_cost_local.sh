#!/usr/bin/env bash
set -euo pipefail

# This wrapper deliberately forwards only explicit inputs. It performs no search,
# glob, recursion, fallback selection, or neighboring-directory inspection.
exec uv run mr-lab-stage4c-zero-cost "$@"
