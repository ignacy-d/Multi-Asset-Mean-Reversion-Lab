#!/usr/bin/env bash
set -euo pipefail

ROOT="${OU_REPLICATION_2023_ROOT:-/mnt/e/mr-lab/corpora/2023/ou-replication-v1}"
exec uv run python -m mr_lab.providers.ou_replication_2023 acquire \
  --root "$ROOT" --instrument "${1:?usage: $0 AUDUSD|EURUSD|GBPUSD}" \
  --acknowledge-external-network
