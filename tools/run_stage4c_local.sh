#!/usr/bin/env bash
set -euo pipefail

INSTRUMENT="${1:-}"
case "$INSTRUMENT" in
  AUDUSD|USDJPY|EURUSD|AUDJPY) ;;
  *)
    echo "Usage: bash tools/run_stage4c_local.sh {AUDUSD|USDJPY|EURUSD|AUDJPY}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MR_ROOT="${MR_ROOT:-/mnt/e/mr-lab}"
CORPUS_DIR="$MR_ROOT/corpora/2024/$INSTRUMENT"
RESULTS_ROOT="$MR_ROOT/results/stage4c"
HEAVY_TMP="$MR_ROOT/tmp"
STAGE4C_SPOOL_ROOT="${STAGE4C_SPOOL_ROOT:-$HOME/.cache/mr-lab/stage4c-spool}"

mkdir -p "$RESULTS_ROOT" "$HEAVY_TMP" "$STAGE4C_SPOOL_ROOT"

if [[ ! -s "$CORPUS_DIR/corpus-manifest.json" ]]; then
  echo "Missing corpus manifest: $CORPUS_DIR/corpus-manifest.json" >&2
  exit 3
fi

# Fail closed against the frozen 2024 registry before any production compute.
export INSTRUMENT CORPUS_DIR
TMPDIR=/tmp uv run python - <<'PY'
import json
import os
from pathlib import Path

instrument = os.environ["INSTRUMENT"]
corpus_dir = Path(os.environ["CORPUS_DIR"])
manifest = json.loads((corpus_dir / "corpus-manifest.json").read_text())
registry = json.loads(Path("configs/stage4a-2024-corpus-registry.json").read_text())
expected = registry["instruments"][instrument]

assert expected["verification_status"] == "verified"
assert manifest["instrument"] == instrument
assert manifest["corpus_id"] == expected["corpus_id"]
assert manifest["assembled_dataset_id"] == expected["assembled_dataset_id"]

print(f"{instrument} frozen corpus identity: PASS")
print("corpus_id:", manifest["corpus_id"])
print("assembled_dataset_id:", manifest["assembled_dataset_id"])
PY

# Keep pytest's capture/temp files on native WSL storage. -s also avoids capture issues
# previously observed with Windows-mounted temporary directories.
echo "=== Stage 4C software smoke ==="
TMPDIR=/tmp uv run pytest -q -s tests/test_stage4c.py
TMPDIR=/tmp uv run ruff check \
  src/mr_lab/stage4c.py \
  src/mr_lab/stage4c_runner.py \
  src/mr_lab/stage4c_reducer.py

# Heavy regenerated Stage 4B -> Stage 4C temporary storage belongs on E:.
export TMPDIR="$HEAVY_TMP"
COMMIT="$(git rev-parse --short=12 HEAD)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULTS_ROOT/${INSTRUMENT}-${COMMIT}-${STAMP}"
export OUT
mkdir -p "$OUT"

echo "============================================================"
echo "REAL STAGE 4C — $INSTRUMENT 2024"
echo "repo:   $REPO_ROOT"
echo "corpus: $CORPUS_DIR"
echo "output: $OUT"
echo "tmp:    $TMPDIR"
echo "spool:  $STAGE4C_SPOOL_ROOT"
echo "============================================================"

/usr/bin/time -v \
  -o "$OUT/resource-usage.txt" \
  uv run mr-lab-stage4c \
    --source-mode regenerated_stage4b \
    --instrument "$INSTRUMENT" \
    --corpus-dir "$CORPUS_DIR" \
    --output-dir "$OUT" \
    --spool-dir "$STAGE4C_SPOOL_ROOT" \
  2>&1 | tee "$OUT/run.log"

for file in \
  stage4c-trade-matrix.csv \
  stage4c-regime-breadth.csv \
  stage4c-cost-scenarios.csv \
  stage4c-summary.json \
  stage4c-report.md \
  execution-audit.json
  do
    [[ -s "$OUT/$file" ]] || {
      echo "Missing required output: $OUT/$file" >&2
      exit 4
    }
  done

[[ ! -e "$OUT/stage4c-debug-trades.jsonl" ]] || {
  echo "Unexpected production debug raw output present" >&2
  exit 5
}
[[ ! -e "$OUT/.stage4c-spool.sqlite3" ]] || {
  echo "Temporary Stage 4C spool was not removed" >&2
  exit 6
}

TMPDIR=/tmp uv run python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
audit = json.loads((out / "execution-audit.json").read_text())
summary = json.loads((out / "stage4c-summary.json").read_text())
instrument = os.environ["INSTRUMENT"]

assert audit["schema_version"] == "stage-4c-report-v1"
assert audit["source_mode"] == "regenerated_stage4b"
assert audit["instrument"] == instrument
assert audit["account_currency"] == "USD"

counts = audit["row_counts"]
assert counts["owned_complete_rows"] == counts["source_complete_rows"]
assert counts["scenario_evaluations"] == 16 * counts["source_complete_rows"]

for name, expected in audit["output_sha256"].items():
    actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert actual == expected, (name, actual, expected)

print("=== AUDIT PASS ===")
print("instrument:", audit["instrument"])
print("source_mode:", audit["source_mode"])
print("stage4b_methodology_id:", audit["stage4b_methodology_id"])
print("stage4b_source_commit:", audit["stage4b_source_commit"])
print("stage4c_source_commit:", audit["stage4c_source_commit"])
print("cost_profile_sha256:", audit["cost_profile_sha256"])
print("source_input_commitment:", audit["source_input_commitment"])
print("row_counts:", counts)
print("=== SUMMARY ===")
for key, value in summary.items():
    print(f"{key}: {value}")
PY

REVIEW="${OUT}.review.tgz"
tar -czf "$REVIEW" \
  -C "$OUT" \
  stage4c-trade-matrix.csv \
  stage4c-regime-breadth.csv \
  stage4c-cost-scenarios.csv \
  stage4c-summary.json \
  stage4c-report.md \
  execution-audit.json \
  resource-usage.txt \
  run.log

echo "============================================================"
echo "$INSTRUMENT STAGE 4C COMPLETE"
echo "OUT=$OUT"
echo "REVIEW=$REVIEW"
ls -lh "$REVIEW"
