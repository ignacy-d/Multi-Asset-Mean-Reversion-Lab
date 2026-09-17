# FINALIZE EXPANDED FX UNIVERSE 2024

Work only on the CURRENT branch.

Read:

* AGENTS.md
* docs/research/WORKFLOW.md
* existing provider/acquisition code
* existing authenticated historical FX registry
* existing expanded-FX implementation on this branch

ABSOLUTE POLICY:
2025 is SEALED OOS.
Do not access, list, search, inspect, hash, reference, or discover any 2025 data/path/artifact/metadata.

Permitted new-data root only:
`/mnt/e/mr-lab/corpora/2024/fx-universe-v2`

Do NOT reacquire market data unless an explicit authenticated completeness check proves a required 2024 item is missing.

The intended expanded universe is exactly:

* EURUSD
* GBPUSD
* AUDUSD
* NZDUSD
* USDJPY
* USDCAD
* USDCHF
* AUDJPY
* EURGBP

TASK:

1. Validate the already-downloaded 2024 corpora for:

   * USDCAD
   * USDCHF
   * NZDUSD
   * EURGBP

2. Reuse existing authenticated frozen identities for:

   * EURUSD
   * GBPUSD
   * AUDUSD
   * USDJPY
   * AUDJPY

3. Build/finalize a NEW authenticated 9-instrument registry:
   `configs/fx-universe-2024-registry-v1.json`

4. Do not modify the historical frozen registry.

5. Validate:

   * exact instrument identity;
   * exact 2024 requested coverage;
   * M1 BID semantics;
   * UTC semantics;
   * timestamp ordering;
   * duplicates;
   * impossible OHLC;
   * missing-bar diagnostics;
   * cross-instrument synchronization statistics;
   * deterministic corpus/dataset identity;
   * provenance integrity.

6. Do not silently repair data.

7. Do not run any trading strategy in this task.

8. Do not run legacy MR, OU, PCA, Trend Exhaustion, or transaction-cost research.

9. Run:

   * targeted tests;
   * full `uv run pytest -q`;
   * `uv run ruff check .`;
   * `uv run ruff format --check .`;
   * `git diff --check`;
   * relevant mypy;
   * registry/provenance validation.

10. Review the complete diff against main.

Do not manage GitHub pull requests.

FINAL REPORT:

* exact registry path and identity/hash;
* all 9 instruments;
* corpus/dataset identity for each;
* completeness;
* missing-bar diagnostics;
* synchronization diagnostics;
* tests/checks;
* files changed;
* branch and HEAD SHA;
* confirmation that historical frozen artifacts were unchanged;
* confirmation that sealed OOS was untouched;
* READY FOR EXTERNAL REVIEW or NOT READY FOR EXTERNAL REVIEW.
