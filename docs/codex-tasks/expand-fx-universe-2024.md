Build and authenticate the expanded 2024 FX research universe end-to-end on the CURRENT branch.

Do not implement or test an alpha hypothesis in this task.

NEW INSTRUMENTS:
- USDCAD
- USDCHF
- NZDUSD
- EURGBP

Existing frozen instruments remain unchanged:
- EURUSD
- GBPUSD
- AUDUSD
- USDJPY
- AUDJPY

ABSOLUTE DATA POLICY:
2025 is SEALED OOS.

Do not access, list, enumerate, search, inspect, hash, reference, load, or otherwise touch any 2025 corpus, path, artifact, manifest, dataset, or workflow metadata.

Only explicit 2024 provider requests and explicit known 2024 corpus paths are permitted.

Do not perform generic filesystem discovery of research-data roots or neighboring years.

Use the Research Engine v1 workflow, AGENTS.md, provenance contracts, spec-lock rules where applicable, and fail-closed semantics.

Do not manage GitHub pull requests.
Do not create, close, merge, reopen, retarget, or otherwise modify PR metadata.

PHASE 1 — PROVIDER CANDIDATES

Extend the existing ProviderInstrumentSpec / Dukascopy candidate-verification architecture.

Add candidate declarations for:

USDCAD:
- provider_symbol = USDCAD
- price_scale = 100000
- price_precision = 5

USDCHF:
- provider_symbol = USDCHF
- price_scale = 100000
- price_precision = 5

NZDUSD:
- provider_symbol = NZDUSD
- price_scale = 100000
- price_precision = 5

EURGBP:
- provider_symbol = EURGBP
- price_scale = 100000
- price_precision = 5

Candidate declaration is NOT proof of production verification.

Do not mark any new instrument decoding_verified=True based only on:
- assumptions;
- synthetic fixtures;
- unit tests;
- expected FX conventions.

Real bounded provider evidence is required.

Preserve all five existing verified instrument specifications and their behavior.

PHASE 2 — BOUNDED REAL-PROVIDER VERIFICATION

Use ONLY these explicit verification dates:

- 2024-01-02
- 2024-06-03
- 2024-10-01

Do not discover substitute dates.

Use the existing candidate-verification acquisition path rather than bypassing it.

For every new instrument verify at minimum:

- provider response is accepted as real Dukascopy data;
- payload is valid LZMA-compressed BI5;
- decoded payload has valid complete-record alignment;
- decoded prices are finite and positive;
- price scale and precision produce realistic FX price levels;
- OHLC relationships are valid;
- canonical instrument and provider symbol are correct;
- source semantics are M1 BID;
- timestamp semantics are UTC and deterministic;
- provenance includes deterministic payload SHA-256;
- M5, M15 and H1 resampling is internally consistent;
- no missing bars are silently synthesized;
- malformed/corrupt payloads fail closed.

Produce a machine-readable verification report for each candidate.

Only after real-provider verification passes may the candidate be explicitly promoted to production verified.

If verification cannot be completed due to a genuine hard blocker such as unavailable network access, leave the instrument unverified and report the blocker.
Do not fake promotion.

PHASE 3 — TESTING BEFORE FULL ACQUISITION

Add meaningful tests for at least:

- candidate declaration vs production-verification separation;
- exact provider symbols;
- exact price_scale / price_precision;
- provider URL generation on explicit 2024 dates;
- non-2024 request rejection for this workflow;
- corrupt LZMA rejection;
- malformed record alignment rejection;
- impossible OHLC rejection;
- invalid price decoding rejection;
- timestamp semantic failures;
- exact range-boundary enforcement;
- explicit instrument allowlist;
- no generic directory discovery;
- existing five verified instruments remain behaviorally unchanged.

Run the relevant targeted suite and full validation BEFORE full-year acquisition.

PHASE 4 — AUTHENTICATED FULL-2024 ACQUISITION

Proceed only if all four new candidates passed real-provider verification.

Acquire exactly:

2024-01-01 through 2024-12-31 inclusive

for:

- USDCAD
- USDCHF
- NZDUSD
- EURGBP

Use explicit root:

/mnt/e/mr-lab/corpora/2024/fx-universe-v2

Reuse the existing Dukascopy range-acquisition architecture.

Do NOT create a parallel downloader unless a proven architecture gap makes extension unavoidable.

Required acquisition properties:

- deterministic;
- resumable where existing architecture supports it;
- fail closed on inconsistent existing files;
- never silently overwrite different raw snapshots;
- preserve raw provider payloads;
- preserve provenance metadata;
- preserve SHA-256 identity;
- report provider no-data days explicitly;
- do not substitute synthetic or proxy data.

Do not access any research-data path outside the explicitly allowed 2024 roots required by this task.

PHASE 5 — EXPANDED 2024 REGISTRY

Create a NEW registry:

configs/fx-universe-2024-registry-v1.json

Do NOT modify:

configs/stage4a-2024-corpus-registry.json

or any other historical frozen registry.

Intended expanded universe:

- EURUSD
- GBPUSD
- AUDUSD
- NZDUSD
- USDJPY
- USDCAD
- USDCHF
- AUDJPY
- EURGBP

For the five historical instruments:
reuse their existing authenticated 2024 identities and semantics.

Do not rebuild or reinterpret frozen historical corpora unless explicitly required for registry compatibility.

For every entry bind at minimum:

- canonical instrument;
- provider;
- price basis;
- timeframe;
- exact requested 2024 range;
- corpus identity;
- assembled dataset identity where available;
- manifest/provenance identity;
- verification state;
- explicit corpus path.

Multiple legitimate semantic identities may reference the same authenticated path, consistent with Research Engine v1.

Fail closed if any required instrument is:
- absent;
- unverified;
- identity-inconsistent;
- semantically incompatible.

Do not invent missing identities.

PHASE 6 — 9-INSTRUMENT SEMANTIC VALIDATION

Validate the complete expanded universe before declaring it research-ready.

Check at minimum:

- exact canonical instrument identity;
- exact provider semantics;
- exact M1 BID basis;
- UTC timestamps;
- exact 2024 bounds;
- timestamp monotonicity;
- duplicate timestamps;
- impossible OHLC;
- nonfinite prices;
- invalid prices;
- missing-bar counts;
- per-instrument coverage;
- cross-instrument synchronized timestamp intersection;
- synchronization loss statistics;
- unexpected large gaps;
- corpus and assembled-dataset identity consistency.

Do NOT forward-fill, back-fill, interpolate, synthesize, or silently repair market data.

Report quality problems rather than hiding them.

Generate a deterministic machine-readable expanded-universe validation report.

PHASE 7 — ADVERSARIAL REVIEWS

Perform two independent reviews after implementation and permitted empirical data work.

A. SOFTWARE / DATA ENGINEERING REVIEW

Look specifically for:

- incorrect provider paths;
- wrong zero-based Dukascopy month handling;
- scale mistakes;
- silent overwrite behavior;
- unsafe filesystem discovery;
- incorrect retry semantics;
- incomplete provenance;
- nondeterministic identity generation;
- duplicate architecture;
- weak failure handling;
- misleading tests.

B. DATA / QUANT SEMANTIC REVIEW

Look specifically for:

- wrong quote direction assumptions;
- BID/ASK mixing;
- timestamp shifts;
- accidental resampling leakage;
- silently manufactured bars;
- inconsistent M1 semantics;
- contaminated date boundaries;
- wrong instrument identity;
- data from a non-permitted year;
- registry identity mismatch.

Fix all legitimate implementation issues found.
Add regression coverage.
Rerun affected and full validation.

Do not change historical empirical results or methodology.

VALIDATION

Run, where applicable:

- targeted provider/instrument/acquisition tests;
- full `uv run pytest -q`;
- `uv run ruff check .`;
- `uv run ruff format --check .`;
- `git diff --check`;
- relevant mypy/type checks;
- provider verification validation;
- corpus/provenance validation;
- expanded registry validation;
- expanded-universe semantic validation.

Do not weaken tests to obtain completion.

COMPLETION REPORT

At completion report:

1. current branch;
2. HEAD SHA;
3. files changed;
4. exact candidate specs;
5. verification evidence for each new instrument;
6. whether each candidate became production verified;
7. exact verification commands;
8. exact full-2024 acquisition commands;
9. acquisition completeness per instrument;
10. explicit missing/no-data diagnostics;
11. exact corpus identities;
12. exact assembled-dataset identities where applicable;
13. expanded registry identity/path;
14. cross-instrument synchronization statistics;
15. semantic-validation findings;
16. targeted/full test results;
17. lint/format/type-check results;
18. adversarial review findings and fixes;
19. any unresolved blocker;
20. confirmation historical frozen registries/corpora were not modified;
21. confirmation no sealed OOS / 2025 data or metadata was accessed;
22. READY / NOT READY FOR EXTERNAL REVIEW.

Work autonomously through the entire task unless a genuinely irreversible or research-semantic decision requires human judgment.

Do not stop after the first working implementation.

Do not manage GitHub pull requests.
