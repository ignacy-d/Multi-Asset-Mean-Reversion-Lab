# Strategy Family Registry

`configs/strategy-family-registry-v1.json` is the persistent governance record
for independently useful strategy families. It records the hierarchy **strategy
family → session → exact alpha rules → exact execution rules → instrument →
evidence → economic status → replication status**. Its schema is
`configs/strategy-family-registry-schema-v1.json`.

## Governance rules

- The registry is append-only history: a new implementation is a new family,
  not a silent replacement. Every current family has `replaces_family_id: null`.
- Families are not globally ranked. Execution rules may differ by family and
  instrument; the registry does not require a universal execution model.
- Status is recorded per instrument. A parameter-family/plateau result must not
  be collapsed into a selected best cell.
- A result summary remembered externally or held locally is distinct from
  repository-authenticated artifact evidence. A missing SHA-256 remains `null`
  and its verification stays `LOCAL_ARTIFACT_AUDIT_PENDING`; metrics must not be
  invented to fill the gap.
- The 2024 work is discovery/follow-up evidence. Independent 2023 replication
  is the next intended validation dataset for London candidates. This statement
  does not authorize access to that dataset before its confirmation stage.
- PR #106 / `OU-LONGUSD-NET-CORE-v2` is not authoritative final family
  selection while the strategy-family audit is unresolved.

## Registered families

1. `FROZEN_STAGE4B_OU_LONDON` preserves the 32-cell London parameter-family
   plateau (four signal definitions × eight exit policies).
2. `OU_LONGUSD_TRADABLE_LONDON_2024_V1` preserves the E0 deterministic-median,
   exact-M1-open, three-tranche execution family and instrument-specific 2024
   evidence. In particular, EURUSD remains an economically positive candidate;
   AUDUSD and GBPUSD surviving 16/16 scenarios is not a global selection rule.
3. `OU_LONGUSD_TRADABLE_NEW_YORK_2024_V1` is parked after costs. Gross
   diagnostics remain recorded as diagnostics, not candidate promotion, and
   this family must not be tuned further on 2024.

## Updating the registry

Add a family alongside the existing records and retain distinct alpha and
execution methodology IDs. Add an instrument row for every assessed instrument,
including negative findings. Supply an artifact SHA-256 only after verifying the
exact evidence artifact; then update the evidence verification status without
rewriting the historical methodology. Changes to the status vocabulary require
an explicit schema and test update.

## Completed local artifact audit

Evidence is now represented as an `evidence_artifacts` array because an
instrument can depend on a study summary, net overlay, and robustness artifact.
Each entry records its path when supplied, SHA-256, and the exact evidence it
supports. A `null` path means the hash is known but its local path was not
supplied; it is not a fabricated location.

The Stage4B comparison locally verifies EURUSD and AUDUSD gross and net values.
It verifies only gross evidence for EURGBP because that replay was
`BLOCKED_MISSING_AUTHENTICATED_COST_PROFILE`; EURGBP net evidence remains
`NOT_ASSESSED` and is not borrowed from the Tradable execution family. The
later GBPUSD cost-artifact identity remains unresolved even though the retained
32-cell robustness counts are known.

Independent 2023 replication is pending for exactly AUDUSD, EURUSD, and GBPUSD
in each of the two London families. Failed London instruments and every parked
New York instrument are excluded from that replication set. The New York
summary hash does not by itself authenticate the locally derived per-instrument
net decomposition, which remains pending a persisted artifact.
