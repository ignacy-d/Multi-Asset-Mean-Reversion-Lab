# Frozen Stage 4B artifact recovery

> [!WARNING]
> These recovered 2024 Stage 4B artifacts are historical
> **VWAP-normalization-v1** artifacts. After the September 2026 VWAP-v2
> methodology correction, they **MUST NOT** be used as production Stage 4C
> inputs. Preserve and recover them for provenance, audit, and comparison only.
> This recovery tool does not upgrade, reinterpret, or recompute them.

`mr-lab-stage4b-artifacts` only reads existing GitHub Actions runs. It cannot
dispatch, rerun, modify, or delete a workflow. The frozen mapping is:

| Instrument | Run ID | Head SHA |
|---|---:|---|
| USDJPY | `32638668214` | `3090682b61090de1b4a30efc18a7547e92fa262e` |
| AUDUSD | `32641981092` | same |
| AUDJPY | `32641988237` | same |
| EURUSD | `32643840766` | same |

GBPUSD is deliberately absent. Artifact names are derived exactly from the run
and the pinned source-corpus artifact IDs in
`configs/stage4a-2024-corpus-registry.json`; no run search or fuzzy matching is
performed. The tool checks run success and head SHA, artifact run/name/expiry
and GitHub digest (when supplied), then checks the internal execution audits,
shard manifests, corpus identity, exact shard set, and every committed file
SHA-256. A mismatch leaves no completed destination.

Authentication is read from `GH_TOKEN` or `GITHUB_TOKEN`. Public API requests
can work without a token but are rate limited. Start with the non-network list,
then perform metadata-only availability verification; neither downloads an
artifact:

```bash
uv run mr-lab-stage4b-artifacts list --instrument all
uv run mr-lab-stage4b-artifacts check --instrument all
```

Downloading is always an explicit operation with an explicit destination. Each
instrument is stored as `<destination>/<INSTRUMENT>/` with `raw-shards/shard-N`,
`compact-shards/shard-N`, `combined-review`, and `provenance/github-metadata.json`.
For example under WSL:

```bash
# USDJPY only (all four raw shards, compact shards, and combined review)
uv run mr-lab-stage4b-artifacts download --instrument USDJPY \
  --destination /mnt/e/mr-lab/frozen/stage4b-2024

# All four completed instruments; potentially multi-GB, so no implicit default download
uv run mr-lab-stage4b-artifacts download --instrument all \
  --destination /mnt/e/mr-lab/frozen/stage4b-2024

# Offline validation after download
uv run mr-lab-stage4b-artifacts verify --instrument all \
  --destination /mnt/e/mr-lab/frozen/stage4b-2024
```

An existing instrument directory is verified and skipped when valid. Invalid
existing content fails closed. `--force` is the only way to replace it; the new
download is staged and fully verified before replacement. The metadata-only
check reports current expiry/availability. Artifact sizes are available in the
saved metadata and GitHub inventory; sum `size_in_bytes` before choosing a disk.
