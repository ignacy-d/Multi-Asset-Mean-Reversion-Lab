# VWAP consecutive-return normalization v2

Status: **methodological correction; production recomputation pending**

## Root cause and corrected invariant

VWAP normalization v1 advanced from one supplied observation to the next and
formed a close-to-close return whenever both were active. It did not prove that
their canonical bars were adjacent. Because absent resampling windows are omitted,
an outage or routine market closure could therefore be represented as one M5,
M15, or H1 return. That contradicted the stated sample-standard-deviation of
consecutive bar returns.

In v2, a return is eligible only if both observations are active, their canonical
timeframes are equal, and `previous.bar.close_time == current.bar.open_time`.
Otherwise no return is synthesized, the rolling-return deque is cleared, and the
current observation is recorded as unavailable. Volatility becomes available only
after a complete configured lookback of newly formed, exactly adjacent returns.
Missing bars are never filled. This rule is point-in-time and prefix invariant.

The canonical-M1 construction continues to change only the session VWAP numerator
and weight. Its volatility denominator is the same v2 native-timeframe denominator,
so native and canonical-M1 gap behavior remains identical. Bollinger already had
an explicit adjacency reset and is unchanged.

## Identity transition

The following v1 values remain historical facts and must remain attached to old
artifacts; they are not aliases for the corrected implementation:

- VWAP strategy schema: `stage-2b-vwap-benchmark-v1`;
- normalization: `relative_deviation_divided_by_sample_stddev_of_consecutive_bar_returns_v1`;
- canonical-M1 robustness schema: `stage-2b-vwap-construction-robustness-v1`;
- Stage 4B methodology ID:
  `sha256:2d0d68ade809aef35cfce0cc39c515c3c1833a3c5aa7da506dd3b765af3ce42d`.

Corrected runs use VWAP strategy schema `stage-2b-vwap-benchmark-v2`, explicit
gap-reset normalization v2, robustness schema
`stage-2b-vwap-construction-robustness-v2`, and a newly hashed Stage 4B methodology
identity committing to the upstream v2 rule. Stage 4A path arithmetic itself does
not change; its outputs nevertheless inherit newly selected VWAP events and v2
strategy IDs. No threshold, 20/40 lookback, entry mode, exit grid, holding period,
session definition, instrument-specific setting, or Bollinger methodology changes.

## Impact and deterministic local recomputation order

Do not run this plan in hosted automation and do not use old result rankings to
alter it. On the approved local research machine:

1. Pin the correction commit and record its revision; verify the existing immutable
   2024 corpus manifests, component hashes, dataset identities, and configuration.
2. Regenerate native-timeframe VWAP features for every predeclared instrument and
   M5/M15/H1 cell at lookbacks 20 and 40, then regenerate canonical-M1 VWAP features
   from the same corpus. Retain v1 artifacts under their original identities.
3. Regenerate Stage 4A event/path outputs for both VWAP families. Bollinger Stage 4A
   artifacts need not be recomputed for this defect.
4. Regenerate Stage 4B candidate, entry, trade, compact-shard, reduction, and audit
   outputs for both VWAP families under the new Stage 4B methodology ID. Combined
   family bundles must be rebuilt; unchanged Bollinger rows may be deterministically
   re-bound only if the tooling proves byte/row identity, otherwise regenerate the
   combined bundle without changing their values.
5. Reassess the 2024 cross-asset interpretation from the complete corrected output
   and publish a new versioned freeze. Do not edit the historical freeze in place.
6. Recreate/authenticate Stage 4C source inputs only after the new freeze, then run
   the separately preregistered cost stage. Any Stage 4C artifact sourced from v1
   VWAP trades is invalid.

Invalidated artifacts are v1 native and canonical-M1 VWAP features and summaries;
their VWAP Stage 4A events; VWAP Stage 4B candidates/trades/shards/reductions;
combined cross-family Stage 4B interpretations; and downstream Stage 4C inputs or
outputs containing those trades. Raw canonical data, corpus manifests, resampled
bars, session assignments, Bollinger features/results, M1 path arithmetic, frozen
thresholds and grids, and v1 records retained for provenance are unaffected.

This correction and its synthetic tests do not access, enumerate, hash, or compute
against sealed 2025 resources. The discovery/recomputation scope remains 2024
in-sample only.
