# Failed Breakout discovery preregistration v1

Scope: 2024 discovery inputs only.

## Hypothesis

A completed close beyond a previously known structural boundary, followed by failure to maintain the new level and a completed close back inside the prior range, has positive forward expectancy in the fade direction.

This family is intentionally independent of VWAP distance and the frozen OU eligibility model. It is evaluated standalone before any combination with Module A.

## Initial anchors

The implementation contract accepts precomputed causal structural levels. The first research pass should populate only these preregistered anchor families:

- previous-day high / previous-day low,
- Asia session high / low,
- first 60 minutes of the London session opening range high / low.

Anchor values and normalization scale must be available before the breakout can qualify.
Generation consumes immutable canonical M1 `Bar` values, requires exact minute
coverage for each construction window, and never fills missing observations.

Each level records explicit `available_at` and `expires_at` UTC instants. The
previous-day pair becomes available only after the complete prior UTC day and
expires at the next UTC midnight. The Asia pair becomes available at the
historical session close and expires at the next Asia open. London OR60 becomes
available only after all first-60-minute bars are complete and expires at that
London session's close. Session boundaries use the existing versioned IANA
timezone session specification, including historical DST.

## Frozen normalization

Every anchor uses the same pre-event scale: the high-low range of the immediately
preceding complete UTC day. That day must have exact M1 coverage, the range must
be positive, and every contributing bar must already be available. Otherwise the
anchor fails closed. This definition is intentionally independent from VWAP,
z-scores, and OU state and is not selected from observed results.

## Detector grid

Fixed reclaim window: 30 minutes.

Minimum breakout depth, normalized by a pre-event scale:

- 0.05,
- 0.15.

No post-hoc rescue of failed anchor/depth cells.

## Event semantics

1. A level is armed only while causally available and unexpired.
2. A breakout episode starts on the first completed M1 close outside the level whose excursion reaches the minimum normalized depth.
3. The event timestamp is the first later completed M1 close back inside the old range within 30 minutes.
4. Upper-level failures produce SHORT events; lower-level failures produce LONG events.
5. Missing M1 intervals cancel the open episode and fail closed.
6. Events contain no VWAP, OU, TP, SL, cost, or trade-execution assumptions.

## Stage-0 diagnostics

For each anchor/depth cell and instrument, measure common discovery outcomes at 15, 30, 60 and 120 minutes, MFE/MAE, sample count, unique timestamps and monthly distribution.

Primary family horizon: 60 minutes.

## Orthogonality test against Module A

Measure timestamp overlap with frozen Module A at:

- exact timestamp,
- +/-15 minutes,
- +/-30 minutes,
- +/-60 minutes.

Operational interpretation before Stage 1:

- strong independence candidate: at least 50% of failed-breakout events remain outside +/-30 minutes of Module A, while standalone directional expectancy survives cross-asset checks;
- high overlap is not automatically a failure, but the family must then be treated as a possible timing/filter layer rather than claimed as an independent alpha.

The 50% threshold is an architecture/research triage rule, not a profitability threshold and must not be tuned to 2024 results.

## Stage-0 family pass criteria

Do not promote on one best cell. Promote only if the family shows a broad, interpretable pattern with adequate event count, at least two instruments with meaningful samples, non-negative median aggregate directional outcome, positive aggregate mean at the primary horizon, and no single instrument or quarter dominating the effect.

If sample size is inadequate, classify as inconclusive rather than lowering thresholds after observing results.
