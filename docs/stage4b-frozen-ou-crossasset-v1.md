# Stage 4B frozen OU cross-asset eligibility v1

This document records the implementation of the already-frozen 2024 hypothesis.
It is an opt-in candidate-time eligibility gate, not a parameter search. The
ordinary Stage 4B run continues to select `none` and retains its existing method.

The signal scope is M15, London, SHORT, either `vwap` or
`vwap-canonical-m1`, and lookback 20 or 40. Both variants use the existing causal
OU process specification with 128 valid transitions and require a valid state and
a directional score strictly greater than 1.5. For SHORT the directional score is
the raw score; for LONG it is its negative. `frozen-ou-crossasset-v1` additionally
requires finite half-life at or below 120 minutes. The score-only control removes
only that cap (it still requires a finite half-life from a valid OU state).

States are computed during the Stage 4B run from the same assembled signal-state
history. Candidate matching uses process identity plus the exact completed signal
timestamp. Missing, unavailable, invalid, non-finite, mismatched, and out-of-scope
states fail closed. No external state file or nearest-time match is accepted.

The associated downstream slice is immediate entry, TP 0.75 or 1.0, SL 0.25 or
0.5, and a 60- or 120-minute time stop. It does not alter Stage 4B exit semantics.
All supported instruments use the same scope and gate; instrument membership is
not encoded in the filter.
