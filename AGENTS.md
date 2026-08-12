# Working rules

- Inspect existing code and instructions before modifying anything; preserve
  working behavior from earlier project stages.
- Keep instruments, timeframes, sessions, strategies, thresholds, holding
  periods, and costs configurable rather than hardcoded.
- Enforce point-in-time semantics: never use information after time `t` to
  produce a value at `t`, and add tests for this and other important invariants.
- Preserve prefix invariance: transforming inputs available by time `t` must
  match the by-`t` outputs obtained from transforming the complete history.
- Use timezone-aware UTC timestamps internally. Session-local time and DST
  conversion belong at explicit boundaries.
- Keep the data layer provider-agnostic, raw inputs immutable, and derived data
  reproducible. Keep research independent from broker/execution integrations.
- Make research deterministic and reproducible where possible; record inputs,
  seeds, dataset identity, code revision, and outputs when that stage arrives.
- Do not silently change research methodology. Document and test intentional
  methodological changes.
- Prefer a simple tested implementation before adding abstraction.
- Treat hypotheses as families: one failed parameter configuration does not
  reject the whole family.
- Clearly distinguish discovery from confirmation and out-of-sample work. Do
  not tune methods to produce attractive backtest results.
