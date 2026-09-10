# Bollinger standalone / incremental alpha study — frozen 2024

Status: **PREREGISTERED; REAL OUTPUTS NOT AVAILABLE IN THIS CHECKOUT**

This study is a read-only reporting transform over authenticated Stage 4B candidate
and trade rows. It never opens a market-data corpus and rejects every signal
timestamp outside 2024. Sealed 2025 data is outside the interface and must not be
used.

The fixed scope is M15, London, SHORT, threshold 2.0, lookbacks 20 and 40,
immediate entry, TP `{0.75, 1.0}`, SL `{0.25, 0.5}`, and time stop `{60, 120}`.
No OU filter is permitted. The comparison benchmark is native VWAP Module A
(`benchmark_family=vwap`), not canonical-M1 VWAP.

An intersection is equality of instrument, causal UTC signal timestamp, direction,
session, timeframe, and lookback. Event classes are reported without changing the
trade engine: Bollinger standalone, Bollinger-only, VWAP-only, and intersection.
Intersection outcomes remain labelled by their originating family because the two
signals can have different centers and displacement-scaled exits. Counts are also
deduplicated by causal timestamp across lookbacks; this is a reporting count, not a
new execution rule.

Every frozen Stage 4C spread statistic and slippage scenario is reported. Gross and
net expectancy, profit factor, win rate, calendar-month expectancy stability,
positive/observed months, worst observed month, and chronological maximum losing
streak are computed from complete adverse-first trade rows. Trades/month always
uses the full 12-month frozen study interval; missing-trade months therefore do not
inflate frequency. No maximum cell is selected: reviewers must assess consistency
across the complete 2 × 2 × 2 exit plateau, both lookbacks, and cost scenarios.

Run once the authenticated raw Stage 4B shard directories are locally available:

```bash
mr-lab-bollinger-incremental \
  --input-dir /path/to/stage4b/shard-0 \
  --input-dir /path/to/stage4b/shard-1 \
  --output-dir results/bollinger-incremental-2024
```

GBPUSD is automatically rejected while its registry entry remains unverified.
The current checkout contains no authenticated Stage 4B raw artifacts, so this PR
does not fabricate numerical findings or claim a plateau conclusion.
