# OU-BIDIRECTIONAL-2024-v1

This study is a **2024 follow-up discovery, not confirmation**. It was motivated
after observing the SHORT-only 2024 result: listed `XXXUSD` and `USDXXX` pairs
appeared to respond differently to the frozen OU gate. It does not change or
reinterpret that historical result.

The only methodological change from `frozen-ou-crossasset-v1` is direction
scope: both LONG and SHORT candidates are evaluated. The process remains listed
price residual `p0 - e0` (not log price and not reciprocal/inversion invariant),
M15 London VWAP/canonical-M1 VWAP, lookbacks 20/40, strict signal boundaries
(`z < -2` for LONG and `z > +2` for SHORT; equality does not signal), 128 valid OU
transitions, strict directional score `> 1.5`, half-life `<= 120` minutes,
immediate entry, TP 0.75/1.0, SL 0.25/0.5, and 60/120 minute stops. Stage 4C's
existing authenticated transform is reused without a new cost formula. The old
OU v1 serialization, identity, and SHORT-only behavior remain unchanged.

Pair direction and USD exposure are distinct. For `USDXXX`, LONG is `LONG_USD`;
for `XXXUSD`, LONG is `SHORT_USD`; SHORT reverses those labels. Crosses without
USD are `NO_USD`. These are descriptive diagnostics only and cannot select the
universe or redefine success.

The runner reads only the explicitly supplied authenticated 2024 registry. No
sealed OOS result or data is accessed, and no filesystem discovery or fallback
corpus search exists. It streams trade rows into compact aggregates, suppresses
`trades.jsonl` persistence, and retains float64 returns only because exact median
is required. The reduced grid has 2 benchmark families × 2 lookbacks × 2
directions × 2 TP × 2 SL × 2 stops = 64 cells per instrument and variant; no
timing estimate is claimed before the external run.

## External WSL run

```bash
cd ~/projects/Multi-Asset-Mean-Reversion-Lab
git fetch origin
git switch codex/create-bidirectional-ou-implementation
git pull --ff-only
mkdir -p results/ou-bidirectional-2024-v1
nohup bash scripts/run_ou_bidirectional_2024_v1.sh \
  > results/ou-bidirectional-2024-v1/launcher.log 2>&1 &
echo $!
tail -f results/ou-bidirectional-2024-v1/replay.log
```

Final output is
`results/ou-bidirectional-2024-v1/output/comparison.json`. Existing work or final
artifacts are never automatically deleted or overwritten.
