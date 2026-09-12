"""CLI for the provisional OU-only portfolio/account Monte Carlo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from mr_lab.ou_monte_carlo import (
    AccountRules,
    MonteCarloInputError,
    PortfolioEvent,
    block_bootstrap_indices,
    deduplicate_events,
    parse_utc,
    r_multiple,
    reject_sealed_path,
    replay_r_days,
)
from mr_lab.stage4c import CostProfile, net_pips

UNIVERSES = {"strict": ("EURUSD", "GBPUSD"), "broad": ("EURUSD", "GBPUSD", "AUDUSD")}
COST_SCENARIOS = (
    ("mean+0", "mean", 0.0),
    ("p75+0.10", "p75", 0.1),
    ("p90+0.25", "p90", 0.25),
    ("p95+0.50", "p95", 0.5),
)
RISK_LEVELS = (0.0025, 0.0035, 0.005, 0.0075)
PORTFOLIO_CAPS = (0.01, 0.015, 0.02)
POLICY_FIELDS = (
    "entry_mode",
    "tp_target_fraction",
    "sl_extension_fraction",
    "time_stop_minutes",
)
REQUIRED_ECONOMIC_FIELDS = (
    "signal_timestamp",
    "entry_timestamp",
    "entry_price",
    "initial_stop_price",
    "exit_timestamp",
)


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MonteCarloInputError(f"cannot read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise MonteCarloInputError(f"expected JSON object: {path}")
    return value


def _audit(instrument: str, audit_path: Path, stage4c_path: Path) -> dict:
    audit, overlay = _json(audit_path), _json(stage4c_path)
    if audit.get("instrument") != instrument:
        raise MonteCarloInputError(f"wrong instrument in audit: expected {instrument}")
    if audit.get("filter_family") != "ornstein-uhlenbeck" or not audit.get(
        "filter_spec_id"
    ):
        raise MonteCarloInputError("invalid OU provenance")
    if not audit.get("process_spec_id"):
        raise MonteCarloInputError("missing OU process provenance")
    if overlay.get("instrument", instrument) != instrument:
        raise MonteCarloInputError(
            f"wrong instrument in Stage4C artifact: expected {instrument}"
        )
    return audit


def _policy(row: dict) -> str:
    missing = [field for field in POLICY_FIELDS if field not in row]
    if missing:
        raise MonteCarloInputError(f"missing policy fields: {missing}")
    return "|".join(f"{field}={row[field]}" for field in POLICY_FIELDS)


def load_trades(
    instrument: str,
    path: Path,
    audit: dict,
    profile: CostProfile,
    statistic: str,
    slippage: float,
) -> list[PortfolioEvent]:
    result = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MonteCarloInputError(
                    f"malformed trade JSON line {line_number}"
                ) from error
            if row.get("instrument") != instrument:
                raise MonteCarloInputError(
                    f"wrong instrument at trade line {line_number}"
                )
            expected = {
                "benchmark_family": {"vwap", "vwap-canonical-m1"},
                "signal_timeframe": {"15m", "M15"},
                "session": {"london"},
                "direction": {"SHORT"},
                "lookback": {20, 40},
            }
            if any(row.get(k) not in allowed for k, allowed in expected.items()):
                raise MonteCarloInputError(
                    f"wrong frozen OU filter at trade line {line_number}"
                )
            missing = [
                field for field in REQUIRED_ECONOMIC_FIELDS if row.get(field) is None
            ]
            if missing:
                available = sorted(row)
                raise MonteCarloInputError(
                    "cannot reconstruct exact initial entry-to-stop risk at "
                    f"line {line_number}; available fields={available}; "
                    f"missing required quantity fields={missing}"
                )
            entry = float(row["entry_price"])
            stop = float(row["initial_stop_price"])
            pip_size = 0.0001
            stop_pips = abs(entry - stop) / pip_size
            gross = row.get("gross_return_pips_adverse_first")
            if not isinstance(gross, int | float) or not math.isfinite(gross):
                raise MonteCarloInputError("missing finite adverse-first gross pips")
            spread, commission = profile.costs(
                instrument, row.get("session"), statistic
            )
            net = net_pips(float(gross), instrument, spread, slippage, commission)
            result.append(
                PortfolioEvent(
                    instrument,
                    str(row.get("candidate_event_id") or ""),
                    parse_utc(row["signal_timestamp"], "signal_timestamp"),
                    parse_utc(row["entry_timestamp"], "entry_timestamp"),
                    parse_utc(row["exit_timestamp"], "exit_timestamp"),
                    _policy(row),
                    float(gross),
                    float(gross) - net,
                    net,
                    stop_pips,
                    r_multiple(net, stop_pips),
                    f"{audit['filter_family']}:{audit['filter_spec_id']}:{audit['process_spec_id']}",
                )
            )
    return deduplicate_events(result)


def _q(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]


def simulate(events, *, paths, seed, block_days, horizon, risk, cap, rules):
    by_day = defaultdict(list)
    for event in events:
        by_day[event.signal_timestamp.date()].append((event.entry_timestamp, event.r))
    days = [by_day[day] for day in sorted(by_day)]
    if not days:
        raise MonteCarloInputError("no comparable trades")
    results = []
    for index in range(paths):
        if paths >= 1000 and index and index % max(1, paths // 10) == 0:
            logging.info("Monte Carlo progress %d/%d", index, paths)
        picks = block_bootstrap_indices(len(days), horizon, block_days, seed + index)
        results.append(replay_r_days([days[pick] for pick in picks], risk, cap, rules))
    return results


def metrics(results, rules):
    n = len(results)
    passing = [
        r
        for r in results
        if r.target_day is not None and not r.daily_breach and not r.total_breach
    ]
    p_pass = len(passing) / n
    fail = 1 - p_pass
    row = {
        "p_pass_target_before_breach": p_pass,
        "p_breach_before_target": fail,
        "p_fail_one_attempt": fail,
        "p_fail_two_attempts": fail**2,
        "p_fail_three_attempts": fail**3,
        "expected_attempts_until_first_pass": (1 / p_pass if p_pass else None),
        "daily_loss_breach_probability_realized_equity": sum(
            r.daily_breach for r in results
        )
        / n,
        "total_loss_breach_probability": sum(r.total_breach for r in results) / n,
    }
    days = [r.target_day for r in passing]
    for limit in (30, 60, 90, 120):
        row[f"p_pass_by_{limit}_days"] = sum(d <= limit for d in days) / n
    for label, p in (
        ("p10", 0.1),
        ("p25", 0.25),
        ("median", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
    ):
        row[f"{label}_days_to_target"] = _q(days, p)
    for day in (20, 60, 120, 252):
        vals = [r.returns[min(day, len(r.returns)) - 1] for r in results]
        row[f"median_return_{day}d"] = _q(vals, 0.5)
        if day == 20:
            for threshold in (0, 0.01, 0.02, 0.03, 0.04):
                row[f"p_20d_return_ge_{threshold}"] = (
                    sum(v >= threshold for v in vals) / n
                )
    for label, p in (
        ("p5", 0.05),
        ("p10", 0.1),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
        ("p95", 0.95),
    ):
        row[f"return_252d_{label}"] = _q([r.returns[-1] for r in results], p)
    for field in ("max_drawdown", "max_losing_streak", "max_losing_days"):
        for label, p in (
            ("median", 0.5),
            ("p75", 0.75),
            ("p90", 0.9),
            ("p95", 0.95),
            ("p99", 0.99),
        ):
            row[f"{field}_{label}"] = _q([getattr(r, field) for r in results], p)
    for months, _day in ((3, 60), (6, 120), (12, 252)):
        row[f"p_survive_{months}_months"] = (
            sum(not r.daily_breach and not r.total_breach for r in results) / n
        )
    monthly = [r.returns[min(20, len(r.returns)) - 1] for r in results]
    row.update(
        monthly_mean=statistics.fmean(monthly),
        monthly_median=_q(monthly, 0.5),
        monthly_p10=_q(monthly, 0.1),
        monthly_p25=_q(monthly, 0.25),
        monthly_p75=_q(monthly, 0.75),
        monthly_p90=_q(monthly, 0.9),
        p_monthly_ge_2pct=sum(x >= 0.02 for x in monthly) / n,
        p_monthly_ge_3pct=sum(x >= 0.03 for x in monthly) / n,
        p_monthly_ge_4pct=sum(x >= 0.04 for x in monthly) / n,
    )
    return row


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    all_paths = [
        value
        for key, value in vars(args).items()
        if key.endswith(("_trades", "_audit", "_stage4c")) and value
    ]
    all_paths += [args.cost_profile, args.output_dir]
    safe = [reject_sealed_path(value) for value in all_paths]
    del safe
    instruments = UNIVERSES[args.universe]
    profile = CostProfile.load(reject_sealed_path(args.cost_profile))
    artifacts = {}
    for instrument in instruments:
        prefix = instrument.lower()
        values = tuple(
            getattr(args, f"{prefix}_{suffix}")
            for suffix in ("trades", "audit", "stage4c")
        )
        if any(value is None for value in values):
            raise MonteCarloInputError(
                f"explicit {instrument} trades, audit, and Stage4C paths are required"
            )
        paths = tuple(reject_sealed_path(value) for value in values)
        artifacts[instrument] = (paths[0], _audit(instrument, paths[1], paths[2]))
    out = reject_sealed_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    portfolio_rows = []
    for scenario, statistic, slippage in COST_SCENARIOS:
        loaded = {
            i: load_trades(i, *artifacts[i], profile, statistic, slippage)
            for i in instruments
        }
        common = set.intersection(
            *(set(e.policy_id for e in loaded[i]) for i in instruments)
        )
        if not common:
            raise MonteCarloInputError("no common/comparable execution policies")
        for policy in sorted(common):
            events = [
                e for i in instruments for e in loaded[i] if e.policy_id == policy
            ]
            if scenario == "p90+0.25":
                portfolio_rows.extend(asdict(e) for e in events)
            for risk in RISK_LEVELS:
                for cap in PORTFOLIO_CAPS:
                    rules = AccountRules(
                        args.starting_balance,
                        args.profit_target,
                        args.max_daily_loss,
                        args.max_total_loss,
                        args.daily_reset_timezone,
                    )
                    result = simulate(
                        events,
                        paths=args.paths,
                        seed=args.seed,
                        block_days=args.block_days,
                        horizon=args.horizon_days,
                        risk=risk,
                        cap=cap,
                        rules=rules,
                    )
                    item = {
                        "universe": args.universe,
                        "policy": policy,
                        "cost_scenario": scenario,
                        "risk_per_trade": risk,
                        "max_portfolio_risk": cap,
                        "block_days": args.block_days,
                        **metrics(result, rules),
                    }
                    if args.challenge_fee is not None:
                        item["expected_fee_spend_until_first_pass"] = (
                            None
                            if item["expected_attempts_until_first_pass"] is None
                            else item["expected_attempts_until_first_pass"]
                            * args.challenge_fee
                        )
                    rows.append(item)
    _write_csv(out / "ou-monte-carlo-challenge.csv", rows)
    _write_csv(out / "ou-monte-carlo-policies.csv", rows)
    _write_csv(out / "ou-monte-carlo-monthly.csv", rows)
    _write_csv(out / "ou-monte-carlo-drawdown.csv", rows)
    _write_csv(out / "portfolio-events.csv", portfolio_rows)
    financial = []
    for row in rows:
        for balance in (100000, 200000, 400000):
            financial.append(
                {
                    "balance": balance,
                    "monthly_median_pct": row["monthly_median"],
                    "monthly_median_usd": balance * row["monthly_median"],
                    "monthly_p25_pct": row["monthly_p25"],
                    "monthly_p75_pct": row["monthly_p75"],
                    "p_monthly_ge_2pct": row["p_monthly_ge_2pct"],
                    "p_monthly_ge_3pct": row["p_monthly_ge_3pct"],
                    "p_monthly_ge_4pct": row["p_monthly_ge_4pct"],
                    "payout_share": args.payout_share,
                    "hypothetical_trader_payout": None
                    if args.payout_share is None
                    else balance * row["monthly_median"] * args.payout_share,
                }
            )
    _write_csv(out / "ou-monte-carlo-financial.csv", financial)
    summary = {
        "status": "PROVISIONAL OU-ONLY economic viability; not final OOS validation",
        "sealed_oos": "untouched",
        "parameters": vars(args),
        "result_count": len(rows),
    }
    (out / "ou-monte-carlo-summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    caveat = (
        "Daily loss is a realized-equity close-time approximation; Stage4B does "
        "not provide a complete floating-equity path."
    )
    params = (
        f"starting_balance={args.starting_balance}; "
        f"profit_target={args.profit_target}; "
        f"max_daily_loss={args.max_daily_loss}; "
        f"max_total_loss={args.max_total_loss}; "
        f"reset_timezone={args.daily_reset_timezone}"
    )
    (out / "report.md").write_text(
        "# PROVISIONAL OU-ONLY Monte Carlo\n\nNot final OOS validation. Uses "
        f"frozen research evidence only.\n\n**Parameters:** {params}\n\n"
        f"**Limitation:** {caveat}\n"
    )
    (out / "financial-summary.md").write_text(
        f"# Provisional financial summary\n\n{params}\n\n{caveat}\n\n"
        "See `ou-monte-carlo-financial.csv`. No payout split is assumed unless "
        "explicitly supplied.\n"
    )
    audit = {
        "seed": args.seed,
        "paths": args.paths,
        "block_days": args.block_days,
        "input_sha256": {
            i: hashlib.sha256(path.read_bytes()).hexdigest()
            for i, (path, _) in artifacts.items()
        },
        "method": "chronological circular trading-day block bootstrap",
    }
    (out / "execution-audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(
        description=(
            "PROVISIONAL OU-only economic viability Monte Carlo; "
            "not final OOS validation"
        )
    )
    p.add_argument("--universe", choices=UNIVERSES, default="strict")
    for instrument in ("eurusd", "gbpusd", "audusd"):
        for suffix in ("trades", "audit", "stage4c"):
            p.add_argument(f"--{instrument}-{suffix}")
    p.add_argument("--cost-profile", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--paths", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20240930)
    p.add_argument("--block-days", type=int, choices=(1, 5), default=5)
    p.add_argument("--horizon-days", type=int, default=252)
    p.add_argument("--starting-balance", type=float, default=100000)
    p.add_argument("--profit-target", type=float, choices=(0.05, 0.10), default=0.10)
    p.add_argument("--max-daily-loss", type=float, default=0.05)
    p.add_argument("--max-total-loss", type=float, default=0.10)
    p.add_argument("--daily-reset-timezone", default="UTC")
    p.add_argument("--challenge-fee", type=float)
    p.add_argument("--payout-share", type=float, choices=(0.8, 0.9))
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args(argv)
    if args.paths <= 0 or args.horizon_days < 252:
        raise SystemExit("--paths must be positive and --horizon-days must be >=252")
    try:
        run(args)
    except (MonteCarloInputError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
