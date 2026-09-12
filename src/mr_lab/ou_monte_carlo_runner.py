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
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.ou_monte_carlo import (
    AccountRules,
    MonteCarloInputError,
    PortfolioEvent,
    SimTrade,
    block_bootstrap_indices,
    deduplicate_events,
    parse_utc,
    r_multiple,
    reject_sealed_path,
    replay_events,
    research_calendar_2024,
)
from mr_lab.stage4b import SIGNAL_THRESHOLD, STAGE4B_METHODOLOGY_ID, pip_size
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
    "benchmark_family",
    "lookback",
    "entry_mode",
    "tp_target_fraction",
    "sl_extension_fraction",
    "time_stop_minutes",
)
REQUIRED_TRADE_FIELDS = ("entry_wait_minutes", "r_at_entry", "exit_timestamp")


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MonteCarloInputError(f"cannot read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise MonteCarloInputError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(
    instrument: str,
    audit_path: Path,
    stage4c_path: Path,
    trades_path: Path,
    candidates_path: Path,
    cost_profile_sha256: str,
) -> dict:
    audit, overlay = _json(audit_path), _json(stage4c_path)
    if audit.get("instrument") != instrument:
        raise MonteCarloInputError(f"wrong instrument in audit: expected {instrument}")
    spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
    expected = {
        "filter_family": "ornstein-uhlenbeck",
        "filter_spec_id": spec.filter_spec_id,
        "process_spec_id": spec.process_spec.process_spec_id,
    }
    if any(audit.get(field) != value for field, value in expected.items()):
        raise MonteCarloInputError("invalid OU provenance")
    if audit.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise MonteCarloInputError("Stage4B methodology mismatch")
    output_hashes = audit.get("output_sha256", {})
    for name, path in (
        ("trades.jsonl", trades_path),
        ("candidate-events.jsonl", candidates_path),
    ):
        if output_hashes.get(name) != _sha256(path):
            raise MonteCarloInputError(f"Stage4B {name} SHA mismatch")
    embedded = audit.get("eligibility_filter_spec")
    canonical_spec = json.loads(json.dumps(asdict(spec)))
    if embedded != canonical_spec:
        raise MonteCarloInputError("frozen OU eligibility specification mismatch")
    if overlay.get("instrument", instrument) != instrument:
        raise MonteCarloInputError(
            f"wrong instrument in Stage4C artifact: expected {instrument}"
        )
    if overlay.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise MonteCarloInputError("Stage4C Stage4B methodology mismatch")
    if any(overlay.get(field) != value for field, value in expected.items()):
        raise MonteCarloInputError("Stage4C OU provenance mismatch")
    source_hashes = overlay.get("source_trade_sha256")
    accepted = (
        set(source_hashes.values())
        if isinstance(source_hashes, dict)
        else {source_hashes}
    )
    if _sha256(trades_path) not in accepted:
        raise MonteCarloInputError("Stage4C source trade SHA mismatch")
    if overlay.get("cost_profile_sha256") != cost_profile_sha256:
        raise MonteCarloInputError("Stage4C cost-profile SHA mismatch")
    return audit


def _policy(row: dict) -> str:
    missing = [field for field in POLICY_FIELDS if field not in row]
    if missing:
        raise MonteCarloInputError(f"missing policy fields: {missing}")
    return "|".join(f"{field}={row[field]}" for field in POLICY_FIELDS)


def load_candidates(instrument: str, path: Path) -> dict[str, dict]:
    candidates = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                identity = row["candidate_event_id"]
                signal = row["signal"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise MonteCarloInputError(
                    f"malformed candidate event line {line_number}"
                ) from error
            if signal.get("instrument") != instrument:
                raise MonteCarloInputError("candidate instrument mismatch")
            if identity in candidates and candidates[identity] != signal:
                raise MonteCarloInputError("incompatible duplicate candidate event")
            candidates[identity] = signal
    return candidates


def load_trades(
    instrument: str,
    path: Path,
    audit: dict,
    profile: CostProfile,
    statistic: str,
    slippage: float,
    candidates: dict[str, dict],
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
            candidate_id = row.get("candidate_event_id")
            if not candidate_id or candidate_id not in candidates:
                raise MonteCarloInputError("trade has no matching candidate event")
            signal = candidates[candidate_id]
            expected = {
                "benchmark_family": {"vwap", "vwap-canonical-m1"},
                "signal_timeframe": {"15m", "M15"},
                "session": {"london"},
                "direction": {"SHORT"},
                "lookback": {20, 40},
                "signal_threshold": {SIGNAL_THRESHOLD},
                "entry_mode": {"immediate"},
                "tp_target_fraction": {0.75, 1.0},
                "sl_extension_fraction": {0.25, 0.5},
                "time_stop_minutes": {60, 120},
            }
            if any(row.get(k) not in allowed for k, allowed in expected.items()):
                raise MonteCarloInputError(
                    f"wrong frozen OU filter at trade line {line_number}"
                )
            if any(
                signal.get(k) != row.get(k)
                for k in (
                    "instrument",
                    "benchmark_family",
                    "signal_timeframe",
                    "lookback",
                    "session",
                )
            ):
                raise MonteCarloInputError(
                    "candidate/trade strategy provenance mismatch"
                )
            if signal.get("direction") != row.get("direction"):
                raise MonteCarloInputError("candidate/trade direction mismatch")
            if signal.get("threshold") != SIGNAL_THRESHOLD:
                raise MonteCarloInputError("candidate signal threshold mismatch")
            if not row.get("complete"):
                continue
            missing = [
                field for field in REQUIRED_TRADE_FIELDS if row.get(field) is None
            ]
            if missing:
                available = sorted(row)
                raise MonteCarloInputError(
                    "cannot reconstruct exact Stage4B entry-to-stop risk at "
                    f"line {line_number}; available fields={available}; "
                    f"missing required quantity fields={missing}"
                )
            try:
                signal_timestamp = parse_utc(
                    signal["signal_timestamp"], "signal_timestamp"
                )
                if signal_timestamp.year != 2024:
                    raise MonteCarloInputError(
                        "candidate is outside 2024 research year"
                    )
                if signal_timestamp.weekday() >= 5:
                    raise MonteCarloInputError(
                        "candidate is outside Monday-Friday research calendar"
                    )
                p0 = float(signal["p0"])
                d0 = float(signal["d0"])
                wait = int(row["entry_wait_minutes"])
                r_at_entry = float(row["r_at_entry"])
                sl_fraction = float(row["sl_extension_fraction"])
            except (KeyError, TypeError, ValueError) as error:
                raise MonteCarloInputError(
                    "malformed Stage4B reconstruction fields"
                ) from error
            if not all(math.isfinite(x) for x in (p0, d0, r_at_entry, sl_fraction)):
                raise MonteCarloInputError("non-finite Stage4B reconstruction field")
            if d0 == 0 or wait < 0 or sl_fraction <= 0:
                raise MonteCarloInputError(
                    "impossible Stage4B entry/stop reconstruction"
                )
            direction = -1 if row["direction"] == "SHORT" else 1
            entry_timestamp = signal_timestamp + timedelta(minutes=wait)
            entry = p0 + direction * r_at_entry * abs(d0)
            stop = p0 - direction * sl_fraction * abs(d0)
            stop_pips = abs(entry - stop) / pip_size(instrument)
            if stop_pips <= 0 or direction * (entry - stop) <= 0:
                raise MonteCarloInputError("initial stop is not adverse to entry")
            gross = row.get("gross_return_pips_adverse_first")
            if not isinstance(gross, int | float) or not math.isfinite(gross):
                raise MonteCarloInputError("missing finite adverse-first gross pips")
            spread, commission = profile.costs(
                instrument, row.get("session"), statistic
            )
            net = net_pips(float(gross), instrument, spread, slippage, commission)
            exit_timestamp = parse_utc(row["exit_timestamp"], "exit_timestamp")
            if exit_timestamp <= entry_timestamp or exit_timestamp.year != 2024:
                raise MonteCarloInputError("impossible Stage4B entry/exit timestamps")
            result.append(
                PortfolioEvent(
                    instrument,
                    str(candidate_id),
                    signal_timestamp,
                    entry_timestamp,
                    exit_timestamp,
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


def bootstrap_samples(paths, seed, block_days, horizon):
    day_count = len(research_calendar_2024())
    return tuple(
        block_bootstrap_indices(day_count, horizon, block_days, seed + index)
        for index in range(paths)
    )


def _synthetic_trades(events, picks, calendar):
    by_day = defaultdict(list)
    for event in events:
        by_day[event.signal_timestamp.date()].append(event)
    result = []
    for synthetic_index, source_index in enumerate(picks):
        source_day = calendar[source_index]
        target_day = calendar[synthetic_index]
        source_midnight = datetime.combine(source_day, time(), UTC)
        target_midnight = datetime.combine(target_day, time(), UTC)
        for event in by_day[source_day]:
            result.append(
                SimTrade(
                    target_midnight + (event.entry_timestamp - source_midnight),
                    target_midnight + (event.exit_timestamp - source_midnight),
                    event.r,
                    f"{synthetic_index}:{event.instrument}:{event.candidate_event_id}",
                )
            )
    return result


def simulate_matrix(events, *, samples, horizon, risks, caps, rules):
    """Batch one policy/cost over shared bootstrap paths and trade reconstruction."""
    calendar = research_calendar_2024()[:horizon]
    if not events:
        raise MonteCarloInputError("no comparable trades")
    results = {(risk, cap): [] for risk in risks for cap in caps}
    source_calendar = research_calendar_2024()
    for index, picks in enumerate(samples):
        if len(samples) >= 1000 and index and index % max(1, len(samples) // 10) == 0:
            logging.info("Monte Carlo progress %d/%d", index, len(samples))
        trades = _synthetic_trades(events, picks, source_calendar)
        for risk in risks:
            for cap in caps:
                results[(risk, cap)].append(
                    replay_events(trades, calendar, risk, cap, rules)
                )
    return results


def metrics(results, rules):
    n = len(results)
    passing = [r for r in results if r.terminal_outcome == "PASS"]
    p_pass = len(passing) / n
    p_breach = sum(r.terminal_outcome == "BREACH" for r in results) / n
    p_unresolved = sum(r.terminal_outcome == "UNRESOLVED" for r in results) / n
    fail = p_breach + p_unresolved
    row = {
        "p_pass_target_before_breach": p_pass,
        "p_breach_before_target": p_breach,
        "p_unresolved_at_horizon": p_unresolved,
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
    for months, day in ((3, 60), (6, 120), (12, 252)):
        row[f"p_survive_{months}_months"] = (
            sum(
                (r.daily_breach_day is None or r.daily_breach_day > day)
                and (r.total_breach_day is None or r.total_breach_day > day)
                for r in results
            )
            / n
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


def policy_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["universe"],
                row["cost_scenario"],
                row["risk_per_trade"],
                row["max_portfolio_risk"],
                row["block_days"],
            )
        ].append(row)
    result = list(rows)
    for key, policies in grouped.items():
        pass_values = [row["p_pass_target_before_breach"] for row in policies]
        monthly_values = [row["monthly_median"] for row in policies]
        result.append(
            {
                "universe": key[0],
                "cost_scenario": key[1],
                "risk_per_trade": key[2],
                "max_portfolio_risk": key[3],
                "block_days": key[4],
                "policy": "ACROSS_POLICIES",
                "policy_count": len(policies),
                "p_pass_policy_p25": _q(pass_values, 0.25),
                "p_pass_policy_median": _q(pass_values, 0.5),
                "p_pass_policy_p75": _q(pass_values, 0.75),
                "p_pass_worst_policy": min(pass_values),
                "monthly_policy_p25": _q(monthly_values, 0.25),
                "monthly_policy_median": _q(monthly_values, 0.5),
                "monthly_policy_p75": _q(monthly_values, 0.75),
                "monthly_worst_policy": min(monthly_values),
                "positive_monthly_plateau_fraction": sum(
                    value > 0 for value in monthly_values
                )
                / len(monthly_values),
            }
        )
    return result


def matrix_filters(args):
    """Resolve optional diagnostic filters without filtering strategy policies."""
    scenarios = (
        tuple(item for item in COST_SCENARIOS if item[0] == args.cost_scenario)
        if args.cost_scenario
        else COST_SCENARIOS
    )
    risks = (args.risk_per_trade,) if args.risk_per_trade is not None else RISK_LEVELS
    caps = (
        (args.max_portfolio_risk,)
        if args.max_portfolio_risk is not None
        else PORTFOLIO_CAPS
    )
    return scenarios, risks, caps


def run(args):
    all_paths = [
        value
        for key, value in vars(args).items()
        if key.endswith(("_trades", "_candidate_events", "_audit", "_stage4c"))
        and value
    ]
    all_paths += [
        value
        for key, value in vars(args).items()
        if key.endswith("_cost_profile") and value
    ]
    all_paths.append(args.output_dir)
    safe = [reject_sealed_path(value) for value in all_paths]
    del safe
    instruments = UNIVERSES[args.universe]
    artifacts = {}
    for instrument in instruments:
        prefix = instrument.lower()
        profile_path_value = getattr(args, f"{prefix}_cost_profile")
        if profile_path_value is None:
            raise MonteCarloInputError(
                f"explicit {instrument} cost-profile path is required"
            )
        profile_path = reject_sealed_path(profile_path_value)
        profile = CostProfile.load(profile_path)
        values = tuple(
            getattr(args, f"{prefix}_{suffix}")
            for suffix in ("trades", "candidate_events", "audit", "stage4c")
        )
        if any(value is None for value in values):
            raise MonteCarloInputError(
                f"explicit {instrument} trades, candidate-events, audit, and "
                "Stage4C paths are required"
            )
        paths = tuple(reject_sealed_path(value) for value in values)
        artifacts[instrument] = (
            paths[0],
            load_candidates(instrument, paths[1]),
            _audit(
                instrument,
                paths[2],
                paths[3],
                paths[0],
                paths[1],
                profile.sha256,
            ),
            profile,
            profile_path,
        )
    out = reject_sealed_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    portfolio_rows = []
    samples = bootstrap_samples(
        args.paths, args.seed, args.block_days, args.horizon_days
    )
    scenarios, risks, caps = matrix_filters(args)
    for scenario, statistic, slippage in scenarios:
        loaded = {
            i: load_trades(
                i,
                artifacts[i][0],
                artifacts[i][2],
                artifacts[i][3],
                statistic,
                slippage,
                artifacts[i][1],
            )
            for i in instruments
        }
        common = set.intersection(
            *(set(e.policy_id for e in loaded[i]) for i in instruments)
        )
        if not common:
            raise MonteCarloInputError("no common/comparable execution policies")
        if len(common) != 32:
            raise MonteCarloInputError(
                f"expected 32 common frozen strategy cells, found {len(common)}"
            )
        for policy in sorted(common):
            events = [
                e for i in instruments for e in loaded[i] if e.policy_id == policy
            ]
            if scenario == "p90+0.25":
                portfolio_rows.extend(asdict(e) for e in events)
            rules = AccountRules(
                args.starting_balance,
                args.profit_target,
                args.max_daily_loss,
                args.max_total_loss,
                args.daily_reset_timezone,
            )
            matrix = simulate_matrix(
                events,
                samples=samples,
                horizon=args.horizon_days,
                risks=risks,
                caps=caps,
                rules=rules,
            )
            for (risk, cap), result in matrix.items():
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
    _write_csv(out / "ou-monte-carlo-policies.csv", policy_summaries(rows))
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
            i: {
                "trades": hashlib.sha256(values[0].read_bytes()).hexdigest(),
                "cost_profile": values[3].sha256,
            }
            for i, values in artifacts.items()
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
        for suffix in ("trades", "candidate-events", "audit", "stage4c"):
            p.add_argument(f"--{instrument}-{suffix}")
        p.add_argument(f"--{instrument}-cost-profile")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--paths",
        type=int,
        default=50000,
        help="Monte Carlo paths; use 2000/5000 for diagnostics, 50000 for final",
    )
    p.add_argument("--seed", type=int, default=20240930)
    p.add_argument("--block-days", type=int, choices=(1, 5), default=5)
    p.add_argument("--cost-scenario", choices=tuple(item[0] for item in COST_SCENARIOS))
    p.add_argument("--risk-per-trade", type=float, choices=RISK_LEVELS)
    p.add_argument("--max-portfolio-risk", type=float, choices=PORTFOLIO_CAPS)
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
    if args.paths <= 0 or not 252 <= args.horizon_days <= 262:
        raise SystemExit(
            "--paths must be positive and --horizon-days must be within 252..262"
        )
    try:
        run(args)
    except (MonteCarloInputError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
