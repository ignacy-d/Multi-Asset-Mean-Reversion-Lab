import json
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise, product
from types import SimpleNamespace

import pytest

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.ou_monte_carlo import (
    AccountRules,
    MonteCarloInputError,
    PortfolioEvent,
    SimTrade,
    account_risk_amount,
    block_bootstrap_indices,
    deduplicate_events,
    r_multiple,
    reject_sealed_path,
    replay_events,
    research_calendar_2024,
)
from mr_lab.ou_monte_carlo_runner import (
    _audit,
    _policy,
    load_trades,
    matrix_filters,
    metrics,
)
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import CostProfile, net_pips

NOW = datetime(2024, 1, 2, 10, tzinfo=UTC)


def test_r_calculation_and_account_risk_sizing():
    assert r_multiple(5, 10) == 0.5
    assert account_risk_amount(100_000, 0.005) == 500
    with pytest.raises(MonteCarloInputError, match="positive"):
        r_multiple(5, 0)


def test_challenge_first_pass_passes_before_later_breach():
    trades = [
        SimTrade(NOW, NOW + timedelta(hours=1), 10, "win"),
        SimTrade(
            NOW + timedelta(days=1), NOW + timedelta(days=1, hours=1), -30, "loss"
        ),
    ]
    result = replay_events(
        trades, [date(2024, 1, 2), date(2024, 1, 3)], 0.01, 0.01, AccountRules()
    )
    assert result.terminal_outcome == "PASS"
    assert result.terminal_day == 1
    assert result.total_breach_day == 2


def test_unresolved_is_not_breach():
    result = replay_events([], [date(2024, 1, 2)], 0.01, 0.01, AccountRules())
    row = metrics([result], AccountRules())
    assert row["p_unresolved_at_horizon"] == 1
    assert row["p_breach_before_target"] == 0


def test_daily_and_total_loss_breach():
    daily = replay_events(
        [SimTrade(NOW, NOW + timedelta(minutes=1), -6)],
        [NOW.date()],
        0.01,
        0.01,
        AccountRules(),
    )
    total = replay_events(
        [SimTrade(NOW, NOW + timedelta(minutes=1), -11)],
        [NOW.date()],
        0.01,
        0.01,
        AccountRules(),
    )
    assert daily.daily_breach
    assert total.total_breach


def test_block_bootstrap_preserves_blocks_and_seed():
    first = block_bootstrap_indices(10, 8, 5, 7)
    assert first == block_bootstrap_indices(10, 8, 5, 7)
    assert all(b == (a + 1) % 10 for a, b in pairwise(first[:5]))


def test_calendar_preserves_no_trade_days():
    calendar = research_calendar_2024()
    assert len(calendar) == 262
    result = replay_events([], calendar[:5], 0.01, 0.01, AccountRules())
    assert result.returns == (0, 0, 0, 0, 0)


def _event(net=1):
    return PortfolioEvent(
        "EURUSD",
        "x",
        NOW,
        NOW,
        NOW + timedelta(minutes=1),
        "p",
        2,
        1,
        net,
        1,
        net,
        "ou",
    )


def test_duplicate_event_handling():
    assert deduplicate_events([_event(), _event()]) == [_event()]
    with pytest.raises(MonteCarloInputError, match="incompatible duplicate"):
        deduplicate_events([_event(), _event(2)])


@pytest.mark.parametrize("value", ["/private/2025/input", r"C:\private\2025\input"])
def test_sealed_path_rejection(value):
    with pytest.raises(MonteCarloInputError, match="sealed"):
        reject_sealed_path(value)


def test_overlapping_positions_respect_portfolio_cap():
    trades = [
        SimTrade(NOW, NOW + timedelta(hours=2), 1, "first"),
        SimTrade(NOW + timedelta(hours=1), NOW + timedelta(hours=3), 1, "second"),
    ]
    result = replay_events(trades, [NOW.date()], 0.01, 0.01, AccountRules())
    assert result.returns[-1] == pytest.approx(0.01)


def test_entry_equity_sizes_after_realized_exit():
    trades = [
        SimTrade(NOW, NOW + timedelta(minutes=1), 1, "first"),
        SimTrade(NOW + timedelta(minutes=2), NOW + timedelta(minutes=3), 1, "second"),
    ]
    result = replay_events(trades, [NOW.date()], 0.01, 0.01, AccountRules())
    assert result.returns[-1] == pytest.approx(0.0201)


def test_timezone_daily_reset_uses_exit_timestamp():
    # Both UTC exits belong to the prior New York reset day.
    trades = [
        SimTrade(
            datetime(2023, 12, 31, 22, tzinfo=UTC),
            datetime(2024, 1, 2, 0, 30, tzinfo=UTC),
            -3,
            "a",
        ),
        SimTrade(
            datetime(2023, 12, 31, 22, tzinfo=UTC),
            datetime(2024, 1, 2, 2, 0, tzinfo=UTC),
            -3,
            "b",
        ),
    ]
    result = replay_events(
        trades,
        [date(2024, 1, 1)],
        0.01,
        0.02,
        AccountRules(daily_reset_timezone="America/New_York"),
    )
    assert result.daily_breach_day == 1


def test_survival_horizons_are_independent():
    base = replay_events([], research_calendar_2024()[:252], 0.01, 0.01, AccountRules())
    r70 = replace(base, daily_breach=True, daily_breach_day=70)
    r130 = replace(base, total_breach=True, total_breach_day=130)
    row = metrics([r70, r130], AccountRules())
    assert row["p_survive_3_months"] == 1
    assert row["p_survive_6_months"] == 0.5
    assert row["p_survive_12_months"] == 0


def test_full_policy_identity_does_not_pool_strategy_cells():
    base = {
        "benchmark_family": "vwap",
        "lookback": 20,
        "entry_mode": "immediate",
        "tp_target_fraction": 0.75,
        "sl_extension_fraction": 0.25,
        "time_stop_minutes": 60,
    }
    assert _policy(base) != _policy(base | {"lookback": 40})
    assert _policy(base) != _policy(base | {"benchmark_family": "vwap-canonical-m1"})


def test_transaction_costs():
    assert net_pips(3, "EURUSD", 0.3, 0.25, 0.5) == pytest.approx(1.95)


class _Profile:
    def costs(self, instrument, session, statistic):
        return 0.3, 0.5


def _trade_and_candidate():
    trade = {
        "instrument": "EURUSD",
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "SHORT",
        "lookback": 20,
        "signal_threshold": 2.0,
        "entry_mode": "immediate",
        "tp_target_fraction": 0.75,
        "sl_extension_fraction": 0.25,
        "time_stop_minutes": 60,
        "candidate_event_id": "x",
        "complete": True,
        "entry_wait_minutes": 5,
        "r_at_entry": -0.1,
        "exit_timestamp": "2024-01-02T11:00:00+00:00",
        "gross_return_pips_adverse_first": 2,
    }
    signal = {
        "instrument": "EURUSD",
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "lookback": 20,
        "session": "london",
        "direction": "SHORT",
        "threshold": 2.0,
        "signal_timestamp": "2024-01-02T10:00:00+00:00",
        "p0": 1.1,
        "d0": 0.002,
    }
    return trade, {"x": signal}


def test_candidate_trade_reconstruction_exact_stop_distance(tmp_path):
    trade, candidates = _trade_and_candidate()
    path = tmp_path / "trades.jsonl"
    path.write_text(json.dumps(trade) + "\n")
    audit = {
        "filter_family": "ornstein-uhlenbeck",
        "filter_spec_id": "f",
        "process_spec_id": "p",
    }
    event = load_trades("EURUSD", path, audit, _Profile(), "p90", 0.25, candidates)[0]
    # entry=1.1002 and stop=1.1005 for SHORT: exactly 3 pips.
    assert event.entry_timestamp == NOW + timedelta(minutes=5)
    assert event.initial_stop_distance_pips == pytest.approx(3)
    assert event.r == pytest.approx(0.95 / 3)


def test_missing_reconstruction_field_fails_closed(tmp_path):
    trade, candidates = _trade_and_candidate()
    del trade["r_at_entry"]
    path = tmp_path / "trades.jsonl"
    path.write_text(json.dumps(trade) + "\n")
    with pytest.raises(MonteCarloInputError, match="missing required quantity"):
        load_trades("EURUSD", path, {}, _Profile(), "p90", 0.25, candidates)


def test_provenance_mismatch_fails_closed(tmp_path):
    spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
    trades = tmp_path / "trades.jsonl"
    trades.write_text("{}\n")
    candidates = tmp_path / "candidate-events.jsonl"
    candidates.write_text("{}\n")
    audit = tmp_path / "audit.json"
    overlay = tmp_path / "overlay.json"
    audit.write_text(json.dumps({"instrument": "EURUSD", "filter_family": "none"}))
    overlay.write_text(
        json.dumps(
            {
                "instrument": "EURUSD",
                "filter_spec_id": spec.filter_spec_id,
                "process_spec_id": spec.process_spec.process_spec_id,
                "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
                "cost_profile_sha256": "cost",
                "source_trade_sha256": "bad",
            }
        )
    )
    with pytest.raises(MonteCarloInputError, match="invalid OU provenance"):
        _audit("EURUSD", audit, overlay, trades, candidates, "cost")


def _minimal_cost_profile(marker):
    return {
        "schema_version": "stage4c-ftmo-cost-profile-v1",
        "cost_scenarios": {
            "spread_statistic": ["mean", "p75", "p90", "p95"],
            "slippage_round_turn_pips": [0.0, 0.1, 0.25, 0.5],
        },
        "marker": marker,
    }


def _authenticated_audits(tmp_path, instrument, profile):
    spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
    trades = tmp_path / f"{instrument}-trades.jsonl"
    candidates = tmp_path / f"{instrument}-candidates.jsonl"
    trades.write_text("{}\n")
    candidates.write_text("{}\n")
    import hashlib

    trade_sha = hashlib.sha256(trades.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidates.read_bytes()).hexdigest()
    provenance = {
        "instrument": instrument,
        "filter_family": "ornstein-uhlenbeck",
        "filter_spec_id": spec.filter_spec_id,
        "process_spec_id": spec.process_spec.process_spec_id,
        "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
    }
    audit = tmp_path / f"{instrument}-audit.json"
    audit.write_text(
        json.dumps(
            provenance
            | {
                "eligibility_filter_spec": json.loads(json.dumps(asdict(spec))),
                "output_sha256": {
                    "trades.jsonl": trade_sha,
                    "candidate-events.jsonl": candidate_sha,
                },
            }
        )
    )
    overlay = tmp_path / f"{instrument}-stage4c.json"
    overlay.write_text(
        json.dumps(
            provenance
            | {
                "source_trade_sha256": {"source": trade_sha},
                "cost_profile_sha256": profile.sha256,
            }
        )
    )
    return audit, overlay, trades, candidates


def test_instruments_authenticate_different_cost_profile_shas(tmp_path):
    profiles = []
    for instrument in ("EURUSD", "GBPUSD"):
        path = tmp_path / f"{instrument}-cost.json"
        path.write_text(json.dumps(_minimal_cost_profile(instrument)))
        profile = CostProfile.load(path)
        profiles.append(profile)
        audit, overlay, trades, candidates = _authenticated_audits(
            tmp_path, instrument, profile
        )
        _audit(
            instrument,
            audit,
            overlay,
            trades,
            candidates,
            profile.sha256,
        )
    assert profiles[0].sha256 != profiles[1].sha256


def test_filtered_diagnostic_retains_all_32_strategy_policies():
    args = SimpleNamespace(
        cost_scenario="p90+0.25",
        risk_per_trade=0.0035,
        max_portfolio_risk=0.015,
    )
    scenarios, risks, caps = matrix_filters(args)
    policies = {
        _policy(
            dict(
                zip(
                    (
                        "benchmark_family",
                        "lookback",
                        "entry_mode",
                        "tp_target_fraction",
                        "sl_extension_fraction",
                        "time_stop_minutes",
                    ),
                    values,
                    strict=True,
                )
            )
        )
        for values in product(
            ("vwap", "vwap-canonical-m1"),
            (20, 40),
            ("immediate",),
            (0.75, 1.0),
            (0.25, 0.5),
            (60, 120),
        )
    }
    assert scenarios == (("p90+0.25", "p90", 0.25),)
    assert risks == (0.0035,)
    assert caps == (0.015,)
    assert len(policies) == 32
