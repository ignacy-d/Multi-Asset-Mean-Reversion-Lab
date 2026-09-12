from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from mr_lab.ou_monte_carlo import (
    AccountRules,
    MonteCarloInputError,
    PortfolioEvent,
    account_risk_amount,
    block_bootstrap_indices,
    deduplicate_events,
    r_multiple,
    reject_sealed_path,
    replay_r_days,
)
from mr_lab.ou_monte_carlo_runner import _audit, load_trades
from mr_lab.stage4c import net_pips

NOW = datetime(2024, 1, 2, 10, tzinfo=UTC)


def test_r_calculation_and_account_risk_sizing():
    assert r_multiple(5, 10) == 0.5
    assert account_risk_amount(100_000, 0.005) == 500
    with pytest.raises(MonteCarloInputError, match="positive"):
        r_multiple(5, 0)


def test_challenge_target_daily_and_total_loss():
    target = replay_r_days([[(NOW, 10)]], 0.01, 0.01, AccountRules(profit_target=0.1))
    assert target.target_day == 1
    daily = replay_r_days([[(NOW, -6)]], 0.01, 0.01, AccountRules())
    assert daily.daily_breach
    total = replay_r_days([[(NOW, -11)]], 0.01, 0.01, AccountRules())
    assert total.total_breach


def test_block_bootstrap_preserves_blocks_and_seed():
    first = block_bootstrap_indices(10, 8, 5, 7)
    assert first == block_bootstrap_indices(10, 8, 5, 7)
    assert all(b == (a + 1) % 10 for a, b in pairwise(first[:5]))


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


def test_cross_asset_simultaneous_events_respect_risk_cap():
    result = replay_r_days([[(NOW, 1), (NOW, 1)]], 0.01, 0.01, AccountRules())
    assert result.returns[-1] == pytest.approx(0.010025)


def test_transaction_costs():
    assert net_pips(3, "EURUSD", 0.3, 0.25, 0.5) == pytest.approx(1.95)


class _Profile:
    def costs(self, instrument, session, statistic):
        return 0.3, 0.5


def test_invalid_ou_provenance_and_missing_stop_fail_closed(tmp_path):
    trade = {
        "instrument": "EURUSD",
        "benchmark_family": "vwap",
        "signal_timeframe": "15m",
        "session": "london",
        "direction": "SHORT",
        "lookback": 20,
        "entry_mode": "immediate",
        "tp_target_fraction": 0.75,
        "sl_extension_fraction": 0.25,
        "time_stop_minutes": 60,
        "candidate_event_id": "x",
        "gross_return_pips_adverse_first": 2,
    }
    path = tmp_path / "trades.jsonl"
    import json

    path.write_text(json.dumps(trade) + "\n")
    audit = {
        "filter_family": "none",
        "filter_spec_id": "none",
        "process_spec_id": "none",
    }
    with pytest.raises(MonteCarloInputError, match="missing required quantity"):
        load_trades("EURUSD", path, audit, _Profile(), "p90", 0.25)


def test_invalid_ou_audit_provenance_rejected(tmp_path):
    import json

    audit = tmp_path / "audit.json"
    overlay = tmp_path / "overlay.json"
    audit.write_text(json.dumps({"instrument": "EURUSD", "filter_family": "none"}))
    overlay.write_text(json.dumps({"instrument": "EURUSD"}))
    with pytest.raises(MonteCarloInputError, match="invalid OU provenance"):
        _audit("EURUSD", audit, overlay)
