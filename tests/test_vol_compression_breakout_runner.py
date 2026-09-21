from __future__ import annotations

import gzip
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data.models import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.stage4c_v2 import CostProfileV2, analyze
from mr_lab.vol_compression_breakout import VolCompressionEvent
from mr_lab.vol_compression_breakout_runner import (
    DEFAULT_COST_PROFILE,
    FROZEN_METHODOLOGY,
    FROZEN_METHODOLOGY_ID,
    HORIZONS,
    VolCompressionRunnerError,
    canonical_m15_bars,
    construct_outcomes,
    cost_session,
    load_authenticated_registry,
)

START = datetime(2024, 1, 2, tzinfo=UTC)


def m1(index: int, *, price: float = 1.0, volume: float = 1.0) -> Bar:
    start = START + timedelta(minutes=index)
    return Bar(
        instrument="EURUSD",
        timeframe=Timeframe("1m"),
        open_time=start,
        close_time=start + timedelta(minutes=1),
        available_at=start + timedelta(minutes=1),
        open=price,
        high=price,
        low=price,
        close=price,
        price_basis=PriceBasis.BID,
        volume=volume,
        volume_semantics=VolumeSemantics.QUOTE_ACTIVITY,
    )


def event(timestamp: datetime) -> VolCompressionEvent:
    return VolCompressionEvent(
        study_id="VOL-COMPRESSION-BREAKOUT-2024-v1",
        event_id="sha256:" + "0" * 64,
        timestamp=timestamp,
        instrument="EURUSD",
        direction="LONG",
        rv_1h_prior=0.0,
        rv_p20_prior=0.0,
        rv_percentile=0.1,
        box_high=1.0,
        box_low=1.0,
        box_width_raw=0.0,
        breakout_close=1.0,
        breakout_distance_raw=0.0,
        breakout_bar_log_return=0.0,
        current_quote_activity=1.0,
        prior_quote_activity=(1.0,) * 4,
    )


def test_registry_is_explicit_authenticated_and_discovery_only(tmp_path) -> None:
    registry = load_authenticated_registry()
    assert len(registry["instruments"]) == 9
    assert registry["requested_end_date"] == "2024-12-31"

    malformed = dict(registry)
    malformed["requested_end_date"] = "2025-12-31"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(malformed))
    with pytest.raises(
        VolCompressionRunnerError, match="registry authentication failed"
    ):
        load_authenticated_registry(path)


def test_m15_requires_exact_contiguous_nonpadding_m1() -> None:
    valid, audit = canonical_m15_bars(
        (m1(i, price=1 + i / 10_000) for i in range(15)), "EURUSD"
    )
    assert len(valid) == 1
    assert valid[0].open == 1.0
    assert valid[0].close == pytest.approx(1.0014)
    assert audit["valid_m15"] == 1

    missing, missing_audit = canonical_m15_bars(
        (m1(i) for i in range(15) if i != 7), "EURUSD"
    )
    assert missing == ()
    assert missing_audit["missing_observation"] == 1

    padded = [m1(i) for i in range(15)]
    padded[7] = replace(padded[7], volume=0.0)
    result, padding_audit = canonical_m15_bars(padded, "EURUSD")
    assert result == ()
    assert padding_audit["provider_padding"] == 1

    wrong, wrong_audit = canonical_m15_bars(
        padded[:7] + [m1(i) for i in range(7, 15)], "GBPUSD"
    )
    assert wrong == ()
    assert wrong_audit["instrument_mismatch"] == 1


def test_delayed_m1_availability_is_preserved_and_prefix_invariant() -> None:
    bars = [m1(i) for i in range(30)]
    delayed = bars[14]
    bars[14] = replace(delayed, available_at=delayed.close_time + timedelta(minutes=2))
    complete, _ = canonical_m15_bars(bars, "EURUSD")
    prefix, _ = canonical_m15_bars(bars[:15], "EURUSD")

    assert complete[0].available_at == START + timedelta(minutes=17)
    assert prefix == complete[:1]
    assert complete[0].available_at > complete[0].close_time

    bars[14] = replace(delayed, available_at=delayed.close_time + timedelta(minutes=20))
    with pytest.raises(VolCompressionRunnerError, match="ordering ambiguity"):
        canonical_m15_bars(bars, "EURUSD")


def test_exact_entry_and_all_frozen_exits_with_no_forward_search() -> None:
    bars = [m1(i, price=1 + i / 10_000) for i in range(100)]
    known = START + timedelta(minutes=15)
    outcomes, audit = construct_outcomes(event(known), bars)
    assert tuple(outcomes) == HORIZONS
    assert outcomes[15].entry_price == bars[15].open
    assert outcomes[15].exit_price == bars[29].close
    assert outcomes[30].exit_price == bars[44].close
    assert outcomes[60].exit_price == bars[74].close
    assert outcomes[30].gross_pips == pytest.approx(29.0)
    assert outcomes[30].gross_signed_bps_return == pytest.approx(
        math.log(bars[44].close / bars[15].open) * 10_000
    )
    assert all(item["reason"] is None for item in audit["horizons"].values())

    no_exact_entry = [bar for bar in bars if bar.open_time != known]
    missing, missing_audit = construct_outcomes(event(known), no_exact_entry)
    assert missing == {}
    assert missing_audit["entry_reason"] == "missing_observation"


def test_cost_session_uses_overall_for_overlap_and_outside() -> None:
    # London-only during winter.
    assert cost_session(datetime(2024, 1, 2, 9, tzinfo=UTC))[0] == "london"
    # London/New York overlap and outside all three both use overall.
    assert cost_session(datetime(2024, 1, 2, 14, tzinfo=UTC))[0] == "overall"
    assert cost_session(datetime(2024, 1, 2, 23, tzinfo=UTC))[0] == "overall"


def test_deterministic_gzip_contract() -> None:
    payload = b'{"a":1}\n'
    first = gzip.compress(payload, mtime=0)
    second = gzip.compress(payload, mtime=0)
    assert first == second


def test_default_cost_overlay_and_frozen_methodology_identity() -> None:
    assert (
        DEFAULT_COST_PROFILE.as_posix() == "configs/stage4c-ftmo-cost-profile-v2.json"
    )
    assert FROZEN_METHODOLOGY["rv_prior_return_count"] == 4
    assert FROZEN_METHODOLOGY["rv_prior_state_count"] == 1920
    assert FROZEN_METHODOLOGY["bootstrap_replicates"] == 10_000
    assert FROZEN_METHODOLOGY_ID.startswith("sha256:")

    outcome = construct_outcomes(
        event(START + timedelta(minutes=15)),
        [m1(i, price=1 + i / 10_000) for i in range(100)],
    )[0][30]
    report = analyze(
        [outcome],
        CostProfileV2.load(DEFAULT_COST_PROFILE),
        bootstrap_replicates=1,
        seed=20240920,
    )
    scenario = report["cost_scenarios"]["mean|0"]
    assert scenario["n"] == 0
    assert scenario["blocked"] == {"BLOCKED_MISSING_SPREAD_PROFILE": 1}
