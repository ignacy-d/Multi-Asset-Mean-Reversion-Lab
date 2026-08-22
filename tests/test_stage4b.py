from datetime import UTC, datetime, timedelta

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe
from mr_lab.research import Direction
from mr_lab.stage4b import (
    ENTRY_MODES,
    SIGNAL_THRESHOLD,
    STAGE4B_METHODOLOGY_ID,
    EligibilityDecision,
    M1Index,
    SignalState,
    construct_eligible_entries,
    construct_entry,
    deduplicate_states,
    path_diagnostic,
    pip_size,
    reversion_fraction,
    simulate_exit,
    target_already_passed,
)

T = datetime(2024, 1, 2, 12, tzinfo=UTC)


def state(minute, z, qualifying=None, direction=Direction.LONG):
    if qualifying is None:
        qualifying = z < -2 if direction is Direction.LONG else z > 2
    return SignalState(
        "EURUSD",
        T + timedelta(minutes=minute),
        "vwap",
        Timeframe("5m"),
        "london",
        direction,
        20,
        1.0,
        1.01 if direction is Direction.LONG else 0.99,
        z,
        "corpus",
        "dataset",
        "spec",
        qualifying,
    )


def bar(minute, o=1.0, h=1.0, low=1.0, c=1.0, instrument="EURUSD"):
    close = T + timedelta(minutes=minute)
    h = max(h, o, c)
    low = min(low, o, c)
    return Bar(
        instrument,
        Timeframe("1m"),
        close - timedelta(minutes=1),
        close,
        close,
        o,
        h,
        low,
        c,
        PriceBasis.BID,
    )


def event():
    return deduplicate_states((state(0, -2.1),))[0]


def test_frozen_matrix_and_methodology_identity():
    assert SIGNAL_THRESHOLD == 2.0 and len(ENTRY_MODES) == 3
    assert STAGE4B_METHODOLOGY_ID.startswith("sha256:")


def test_continuous_above_is_one_candidate():
    assert len(deduplicate_states((state(0, -2.1), state(5, -2.2), state(10, -3)))) == 1


def test_real_below_threshold_rearms():
    assert (
        len(
            deduplicate_states(
                (
                    state(0, -2.1),
                    state(5, -2.2),
                    state(10, -1.9, False),
                    state(15, -2.1),
                )
            )
        )
        == 2
    )


def test_no_valid_below_no_false_rearm():
    assert len(deduplicate_states((state(0, -2.1), state(30, -2.1)))) == 1


def test_direction_normalized_geometry():
    e = event()
    assert reversion_fraction(e.signal, 1.005) == pytest.approx(0.5)
    short = deduplicate_states((state(0, 2.1, direction=Direction.SHORT),))[0]
    assert reversion_fraction(short.signal, 0.995) == pytest.approx(0.5)


def test_immediate_entry():
    x = construct_entry(event(), M1Index(()), "immediate")
    assert x.executed and x.price == 1.0


def test_reclaim_uses_first_completed_close_and_no_future():
    index = M1Index(
        (
            bar(1, h=1.01, low=0.99, c=0.999),
            bar(2, h=1.01, low=0.99, c=1.001),
            bar(3, c=1.002),
        )
    )
    x = construct_entry(event(), index, "m1-reclaim-p0")
    assert x.timestamp == T + timedelta(minutes=2)


def test_extension_and_reclaim_can_be_same_bar():
    x = construct_entry(
        event(),
        M1Index((bar(1, h=1.002, low=0.997, c=1.001),)),
        "extension25-then-reclaim-p0",
    )
    assert x.executed and x.time_from_extension25_to_entry == 0


def test_entry_timeouts():
    index = M1Index(tuple(bar(i, c=0.999) for i in range(1, 31)))
    assert not construct_entry(event(), index, "m1-reclaim-p0").executed
    assert not construct_entry(event(), index, "extension25-then-reclaim-p0").executed


def test_target_already_passed():
    x = construct_entry(
        event(), M1Index((bar(1, h=1.004, low=0.999, c=1.003),)), "m1-reclaim-p0"
    )
    assert target_already_passed(x, 0.25)


def test_tp_sl_and_ambiguity_first_exit():
    e = event()
    entry = construct_entry(e, M1Index(()), "immediate")
    tp = simulate_exit(
        e,
        entry,
        (bar(1, h=1.003, low=0.999, c=1.002),),
        tp_fraction=0.25,
        sl_fraction=0.5,
        time_stop_minutes=30,
    )
    sl = simulate_exit(
        e,
        entry,
        (bar(1, h=1.001, low=0.994, c=0.995),),
        tp_fraction=0.5,
        sl_fraction=0.5,
        time_stop_minutes=30,
    )
    both = simulate_exit(
        e,
        entry,
        (bar(1, h=1.003, low=0.997, c=1),),
        tp_fraction=0.25,
        sl_fraction=0.25,
        time_stop_minutes=30,
    )
    assert (
        tp.exit_reason == "tp"
        and sl.exit_reason == "sl"
        and both.exit_ordering == "ambiguous_same_minute"
    )
    assert tp.gross_return_pips_adverse_first == tp.gross_return_pips_favorable_first
    assert sl.gross_return_pips_adverse_first == sl.gross_return_pips_favorable_first
    assert both.gross_return_pips_adverse_first < 0
    assert both.gross_return_pips_favorable_first > 0
    assert tp.mfe_price_certain == pytest.approx(0.0025)
    assert sl.mae_price_certain == pytest.approx(0.005)


def test_time_stop_exact_close_and_missing_is_incomplete():
    e = event()
    entry = construct_entry(e, M1Index(()), "immediate")
    path = tuple(bar(i, c=1.0001) for i in range(1, 31))
    assert (
        simulate_exit(
            e, entry, path, tp_fraction=1, sl_fraction=None, time_stop_minutes=30
        ).exit_reason
        == "time_stop"
    )
    assert not simulate_exit(
        e, entry, path[:-1], tp_fraction=1, sl_fraction=None, time_stop_minutes=30
    ).complete


def test_exit_bar_extrema_are_bounds_not_claimed_exact():
    e = event()
    entry = construct_entry(e, M1Index(()), "immediate")
    x = simulate_exit(
        e,
        entry,
        (bar(1, h=1.009, low=0.999, c=1.003),),
        tp_fraction=0.25,
        sl_fraction=None,
        time_stop_minutes=30,
    )
    assert x.mfe_price_certain == pytest.approx(0.0025)
    assert x.mfe_price_upper_bound == pytest.approx(0.009)
    assert x.exit_bar_path_ambiguous


def test_path_diagnostic_same_minute():
    rows = path_diagnostic(event(), M1Index((bar(1, h=1.006, low=0.994, c=1),)))
    assert rows[0]["ordering_50"] == "ambiguous_same_minute"


def test_exact_pips():
    assert pip_size("EURUSD") == 0.0001 and pip_size("USDJPY") == 0.01


def test_m1_index_exact_path_and_duplicate_validation():
    index = M1Index((bar(1), bar(3)))
    assert [b.available_at for b in index.path(T, 3)] == [
        T + timedelta(minutes=1),
        T + timedelta(minutes=3),
    ]
    with pytest.raises(ValueError):
        M1Index((bar(1), bar(1)))


def test_filter_decision_baseline_shape():
    assert EligibilityDecision().filter_family == "none"


def test_filter_is_invoked_before_entry_construction():
    class Reject:
        calls = 0

        def evaluate(self, candidate):
            self.calls += 1
            return EligibilityDecision(False, "future-test", "reject-v1")

    filter_ = Reject()
    decision, entries = construct_eligible_entries(event(), M1Index(()), filter_)
    assert filter_.calls == 1 and not decision.eligible and entries == {}
