from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe
from mr_lab.sessions import (
    DEFAULT_SESSION_SPEC,
    SessionSpec,
    SessionSpecError,
    TimeWindow,
    classify_bar,
    classify_bars,
    classify_timestamp,
)


def at(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def make_bar(open_time: datetime, price: float = 1.0) -> Bar:
    return Bar(
        instrument="EURUSD",
        timeframe=Timeframe("15m"),
        open_time=open_time,
        close_time=open_time + timedelta(minutes=15),
        available_at=open_time + timedelta(minutes=15),
        open=price,
        high=price,
        low=price,
        close=price,
        price_basis=PriceBasis.BID,
    )


@pytest.mark.parametrize(
    ("timestamp", "session"),
    [
        (at(2024, 1, 15, 8), "london"),  # GMT winter opening
        (at(2024, 7, 15, 7), "london"),  # BST summer opening
        (at(2024, 1, 15, 13), "new_york"),  # EST winter opening
        (at(2024, 7, 15, 12), "new_york"),  # EDT summer opening
    ],
)
def test_local_opening_clock_maps_to_historical_utc(
    timestamp: datetime, session: str
) -> None:
    classification = classify_timestamp(timestamp, DEFAULT_SESSION_SPEC)

    assert classification.session_membership[session]
    assert classification.local_times[session].hour == 8
    before = classify_timestamp(
        timestamp - timedelta(microseconds=1), DEFAULT_SESSION_SPEC
    )
    assert not before.session_membership[session]


def test_london_new_york_overlap_uses_actual_dst_membership() -> None:
    # During the March mismatch, New York is already on EDT while London is on GMT.
    mismatch = classify_timestamp(at(2024, 3, 18, 16), DEFAULT_SESSION_SPEC)
    after_uk_change = classify_timestamp(at(2024, 4, 8, 16), DEFAULT_SESSION_SPEC)

    assert mismatch.regime == "london_new_york_overlap"
    assert after_uk_change.regime == "new_york_only"
    assert classify_timestamp(at(2024, 3, 18, 11, 59), DEFAULT_SESSION_SPEC).regime == (
        "london_only"
    )
    assert classify_timestamp(at(2024, 4, 8, 11, 59), DEFAULT_SESSION_SPEC).regime == (
        "london_only"
    )


def test_autumn_dst_mismatch_is_historical_too() -> None:
    # The UK has returned to GMT, while New York remains on EDT until November 3.
    result = classify_timestamp(at(2024, 10, 28, 12), DEFAULT_SESSION_SPEC)

    assert result.regime == "london_new_york_overlap"
    assert result.local_times["london"].hour == 12
    assert result.local_times["new_york"].hour == 8


def test_half_open_boundaries_and_bar_open_time_rule() -> None:
    before = make_bar(at(2024, 1, 15, 7, 45))
    opening = make_bar(at(2024, 1, 15, 8))

    assert not classify_bar(before, DEFAULT_SESSION_SPEC).session_membership["london"]
    assert classify_bar(opening, DEFAULT_SESSION_SPEC).session_membership["london"]
    assert classify_timestamp(
        at(2024, 1, 15, 16, 59), DEFAULT_SESSION_SPEC
    ).session_membership["london"]
    assert not classify_timestamp(
        at(2024, 1, 15, 17), DEFAULT_SESSION_SPEC
    ).session_membership["london"]


def test_display_timezone_cannot_change_canonical_classification() -> None:
    timestamp = at(2024, 7, 15, 12, 30)
    expected = classify_timestamp(timestamp, DEFAULT_SESSION_SPEC)

    for display_timezone in (
        "UTC",
        "Europe/Warsaw",
        "Europe/London",
        "America/New_York",
    ):
        displayed = timestamp.astimezone(ZoneInfo(display_timezone))
        # Presentation is reversible; the classifier receives canonical UTC only.
        actual = classify_timestamp(displayed.astimezone(UTC), DEFAULT_SESSION_SPEC)
        assert actual == expected


def test_cross_midnight_candidate_start_inclusive_and_end_exclusive() -> None:
    # January ET is UTC-5: 20:00 ET is 01:00 UTC on the following UTC day.
    start = classify_timestamp(at(2024, 1, 3, 1), DEFAULT_SESSION_SPEC)
    near_end = classify_timestamp(at(2024, 1, 3, 4, 59), DEFAULT_SESSION_SPEC)
    end = classify_timestamp(at(2024, 1, 3, 5), DEFAULT_SESSION_SPEC)

    assert "asian_kz_20_00_et" in start.active_named_windows
    assert "asian_kz_20_00_et" in near_end.active_named_windows
    assert "asian_kz_20_00_et" not in end.active_named_windows


@pytest.mark.parametrize("timestamp", [at(2024, 1, 15, 7), at(2024, 7, 15, 6)])
def test_london_candidate_tracks_est_and_edt(timestamp: datetime) -> None:
    result = classify_timestamp(timestamp, DEFAULT_SESSION_SPEC)

    assert "london_kz_02_05_et" in result.active_named_windows
    assert result.local_times["london_kz_02_05_et"].hour == 2


def test_overlapping_named_candidates_and_lunch_are_independent() -> None:
    overlap = classify_timestamp(at(2024, 1, 15, 14), DEFAULT_SESSION_SPEC)
    lunch = classify_timestamp(at(2024, 1, 15, 16), DEFAULT_SESSION_SPEC)

    assert {"new_york_kz_07_10_et", "new_york_kz_0830_1100_et"} <= set(
        overlap.active_named_windows
    )
    assert "new_york_lunch_11_13_et" in lunch.active_named_windows
    assert overlap.session_membership["new_york"]
    without_windows = classify_timestamp(
        overlap.timestamp,
        replace(DEFAULT_SESSION_SPEC, named_windows=()),
    )
    assert overlap.session_membership == without_windows.session_membership
    assert overlap.regime == without_windows.regime


def test_generic_three_way_overlap_regime() -> None:
    windows = tuple(
        TimeWindow(name, "UTC", time(0), time(23))
        for name in ("charlie", "alpha", "bravo")
    )
    result = classify_timestamp(at(2024, 1, 1, 12), SessionSpec("test-v1", windows))

    assert result.active_sessions == ("alpha", "bravo", "charlie")
    assert result.regime == "alpha_bravo_charlie_overlap"


def test_weekdays_use_local_start_date_for_cross_midnight_window() -> None:
    monday_only = SessionSpec(
        "test-v1",
        (),
        (TimeWindow("overnight", "America/New_York", time(20), time(0), (0,)),),
    )

    monday_evening = classify_timestamp(at(2024, 1, 9, 1), monday_only)
    tuesday_end = classify_timestamp(at(2024, 1, 9, 5), monday_only)

    assert monday_evening.active_named_windows == ("overnight",)
    assert tuesday_end.active_named_windows == ()


def test_spec_identity_is_stable_and_declaration_order_independent() -> None:
    reordered = SessionSpec(
        DEFAULT_SESSION_SPEC.version,
        tuple(reversed(DEFAULT_SESSION_SPEC.major_sessions)),
        tuple(reversed(DEFAULT_SESSION_SPEC.named_windows)),
    )
    changed = replace(
        DEFAULT_SESSION_SPEC,
        named_windows=DEFAULT_SESSION_SPEC.named_windows[:-1],
    )

    assert reordered.to_json() == DEFAULT_SESSION_SPEC.to_json()
    assert reordered.session_spec_id == DEFAULT_SESSION_SPEC.session_spec_id
    assert changed.session_spec_id != DEFAULT_SESSION_SPEC.session_spec_id
    assert DEFAULT_SESSION_SPEC.session_spec_id.startswith("sha256:")


def test_classification_is_prefix_invariant_and_price_independent() -> None:
    bars = tuple(make_bar(at(2024, 1, 2, hour), float(hour)) for hour in range(8, 18))
    full = classify_bars(bars, DEFAULT_SESSION_SPEC)
    prefix = classify_bars(bars[:4], DEFAULT_SESSION_SPEC)
    changed_future = bars[:4] + tuple(
        make_bar(bar.open_time, 10_000.0) for bar in bars[4:]
    )

    assert prefix == full[:4]
    assert classify_bars(changed_future, DEFAULT_SESSION_SPEC)[:4] == full[:4]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (datetime(2024, 1, 1), "canonical UTC"),
        (datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/Warsaw")), "canonical UTC"),
    ],
)
def test_noncanonical_timestamp_is_rejected(value: datetime, message: str) -> None:
    with pytest.raises(SessionSpecError, match=message):
        classify_timestamp(value, DEFAULT_SESSION_SPEC)
