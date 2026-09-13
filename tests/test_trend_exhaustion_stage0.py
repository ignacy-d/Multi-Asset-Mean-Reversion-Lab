from datetime import UTC, datetime, timedelta

from mr_lab.trend_exhaustion_stage0 import (
    ModuleAReference,
    calculate_overlap,
    classify,
    plateau_acceptance,
)


def _rows(neighbor: float = 0.6, count: int = 100):
    common = {
        "scope": "threshold",
        "event_count": count,
        "n_h60": count,
        "median_h60": 0.0,
        "instrument_counts": {"EURUSD": 50, "GBPUSD": 50},
        "instrument_expectancy_h60": {"EURUSD": 1.0, "GBPUSD": 0.5},
        "max_instrument_concentration": 0.5,
        "max_quarter_concentration": 0.5,
    }
    return tuple(
        common
        | {
            "threshold": threshold,
            "mean_h60": 1.0 if threshold == 1.5 else neighbor,
        }
        for threshold in (1.25, 1.5, 1.75)
    )


def test_plateau_and_deterministic_classification() -> None:
    assert plateau_acceptance(_rows())
    assert classify(_rows()) == "PASS"
    assert not plateau_acceptance(_rows(-1.0))
    assert classify(_rows(-1.0)) == "KILL"
    assert classify(_rows(count=99)) == "INCONCLUSIVE"


def test_session_diagnostics_cannot_change_classification() -> None:
    baseline = _rows()
    annotated = tuple(
        row
        | {
            "session_label": "london",
            "minutes_from_session_open": 120,
            "minutes_to_session_close": 420,
        }
        for row in baseline
    )

    assert classify(annotated) == classify(baseline) == "PASS"


class _Event:
    displacement_threshold = 1.5
    direction = None

    def __init__(self, instrument, timestamp):
        self.instrument = instrument
        self.signal_timestamp = timestamp


class _Observation:
    def __init__(self, event):
        self.event = event


def test_module_a_overlap_is_same_instrument_and_exact_windows() -> None:
    now = datetime(2024, 2, 1, tzinfo=UTC)
    observations = (_Observation(_Event("EURUSD", now)),)
    references = (
        ModuleAReference("GBPUSD", now),
        ModuleAReference("EURUSD", now + timedelta(minutes=30)),
    )
    aggregate = calculate_overlap(observations, references)[0]
    assert aggregate["overlap_count_w15"] == 0
    assert aggregate["overlap_count_w30"] == 1
