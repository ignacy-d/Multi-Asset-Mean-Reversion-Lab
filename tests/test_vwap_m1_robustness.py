import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pytest
from test_vwap_benchmark import synthetic_corpus

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics, resample_bars
from mr_lab.research import build_research_observations
from mr_lab.sessions import DEFAULT_SESSION_SPEC, SessionSpec, TimeWindow
from mr_lab.vwap_benchmark import (
    DEFAULT_THRESHOLDS,
    DEFAULT_VOLATILITY_LOOKBACKS,
    VwapBenchmarkError,
    VwapStrategySpec,
    build_vwap_features,
)
from mr_lab.vwap_m1_robustness import (
    CANONICAL_M1,
    NATIVE_TIMEFRAME,
    VwapRobustnessStrategySpec,
    build_canonical_m1_vwap_features,
    main,
)


def bar(
    opened: datetime,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1,
    timeframe: str = "1m",
) -> Bar:
    duration = Timeframe(timeframe).duration
    return Bar(
        "EURUSD",
        Timeframe(timeframe),
        opened,
        opened + duration,
        opened + duration,
        close,
        high if high is not None else close,
        low if low is not None else close,
        close,
        PriceBasis.BID,
        volume,
        VolumeSemantics.QUOTE_ACTIVITY,
    )


def obs(bars, spec=DEFAULT_SESSION_SPEC):
    return build_research_observations(bars, spec)


def features(m1, timeframe="5m", spec=DEFAULT_SESSION_SPEC):
    research = resample_bars(tuple(m1), Timeframe(timeframe)).bars
    return build_canonical_m1_vwap_features(obs(m1, spec), obs(research, spec), spec, 2)


@pytest.mark.parametrize(("timeframe", "count"), (("5m", 5), ("15m", 15), ("1h", 60)))
def test_exact_completed_timestamp_sampling_and_hlc3_arithmetic(
    timeframe: str, count: int
) -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    m1 = [bar(start + timedelta(minutes=i), 10, high=13, low=7) for i in range(count)]
    m1.append(bar(start + timedelta(minutes=count), 1000))

    sampled = features(m1[:count], timeframe)
    london = next(item for item in sampled if item.anchor_session == "london")

    assert london.observation.available_at == start + timedelta(minutes=count)
    assert london.vwap == 10  # HLC3 is (13 + 7 + 10) / 3.
    assert (
        features(m1, timeframe)[0].vwap == london.vwap
    )  # incomplete future M1 ignored


def test_weighting_activity_missing_and_zero_volume_semantics() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    m1 = [
        bar(start, 2, high=4, low=1, volume=2),
        bar(start + timedelta(minutes=1), 5, high=7, low=3),
        bar(start + timedelta(minutes=2), 9, high=10, volume=0),  # active
        bar(start + timedelta(minutes=3), 9, volume=0),  # inactive filler
        bar(start + timedelta(minutes=4), 9, volume=0),
    ]
    london = next(item for item in features(m1) if item.anchor_session == "london")

    assert obs(m1)[2].is_research_active
    assert not obs(m1)[3].is_research_active
    assert london.vwap == pytest.approx(((7 / 3) * 2 + 5) / 3)

    unavailable = [
        bar(start + timedelta(minutes=i), 10 + i, volume=0) for i in range(5)
    ]
    assert (
        next(
            item for item in features(unavailable) if item.anchor_session == "london"
        ).vwap
        is None
    )


def test_overlap_streams_reset_and_session_instance_isolation() -> None:
    spec = SessionSpec(
        "test",
        (
            TimeWindow("first", "UTC", time(8), time(10)),
            TimeWindow("second", "UTC", time(8, 2), time(11)),
        ),
    )
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    day_one = [bar(start + timedelta(minutes=i), 10 + i) for i in range(10)]
    day_two = [bar(start + timedelta(days=1, minutes=i), 30) for i in range(10)]
    output = features(day_one + day_two, spec=spec)
    first_bar = [x for x in output if x.observation.bar.open_time.date().day == 2]
    second_bar = [x for x in output if x.observation.bar.open_time.date().day == 3]

    overlap = [x for x in first_bar if x.observation.bar.open_time.minute == 5]
    assert {x.anchor_session: x.vwap for x in overlap} == {
        "first": 14.5,
        "second": 15.5,
    }
    assert {x.vwap for x in second_bar} == {30}


def test_future_m1_and_research_prefixes_cannot_change_history() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    m1 = [bar(start + timedelta(minutes=i), 100 + i * i) for i in range(15)]
    research = resample_bars(tuple(m1), Timeframe("5m")).bars
    prefix = build_canonical_m1_vwap_features(
        obs(m1[:10]), obs(research[:2]), DEFAULT_SESSION_SPEC, 2
    )
    full = build_canonical_m1_vwap_features(
        obs(m1), obs(research), DEFAULT_SESSION_SPEC, 2
    )

    assert prefix == full[: len(prefix)]
    changed = m1[:10] + [bar(start + timedelta(minutes=10 + i), 999) for i in range(5)]
    assert (
        build_canonical_m1_vwap_features(
            obs(changed), obs(research), DEFAULT_SESSION_SPEC, 2
        )[: len(prefix)]
        == prefix
    )


def test_dst_membership_is_inherited_from_stage_1c() -> None:
    before = [
        bar(datetime(2024, 3, 29, 8, tzinfo=UTC) + timedelta(minutes=i), 10)
        for i in range(5)
    ]
    after = [
        bar(datetime(2024, 4, 1, 7, tzinfo=UTC) + timedelta(minutes=i), 20)
        for i in range(5)
    ]
    london = [x for x in features(before + after) if x.anchor_session == "london"]
    assert [x.session_instance.isoformat() for x in london] == [
        "2024-03-29",
        "2024-04-01",
    ]


def test_robustness_identity_is_deterministic_source_sensitive_and_separate() -> None:
    canonical = VwapRobustnessStrategySpec(20, 1.5, CANONICAL_M1)
    same = VwapRobustnessStrategySpec(20, 1.5, CANONICAL_M1)
    native = VwapRobustnessStrategySpec(20, 1.5, NATIVE_TIMEFRAME)

    assert canonical.strategy_spec_id == same.strategy_spec_id
    assert canonical.strategy_spec_id != native.strategy_spec_id
    assert canonical.strategy_spec_id != VwapStrategySpec(20, 1.5).strategy_spec_id
    assert (
        VwapStrategySpec(20, 1.5).strategy_spec_id
        == VwapStrategySpec(20, 1.5).strategy_spec_id
    )
    with pytest.raises(VwapBenchmarkError, match="construction"):
        VwapRobustnessStrategySpec(20, 1.5, "daily")


def test_frozen_grid_and_native_feature_behavior_are_unchanged() -> None:
    assert DEFAULT_THRESHOLDS == (1.0, 1.5, 2.0, 2.5)
    assert DEFAULT_VOLATILITY_LOOKBACKS == (20, 40)
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    native = obs(
        [
            bar(start + timedelta(minutes=5 * i), 100 + i, timeframe="5m")
            for i in range(3)
        ]
    )
    assert build_vwap_features(native, DEFAULT_SESSION_SPEC, 2) == build_vwap_features(
        native, DEFAULT_SESSION_SPEC, 2
    )


def test_offline_execution_is_deterministic_no_network_and_rejects_2025_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    synthetic_corpus(corpus)
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    assert (
        main(["--corpus-dir", str(corpus), "--timeframe", "M5", "--output", str(one)])
        == 0
    )
    assert (
        main(["--corpus-dir", str(corpus), "--timeframe", "M5", "--output", str(two)])
        == 0
    )
    assert one.read_bytes() == two.read_bytes()
    rows = json.loads(one.read_text())
    assert {row["vwap_construction_source"] for row in rows} == {CANONICAL_M1}

    manifest_path = corpus / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["requested_end_date"] = "2025-01-01"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "mr_lab.vwap_m1_robustness.load_offline_corpus",
        lambda _: pytest.fail("component loader/network must not be reached"),
    )
    with pytest.raises(VwapBenchmarkError, match="2024"):
        main(["--corpus-dir", str(corpus), "--timeframe", "M5", "--output", str(one)])
