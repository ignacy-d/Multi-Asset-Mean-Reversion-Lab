import hashlib
import json
import lzma
import statistics
import struct
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from mr_lab.bollinger_benchmark import (
    CENTER_DEFINITION,
    DISPERSION_DEFINITION,
    INACTIVE_WINDOW_RULE,
    PRICE_DEFINITION,
    BollingerBenchmarkError,
    BollingerFeature,
    BollingerStrategySpec,
    build_bollinger_features,
    compatible_horizons,
    main,
    run_offline_benchmark,
    signal_direction,
    summarize_benchmark,
)
from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import build_corpus_manifest
from mr_lab.research import (
    ResearchSpec,
    build_forward_outcomes,
    build_research_observations,
)
from mr_lab.sessions import DEFAULT_SESSION_SPEC


def bar(
    opened: datetime,
    close: float,
    *,
    volume: float = 1.0,
    open_: float | None = None,
    timeframe: str = "5m",
) -> Bar:
    duration = Timeframe(timeframe).duration
    opening = close if open_ is None else open_
    return Bar(
        "EURUSD",
        Timeframe(timeframe),
        opened,
        opened + duration,
        opened + duration,
        opening,
        max(opening, close),
        min(opening, close),
        close,
        PriceBasis.BID,
        volume,
        VolumeSemantics.QUOTE_ACTIVITY,
    )


def observations(*bars: Bar):
    return build_research_observations(bars, DEFAULT_SESSION_SPEC)


def test_rolling_arithmetic_sample_stddev_z_current_bar_and_warmup() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = tuple(
        bar(start + i * timedelta(minutes=5), value)
        for i, value in enumerate((1, 2, 4))
    )
    features = build_bollinger_features(observations(*bars), 3)

    assert features[0].middle is None and features[1].middle is None
    assert features[2].middle == pytest.approx(7 / 3)
    assert features[2].sample_stddev == pytest.approx(statistics.stdev((1, 2, 4)))
    assert features[2].bollinger_z == pytest.approx(
        (4 - 7 / 3) / statistics.stdev((1, 2, 4))
    )


def test_zero_dispersion_is_unavailable() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    feature = build_bollinger_features(
        observations(*(bar(start + i * timedelta(minutes=5), 1) for i in range(3))), 3
    )[-1]
    assert feature.middle == 1
    assert feature.sample_stddev is None
    assert feature.bollinger_z is None


def test_inactive_window_and_moving_zero_activity_semantics() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = (
        bar(start, 1),
        bar(start + timedelta(minutes=5), 2),
        bar(start + timedelta(minutes=10), 2, volume=0),  # inactive flat
        bar(start + timedelta(minutes=15), 4, volume=0, open_=3),  # active moving
        bar(start + timedelta(minutes=20), 8),
        bar(start + timedelta(minutes=25), 16),
    )
    features = build_bollinger_features(observations(*bars), 3)

    assert len(features) == len(bars)
    assert not features[2].observation.is_research_active
    assert features[2].middle is None
    assert features[3].observation.is_research_active
    assert features[3].middle is None and features[4].middle is None
    assert features[5].middle == pytest.approx((4 + 8 + 16) / 3)


def test_prefix_invariance_future_addition_and_change() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    prefix_bars = tuple(bar(start + i * timedelta(minutes=5), i + 1) for i in range(4))
    prefix = build_bollinger_features(observations(*prefix_bars), 3)
    full = build_bollinger_features(
        observations(*prefix_bars, bar(start + timedelta(minutes=20), 5)), 3
    )
    changed = build_bollinger_features(
        observations(*prefix_bars, bar(start + timedelta(minutes=20), 999)), 3
    )
    assert prefix == full[:4] == changed[:4]


def test_missing_canonical_interval_does_not_compress_elapsed_time() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = (
        bar(start, 1),
        bar(start + timedelta(minutes=5), 2),
        bar(start + timedelta(minutes=15), 4),
        bar(start + timedelta(minutes=20), 8),
        bar(start + timedelta(minutes=25), 16),
    )
    features = build_bollinger_features(observations(*bars), 3)
    assert features[2].middle is None and features[3].middle is None
    assert features[4].middle == pytest.approx((4 + 8 + 16) / 3)


def test_strict_symmetric_signal_boundaries_and_inactive_guard() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    active, inactive = observations(
        bar(start, 1), bar(start + timedelta(minutes=5), 1, volume=0)
    )
    feature = BollingerFeature(active, 2, 1, 1, -1.5)
    assert signal_direction(feature, 1.5) is None
    assert signal_direction(BollingerFeature(active, 2, 1, 1, 1.5), 1.5) is None
    assert (
        signal_direction(BollingerFeature(active, 2, 1, 1, -1.51), 1.5).name == "LONG"
    )
    assert (
        signal_direction(BollingerFeature(active, 2, 1, 1, 1.51), 1.5).name == "SHORT"
    )
    assert signal_direction(BollingerFeature(active, 2, 1, 1, 0), 1.5) is None
    assert signal_direction(BollingerFeature(inactive, 2, 1, 1, 2), 1.5) is None


def test_strategy_identity_is_deterministic_sensitive_and_distinct() -> None:
    first = BollingerStrategySpec(20, 1.5)
    same = BollingerStrategySpec(deviation_threshold=1.5, rolling_lookback=20)
    assert first.strategy_spec_id == same.strategy_spec_id
    assert first.strategy_spec_id != BollingerStrategySpec(40, 1.5).strategy_spec_id
    assert first.strategy_spec_id != BollingerStrategySpec(20, 2.0).strategy_spec_id
    assert (
        not {"dataset_id", "session_spec_id", "research_spec_id"}
        & first.as_dict().keys()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("price_definition", PRICE_DEFINITION),
        ("center_definition", CENTER_DEFINITION),
        ("dispersion_definition", DISPERSION_DEFINITION),
        ("inactive_window_rule", INACTIVE_WINDOW_RULE),
    ),
)
def test_strategy_rejects_unsupported_methodology(field: str, value: str) -> None:
    with pytest.raises(BollingerBenchmarkError, match=f"unsupported {field}"):
        BollingerStrategySpec(20, 1.5, **{field: f"not-{value}"})


@pytest.mark.parametrize(
    ("timeframe", "seconds"),
    (
        ("5m", (900, 1800, 3600, 7200)),
        ("15m", (900, 1800, 3600, 7200)),
        ("1h", (3600, 7200)),
    ),
)
def test_compatible_horizons(timeframe: str, seconds: tuple[int, ...]) -> None:
    assert (
        tuple(int(x.total_seconds()) for x in compatible_horizons(Timeframe(timeframe)))
        == seconds
    )


def test_exact_clock_signed_outcomes_statistics_and_overlapping_contexts() -> None:
    # 13:00 UTC is simultaneously London and New York on this winter date.
    start = datetime(2024, 1, 2, 12, 55, tzinfo=UTC)
    bars = tuple(
        bar(start + i * timedelta(minutes=5), value)
        for i, value in enumerate((100, 90, 99, 80))
    )
    obs = observations(*bars)
    research = ResearchSpec((timedelta(minutes=5),))
    outcomes = build_forward_outcomes(obs, research)
    features = tuple(
        BollingerFeature(item, 2, 95, 1, z)
        for item, z in zip(obs, (0, -2, 2, -2), strict=True)
    )
    rows = summarize_benchmark(
        dataset_id="dataset",
        timeframe=Timeframe("5m"),
        observations=obs,
        features=features,
        outcomes=outcomes,
        research_spec=research,
        strategy_spec=BollingerStrategySpec(2, 1),
    )
    overall = next(
        row
        for row in rows
        if row["context_session"] is None and row["direction"] == "all"
    )
    london = next(
        row
        for row in rows
        if row["context_session"] == "london" and row["direction"] == "all"
    )
    new_york = next(
        row
        for row in rows
        if row["context_session"] == "new_york" and row["direction"] == "all"
    )
    expected = (99 / 90 - 1, -(80 / 99 - 1))
    assert overall["signal_count"] == 3
    assert overall["valid_outcome_count"] == 2
    assert overall["unavailable_count"] == 1  # exact future target absent
    assert overall["long_count"] == 2 and overall["short_count"] == 1
    assert overall["mean_signed_return"] == pytest.approx(statistics.fmean(expected))
    assert overall["median_signed_return"] == pytest.approx(statistics.median(expected))
    assert overall["win_rate"] == 1
    assert overall["stddev"] == pytest.approx(statistics.stdev(expected))
    assert overall["standard_error"] == pytest.approx(
        statistics.stdev(expected) / 2**0.5
    )
    assert overall["t_stat"] == pytest.approx(
        overall["mean_signed_return"] / overall["standard_error"]
    )
    assert london["signal_count"] == new_york["signal_count"] == 3
    assert len(obs[1].sessions.active_named_windows) >= 0
    assert (
        len(
            {
                overall[key]
                for key in (
                    "dataset_id",
                    "session_spec_id",
                    "research_spec_id",
                    "strategy_spec_id",
                )
            }
        )
        == 4
    )


RECORD = struct.Struct(">5if")


def synthetic_corpus(path: Path) -> None:
    day = date(2024, 1, 2)
    records = []
    for minute in range(480, 900):
        price = 100_000 + (minute % 13)
        records.append(RECORD.pack(minute * 60, price, price, price, price, 1.0))
    raw = lzma.compress(b"".join(records))
    manifest = build_corpus_manifest(day, day, [DailyPayload(day, raw)], [])
    path.mkdir()
    stem = path / f"EURUSD-{day.isoformat()}-M1-BID"
    stem.with_suffix(".bi5").write_bytes(raw)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "requested_date": day.isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    (path / "corpus-manifest.json").write_text(manifest.to_json())


def test_offline_cli_is_deterministic_and_does_not_use_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    synthetic_corpus(corpus)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("network used")
    )
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    assert (
        main(
            [
                "--corpus-dir",
                str(corpus),
                "--timeframe",
                "M5",
                "--output",
                str(first),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert (
        main(
            ["--corpus-dir", str(corpus), "--timeframe", "M5", "--output", str(second)]
        )
        == 0
    )
    assert first.read_bytes() == second.read_bytes()
    rows = json.loads(first.read_text())
    assert {row["bollinger_lookback"] for row in rows} == {20, 40}
    assert {row["threshold"] for row in rows} == {1.0, 1.5, 2.0, 2.5}


def test_discovery_guard_precedes_component_loading_and_rejects_2025(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    synthetic_corpus(corpus)
    manifest_path = corpus / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["requested_end_date"] = "2025-01-01"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "mr_lab.bollinger_benchmark.load_offline_corpus",
        lambda _: pytest.fail("components loaded"),
    )
    with pytest.raises(BollingerBenchmarkError, match="only the frozen 2024"):
        run_offline_benchmark(corpus, "M5")
