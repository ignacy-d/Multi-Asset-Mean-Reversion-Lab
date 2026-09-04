import hashlib
import json
import lzma
import statistics
import struct
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.providers.dukascopy_multiday import DailyPayload
from mr_lab.providers.dukascopy_range import build_corpus_manifest
from mr_lab.research import (
    ResearchSpec,
    build_forward_outcomes,
    build_research_observations,
)
from mr_lab.sessions import DEFAULT_SESSION_SPEC, SessionSpec, TimeWindow
from mr_lab.vwap_benchmark import (
    LEGACY_NORMALIZATION_DEFINITION,
    LEGACY_STRATEGY_SCHEMA_VERSION,
    NORMALIZATION_DEFINITION,
    RESET_MODE,
    VWAP_PRICE_DEFINITION,
    WEIGHT_SEMANTICS,
    VwapBenchmarkError,
    VwapStrategySpec,
    build_vwap_features,
    main,
    signal_direction,
    summarize_benchmark,
    typical_price,
)


def bar(
    opened: datetime,
    *,
    close: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1.0,
    timeframe: str = "5m",
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


def observations(*bars: Bar, spec: SessionSpec = DEFAULT_SESSION_SPEC):
    return build_research_observations(bars, spec)


def test_hlc3_and_weighted_vwap_activity_arithmetic() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = (
        bar(start, close=2, high=4, low=1, volume=2),
        bar(start + timedelta(minutes=5), close=5, high=7, low=3, volume=1),
        bar(
            start + timedelta(minutes=10), close=9, high=10, volume=0
        ),  # moving, zero weight
        bar(start + timedelta(minutes=15), close=9, volume=0),  # inactive flat filler
    )
    features = build_vwap_features(observations(*bars), DEFAULT_SESSION_SPEC, 2)
    london = [feature for feature in features if feature.anchor_session == "london"]

    assert typical_price(bars[0]) == pytest.approx(7 / 3)
    assert london[1].vwap == pytest.approx(((7 / 3) * 2 + 5) / 3)
    assert london[2].observation.is_research_active
    assert london[2].vwap == london[1].vwap
    assert not london[3].observation.is_research_active
    assert london[3].vwap == london[2].vwap


def test_session_reset_no_contamination_and_overlap_streams() -> None:
    custom = SessionSpec(
        "test",
        (
            TimeWindow("first", "UTC", time(8), time(10)),
            TimeWindow("second", "UTC", time(9), time(11)),
        ),
    )
    bars = (
        bar(datetime(2024, 1, 2, 8, tzinfo=UTC), close=10),
        bar(datetime(2024, 1, 2, 9, tzinfo=UTC), close=20),
        bar(datetime(2024, 1, 3, 8, tzinfo=UTC), close=30),
    )
    features = build_vwap_features(observations(*bars, spec=custom), custom, 2)
    overlap = [item for item in features if item.observation.bar is bars[1]]

    assert {item.anchor_session for item in overlap} == {"first", "second"}
    assert {item.anchor_session: item.vwap for item in overlap} == {
        "first": 15,
        "second": 20,
    }
    assert [item.vwap for item in features if item.observation.bar is bars[2]] == [30]


def test_dst_session_instance_reuses_historical_local_semantics() -> None:
    before = bar(datetime(2024, 3, 29, 8, tzinfo=UTC))
    after = bar(datetime(2024, 4, 1, 7, tzinfo=UTC))
    features = build_vwap_features(observations(before, after), DEFAULT_SESSION_SPEC, 2)
    london = [item for item in features if item.anchor_session == "london"]

    assert len(london) == 2
    assert london[0].session_instance == date(2024, 3, 29)
    assert london[1].session_instance == date(2024, 4, 1)


def test_prefix_invariance_deviation_and_trailing_volatility() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = tuple(
        bar(start + timedelta(minutes=5 * i), close=100 + i * i) for i in range(5)
    )
    prefix = build_vwap_features(observations(*bars[:4]), DEFAULT_SESSION_SPEC, 2)
    full = build_vwap_features(observations(*bars), DEFAULT_SESSION_SPEC, 2)
    changed = build_vwap_features(
        observations(*bars[:4], bar(start + timedelta(minutes=20), close=999)),
        DEFAULT_SESSION_SPEC,
        2,
    )

    assert prefix == full[: len(prefix)] == changed[: len(prefix)]
    latest = next(
        item
        for item in full
        if item.observation.bar is bars[3] and item.anchor_session == "london"
    )
    returns = (bars[2].close / bars[1].close - 1, bars[3].close / bars[2].close - 1)
    expected_volatility = (
        (returns[0] - sum(returns) / 2) ** 2 + (returns[1] - sum(returns) / 2) ** 2
    ) ** 0.5
    assert latest.relative_deviation == pytest.approx(latest.price / latest.vwap - 1)
    assert latest.rolling_volatility == pytest.approx(expected_volatility)
    assert latest.vwap_deviation_z == pytest.approx(
        latest.relative_deviation / expected_volatility
    )


@pytest.mark.parametrize("timeframe", ["5m", "15m", "1h"])
def test_only_exactly_adjacent_bars_enter_volatility(timeframe: str) -> None:
    duration = Timeframe(timeframe).duration
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = tuple(
        bar(start + duration * i, close=100 + i * i, timeframe=timeframe)
        for i in range(4)
    )
    output = build_vwap_features(observations(*bars), DEFAULT_SESSION_SPEC, 2)
    by_bar = {
        item.observation.bar: item
        for item in output
        if item.anchor_session == "london"
    }
    expected = statistics.stdev(
        (bars[2].close / bars[1].close - 1, bars[3].close / bars[2].close - 1)
    )
    assert by_bar[bars[3]].rolling_volatility == pytest.approx(expected)


@pytest.mark.parametrize("missing_bars", [1, 3, 12 * 24 * 3])
def test_gap_resets_and_requires_complete_new_return_window(missing_bars: int) -> None:
    duration = Timeframe("5m").duration
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    offsets = (0, 1, 2 + missing_bars, 3 + missing_bars, 4 + missing_bars)
    bars = tuple(
        bar(start + duration * offset, close=100 + index * index)
        for index, offset in enumerate(offsets)
    )
    output = build_vwap_features(observations(*bars), DEFAULT_SESSION_SPEC, 2)
    volatility = {
        item.observation.bar: item.rolling_volatility
        for item in output
        if item.anchor_session == "london"
    }
    assert volatility[bars[2]] is None
    assert volatility[bars[3]] is None
    assert volatility[bars[4]] is not None


def test_inactive_observation_resets_return_window() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = tuple(
        bar(
            start + timedelta(minutes=5 * i),
            close=value,
            volume=0 if i == 2 else 1,
        )
        for i, value in enumerate((100, 101, 101, 102, 104, 107))
    )
    output = build_vwap_features(observations(*bars), DEFAULT_SESSION_SPEC, 2)
    volatility = [
        item.rolling_volatility for item in output if item.anchor_session == "london"
    ]
    assert volatility[2:5] == [None, None, None]
    assert volatility[5] is not None


def test_v2_identity_does_not_reuse_historical_v1_identity() -> None:
    current = VwapStrategySpec(20, 1.5)
    historical = {
        **current.as_dict(),
        "strategy_schema_version": LEGACY_STRATEGY_SCHEMA_VERSION,
        "normalized_deviation_definition": LEGACY_NORMALIZATION_DEFINITION,
    }
    historical_id = "sha256:" + hashlib.sha256(
        json.dumps(historical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert historical_id == (
        "sha256:6aca0e037562736c266ebcf1e8a518383b1a32961dad47082c02d423ad3ce7eb"
    )
    assert current.strategy_spec_id != historical_id


def test_warmup_and_strict_signal_boundaries() -> None:
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = tuple(
        bar(start + timedelta(minutes=5 * i), close=value)
        for i, value in enumerate((100, 101, 99))
    )
    features = build_vwap_features(observations(*bars), DEFAULT_SESSION_SPEC, 2)
    london = [item for item in features if item.anchor_session == "london"]

    assert london[0].vwap_deviation_z is None
    assert london[1].vwap_deviation_z is None
    feature = london[2]
    object.__setattr__(feature, "vwap_deviation_z", -1.5)
    assert signal_direction(feature, 1.5) is None
    object.__setattr__(feature, "vwap_deviation_z", -1.5001)
    assert signal_direction(feature, 1.5).name == "LONG"
    object.__setattr__(feature, "vwap_deviation_z", 1.5001)
    assert signal_direction(feature, 1.5).name == "SHORT"
    object.__setattr__(feature, "vwap_deviation_z", 0.2)
    assert signal_direction(feature, 1.5) is None


def test_strategy_identity_is_canonical_sensitive_and_separate() -> None:
    first = VwapStrategySpec(20, 1.5)
    same = VwapStrategySpec(deviation_threshold=1.5, volatility_lookback=20)

    assert first.strategy_spec_id == same.strategy_spec_id
    assert first.strategy_spec_id != VwapStrategySpec(40, 1.5).strategy_spec_id
    assert first.strategy_spec_id != VwapStrategySpec(20, 2.0).strategy_spec_id
    assert "dataset_id" not in first.to_json()
    assert "session_spec_id" not in first.to_json()
    assert "research_spec_id" not in first.to_json()


@pytest.mark.parametrize(
    ("field", "implemented"),
    (
        ("vwap_price_definition", VWAP_PRICE_DEFINITION),
        ("weight_semantics", WEIGHT_SEMANTICS),
        ("reset_mode", RESET_MODE),
        ("normalized_deviation_definition", NORMALIZATION_DEFINITION),
    ),
)
def test_strategy_spec_rejects_unimplemented_methodology(
    field: str, implemented: str
) -> None:
    with pytest.raises(VwapBenchmarkError, match=f"unsupported {field}"):
        VwapStrategySpec(20, 1.5, **{field: f"not-{implemented}"})


@pytest.mark.parametrize("timeframe", ["5m", "15m", "1h"])
def test_timeframes_forward_reuse_signed_outcomes_and_summary(timeframe: str) -> None:
    duration = Timeframe(timeframe).duration
    start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    bars = tuple(
        bar(start + duration * i, close=value, timeframe=timeframe)
        for i, value in enumerate((100, 102, 99, 101))
    )
    obs = observations(*bars)
    research = ResearchSpec((duration,))
    outcomes = build_forward_outcomes(obs, research)
    features = list(build_vwap_features(obs, DEFAULT_SESSION_SPEC, 2))
    for feature in features:
        object.__setattr__(feature, "vwap_deviation_z", -2.0)
    strategy = VwapStrategySpec(2, 1.0)
    rows = summarize_benchmark(
        dataset_id="dataset",
        timeframe=Timeframe(timeframe),
        observations=obs,
        features=features,
        outcomes=outcomes,
        research_spec=research,
        strategy_spec=strategy,
    )
    london_all = next(
        row
        for row in rows
        if row["anchor_session"] == "london" and row["direction"] == "all"
    )

    assert london_all["signal_count"] >= london_all["valid_outcome_count"]
    assert london_all["unavailable_count"] >= 1
    assert london_all["research_spec_id"] != london_all["strategy_spec_id"]
    assert london_all["mean_signed_return"] is not None
    assert london_all["monthly_signal_frequency"] == london_all["signal_count"]


RECORD = struct.Struct(">5if")


def synthetic_corpus(path: Path) -> None:
    day = date(2024, 1, 2)
    records = []
    for minute in range(480, 660):
        price = 100_000 + (minute % 7)
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


def test_offline_cli_is_deterministic_and_rejects_holdout_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    synthetic_corpus(corpus)
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    assert (
        main(["--corpus-dir", str(corpus), "--timeframe", "M5", "--output", str(first)])
        == 0
    )
    assert (
        main(
            ["--corpus-dir", str(corpus), "--timeframe", "M5", "--output", str(second)]
        )
        == 0
    )
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())

    manifest = json.loads((corpus / "corpus-manifest.json").read_text())
    manifest["successful_component_dates"] = ["2025-01-02"]
    (corpus / "corpus-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "mr_lab.vwap_benchmark.load_offline_corpus",
        lambda _: pytest.fail("must not load 2025"),
    )
    with pytest.raises(VwapBenchmarkError, match="2024"):
        main(["--corpus-dir", str(corpus), "--timeframe", "M5", "--output", str(first)])


def test_h1_cli_emits_only_compatible_default_horizons(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    synthetic_corpus(corpus)
    output = tmp_path / "h1.json"

    assert (
        main(
            ["--corpus-dir", str(corpus), "--timeframe", "H1", "--output", str(output)]
        )
        == 0
    )

    rows = json.loads(output.read_text())
    assert {row["forward_horizon_seconds"] for row in rows} == {3600, 7200}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("requested_end_date", "2025-01-01"),
        ("confirmed_absent_dates", ["2025-01-01"]),
    ),
)
def test_discovery_guard_checks_all_manifest_dates_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str | list[str],
) -> None:
    corpus = tmp_path / "corpus"
    synthetic_corpus(corpus)
    manifest_path = corpus / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "mr_lab.vwap_benchmark.load_offline_corpus",
        lambda _: pytest.fail("loader must not read BI5 for a non-2024 manifest"),
    )

    with pytest.raises(VwapBenchmarkError, match="2024"):
        main(
            [
                "--corpus-dir",
                str(corpus),
                "--timeframe",
                "M5",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
