import inspect
import lzma
import struct
from datetime import UTC, date, datetime

import pytest

from mr_lab.data import PriceBasis, Timeframe, VolumeSemantics, resample_bars
from mr_lab.providers.dukascopy import build_url
from mr_lab.providers.dukascopy_bi5 import (
    build_dataset_metadata,
    parse_m1_bid_bars,
)
from mr_lab.providers.dukascopy_multiday import (
    DailyPayload,
    MultiDayAssemblyError,
    assemble_daily_payloads,
)
from mr_lab.providers.dukascopy_range import (
    RangeAcquisitionError,
    acquire_range,
    build_corpus_manifest,
)
from mr_lab.providers.instruments import (
    SUPPORTED_INSTRUMENTS,
    InstrumentSpecError,
    ProviderInstrumentSpec,
    get_instrument_spec,
)
from mr_lab.research import build_research_observations
from mr_lab.sessions import DEFAULT_SESSION_SPEC

DAY = date(2024, 1, 2)


def payload(encoded_price: int, volume: float = 1.0) -> bytes:
    records = b"".join(
        struct.pack(
            ">5if",
            offset,
            encoded_price,
            encoded_price,
            encoded_price,
            encoded_price,
            volume,
        )
        for offset in range(0, 3600, 60)
    )
    return lzma.compress(records)


@pytest.mark.parametrize(
    ("instrument", "scale", "encoded", "decoded"),
    (
        ("EURUSD", 100_000, 110_123, 1.10123),
        ("GBPUSD", 100_000, 127_456, 1.27456),
        ("USDJPY", 1_000, 143_210, 143.21),
        ("AUDUSD", 100_000, 68_765, 0.68765),
        ("AUDJPY", 1_000, 98_765, 98.765),
    ),
)
def test_verified_specs_and_decoding(instrument, scale, encoded, decoded):
    spec = get_instrument_spec(instrument)
    assert spec.price_scale == scale
    assert spec.decode_price(encoded) == decoded
    assert spec.to_json() == get_instrument_spec(instrument.lower()).to_json()
    bar = parse_m1_bid_bars(payload(encoded), DAY, instrument)[0]
    assert bar.instrument == instrument
    assert bar.open == decoded
    assert bar.price_basis is PriceBasis.BID
    assert bar.volume_semantics is VolumeSemantics.QUOTE_ACTIVITY
    assert bar.open_time == datetime(2024, 1, 2, tzinfo=UTC)
    assert bar.available_at == datetime(2024, 1, 2, 0, 1, tzinfo=UTC)


def test_supported_universe_urls_and_rejections():
    assert SUPPORTED_INSTRUMENTS == ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDJPY")
    for instrument in SUPPORTED_INSTRUMENTS:
        assert build_url(instrument, DAY) == (
            f"https://datafeed.dukascopy.com/datafeed/{instrument}/2024/00/02/"
            "BID_candles_min_1.bi5"
        )
    with pytest.raises(InstrumentSpecError, match="unsupported instrument"):
        get_instrument_spec("NZDUSD")
    with pytest.raises(InstrumentSpecError, match="price_scale"):
        ProviderInstrumentSpec("TEST", "TEST", 100, 5)
    with pytest.raises(InstrumentSpecError, match="unverified"):
        ProviderInstrumentSpec("TEST", "TEST", 100_000, 5, decoding_verified=False)


def test_zero_volume_point_in_time_prefix_and_generic_resampling():
    raw = payload(143_210, volume=0.0)
    bars = parse_m1_bid_bars(raw, DAY, "USDJPY")
    assert bars[0].volume == 0.0
    observations = build_research_observations(bars, DEFAULT_SESSION_SPEC)
    prefix = build_research_observations(bars[:20], DEFAULT_SESSION_SPEC)
    assert observations[:20] == prefix
    for timeframe, count in (("5m", 12), ("15m", 4), ("1h", 1)):
        result = resample_bars(bars, Timeframe(timeframe))
        assert len(result.bars) == count
        assert all(bar.instrument == "USDJPY" for bar in result.bars)


def test_instrument_sensitive_identities_and_mixed_rejection():
    raw = payload(100_000)
    eur_daily = build_dataset_metadata(raw, DAY, "EURUSD")
    gbp_daily = build_dataset_metadata(raw, DAY, "GBPUSD")
    assert eur_daily.dataset_id != gbp_daily.dataset_id
    eur = DailyPayload(DAY, raw, "EURUSD")
    gbp = DailyPayload(DAY, raw, "GBPUSD")
    assert (
        assemble_daily_payloads([eur]).metadata.dataset_id
        != assemble_daily_payloads([gbp]).metadata.dataset_id
    )
    assert (
        build_corpus_manifest(DAY, DAY, [eur], [], "EURUSD").corpus_id
        != build_corpus_manifest(DAY, DAY, [gbp], [], "GBPUSD").corpus_id
    )
    with pytest.raises(MultiDayAssemblyError, match="exactly one instrument"):
        assemble_daily_payloads([eur, DailyPayload(date(2024, 1, 3), raw, "GBPUSD")])


def test_frozen_strategy_modules_do_not_branch_on_instrument():
    import mr_lab.bollinger_benchmark as bollinger
    import mr_lab.research as research
    import mr_lab.vwap_benchmark as vwap
    import mr_lab.vwap_m1_robustness as robustness

    for module in (research, vwap, bollinger, robustness):
        source = inspect.getsource(module)
        assert "instrument ==" not in source
        assert "instrument-specific" not in source


def test_acquisition_rejects_2025_before_provider_access(tmp_path):
    called = False

    def acquire_day(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(RangeAcquisitionError, match="only 2024"):
        acquire_range(
            tmp_path,
            date(2025, 1, 1),
            date(2025, 1, 1),
            acquire_day=acquire_day,
        )
    assert not called
