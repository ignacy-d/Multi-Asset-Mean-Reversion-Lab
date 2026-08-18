import inspect
import lzma
import struct
from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from mr_lab.data import PriceBasis, Timeframe, VolumeSemantics, resample_bars
from mr_lab.providers.dukascopy import (
    acquire,
    acquire_verification_sample,
    build_url,
)
from mr_lab.providers.dukascopy_bi5 import (
    build_dataset_metadata,
    candidate_canonicalization_audit,
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
    get_candidate_instrument_spec,
    get_instrument_spec,
    require_verified,
)
from mr_lab.replication import run_frozen_replication
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
def test_candidate_specs_and_verification_decoding(instrument, scale, encoded, decoded):
    spec = get_candidate_instrument_spec(instrument)
    assert spec.price_scale == scale
    assert spec.decode_price(encoded) == decoded
    assert spec.to_json() == get_candidate_instrument_spec(instrument.lower()).to_json()
    audit = candidate_canonicalization_audit(payload(encoded), DAY, instrument)
    assert audit["instrument"] == instrument
    assert audit["first_bar_ohlc"] == [decoded] * 4
    assert audit["price_basis"] == PriceBasis.BID.value
    assert audit["volume_semantics"] == VolumeSemantics.QUOTE_ACTIVITY.value
    assert audit["first_m1_open_time"] == datetime(2024, 1, 2, tzinfo=UTC).isoformat()
    assert (
        audit["final_m1_close_time"] == datetime(2024, 1, 2, 1, tzinfo=UTC).isoformat()
    )


def test_supported_universe_urls_and_rejections():
    assert SUPPORTED_INSTRUMENTS == ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDJPY")
    assert build_url("EURUSD", DAY).endswith("/EURUSD/2024/00/02/BID_candles_min_1.bi5")
    for instrument in SUPPORTED_INSTRUMENTS[1:]:
        with pytest.raises(InstrumentSpecError, match="not production-verified"):
            build_url(instrument, DAY)
    with pytest.raises(InstrumentSpecError, match="unsupported instrument"):
        get_instrument_spec("NZDUSD")
    with pytest.raises(InstrumentSpecError, match="price_scale"):
        ProviderInstrumentSpec("TEST", "TEST", 100, 5)
    candidate = ProviderInstrumentSpec(
        "TEST", "TEST", 100_000, 5, decoding_verified=False
    )
    with pytest.raises(InstrumentSpecError, match="not production-verified"):
        require_verified(candidate)


def test_verification_status_changes_eligibility_not_decoding_facts():
    candidate = get_candidate_instrument_spec("USDJPY")
    verified = replace(candidate, decoding_verified=True)
    assert not candidate.decoding_verified
    assert require_verified(verified) is verified
    assert candidate.as_dict() | {"decoding_verified": True} == verified.as_dict()
    assert candidate.decode_price(143_210) == verified.decode_price(143_210)


def test_normal_acquisition_rejects_candidate_before_network(tmp_path):
    called = False

    def getter(_url, _timeout):
        nonlocal called
        called = True

    with pytest.raises(InstrumentSpecError, match="not production-verified"):
        acquire(tmp_path, DAY, instrument="GBPUSD", getter=getter)
    assert not called
    with pytest.raises(InstrumentSpecError, match="not production-verified"):
        parse_m1_bid_bars(payload(127_456), DAY, "GBPUSD")
    with pytest.raises(MultiDayAssemblyError, match="not production-verified"):
        DailyPayload(DAY, payload(127_456), "GBPUSD")
    with pytest.raises(InstrumentSpecError, match="not production-verified"):
        run_frozen_replication(tmp_path, tmp_path / "results", "GBPUSD")


def test_verification_acquisition_uses_candidate_provider_path(tmp_path):
    urls = []
    raw = payload(127_456)

    def getter(url, _timeout):
        urls.append(url)
        return 200, raw

    acquired, _ = acquire_verification_sample(tmp_path, DAY, "GBPUSD", getter=getter)
    assert acquired.read_bytes() == raw
    assert urls == [
        "https://datafeed.dukascopy.com/datafeed/GBPUSD/2024/00/02/"
        "BID_candles_min_1.bi5"
    ]


def test_zero_volume_point_in_time_prefix_and_generic_resampling():
    raw = payload(110_123, volume=0.0)
    bars = parse_m1_bid_bars(raw, DAY, "EURUSD")
    assert bars[0].volume == 0.0
    observations = build_research_observations(bars, DEFAULT_SESSION_SPEC)
    prefix = build_research_observations(bars[:20], DEFAULT_SESSION_SPEC)
    assert observations[:20] == prefix
    for timeframe, count in (("5m", 12), ("15m", 4), ("1h", 1)):
        result = resample_bars(bars, Timeframe(timeframe))
        assert len(result.bars) == count
        assert all(bar.instrument == "EURUSD" for bar in result.bars)


def test_instrument_sensitive_identities_and_mixed_rejection(monkeypatch):
    import mr_lab.providers.instruments as instruments

    monkeypatch.setitem(
        instruments._SPECS,
        "GBPUSD",
        replace(get_candidate_instrument_spec("GBPUSD"), decoding_verified=True),
    )
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
