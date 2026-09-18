import json
from pathlib import Path

import pytest

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import INSTRUMENTS as FROZEN_COST_INSTRUMENTS
from mr_lab.stage4c import adjusted_execution_pips
from mr_lab.stage4c_zero_cost import (
    INSTRUMENTS,
    MODE,
    OU_FILTER,
    VARIANTS,
    _validate,
    main,
    report,
    zero_cost_trade,
)

REGISTRY = Path("configs/fx-universe-2024-registry-v1.json")


def _trade(instrument, direction="LONG", complete=True):
    return {
        "instrument": instrument,
        "complete": complete,
        "direction": direction,
        "signal_timeframe": "15m",
        "session": "london",
        "benchmark_family": "vwap",
        "lookback": 20,
        "signal_threshold": 2.0,
        "filter_family": "none",
        "filter_spec_id": "none-v1",
        "entry_mode": "immediate",
        "tp_target_fraction": 0.75,
        "sl_extension_fraction": 0.25,
        "time_stop_minutes": 60,
        "gross_return_pips_adverse_first": -1.25,
        "gross_return_pips_favorable_first": 2.5,
        "mfe_pips_certain": 3.0,
        "mae_pips_certain": 1.5,
    }


def _runner(_corpus, _out, instrument, _registry, **kwargs):
    consumer = kwargs["trade_row_consumer"]
    row = _trade(instrument)
    if kwargs["eligibility_filter"]:
        spec = frozen_ou_eligibility_spec(OU_FILTER)
        row |= {
            "filter_family": "ornstein-uhlenbeck",
            "filter_spec_id": spec.filter_spec_id,
        }
    consumer(row)
    consumer(_trade(instrument, "SHORT"))
    consumer(_trade(instrument, complete=False))
    return {}


def test_zero_cost_is_exact_gross_and_has_no_conversion():
    row = _trade("EURGBP")
    result = zero_cost_trade(row)
    assert result["zero_cost_return_pips_adverse_first"] == -1.25
    assert result["zero_cost_return_pips_favorable_first"] == 2.5
    assert (
        result["zero_cost_return_pips_adverse_first"]
        == row["gross_return_pips_adverse_first"]
    )
    assert (
        result["zero_cost_return_pips_favorable_first"]
        == row["gross_return_pips_favorable_first"]
    )


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_each_new_pair_is_accepted_from_authenticated_registry(instrument):
    registry = _validate(REGISTRY, INSTRUMENTS)
    assert registry["instruments"][instrument]["instrument"] == instrument


@pytest.mark.parametrize(
    "instruments",
    [(), ("CHFUSD",), (*INSTRUMENTS[:-1], "CHFUSD"), (*INSTRUMENTS, "EURUSD")],
)
def test_instrument_subset_is_explicit_exact_and_never_substituted(instruments):
    with pytest.raises(ValueError, match="instruments"):
        _validate(REGISTRY, instruments)


def test_non_2024_registry_rejected_before_runner_or_corpus_inspection(tmp_path):
    raw = json.loads(REGISTRY.read_text())
    raw["requested_end_date"] = "2025-12-31"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(raw))
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="2024"):
        report(path, INSTRUMENTS, tmp_path / "out", runner=forbidden)
    assert not called


def test_mixed_instrument_rows_fail_closed(tmp_path):
    def mixed(_corpus, _out, instrument, _registry, **kwargs):
        kwargs["trade_row_consumer"](
            _trade("EURUSD" if instrument != "EURUSD" else "GBPUSD")
        )

    with pytest.raises(ValueError, match="mixed-instrument"):
        report(REGISTRY, INSTRUMENTS, tmp_path / "out", runner=mixed)


def test_frozen_identities_and_stage4c_a_are_unchanged():
    assert STAGE4B_METHODOLOGY_ID == (
        "sha256:cd89a2524ccc17e221ea921a5e7651f0131eeed4b5ec6332221f2603c60e36bf"
    )
    crossasset = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
    assert crossasset.direction == "SHORT"
    assert crossasset.filter_spec_id == (
        "sha256:0d061cc590454c56f3dca0d69cf79b05d3c0d53c23676a5657f9562da5fb6b36"
    )
    bidirectional = frozen_ou_eligibility_spec(OU_FILTER)
    assert bidirectional.direction == "BOTH"
    assert bidirectional.filter_spec_id == (
        "sha256:f1ae8109bce393d09fe21456b5c06a7fb06dc605efcea32112f5076011b9a912"
    )
    assert FROZEN_COST_INSTRUMENTS == (
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "AUDJPY",
    )
    assert adjusted_execution_pips(10.0, "USDJPY") == 9.93


def test_report_is_deterministic_and_complete(tmp_path):
    first = report(REGISTRY, INSTRUMENTS, tmp_path / "one", runner=_runner)
    second = report(REGISTRY, INSTRUMENTS, tmp_path / "two", runner=_runner)
    assert [path.read_bytes() for path in first] == [
        path.read_bytes() for path in second
    ]
    payload = json.loads(first[1].read_text())
    assert payload["mode"] == MODE
    assert payload["variants"] == list(VARIANTS)
    assert len(payload["rows"]) == 4 * 2 * 3
    audit = json.loads(first[3].read_text())
    assert audit["currency_conversion_adjustment"] is False
    assert audit["spread_pips"] == audit["commission_pips"] == 0
    assert audit["slippage_pips"] == 0


def test_cli_requires_explicit_instruments(tmp_path):
    with pytest.raises(SystemExit):
        main(["--registry", str(REGISTRY), "--output-dir", str(tmp_path)])
