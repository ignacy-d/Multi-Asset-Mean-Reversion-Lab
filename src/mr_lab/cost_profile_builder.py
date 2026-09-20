"""Deterministic Stage4C-v2 spread-profile builder using explicit CSV paths only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mr_lab.sessions import DEFAULT_SESSION_SPEC, classify_timestamp
from mr_lab.stage4c_v2 import (
    FX_INSTRUMENTS,
    PROFILE_SCHEMA,
    SLIPPAGES,
    SPREAD_STATISTICS,
)


def _pick(row: dict[str, str], *names: str) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise ValueError(f"CSV missing required column: {'/'.join(names)}")


def _timestamp(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("CSV timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return (
        ordered[low]
        if low == high
        else ordered[low] * (high - position) + ordered[high] * (position - low)
    )


def _stats(rows: list[tuple[float, float, bool, bool]]) -> dict[str, float | int]:
    spreads = [max(row[0], 0.0) for row in rows]
    if not spreads:
        raise ValueError("each requested profile must contain observations")
    return {
        "mean_pips": sum(spreads) / len(spreads),
        "median_pips": _quantile(spreads, 0.5),
        **{
            f"p{int(p * 100)}_pips": _quantile(spreads, p)
            for p in (0.75, 0.9, 0.95, 0.99)
        },
        "max_pips": max(spreads),
        "sample_count": len(rows),
        "mean_mid": sum(x[1] for x in rows) / len(rows),
        "mean_inverse_mid": sum(1 / x[1] for x in rows) / len(rows),
        "negative_spread_rows_raw": sum(x[2] for x in rows),
        "zero_spread_rows_raw": sum(x[3] for x in rows),
    }


def inspect_csv(instrument: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read exactly ``path``; this function never searches its parent directory."""
    meta = FX_INSTRUMENTS[instrument]
    groups: dict[str, list[tuple[float, float, bool, bool]]] = {
        name: [] for name in ("overall", "asia", "london", "new_york")
    }
    timestamps: list[datetime] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            timestamp = _timestamp(_pick(row, "timestamp", "timestamp_utc", "time"))
            bid, ask = float(_pick(row, "bid")), float(_pick(row, "ask"))
            if not all(math.isfinite(x) and x > 0 for x in (bid, ask)):
                raise ValueError("bid and ask must be finite and positive")
            raw = (ask - bid) / meta.pip_size
            item = (raw, (ask + bid) / 2, raw < 0, raw == 0)
            timestamps.append(timestamp)
            groups["overall"].append(item)
            for session in classify_timestamp(
                timestamp, DEFAULT_SESSION_SPEC
            ).active_sessions:
                groups[session].append(item)
    if not timestamps:
        raise ValueError("CSV must contain data rows")
    profiles = {name: _stats(rows) for name, rows in groups.items() if rows}
    provenance = {
        "file_name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "row_count": len(timestamps),
        "first_timestamp_utc": min(timestamps).isoformat().replace("+00:00", "Z"),
        "last_timestamp_utc": max(timestamps).isoformat().replace("+00:00", "Z"),
        "pip_size": meta.pip_size,
        "negative_spread_rows_raw": profiles["overall"]["negative_spread_rows_raw"],
        "zero_spread_rows_raw": profiles["overall"]["zero_spread_rows_raw"],
    }
    return provenance, profiles


def _conversion_profiles(
    instrument: str,
    reference_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build authenticated session conversions without discovering references."""
    meta = FX_INSTRUMENTS[instrument]
    if meta.quote_currency == "USD":
        return {
            key: {"authenticated": True, "quote_currency_per_usd": 1.0}
            for key in ("overall", "asia", "london", "new_york")
        }
    reference_symbol = f"USD{meta.quote_currency}"
    reciprocal = False
    if meta.quote_currency == "GBP":
        reference_symbol, reciprocal = "GBPUSD", True
    source = reference_profiles.get(reference_symbol)
    if source is None:
        return {}
    profiles = {}
    for key, values in source.items():
        profiles[key] = {
            "authenticated": True,
            "quote_currency_per_usd": (
                float(values["mean_inverse_mid"])
                if reciprocal
                else float(values["mean_mid"])
            ),
            "reference_instrument": reference_symbol,
            "method": "mean_reciprocal_mid" if reciprocal else "mean_mid",
        }
    return profiles


def build_profile(
    inputs: dict[str, Path],
    reference_inputs: dict[str, Path] | None = None,
    *,
    non_usd_adjustment_rate: float | None = None,
) -> dict[str, Any]:
    """Build from only the supplied spread and conversion-reference CSV paths."""
    reference_inputs = reference_inputs or {}
    allowed_references = {"USDJPY", "USDCAD", "USDCHF", "GBPUSD"}
    unsupported = set(reference_inputs) - allowed_references
    if unsupported:
        raise ValueError(f"unsupported conversion reference: {sorted(unsupported)}")
    if non_usd_adjustment_rate is not None and not (
        math.isfinite(non_usd_adjustment_rate) and 0 <= non_usd_adjustment_rate < 1
    ):
        raise ValueError("non-USD adjustment rate must be finite and in [0, 1)")
    reference_profiles = {
        symbol: inspect_csv(symbol, path)[1]
        for symbol, path in sorted(reference_inputs.items())
    }
    inspected = {
        instrument: inspect_csv(instrument, path)
        for instrument, path in sorted(inputs.items())
    }
    for instrument, (_, profiles) in inspected.items():
        if FX_INSTRUMENTS[instrument].base_currency == "USD":
            reference_profiles.setdefault(instrument, profiles)
    result: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA,
        "account_currency": "USD",
        "null_session_cost_profile": "overall",
        "commission_usd_round_turn": 5.0,
        "cost_scenarios": {
            "spread_statistic": list(SPREAD_STATISTICS),
            "slippage_round_turn_pips": list(SLIPPAGES),
        },
        "instruments": {},
    }
    for instrument, (provenance, profiles) in inspected.items():
        conversion_profiles = _conversion_profiles(instrument, reference_profiles)
        quote_is_usd = FX_INSTRUMENTS[instrument].quote_currency == "USD"
        adjustment_defined = quote_is_usd or non_usd_adjustment_rate is not None
        result["instruments"][instrument] = {
            "metadata": FX_INSTRUMENTS[instrument].__dict__
            if hasattr(FX_INSTRUMENTS[instrument], "__dict__")
            else {
                "base_currency": FX_INSTRUMENTS[instrument].base_currency,
                "quote_currency": FX_INSTRUMENTS[instrument].quote_currency,
                "pip_size": FX_INSTRUMENTS[instrument].pip_size,
                "standard_lot_size": FX_INSTRUMENTS[instrument].standard_lot_size,
            },
            "raw_source": provenance,
            "spread_profiles": {
                name: values | {"authenticated": True}
                for name, values in profiles.items()
            },
            "commission_conversion": {
                "profiles": conversion_profiles,
            },
            "conversion_adjustment": {
                "authenticated": adjustment_defined,
                "defined": adjustment_defined,
                "rate": 0.0 if quote_is_usd else non_usd_adjustment_rate,
            },
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True, metavar="INSTRUMENT=CSV"
    )
    parser.add_argument(
        "--non-usd-adjustment-rate",
        type=float,
        help="explicit authenticated cTrader rate, currently 0.007 when applicable",
    )
    parser.add_argument(
        "--conversion-reference",
        action="append",
        default=[],
        metavar="INSTRUMENT=CSV",
        help="explicit authenticated USDJPY/USDCAD/USDCHF/GBPUSD reference",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs: dict[str, Path] = {}
    for item in args.input:
        instrument, separator, raw_path = item.partition("=")
        if not separator or instrument not in FX_INSTRUMENTS or instrument in inputs:
            parser.error("each --input must be a unique supported INSTRUMENT=CSV")
        inputs[instrument] = Path(raw_path)
    references: dict[str, Path] = {}
    for item in args.conversion_reference:
        instrument, separator, raw_path = item.partition("=")
        if not separator or instrument in references:
            parser.error("each conversion reference must be a unique INSTRUMENT=CSV")
        references[instrument] = Path(raw_path)
    args.output.write_text(
        json.dumps(
            build_profile(
                inputs,
                references,
                non_usd_adjustment_rate=args.non_usd_adjustment_rate,
            ),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
