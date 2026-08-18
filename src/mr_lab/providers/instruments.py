"""Explicit provider decoding candidates and production eligibility."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from mr_lab.data import PriceBasis, Timeframe, VolumeSemantics

_M1 = Timeframe("1m")


class InstrumentSpecError(ValueError):
    """Raised when an instrument specification is unsupported or unsafe."""


@dataclass(frozen=True, slots=True)
class ProviderInstrumentSpec:
    """Provider-bound facts required to locate and decode one instrument."""

    instrument: str
    provider_symbol: str
    price_scale: int
    price_precision: int
    native_timeframe: Timeframe = _M1
    price_basis: PriceBasis = PriceBasis.BID
    volume_semantics: VolumeSemantics = VolumeSemantics.QUOTE_ACTIVITY
    provider: str = "Dukascopy"
    decoding_verified: bool = True

    def __post_init__(self) -> None:
        if (
            not self.instrument
            or self.instrument != self.instrument.strip().upper()
            or not self.provider_symbol
            or self.provider_symbol != self.provider_symbol.strip().upper()
        ):
            raise InstrumentSpecError("instrument symbols must be non-empty uppercase")
        if type(self.price_scale) is not int or self.price_scale <= 0:
            raise InstrumentSpecError("price_scale must be a positive integer")
        if (
            type(self.price_precision) is not int
            or self.price_precision < 0
            or self.price_scale != 10**self.price_precision
        ):
            raise InstrumentSpecError(
                "price_scale must equal 10 raised to price_precision"
            )
        if self.native_timeframe != _M1:
            raise InstrumentSpecError("only verified native M1 data is supported")
        if self.price_basis is not PriceBasis.BID:
            raise InstrumentSpecError("only verified BID data is supported")
        if self.volume_semantics is not VolumeSemantics.QUOTE_ACTIVITY:
            raise InstrumentSpecError("only QUOTE_ACTIVITY volume is supported")
        if self.provider != "Dukascopy":
            raise InstrumentSpecError(
                "only the verified Dukascopy provider is supported"
            )

    def as_dict(self) -> dict[str, object]:
        """Return stable decoding identity inputs."""
        return {
            "decoding_verified": self.decoding_verified,
            "instrument": self.instrument,
            "native_timeframe": self.native_timeframe.value,
            "price_basis": self.price_basis.value,
            "price_precision": self.price_precision,
            "price_scale": self.price_scale,
            "provider": self.provider,
            "provider_symbol": self.provider_symbol,
            "volume_semantics": self.volume_semantics.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    def decode_price(self, value: int) -> float:
        if type(value) is not int:
            raise InstrumentSpecError("encoded price must be an integer")
        decoded = value / self.price_scale
        if not math.isfinite(decoded) or decoded <= 0:
            raise InstrumentSpecError("decoded price must be finite and positive")
        return decoded


_SPECS = {
    # EURUSD is the historically verified Stage 1A contract. New-instrument
    # scales remain candidates until bounded real-provider verification succeeds.
    "EURUSD": ProviderInstrumentSpec("EURUSD", "EURUSD", 100_000, 5),
    "GBPUSD": ProviderInstrumentSpec(
        "GBPUSD", "GBPUSD", 100_000, 5, decoding_verified=False
    ),
    "USDJPY": ProviderInstrumentSpec(
        "USDJPY", "USDJPY", 1_000, 3, decoding_verified=False
    ),
    "AUDUSD": ProviderInstrumentSpec(
        "AUDUSD", "AUDUSD", 100_000, 5, decoding_verified=False
    ),
    "AUDJPY": ProviderInstrumentSpec(
        "AUDJPY", "AUDJPY", 1_000, 3, decoding_verified=False
    ),
}

SUPPORTED_INSTRUMENTS = tuple(_SPECS)


def get_candidate_instrument_spec(instrument: str) -> ProviderInstrumentSpec:
    """Resolve declared decoding facts, including verification-only candidates."""
    if not isinstance(instrument, str):
        raise InstrumentSpecError("instrument must be a string")
    symbol = instrument.strip().upper()
    try:
        return _SPECS[symbol]
    except KeyError as error:
        raise InstrumentSpecError(f"unsupported instrument: {instrument!r}") from error


def require_verified(spec: ProviderInstrumentSpec) -> ProviderInstrumentSpec:
    """Return a production-eligible spec or fail closed."""
    if not spec.decoding_verified:
        raise InstrumentSpecError(
            f"instrument decoding is not production-verified: {spec.instrument}"
        )
    return spec


def get_instrument_spec(instrument: str) -> ProviderInstrumentSpec:
    """Resolve only production-verified instrument decoding specifications."""
    return require_verified(get_candidate_instrument_spec(instrument))
