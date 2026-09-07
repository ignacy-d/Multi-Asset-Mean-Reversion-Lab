"""Optional independent-decoder comparison; no external runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass

from mr_lab.data import Bar


@dataclass(frozen=True, slots=True)
class OracleComparison:
    local_count: int
    reference_count: int
    timestamp_mismatches: int
    ohlc_mismatches: int
    missing_local: int
    missing_reference: int


def compare_bars(
    local: tuple[Bar, ...], reference: tuple[Bar, ...], *, tolerance: float = 1e-9
) -> OracleComparison:
    """Compare bounded, already-decoded fixtures from an independent tool."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    left, right = ({x.open_time: x for x in local}, {x.open_time: x for x in reference})
    common = left.keys() & right.keys()
    ohlc = sum(
        any(
            abs(a - b) > tolerance
            for a, b in zip(
                (left[t].open, left[t].high, left[t].low, left[t].close),
                (right[t].open, right[t].high, right[t].low, right[t].close),
                strict=True,
            )
        )
        for t in common
    )
    return OracleComparison(
        len(local),
        len(reference),
        len(left.keys() ^ right.keys()),
        ohlc,
        len(right.keys() - left.keys()),
        len(left.keys() - right.keys()),
    )
