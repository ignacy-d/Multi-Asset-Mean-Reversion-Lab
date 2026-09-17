"""Frozen-2024, zero-cost Stage 4C diagnostic primitives.

This module is intentionally independent of the Stage 4C-A cost transform.  In
particular, it never calls ``adjusted_execution_pips``: the evaluated value is
the immutable Stage 4B gross-pip value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import CONFIG_FIELDS, REGIME_FIELDS

MODE = "STAGE4C_ZERO_COST_DIAGNOSTIC"
SCHEMA_VERSION = "stage4c-zero-cost-diagnostic-v1"
_FX_TICKER = re.compile(r"[A-Z]{6}")
SEMANTICS = {
    "mode": MODE,
    "source": "complete-frozen-stage4b-trade-rows",
    "evaluated_values": "identity-from-stage4b-gross-pips",
    "spread_pips": 0,
    "commission_pips": 0,
    "slippage_pips": 0,
    "currency_conversion_adjustment": False,
    "configuration_fields": CONFIG_FIELDS,
    "regime_fields": REGIME_FIELDS,
    "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
}
METHODOLOGY_ID = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(SEMANTICS, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
)


class ZeroCostDiagnosticError(ValueError):
    """An input violates the zero-cost diagnostic contract."""


def validate_instrument(instrument: str) -> str:
    if not isinstance(instrument, str) or not _FX_TICKER.fullmatch(instrument):
        raise ZeroCostDiagnosticError(
            "instrument must be an explicit six-letter uppercase FX ticker"
        )
    if instrument[:3] == instrument[3:]:
        raise ZeroCostDiagnosticError("base and quote currencies must differ")
    return instrument


def transform_trade(row: dict, instrument: str) -> dict:
    """Copy one complete Stage 4B row and expose gross pips without adjustment."""
    validate_instrument(instrument)
    if row.get("instrument") != instrument:
        raise ZeroCostDiagnosticError("mixed or unexpected instrument in Stage 4B rows")
    if not row.get("candidate_event_id"):
        raise ZeroCostDiagnosticError("missing candidate identity")
    if not row.get("complete"):
        raise ZeroCostDiagnosticError("only complete Stage 4B rows can be evaluated")
    values = {}
    for ordering in ("adverse_first", "favorable_first"):
        source = f"gross_return_pips_{ordering}"
        value = row.get(source)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ZeroCostDiagnosticError(f"malformed gross value: {source}")
        if not math.isfinite(value):
            raise ZeroCostDiagnosticError(f"non-finite gross value: {source}")
        values[f"zero_cost_pips_{ordering}"] = value
    return row.copy() | values
