"""Bounded provider-format verification; never runs research strategies."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from mr_lab.providers.dukascopy import acquire_verification_sample
from mr_lab.providers.dukascopy_bi5 import candidate_canonicalization_audit

VERIFICATION_DAY = date(2024, 1, 2)
VERIFICATION_INSTRUMENTS = ("GBPUSD", "USDJPY", "AUDUSD", "AUDJPY")
# Deliberately broad sanity bounds, not research thresholds or market forecasts.
PLAUSIBLE_PRICE_BOUNDS = {
    "GBPUSD": (0.5, 2.5),
    "USDJPY": (50.0, 250.0),
    "AUDUSD": (0.3, 1.5),
    "AUDJPY": (30.0, 200.0),
}


def verify(output_dir: Path) -> Path:
    """Acquire exactly one frozen 2024 day per candidate and audit its format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audits = []
    for instrument in VERIFICATION_INSTRUMENTS:
        raw_path, _ = acquire_verification_sample(
            output_dir, VERIFICATION_DAY, instrument
        )
        audit = candidate_canonicalization_audit(
            raw_path.read_bytes(), VERIFICATION_DAY, instrument
        )
        lower, upper = PLAUSIBLE_PRICE_BOUNDS[instrument]
        if not (
            lower <= audit["minimum_price"]
            and audit["maximum_price"] <= upper
            and audit["m1_count"] > 0
            and audit["volume"]["minimum"] >= 0
        ):
            raise ValueError(f"implausible provider sample for {instrument}")
        audits.append(audit)
    path = output_dir / "stage-3a-instrument-verification.json"
    path.write_text(
        json.dumps(audits, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify bounded 2024 BI5 samples")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"verification={verify(args.output_dir)}")


if __name__ == "__main__":
    main()
