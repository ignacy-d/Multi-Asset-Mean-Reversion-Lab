"""Bounded real-provider verification for declared Dukascopy candidates.

This module never runs research strategies and accepts no substitute dates.
Candidate declarations remain distinct from production eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from mr_lab.data import Timeframe, resample_bars
from mr_lab.providers.dukascopy import acquire_verification_sample, build_candidate_url
from mr_lab.providers.dukascopy_bi5 import (
    candidate_canonicalization_audit,
    parse_candidate_m1_bid_bars,
)
from mr_lab.providers.instruments import get_candidate_instrument_spec

VERIFICATION_DAYS = (
    date(2024, 1, 2),
    date(2024, 6, 3),
    date(2024, 10, 1),
)
VERIFICATION_INSTRUMENTS = ("USDCAD", "USDCHF", "NZDUSD", "EURGBP")
VERIFICATION_REPORT_SCHEMA_VERSION = "fx-universe-candidate-verification-v1"

# Broad format-sanity bounds, not alpha thresholds or market forecasts.
PLAUSIBLE_PRICE_BOUNDS = {
    "USDCAD": (0.5, 2.0),
    "USDCHF": (0.5, 1.5),
    "NZDUSD": (0.3, 1.5),
    "EURGBP": (0.5, 1.5),
}


class CandidateVerificationError(ValueError):
    """Raised when the exact bounded provider contract is not satisfied."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_request(instrument: str, days: Sequence[date]) -> tuple[date, ...]:
    spec = get_candidate_instrument_spec(instrument)
    requested = tuple(days)
    if spec.instrument not in VERIFICATION_INSTRUMENTS:
        raise CandidateVerificationError(
            "instrument is not an expanded-universe candidate"
        )
    if requested != VERIFICATION_DAYS:
        raise CandidateVerificationError(
            "candidate verification requires the exact three authorized 2024 dates"
        )
    return requested


def build_url_for_candidate(instrument: str, day: date) -> str:
    """Build a candidate URL without granting production eligibility."""
    if day.year != 2024:
        raise CandidateVerificationError("candidate verification accepts only 2024")
    if instrument.strip().upper() not in VERIFICATION_INSTRUMENTS:
        raise CandidateVerificationError(
            "instrument is not an expanded-universe candidate"
        )
    return build_candidate_url(instrument, day)


def verify_candidate(
    output_dir: Path,
    instrument: str,
    *,
    days: Sequence[date] = VERIFICATION_DAYS,
    acquire_sample: Callable[..., tuple[Path, Path]] = acquire_verification_sample,
) -> Path:
    """Acquire and audit one candidate on exactly the authorized provider dates."""
    requested = _validate_request(instrument, days)
    spec = get_candidate_instrument_spec(instrument)
    candidate_dir = output_dir / spec.instrument
    samples: list[dict[str, object]] = []
    lower, upper = PLAUSIBLE_PRICE_BOUNDS[spec.instrument]

    for day in requested:
        raw_path, provenance_path = acquire_sample(candidate_dir, day, spec.instrument)
        payload = raw_path.read_bytes()
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CandidateVerificationError(
                "invalid acquisition provenance"
            ) from error
        audit = candidate_canonicalization_audit(payload, day, spec.instrument)
        bars = parse_candidate_m1_bid_bars(payload, day, spec.instrument)
        resamples = audit.get("resamples")
        if not isinstance(resamples, dict):
            raise CandidateVerificationError("invalid resampling audit")
        resampling_consistent = True
        for name, timeframe in (("m5", "5m"), ("m15", "15m"), ("h1", "1h")):
            result = resample_bars(bars, Timeframe(timeframe))
            expected_components = (
                Timeframe(timeframe).duration // Timeframe("1m").duration
            )
            accounted_components = len(result.bars) * expected_components + sum(
                window.observed_components for window in result.incomplete_windows
            )
            summary = resamples.get(name)
            if not isinstance(summary, dict):
                raise CandidateVerificationError("invalid resampling audit")
            resampling_consistent = resampling_consistent and (
                accounted_components == len(bars)
                and summary["count"] == len(result.bars)
                and summary["incomplete_window_count"] == len(result.incomplete_windows)
            )
        if not resampling_consistent:
            raise CandidateVerificationError("resampling consistency failure")
        expected_url = build_url_for_candidate(spec.instrument, day)
        expected_provenance = {
            "canonical_instrument": spec.instrument,
            "provider_symbol": spec.provider_symbol,
            "price_scale": spec.price_scale,
            "price_precision": spec.price_precision,
            "source_timeframe": "M1",
            "price_side": "BID",
            "requested_date": day.isoformat(),
            "requested_url": expected_url,
            "http_status": 200,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if not isinstance(provenance, dict) or any(
            provenance.get(key) != value for key, value in expected_provenance.items()
        ):
            raise CandidateVerificationError("provider provenance contract mismatch")
        volume = audit["volume"]
        if not isinstance(volume, dict):  # pragma: no cover - internal audit contract
            raise CandidateVerificationError("invalid audit volume summary")
        minimum_price = audit["minimum_price"]
        maximum_price = audit["maximum_price"]
        m1_count = audit["m1_count"]
        minimum_volume = volume.get("minimum")
        if not (
            isinstance(minimum_price, int | float)
            and not isinstance(minimum_price, bool)
            and isinstance(maximum_price, int | float)
            and not isinstance(maximum_price, bool)
            and isinstance(m1_count, int)
            and not isinstance(m1_count, bool)
            and isinstance(minimum_volume, int | float)
            and not isinstance(minimum_volume, bool)
        ):
            raise CandidateVerificationError("invalid numeric audit summary")
        if not (
            lower <= minimum_price
            and maximum_price <= upper
            and m1_count > 0
            and minimum_volume >= 0
        ):
            raise CandidateVerificationError(
                f"implausible provider sample for {spec.instrument} on {day}"
            )
        if audit["instrument"] != spec.instrument:
            raise CandidateVerificationError("canonical instrument mismatch")
        if (
            audit["provider"] != spec.provider
            or audit["price_basis"] != "bid"
            or audit["source_timezone"] != "UTC"
        ):
            raise CandidateVerificationError("provider source semantics mismatch")
        samples.append(
            {
                "audit": audit,
                "provider_response": expected_provenance,
                "resampling_consistent": resampling_consistent,
            }
        )

    report_body: dict[str, object] = {
        "report_schema_version": VERIFICATION_REPORT_SCHEMA_VERSION,
        "candidate_spec": spec.as_dict(),
        "authorized_dates": [day.isoformat() for day in VERIFICATION_DAYS],
        "plausible_price_bounds": [lower, upper],
        "sample_count": len(samples),
        "samples": samples,
        "verification_passed": True,
    }
    report_id = (
        "sha256:" + hashlib.sha256(_canonical_json(report_body).encode()).hexdigest()
    )
    report = {"verification_report_id": report_id, **report_body}
    report_path = candidate_dir / "verification-report.json"
    if report_path.exists():
        raise CandidateVerificationError("refusing to overwrite a verification report")
    report_path.write_text(_canonical_json(report) + "\n", encoding="utf-8")
    return report_path


def verify(output_dir: Path) -> Path:
    """Produce four reports plus one deterministic aggregate report."""
    reports = []
    for instrument in VERIFICATION_INSTRUMENTS:
        path = verify_candidate(output_dir, instrument)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    aggregate_body = {
        "report_schema_version": "fx-universe-candidate-verification-index-v1",
        "authorized_dates": [day.isoformat() for day in VERIFICATION_DAYS],
        "candidate_reports": [
            {
                "instrument": report["candidate_spec"]["instrument"],
                "verification_report_id": report["verification_report_id"],
                "verification_passed": report["verification_passed"],
            }
            for report in reports
        ],
    }
    aggregate = {
        "verification_index_id": (
            "sha256:"
            + hashlib.sha256(_canonical_json(aggregate_body).encode()).hexdigest()
        ),
        **aggregate_body,
    }
    path = output_dir / "verification-index.json"
    if path.exists():
        raise CandidateVerificationError("refusing to overwrite verification index")
    path.write_text(_canonical_json(aggregate) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify expanded FX candidates on exact bounded 2024 dates"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"verification_index={verify(args.output_dir)}")


if __name__ == "__main__":
    main()
