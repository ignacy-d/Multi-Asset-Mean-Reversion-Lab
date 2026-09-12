"""Fail-fast exact parity records for the frozen OU module."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from mr_lab.signals import AlphaSignal, SignalProvenance, SignalReference


@dataclass(frozen=True, slots=True)
class FrozenOuDecisionDetails:
    """Typed OU-only details kept outside the generic alpha envelope."""

    benchmark_family: str
    lookback: int
    p0: float
    e0: float
    z: float
    process_id: str
    process_spec_id: str
    process_status: str
    process_is_structurally_valid: bool
    process_invalid_reason: str | None
    ou_score: float | None
    half_life_minutes: float | None
    eligible: bool
    eligibility_reason: str
    filter_spec_id: str


@dataclass(frozen=True, slots=True)
class FrozenOuDecision:
    signal: AlphaSignal
    details: FrozenOuDecisionDetails


@dataclass(frozen=True, slots=True)
class ParityMismatch:
    index: int
    field: str
    reference: object
    replay: object


@dataclass(frozen=True, slots=True)
class ParityReport:
    reference_event_count: int
    replay_event_count: int
    mismatch_count: int
    first_mismatch: ParityMismatch | None
    signal_timestamps: tuple[str, ...]
    identity_mismatches: int
    process_state_mismatches: int
    eligibility_mismatches: int

    @property
    def passed(self) -> bool:
        return self.mismatch_count == 0


IDENTITY_FIELDS = {
    "signal.source_event_id",
    "signal.timestamp",
    "signal.direction",
    "details.p0",
    "details.e0",
    "details.z",
}
PROCESS_FIELDS = {
    "details.process_id",
    "details.process_spec_id",
    "details.process_status",
    "details.process_is_structurally_valid",
    "details.process_invalid_reason",
    "details.ou_score",
    "details.half_life_minutes",
}
ELIGIBILITY_FIELDS = {
    "details.eligible",
    "details.eligibility_reason",
    "details.filter_spec_id",
}


class ParityError(AssertionError):
    def __init__(self, report: ParityReport):
        super().__init__(f"research/replay parity failed: {report.first_mismatch}")
        self.report = report


def _fields(record: FrozenOuDecision):
    for section in ("signal", "details"):
        value = getattr(record, section)
        for field, item in asdict(value).items():
            yield f"{section}.{field}", item


def compare_event_streams(
    reference: Iterable[FrozenOuDecision],
    replay: Iterable[FrozenOuDecision],
    *,
    fail=True,
) -> ParityReport:
    """Compare every envelope and typed-details field, stopping at the first."""
    expected, actual = tuple(reference), tuple(replay)
    mismatch = None
    category = {"identity": 0, "process": 0, "eligibility": 0}
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected) or index >= len(actual):
            mismatch = ParityMismatch(index, "event_count", len(expected), len(actual))
            break
        for (field, left), (_, right) in zip(
            _fields(expected[index]), _fields(actual[index]), strict=True
        ):
            if left != right:
                mismatch = ParityMismatch(index, field, left, right)
                if field in IDENTITY_FIELDS:
                    category["identity"] = 1
                elif field in PROCESS_FIELDS:
                    category["process"] = 1
                elif field in ELIGIBILITY_FIELDS:
                    category["eligibility"] = 1
                break
        if mismatch:
            break
    report = ParityReport(
        len(expected),
        len(actual),
        int(mismatch is not None),
        mismatch,
        tuple(item.signal.timestamp.isoformat() for item in actual),
        category["identity"],
        category["process"],
        category["eligibility"],
    )
    if fail and not report.passed:
        raise ParityError(report)
    return report


def _load_jsonl(path: Path) -> tuple[FrozenOuDecision, ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            signal = value["signal"]
            signal["timestamp"] = datetime.fromisoformat(signal["timestamp"])
            if signal.get("reference") is not None:
                signal["reference"] = SignalReference(**signal["reference"])
            if signal.get("provenance") is not None:
                signal["provenance"] = SignalProvenance(**signal["provenance"])
            rows.append(
                FrozenOuDecision(
                    AlphaSignal(**signal), FrozenOuDecisionDetails(**value["details"])
                )
            )
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    """Compare explicitly supplied research and replay JSONL files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args(argv)
    report = compare_event_streams(
        _load_jsonl(args.reference), _load_jsonl(args.replay), fail=False
    )
    print(json.dumps(asdict(report), default=str, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
