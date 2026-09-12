"""Fail-fast, exact field parity for research and replay event streams."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from mr_lab.signals import AlphaSignal


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


IDENTITY_FIELDS = {"source_event_id", "signal_timestamp", "direction"}
PROCESS_FIELDS = {
    "process_id",
    "process_spec_id",
    "process_status",
    "process_is_structurally_valid",
    "process_invalid_reason",
    "ou_score",
    "half_life_minutes",
}
ELIGIBILITY_FIELDS = {"eligible", "eligibility_reason", "filter_spec_id"}


class ParityError(AssertionError):
    def __init__(self, report: ParityReport):
        super().__init__(f"research/replay parity failed: {report.first_mismatch}")
        self.report = report


def compare_event_streams(
    reference: Iterable[AlphaSignal], replay: Iterable[AlphaSignal], *, fail=True
) -> ParityReport:
    """Compare every contract field exactly, stopping at the first mismatch."""
    expected, actual = tuple(reference), tuple(replay)
    mismatch = None
    category = {"identity": 0, "process": 0, "eligibility": 0}
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected) or index >= len(actual):
            mismatch = ParityMismatch(index, "event_count", len(expected), len(actual))
            break
        left, right = asdict(expected[index]), asdict(actual[index])
        for field in left:
            if left[field] != right[field]:
                mismatch = ParityMismatch(index, field, left[field], right[field])
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
        tuple(item.signal_timestamp.isoformat() for item in actual),
        category["identity"],
        category["process"],
        category["eligibility"],
    )
    if fail and not report.passed:
        raise ParityError(report)
    return report


def _load_jsonl(path: Path) -> tuple[AlphaSignal, ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            value["signal_timestamp"] = datetime.fromisoformat(
                value["signal_timestamp"]
            )
            rows.append(AlphaSignal(**value))
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    """Compare two AlphaSignal JSONL files without loading market data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args(argv)
    report = compare_event_streams(
        _load_jsonl(args.reference), _load_jsonl(args.replay), fail=False
    )
    print(json.dumps(asdict(report), default=str, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a command boundary
    raise SystemExit(main())
