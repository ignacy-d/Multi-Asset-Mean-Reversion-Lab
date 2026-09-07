"""Serialized contract boundary for later NautilusTrader replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from mr_lab.identity import canonical_json, methodology_id


@dataclass(frozen=True, slots=True)
class FrozenSignal:
    signal_id: str
    instrument: str
    event_time: datetime
    available_at: datetime
    direction: str
    strategy_spec_id: str
    dataset_id: str

    def __post_init__(self) -> None:
        if self.direction not in ("LONG", "SHORT"):
            raise ValueError("direction must be LONG or SHORT")
        for value in (self.event_time, self.available_at):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError("signal times must be timezone-aware UTC")
        if self.available_at < self.event_time:
            raise ValueError("signal cannot be available before its event")


def export_frozen_signals(signals: tuple[FrozenSignal, ...]) -> str:
    """Return deterministic JSONL consumable without Nautilus being installed."""
    ordered = sorted(signals, key=lambda x: (x.available_at, x.signal_id))
    header = canonical_json(
        {
            "schema_version": "mr-lab-frozen-signals-v1",
            "export_id": methodology_id(
                "frozen-signal-export-v1", {"signals": [asdict(x) for x in ordered]}
            ),
        }
    )
    return (
        "\n".join((header, *(canonical_json(asdict(signal)) for signal in ordered)))
        + "\n"
    )
