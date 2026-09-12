"""Deterministic candle replay and research parity support."""

from mr_lab.replay.engine import (
    FrozenOuReplayEngine,
    FrozenOuResearchBuilder,
    IncrementalResampler,
    ReplayPrefix,
    build_frozen_ou_decisions,
)
from mr_lab.replay.parity import (
    FrozenOuDecision,
    FrozenOuDecisionDetails,
    ParityMismatch,
    ParityReport,
    compare_event_streams,
)

__all__ = [
    "FrozenOuDecision",
    "FrozenOuDecisionDetails",
    "FrozenOuReplayEngine",
    "FrozenOuResearchBuilder",
    "IncrementalResampler",
    "ParityMismatch",
    "ParityReport",
    "ReplayPrefix",
    "build_frozen_ou_decisions",
    "compare_event_streams",
]
