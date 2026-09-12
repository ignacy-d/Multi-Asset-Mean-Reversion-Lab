"""Run research-versus-replay parity for an explicitly supplied M1 JSONL file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from mr_lab.data import Bar, PriceBasis, Timeframe, VolumeSemantics
from mr_lab.replay.engine import (
    FrozenOuReplayEngine,
    FrozenOuResearchBuilder,
    build_frozen_ou_decisions,
)
from mr_lab.replay.parity import compare_event_streams
from mr_lab.stage4b_runner import assemble_signal_states


def _load_bars(path: Path) -> tuple[Bar, ...]:
    bars = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            for field in ("open_time", "close_time", "available_at"):
                value[field] = datetime.fromisoformat(value[field])
            value["timeframe"] = Timeframe(value["timeframe"])
            value["price_basis"] = PriceBasis(value["price_basis"])
            value["volume_semantics"] = VolumeSemantics(value["volume_semantics"])
            bars.append(Bar(**value))
    return tuple(bars)


def _frozen_scope(states):
    return tuple(
        state
        for state in states
        if str(state.signal_timeframe) == "15m"
        and state.session == "london"
        and state.direction.name == "SHORT"
        and state.benchmark_family in {"vwap", "vwap-canonical-m1"}
        and state.lookback in {20, 40}
    )


def run_parity(m1: tuple[Bar, ...], manifest: dict):
    """Use complete Stage4B assembly as reference and causal replay as actual."""
    reference_states = _frozen_scope(
        assemble_signal_states(SimpleNamespace(bars=m1), manifest)
    )
    reference = build_frozen_ou_decisions(reference_states)
    replay = FrozenOuReplayEngine(FrozenOuResearchBuilder(manifest)).run(m1)
    return compare_event_streams(reference, replay, fail=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m1_jsonl", type=Path)
    parser.add_argument("manifest_json", type=Path)
    args = parser.parse_args(argv)
    m1 = _load_bars(args.m1_jsonl)
    with args.manifest_json.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    report = run_parity(m1, manifest)
    print(json.dumps(asdict(report), default=str, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
