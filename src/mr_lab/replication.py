"""Stage 3 reusable, instrument-labelled frozen replication interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from mr_lab.bollinger_benchmark import run_offline_benchmark as run_bollinger
from mr_lab.providers.dukascopy_range import load_offline_corpus
from mr_lab.providers.instruments import get_instrument_spec
from mr_lab.vwap_benchmark import run_offline_benchmark as run_vwap
from mr_lab.vwap_m1_robustness import run_offline_robustness

TIMEFRAMES = ("M5", "M15", "H1")


class ReplicationError(ValueError):
    """Raised when a corpus does not match the selected replication asset."""


def run_frozen_replication(
    corpus_dir: Path,
    output_dir: Path,
    instrument: str,
    *,
    include_canonical_m1_vwap: bool = False,
) -> tuple[Path, ...]:
    """Run unchanged benchmark families for one already-frozen 2024 corpus."""
    spec = get_instrument_spec(instrument)
    dataset = load_offline_corpus(corpus_dir)
    if dataset.metadata.instrument != spec.instrument:
        raise ReplicationError(
            "selected instrument does not match the frozen corpus instrument"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Callable[[Path, str], tuple[dict[str, object], ...]]]] = [
        ("vwap", run_vwap),
        ("bollinger", run_bollinger),
    ]
    if include_canonical_m1_vwap:
        jobs.append(("vwap-canonical-m1", run_offline_robustness))
    written = []
    for family, runner in jobs:
        for timeframe in TIMEFRAMES:
            rows = runner(corpus_dir, timeframe)
            document = {
                "instrument": spec.instrument,
                "benchmark_family": family,
                "timeframe": timeframe,
                "assembled_dataset_id": dataset.metadata.dataset_id,
                "rows": rows,
            }
            path = output_dir / (
                f"stage-3b-{spec.instrument.lower()}-{family}-{timeframe.lower()}.json"
            )
            path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            written.append(path)
    return tuple(written)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen multi-asset replication")
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-m1-vwap", action="store_true")
    args = parser.parse_args(argv)
    paths = run_frozen_replication(
        args.corpus_dir,
        args.output_dir,
        args.instrument,
        include_canonical_m1_vwap=args.canonical_m1_vwap,
    )
    for path in paths:
        print(f"result={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
