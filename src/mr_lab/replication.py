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
SESSION_DISPLAY_ORDER = ("asia", "new_york", "london", "overall")


class ReplicationError(ValueError):
    """Raised when a corpus does not match the selected replication asset."""


def _session_name(row: dict[str, object]) -> str:
    """Return the comparable session label without changing benchmark rows."""
    if "anchor_session" in row:
        return str(row["anchor_session"])
    context = row.get("context_session")
    return "overall" if context is None else str(context)


def _write_comparison_report(
    output_dir: Path,
    instrument: str,
    dataset_id: str,
    documents: Sequence[tuple[str, str, Sequence[dict[str, object]], Path]],
) -> Path:
    """Write a deterministic descriptive inventory, never a ranked analysis."""
    lines = [
        f"# Stage 3B {instrument} 2024 frozen replication",
        "",
        f"- Assembled dataset ID: `{dataset_id}`",
        "- Construction order: native-timeframe VWAP, canonical-M1 VWAP "
        "robustness, Bollinger.",
        "- Session display order: Asia, New York, London, then any remaining contexts.",
        "- Values below are descriptive inventory totals, not ranked evidence "
        "and not a methodology change.",
        "",
        "| Construction | Timeframe | Session | Grid rows | "
        "All-direction signal total* |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    family_order = {
        name: index
        for index, name in enumerate(("vwap", "vwap-canonical-m1", "bollinger"))
    }
    timeframe_order = {name: index for index, name in enumerate(TIMEFRAMES)}
    session_order = {name: index for index, name in enumerate(SESSION_DISPLAY_ORDER)}
    summaries = []
    for family, timeframe, rows, _path in documents:
        sessions = sorted(
            {_session_name(row) for row in rows},
            key=lambda name: (session_order.get(name, len(session_order)), name),
        )
        for session in sessions:
            selected = [row for row in rows if _session_name(row) == session]
            summaries.append(
                (
                    family_order[family],
                    timeframe_order[timeframe],
                    session_order.get(session, len(session_order)),
                    session,
                    family,
                    timeframe,
                    len(selected),
                    sum(
                        int(row["signal_count"])
                        for row in selected
                        if row.get("direction") == "all"
                    ),
                )
            )
    for _, _, _, session, family, timeframe, row_count, signals in sorted(summaries):
        lines.append(
            f"| {family} | {timeframe} | {session} | {row_count} | {signals} |"
        )
    lines.extend(
        (
            "",
            "\\* Sum across every frozen threshold, lookback, and compatible "
            "horizon. Signals therefore recur across horizons and this total is "
            "only a completeness-oriented comparison, not a unique-event count.",
            "",
            "## Complete result documents",
            "",
        )
    )
    for family, timeframe, rows, path in sorted(
        documents, key=lambda item: (family_order[item[0]], timeframe_order[item[1]])
    ):
        lines.append(f"- `{path.name}` — {family}, {timeframe}, {len(rows)} rows")
    path = output_dir / f"stage-3b-{instrument.lower()}-comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    ]
    if include_canonical_m1_vwap:
        jobs.append(("vwap-canonical-m1", run_offline_robustness))
    jobs.append(("bollinger", run_bollinger))
    written = []
    documents = []
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
            documents.append((family, timeframe, rows, path))
    written.append(
        _write_comparison_report(
            output_dir, spec.instrument, dataset.metadata.dataset_id, documents
        )
    )
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
