"""Stage 2B robustness benchmark using point-in-time canonical-M1 session VWAP."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import TextIO

from mr_lab.config import normalize_timeframe
from mr_lab.data import Timeframe, VolumeSemantics, resample_bars
from mr_lab.providers.dukascopy_range import (
    DISCOVERY_END,
    DISCOVERY_START,
    load_offline_corpus,
)
from mr_lab.research import (
    ResearchObservation,
    ResearchSpec,
    build_forward_outcomes,
    build_research_observations,
)
from mr_lab.sessions import DEFAULT_SESSION_SPEC, SessionSpec
from mr_lab.vwap_benchmark import (
    DEFAULT_THRESHOLDS,
    DEFAULT_VOLATILITY_LOOKBACKS,
    NORMALIZATION_DEFINITION,
    RESET_MODE,
    VWAP_PRICE_DEFINITION,
    WEIGHT_SEMANTICS,
    VwapBenchmarkError,
    VwapFeature,
    VwapStrategySpec,
    _compatible_horizons,
    _manifest_declared_dates,
    _session_instance,
    build_vwap_features,
    summarize_benchmark,
    typical_price,
)

LEGACY_ROBUSTNESS_SCHEMA_VERSION = "stage-2b-vwap-construction-robustness-v1"
ROBUSTNESS_SCHEMA_VERSION = "stage-2b-vwap-construction-robustness-v2"
NATIVE_TIMEFRAME = "native_timeframe"
CANONICAL_M1 = "canonical_m1"
SUPPORTED_CONSTRUCTION_SOURCES = frozenset((NATIVE_TIMEFRAME, CANONICAL_M1))


@dataclass(frozen=True, slots=True)
class VwapRobustnessStrategySpec:
    """Distinct identity for a frozen VWAP construction comparison cell.

    This deliberately does not modify ``VwapStrategySpec`` or its historical IDs.
    """

    volatility_lookback: int
    deviation_threshold: float
    vwap_construction_source: str
    robustness_schema_version: str = ROBUSTNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Reuse all frozen Stage 2B validation without changing its identity schema.
        VwapStrategySpec(self.volatility_lookback, self.deviation_threshold)
        if self.robustness_schema_version != ROBUSTNESS_SCHEMA_VERSION:
            raise VwapBenchmarkError("unsupported robustness schema version")
        if self.vwap_construction_source not in SUPPORTED_CONSTRUCTION_SOURCES:
            raise VwapBenchmarkError("unsupported vwap_construction_source")

    def as_dict(self) -> dict[str, object]:
        return {
            "deviation_threshold": self.deviation_threshold,
            "normalized_deviation_definition": NORMALIZATION_DEFINITION,
            "reset_mode": RESET_MODE,
            "robustness_schema_version": self.robustness_schema_version,
            "volatility_lookback": self.volatility_lookback,
            "vwap_construction_source": self.vwap_construction_source,
            "vwap_price_definition": VWAP_PRICE_DEFINITION,
            "weight_semantics": WEIGHT_SEMANTICS,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def strategy_spec_id(self) -> str:
        return f"sha256:{sha256(self.to_json().encode()).hexdigest()}"


def build_canonical_m1_vwap_features(
    m1_observations: Iterable[ResearchObservation],
    research_observations: Iterable[ResearchObservation],
    session_spec: SessionSpec,
    volatility_lookback: int,
) -> tuple[VwapFeature, ...]:
    """Sample independent M1 session VWAP streams at research availability times.

    Only M1 bars with ``available_at <= research.available_at`` are admitted. The
    sampled stream must match the research observation's session instance, so no
    value is carried across a reset or structurally unavailable session state.
    """
    m1_items = tuple(m1_observations)
    research_items = tuple(research_observations)
    if any(item.bar.timeframe != Timeframe("1m") for item in m1_items):
        raise VwapBenchmarkError("canonical VWAP source must contain only M1 bars")
    if any(
        left.available_at > right.available_at for left, right in pairwise(m1_items)
    ) or any(
        left.available_at > right.available_at
        for left, right in pairwise(research_items)
    ):
        raise VwapBenchmarkError("observations must be ordered by available_at")

    # Native feature construction supplies the frozen Stage 2B trailing-volatility
    # denominator. Only its equilibrium fields are replaced below.
    native = build_vwap_features(research_items, session_spec, volatility_lookback)
    native_by_key = {
        (item.observation.available_at, item.anchor_session): item for item in native
    }
    windows = {window.name: window for window in session_spec.major_sessions}
    totals: dict[tuple[str, date], list[float]] = defaultdict(lambda: [0.0, 0.0])
    cursor = 0
    output: list[VwapFeature] = []
    for research in research_items:
        while cursor < len(m1_items) and (
            m1_items[cursor].available_at <= research.available_at
        ):
            source = m1_items[cursor]
            for anchor in source.sessions.active_sessions:
                key = (
                    anchor,
                    _session_instance(source.bar.open_time, windows[anchor]),
                )
                if source.is_research_active:
                    if (
                        source.bar.volume_semantics
                        is not VolumeSemantics.QUOTE_ACTIVITY
                    ):
                        raise VwapBenchmarkError("VWAP requires QUOTE_ACTIVITY volume")
                    if source.bar.volume is None:
                        raise VwapBenchmarkError("VWAP requires numeric volume")
                    totals[key][0] += typical_price(source.bar) * source.bar.volume
                    totals[key][1] += source.bar.volume
            cursor += 1
        for anchor in research.sessions.active_sessions:
            instance = _session_instance(research.bar.open_time, windows[anchor])
            numerator, weight = totals[(anchor, instance)]
            vwap = numerator / weight if weight > 0 else None
            base = native_by_key[(research.available_at, anchor)]
            absolute = research.bar.close - vwap if vwap is not None else None
            relative = research.bar.close / vwap - 1.0 if vwap else None
            z = (
                relative / base.rolling_volatility
                if relative is not None and base.rolling_volatility
                else None
            )
            output.append(
                VwapFeature(
                    research,
                    anchor,
                    instance,
                    vwap,
                    research.bar.close,
                    absolute,
                    relative,
                    volatility_lookback,
                    base.rolling_volatility,
                    z,
                )
            )
    return tuple(output)


def _summarize(
    *,
    dataset_id: str,
    timeframe: Timeframe,
    observations: Sequence[ResearchObservation],
    features: Sequence[VwapFeature],
    outcomes,
    research_spec: ResearchSpec,
    strategy_spec: VwapRobustnessStrategySpec,
) -> tuple[dict[str, object], ...]:
    legacy = VwapStrategySpec(
        strategy_spec.volatility_lookback, strategy_spec.deviation_threshold
    )
    rows = summarize_benchmark(
        dataset_id=dataset_id,
        timeframe=timeframe,
        observations=observations,
        features=features,
        outcomes=outcomes,
        research_spec=research_spec,
        strategy_spec=legacy,
    )
    return tuple(
        {
            **row,
            "strategy_spec_id": strategy_spec.strategy_spec_id,
            "vwap_construction_source": strategy_spec.vwap_construction_source,
        }
        for row in rows
    )


def run_offline_robustness(
    corpus_dir: Path, timeframe: str
) -> tuple[dict[str, object], ...]:
    """Run only canonical-M1 robustness against the frozen offline 2024 corpus."""
    manifest_path = corpus_dir / "corpus-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VwapBenchmarkError("invalid offline corpus manifest") from error
    dates = _manifest_declared_dates(manifest)
    if not dates or min(dates) < DISCOVERY_START or max(dates) > DISCOVERY_END:
        raise VwapBenchmarkError("robustness benchmark accepts only frozen 2024")
    dataset = load_offline_corpus(corpus_dir)
    target = normalize_timeframe(timeframe)
    if target not in (Timeframe("5m"), Timeframe("15m"), Timeframe("1h")):
        raise VwapBenchmarkError("timeframe must be M5, M15, or H1")
    m1_observations = build_research_observations(dataset.bars, DEFAULT_SESSION_SPEC)
    bars = resample_bars(dataset.bars, target).bars
    observations = build_research_observations(bars, DEFAULT_SESSION_SPEC)
    research_spec = ResearchSpec(_compatible_horizons(target))
    outcomes = build_forward_outcomes(observations, research_spec)
    rows = []
    for lookback in DEFAULT_VOLATILITY_LOOKBACKS:
        features = build_canonical_m1_vwap_features(
            m1_observations, observations, DEFAULT_SESSION_SPEC, lookback
        )
        for threshold in DEFAULT_THRESHOLDS:
            strategy = VwapRobustnessStrategySpec(lookback, threshold, CANONICAL_M1)
            rows.extend(
                _summarize(
                    dataset_id=dataset.metadata.dataset_id,
                    timeframe=target,
                    observations=observations,
                    features=features,
                    outcomes=outcomes,
                    research_spec=research_spec,
                    strategy_spec=strategy,
                )
            )
    return tuple(rows)


def _write_csv(rows: Sequence[dict[str, object]], output: TextIO) -> None:
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run canonical-M1 VWAP robustness")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--timeframe", choices=("M5", "M15", "H1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args(argv)
    rows = run_offline_robustness(args.corpus_dir, args.timeframe)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        if args.format == "json":
            json.dump(
                rows, output, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            output.write("\n")
        else:
            _write_csv(rows, output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
