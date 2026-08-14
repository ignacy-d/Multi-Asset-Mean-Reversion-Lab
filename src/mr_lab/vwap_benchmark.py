"""Stage 2B point-in-time session-reset VWAP discovery benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

from mr_lab.config import normalize_timeframe
from mr_lab.data import Bar, Timeframe, VolumeSemantics, resample_bars
from mr_lab.providers.dukascopy_range import (
    DISCOVERY_END,
    DISCOVERY_START,
    load_offline_corpus,
)
from mr_lab.research import (
    Direction,
    ForwardOutcome,
    ResearchObservation,
    ResearchSpec,
    build_forward_outcomes,
    build_research_observations,
)
from mr_lab.sessions import DEFAULT_SESSION_SPEC, SessionSpec, TimeWindow

STRATEGY_SCHEMA_VERSION = "stage-2b-vwap-benchmark-v1"
VWAP_PRICE_DEFINITION = "hlc3"
WEIGHT_SEMANTICS = "quote_activity"
RESET_MODE = "major_session_instance"
NORMALIZATION_DEFINITION = (
    "relative_deviation_divided_by_sample_stddev_of_consecutive_bar_returns_v1"
)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_VOLATILITY_LOOKBACKS = (20, 40)
DEFAULT_HORIZONS = tuple(timedelta(minutes=value) for value in (15, 30, 60, 120))


class VwapBenchmarkError(ValueError):
    """Raised when Stage 2B inputs violate its frozen methodology."""


def typical_price(bar: Bar) -> float:
    """Return HLC3 for a canonical completed bar."""
    if not isinstance(bar, Bar):
        raise VwapBenchmarkError("bar must be a canonical Bar")
    return (bar.high + bar.low + bar.close) / 3.0


def _session_instance(timestamp: datetime, window: TimeWindow) -> date:
    local = timestamp.astimezone(ZoneInfo(window.timezone))
    instance = local.date()
    if window.start > window.end and local.timetz().replace(tzinfo=None) < window.end:
        instance -= timedelta(days=1)
    return instance


@dataclass(frozen=True, slots=True)
class VwapFeature:
    """One independently anchored, point-in-time activity-weighted VWAP feature."""

    observation: ResearchObservation
    anchor_session: str
    session_instance: date
    vwap: float | None
    price: float
    absolute_deviation: float | None
    relative_deviation: float | None
    volatility_lookback: int
    rolling_volatility: float | None
    vwap_deviation_z: float | None


def build_vwap_features(
    observations: Iterable[ResearchObservation],
    session_spec: SessionSpec,
    volatility_lookback: int,
) -> tuple[VwapFeature, ...]:
    """Build session streams and trailing volatility without deleting canonical time.

    Volatility is the sample standard deviation of exactly ``lookback`` consecutive
    close-to-close arithmetic bar returns ending at the current bar. A return is
    unavailable when either adjacent canonical observation is inactive; the rolling
    denominator then remains unavailable until a complete consecutive window exists.
    """
    if type(volatility_lookback) is not int or volatility_lookback < 2:
        raise VwapBenchmarkError("volatility_lookback must be an integer of at least 2")
    items = tuple(observations)
    windows = {window.name: window for window in session_spec.major_sessions}
    returns: deque[float | None] = deque(maxlen=volatility_lookback)
    cumulative: dict[tuple[str, date], list[float]] = defaultdict(lambda: [0.0, 0.0])
    features: list[VwapFeature] = []
    previous: ResearchObservation | None = None
    for observation in items:
        current_return = None
        if previous and previous.is_research_active and observation.is_research_active:
            if previous.bar.close == 0:
                raise VwapBenchmarkError("bar return is undefined for zero close")
            current_return = observation.bar.close / previous.bar.close - 1.0
        returns.append(current_return)
        volatility = None
        if len(returns) == volatility_lookback and all(x is not None for x in returns):
            volatility = statistics.stdev(x for x in returns if x is not None)
            if volatility == 0:
                volatility = None
        for anchor in observation.sessions.active_sessions:
            window = windows[anchor]
            instance = _session_instance(observation.bar.open_time, window)
            totals = cumulative[(anchor, instance)]
            # Inactive filler contributes neither numerator nor weight. A moving
            # zero-weight bar remains active but naturally adds zero VWAP weight.
            if observation.is_research_active:
                if (
                    observation.bar.volume_semantics
                    is not VolumeSemantics.QUOTE_ACTIVITY
                ):
                    raise VwapBenchmarkError("VWAP requires QUOTE_ACTIVITY volume")
                weight = observation.bar.volume
                if weight is None:
                    raise VwapBenchmarkError("VWAP requires numeric volume")
                totals[0] += typical_price(observation.bar) * weight
                totals[1] += weight
            vwap = totals[0] / totals[1] if totals[1] > 0 else None
            absolute = observation.bar.close - vwap if vwap is not None else None
            relative = observation.bar.close / vwap - 1.0 if vwap else None
            z = relative / volatility if relative is not None and volatility else None
            features.append(
                VwapFeature(
                    observation,
                    anchor,
                    instance,
                    vwap,
                    observation.bar.close,
                    absolute,
                    relative,
                    volatility_lookback,
                    volatility,
                    z,
                )
            )
        previous = observation
    return tuple(features)


def signal_direction(feature: VwapFeature, threshold: float) -> Direction | None:
    """Apply strict symmetric boundaries: equality does not signal."""
    if (
        not isinstance(threshold, int | float)
        or isinstance(threshold, bool)
        or threshold <= 0
    ):
        raise VwapBenchmarkError("threshold must be positive")
    if not feature.observation.is_research_active or feature.vwap_deviation_z is None:
        return None
    if feature.vwap_deviation_z < -threshold:
        return Direction.LONG
    if feature.vwap_deviation_z > threshold:
        return Direction.SHORT
    return None


@dataclass(frozen=True, slots=True)
class VwapStrategySpec:
    """Immutable identity of one predeclared benchmark configuration."""

    volatility_lookback: int
    deviation_threshold: float
    strategy_schema_version: str = STRATEGY_SCHEMA_VERSION
    vwap_price_definition: str = VWAP_PRICE_DEFINITION
    weight_semantics: str = WEIGHT_SEMANTICS
    reset_mode: str = RESET_MODE
    normalized_deviation_definition: str = NORMALIZATION_DEFINITION

    def __post_init__(self) -> None:
        if self.strategy_schema_version != STRATEGY_SCHEMA_VERSION:
            raise VwapBenchmarkError("unsupported strategy schema version")
        if type(self.volatility_lookback) is not int or self.volatility_lookback < 2:
            raise VwapBenchmarkError("volatility lookback must be at least 2")
        if (
            not isinstance(self.deviation_threshold, int | float)
            or isinstance(self.deviation_threshold, bool)
            or not math.isfinite(self.deviation_threshold)
            or self.deviation_threshold <= 0
        ):
            raise VwapBenchmarkError("deviation threshold must be finite and positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "deviation_threshold": self.deviation_threshold,
            "normalized_deviation_definition": self.normalized_deviation_definition,
            "reset_mode": self.reset_mode,
            "strategy_schema_version": self.strategy_schema_version,
            "volatility_lookback": self.volatility_lookback,
            "vwap_price_definition": self.vwap_price_definition,
            "weight_semantics": self.weight_semantics,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def strategy_spec_id(self) -> str:
        return f"sha256:{sha256(self.to_json().encode()).hexdigest()}"


def _stats(values: Sequence[float]) -> tuple[float | None, ...]:
    if not values:
        return (None,) * 6
    mean = statistics.fmean(values)
    median = statistics.median(values)
    win_rate = sum(value > 0 for value in values) / len(values)
    stddev = statistics.stdev(values) if len(values) >= 2 else None
    error = stddev / math.sqrt(len(values)) if stddev is not None else None
    t_stat = mean / error if error not in (None, 0) else None
    return mean, median, win_rate, stddev, error, t_stat


def summarize_benchmark(
    *,
    dataset_id: str,
    timeframe: Timeframe,
    observations: Sequence[ResearchObservation],
    features: Sequence[VwapFeature],
    outcomes: Sequence[ForwardOutcome],
    research_spec: ResearchSpec,
    strategy_spec: VwapStrategySpec,
) -> tuple[dict[str, object], ...]:
    """Return deterministic, unranked all/long/short configuration rows."""
    outcome_map = {(item.source.available_at, item.horizon): item for item in outcomes}
    months = {
        (item.bar.open_time.year, item.bar.open_time.month)
        for item in observations
        if item.is_research_active
    }
    month_count = len(months)
    rows = []
    for anchor in sorted({item.anchor_session for item in features}):
        anchor_signals = [
            (feature, signal_direction(feature, strategy_spec.deviation_threshold))
            for feature in features
            if feature.anchor_session == anchor
        ]
        anchor_signals = [
            (feature, direction) for feature, direction in anchor_signals if direction
        ]
        for horizon in research_spec.horizons:
            for label, selected in (
                ("all", anchor_signals),
                ("long", [(f, d) for f, d in anchor_signals if d is Direction.LONG]),
                ("short", [(f, d) for f, d in anchor_signals if d is Direction.SHORT]),
            ):
                signed = []
                unavailable = 0
                for feature, direction in selected:
                    outcome = outcome_map[(feature.observation.available_at, horizon)]
                    value = outcome.signed_forward_return(direction)
                    if value is None:
                        unavailable += 1
                    else:
                        signed.append(value)
                mean, median, win_rate, stddev, error, t_stat = _stats(signed)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "session_spec_id": observations[0].session_spec_id,
                        "research_spec_id": research_spec.research_spec_id,
                        "strategy_spec_id": strategy_spec.strategy_spec_id,
                        "timeframe": str(timeframe),
                        "anchor_session": anchor,
                        "threshold": strategy_spec.deviation_threshold,
                        "volatility_lookback": strategy_spec.volatility_lookback,
                        "forward_horizon_seconds": int(horizon.total_seconds()),
                        "direction": label,
                        "signal_count": len(selected),
                        "long_count": sum(d is Direction.LONG for _, d in selected),
                        "short_count": sum(d is Direction.SHORT for _, d in selected),
                        "valid_outcome_count": len(signed),
                        "unavailable_count": unavailable,
                        "mean_signed_return": mean,
                        "median_signed_return": median,
                        "win_rate": win_rate,
                        "stddev": stddev,
                        "standard_error": error,
                        "t_stat": t_stat,
                        "available_research_months": month_count,
                        "monthly_signal_frequency": len(selected) / month_count
                        if month_count
                        else None,
                    }
                )
    return tuple(rows)


def run_offline_benchmark(
    corpus_dir: Path, timeframe: str
) -> tuple[dict[str, object], ...]:
    """Run the frozen grid against an explicitly discovery-only offline corpus."""
    manifest_path = corpus_dir / "corpus-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = tuple(
            date.fromisoformat(value)
            for value in manifest["successful_component_dates"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VwapBenchmarkError("invalid offline corpus manifest") from error
    if not dates or min(dates) < DISCOVERY_START or max(dates) > DISCOVERY_END:
        raise VwapBenchmarkError(
            "Stage 2B accepts only the frozen 2024 discovery corpus"
        )
    dataset = load_offline_corpus(corpus_dir)
    target = normalize_timeframe(timeframe)
    if target not in (Timeframe("5m"), Timeframe("15m"), Timeframe("1h")):
        raise VwapBenchmarkError("timeframe must be M5, M15, or H1")
    bars = resample_bars(dataset.bars, target).bars
    observations = build_research_observations(bars, DEFAULT_SESSION_SPEC)
    research_spec = ResearchSpec(DEFAULT_HORIZONS)
    outcomes = build_forward_outcomes(observations, research_spec)
    rows = []
    for lookback in DEFAULT_VOLATILITY_LOOKBACKS:
        features = build_vwap_features(observations, DEFAULT_SESSION_SPEC, lookback)
        for threshold in DEFAULT_THRESHOLDS:
            strategy = VwapStrategySpec(lookback, threshold)
            rows.extend(
                summarize_benchmark(
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
    if not rows:
        return
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline Stage 2B benchmark")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--timeframe", choices=("M5", "M15", "H1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args(argv)
    rows = run_offline_benchmark(args.corpus_dir, args.timeframe)
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
