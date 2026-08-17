"""Stage 2C point-in-time rolling Bollinger deviation discovery benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from typing import TextIO

from mr_lab.config import normalize_timeframe
from mr_lab.data import Timeframe, resample_bars
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
from mr_lab.sessions import DEFAULT_SESSION_SPEC

STRATEGY_SCHEMA_VERSION = "stage-2c-bollinger-benchmark-v1"
PRICE_DEFINITION = "close"
CENTER_DEFINITION = "arithmetic_rolling_mean_of_close"
DISPERSION_DEFINITION = "sample_stddev_of_close"
INACTIVE_WINDOW_RULE = "exact_consecutive_canonical_active_observations-v1"
DEFAULT_LOOKBACKS = (20, 40)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_HORIZONS = tuple(timedelta(minutes=value) for value in (15, 30, 60, 120))
CONTEXT_SESSIONS = (None, "asia", "london", "new_york")


class BollingerBenchmarkError(ValueError):
    """Raised when Stage 2C inputs violate its frozen methodology."""


@dataclass(frozen=True, slots=True)
class BollingerFeature:
    """A completed-bar feature retaining its canonical observation and context."""

    observation: ResearchObservation
    rolling_lookback: int
    middle: float | None
    sample_stddev: float | None
    bollinger_z: float | None


def build_bollinger_features(
    observations: Iterable[ResearchObservation], rolling_lookback: int
) -> tuple[BollingerFeature, ...]:
    """Use exactly N consecutive canonical active closes, including the current bar."""
    if type(rolling_lookback) is not int or rolling_lookback < 2:
        raise BollingerBenchmarkError(
            "rolling_lookback must be an integer of at least 2"
        )
    window: deque[float | None] = deque(maxlen=rolling_lookback)
    features = []
    previous: ResearchObservation | None = None
    for observation in observations:
        if (
            previous is not None
            and observation.bar.open_time
            != previous.bar.open_time + previous.bar.timeframe.duration
        ):
            # Missing canonical intervals are boundaries, never permission to pull
            # older closes forward and silently compress elapsed time.
            window.clear()
        window.append(observation.bar.close if observation.is_research_active else None)
        middle = dispersion = z = None
        if len(window) == rolling_lookback and all(
            value is not None for value in window
        ):
            closes = tuple(value for value in window if value is not None)
            middle = statistics.fmean(closes)
            dispersion = statistics.stdev(closes)
            if dispersion == 0:
                dispersion = None
            else:
                z = (observation.bar.close - middle) / dispersion
        features.append(
            BollingerFeature(observation, rolling_lookback, middle, dispersion, z)
        )
        previous = observation
    return tuple(features)


def signal_direction(feature: BollingerFeature, threshold: float) -> Direction | None:
    """Apply strict symmetric boundaries; equality and inactive bars do not signal."""
    if (
        not isinstance(threshold, int | float)
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise BollingerBenchmarkError("threshold must be finite and positive")
    if not feature.observation.is_research_active or feature.bollinger_z is None:
        return None
    if feature.bollinger_z < -threshold:
        return Direction.LONG
    if feature.bollinger_z > threshold:
        return Direction.SHORT
    return None


@dataclass(frozen=True, slots=True)
class BollingerStrategySpec:
    """Immutable identity for one frozen Bollinger benchmark configuration."""

    rolling_lookback: int
    deviation_threshold: float
    strategy_schema_version: str = STRATEGY_SCHEMA_VERSION
    price_definition: str = PRICE_DEFINITION
    center_definition: str = CENTER_DEFINITION
    dispersion_definition: str = DISPERSION_DEFINITION
    inactive_window_rule: str = INACTIVE_WINDOW_RULE

    def __post_init__(self) -> None:
        if self.strategy_schema_version != STRATEGY_SCHEMA_VERSION:
            raise BollingerBenchmarkError("unsupported strategy_schema_version")
        for field, expected in (
            ("price_definition", PRICE_DEFINITION),
            ("center_definition", CENTER_DEFINITION),
            ("dispersion_definition", DISPERSION_DEFINITION),
            ("inactive_window_rule", INACTIVE_WINDOW_RULE),
        ):
            if getattr(self, field) != expected:
                raise BollingerBenchmarkError(f"unsupported {field}")
        if type(self.rolling_lookback) is not int or self.rolling_lookback < 2:
            raise BollingerBenchmarkError("rolling lookback must be at least 2")
        if (
            not isinstance(self.deviation_threshold, int | float)
            or isinstance(self.deviation_threshold, bool)
            or not math.isfinite(self.deviation_threshold)
            or self.deviation_threshold <= 0
        ):
            raise BollingerBenchmarkError(
                "deviation threshold must be finite and positive"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "center_definition": self.center_definition,
            "deviation_threshold": self.deviation_threshold,
            "dispersion_definition": self.dispersion_definition,
            "inactive_window_rule": self.inactive_window_rule,
            "price_definition": self.price_definition,
            "rolling_lookback": self.rolling_lookback,
            "strategy_schema_version": self.strategy_schema_version,
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
    features: Sequence[BollingerFeature],
    outcomes: Sequence[ForwardOutcome],
    research_spec: ResearchSpec,
    strategy_spec: BollingerStrategySpec,
) -> tuple[dict[str, object], ...]:
    """Return unranked overall and overlapping session-context summary rows."""
    if not observations:
        raise BollingerBenchmarkError("observations must not be empty")
    outcome_map = {(item.source.available_at, item.horizon): item for item in outcomes}
    month_count = len(
        {
            (item.bar.open_time.year, item.bar.open_time.month)
            for item in observations
            if item.is_research_active
        }
    )
    signals = [
        (feature, direction)
        for feature in features
        if (direction := signal_direction(feature, strategy_spec.deviation_threshold))
    ]
    rows = []
    for context in CONTEXT_SESSIONS:
        contextual = [
            (feature, direction)
            for feature, direction in signals
            if context is None
            or context in feature.observation.sessions.active_sessions
        ]
        for horizon in research_spec.horizons:
            for label, selected in (
                ("all", contextual),
                ("long", [(f, d) for f, d in contextual if d is Direction.LONG]),
                ("short", [(f, d) for f, d in contextual if d is Direction.SHORT]),
            ):
                signed, unavailable = [], 0
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
                        "bollinger_lookback": strategy_spec.rolling_lookback,
                        "threshold": strategy_spec.deviation_threshold,
                        "forward_horizon_seconds": int(horizon.total_seconds()),
                        "direction": label,
                        "context_session": context,
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


def compatible_horizons(timeframe: Timeframe) -> tuple[timedelta, ...]:
    return tuple(
        horizon
        for horizon in DEFAULT_HORIZONS
        if horizon >= timeframe.duration
        and horizon % timeframe.duration == timedelta(0)
    )


def _manifest_declared_dates(manifest: object) -> tuple[date, ...]:
    if not isinstance(manifest, dict):
        raise BollingerBenchmarkError("invalid offline corpus manifest")
    try:
        values = (
            manifest["requested_start_date"],
            manifest["requested_end_date"],
            *manifest["successful_component_dates"],
            *manifest["confirmed_absent_dates"],
            *(component["requested_day"] for component in manifest["components"]),
        )
        if not all(isinstance(value, str) for value in values):
            raise TypeError
        return tuple(date.fromisoformat(value) for value in values)
    except (KeyError, TypeError, ValueError) as error:
        raise BollingerBenchmarkError("invalid offline corpus manifest") from error


def run_offline_benchmark(
    corpus_dir: Path, timeframe: str
) -> tuple[dict[str, object], ...]:
    """Guard the manifest before loading and run the complete frozen 2024 grid."""
    try:
        manifest = json.loads(
            (corpus_dir / "corpus-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise BollingerBenchmarkError("invalid offline corpus manifest") from error
    dates = _manifest_declared_dates(manifest)
    if not dates or min(dates) < DISCOVERY_START or max(dates) > DISCOVERY_END:
        raise BollingerBenchmarkError(
            "Stage 2C accepts only the frozen 2024 discovery corpus"
        )
    dataset = load_offline_corpus(corpus_dir)
    target = normalize_timeframe(timeframe)
    if target not in (Timeframe("5m"), Timeframe("15m"), Timeframe("1h")):
        raise BollingerBenchmarkError("timeframe must be M5, M15, or H1")
    bars = resample_bars(dataset.bars, target).bars
    observations = build_research_observations(bars, DEFAULT_SESSION_SPEC)
    research_spec = ResearchSpec(compatible_horizons(target))
    outcomes = build_forward_outcomes(observations, research_spec)
    rows = []
    for lookback in DEFAULT_LOOKBACKS:
        features = build_bollinger_features(observations, lookback)
        for threshold in DEFAULT_THRESHOLDS:
            strategy = BollingerStrategySpec(lookback, threshold)
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
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline Stage 2C benchmark")
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
