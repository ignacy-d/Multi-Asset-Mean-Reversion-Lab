"""Causal, provider-independent rolling PCA residual primitives.

The engine accepts only an already synchronized in-memory close panel.  For a
return stamped ``t``, every fitted quantity uses returns/residuals stamped
strictly before ``t``.  The first residual is available after
``pca_training_window`` prior returns; published observations begin after a
further ``residual_normalization_window`` prior residuals have accumulated.
No missing-value repair or data loading is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from math import isfinite

import numpy as np
from numpy.typing import NDArray


class PCAResidualError(ValueError):
    """Raised when configuration, panel data, or rolling state is invalid."""


class Direction(IntEnum):
    NEGATIVE = -1
    NONE = 0
    POSITIVE = 1


@dataclass(frozen=True, slots=True)
class PCAResidualConfig:
    pca_training_window: int = 1024
    components: int = 2
    residual_normalization_window: int = 256
    shock_threshold: float = 2.0
    rearm_threshold: float = 1.0
    variance_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pca_training_window, int)
            or isinstance(self.pca_training_window, bool)
            or self.pca_training_window < 2
        ):
            raise PCAResidualError("pca_training_window must be an integer >= 2")
        if (
            not isinstance(self.components, int)
            or isinstance(self.components, bool)
            or self.components < 1
        ):
            raise PCAResidualError("components must be a positive integer")
        if (
            not isinstance(self.residual_normalization_window, int)
            or isinstance(self.residual_normalization_window, bool)
            or self.residual_normalization_window < 2
        ):
            raise PCAResidualError(
                "residual_normalization_window must be an integer >= 2"
            )
        values = (self.shock_threshold, self.rearm_threshold, self.variance_epsilon)
        if any(
            isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v)
            for v in values
        ):
            raise PCAResidualError("thresholds and variance_epsilon must be finite")
        if self.shock_threshold <= 0:
            raise PCAResidualError("shock_threshold must be positive")
        if self.rearm_threshold < 0 or self.rearm_threshold >= self.shock_threshold:
            raise PCAResidualError(
                "rearm_threshold must be non-negative and below shock_threshold"
            )
        if self.variance_epsilon <= 0:
            raise PCAResidualError("variance_epsilon must be positive")


@dataclass(frozen=True, slots=True)
class SynchronizedClosePanel:
    """Immutable logical contract for ordered UTC timestamps and close columns."""

    timestamps: tuple[datetime, ...]
    instruments: tuple[str, ...]
    closes: NDArray[np.float64]

    def validated_copy(self) -> NDArray[np.float64]:
        if not self.instruments or any(
            not isinstance(x, str) or not x.strip() for x in self.instruments
        ):
            raise PCAResidualError("instruments must be non-empty names")
        if len(set(self.instruments)) != len(self.instruments):
            raise PCAResidualError("instrument names must be unique")
        if len(self.timestamps) < 2:
            raise PCAResidualError("at least two timestamps are required")
        for stamp in self.timestamps:
            if (
                not isinstance(stamp, datetime)
                or stamp.tzinfo is None
                or stamp.utcoffset() != timedelta(0)
            ):
                raise PCAResidualError(
                    "timestamps must be timezone-aware UTC datetimes"
                )
        if any(
            right <= left
            for left, right in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise PCAResidualError("timestamps must be unique and strictly increasing")
        values = np.asarray(self.closes, dtype=np.float64)
        expected = (len(self.timestamps), len(self.instruments))
        if values.ndim != 2 or values.shape != expected:
            raise PCAResidualError(f"closes must have shape {expected}")
        if not np.isfinite(values).all():
            raise PCAResidualError("closes must contain only finite observations")
        if (values <= 0).any():
            raise PCAResidualError("close prices must be positive")
        copy = values.copy()
        copy.setflags(write=False)
        return copy


@dataclass(frozen=True, slots=True)
class ResidualObservation:
    timestamp: datetime
    instrument: str
    log_return: float
    standardized_return: float
    factor_reconstruction: float
    residual: float
    residual_zscore: float
    shock_active: bool
    event_emitted: bool
    residual_shock_sign: Direction
    fade_direction: Direction


@dataclass(frozen=True, slots=True)
class ShockState:
    active: bool
    event_emitted: bool
    shock_sign: Direction
    fade_direction: Direction


class ShockStateDetector:
    """Independent per-series threshold/re-arm state machine.

    ``shock_sign`` and ``fade_direction`` describe the episode entry and remain
    fixed until re-arm.  They deliberately do not represent the instantaneous
    sign of subsequent z-scores within an active episode.
    """

    def __init__(self, shock_threshold: float = 2.0, rearm_threshold: float = 1.0):
        if not isfinite(shock_threshold) or not isfinite(rearm_threshold):
            raise PCAResidualError("shock thresholds must be finite")
        if (
            shock_threshold <= 0
            or rearm_threshold < 0
            or rearm_threshold >= shock_threshold
        ):
            raise PCAResidualError("require 0 <= rearm_threshold < shock_threshold")
        self.shock_threshold = shock_threshold
        self.rearm_threshold = rearm_threshold
        self._active = False
        self._episode_sign = Direction.NONE

    def update(self, zscore: float) -> ShockState:
        if not isfinite(zscore):
            raise PCAResidualError("zscore must be finite")
        event = False
        if self._active and abs(zscore) < self.rearm_threshold:
            self._active = False
            self._episode_sign = Direction.NONE
        if not self._active and abs(zscore) >= self.shock_threshold:
            self._active = True
            event = True
            self._episode_sign = (
                Direction.POSITIVE if zscore > 0 else Direction.NEGATIVE
            )
        fade = Direction(-self._episode_sign) if self._active else Direction.NONE
        return ShockState(self._active, event, self._episode_sign, fade)


def reconstruct_from_components(
    value: NDArray[np.float64],
    mean: NDArray[np.float64],
    components: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project and inverse-transform; paired component sign flips cancel."""
    centered = value - mean
    return mean + (centered @ components.T) @ components


class PCAResidualEngine:
    """Compute deterministic prior-only rolling PCA residual observations."""

    def __init__(self, config: PCAResidualConfig | None = None):
        self.config = config or PCAResidualConfig()

    def run(self, panel: SynchronizedClosePanel) -> tuple[ResidualObservation, ...]:
        prices = panel.validated_copy()
        n_instruments = prices.shape[1]
        if self.config.components > n_instruments:
            raise PCAResidualError("components cannot exceed instrument count")
        if self.config.components >= n_instruments:
            raise PCAResidualError(
                "components must leave at least one residual dimension"
            )
        returns = np.diff(np.log(prices), axis=0)
        residual_history: list[NDArray[np.float64]] = []
        detectors = [
            ShockStateDetector(self.config.shock_threshold, self.config.rearm_threshold)
            for _ in panel.instruments
        ]
        output: list[ResidualObservation] = []
        w = self.config.pca_training_window
        rw = self.config.residual_normalization_window
        for index in range(w, len(returns)):
            prior = returns[index - w : index]
            mean = prior.mean(axis=0)
            scale = prior.std(axis=0, ddof=1)
            if (scale <= self.config.variance_epsilon).any():
                raise PCAResidualError("prior return variance is zero or near zero")
            training = (prior - mean) / scale
            pca_mean = training.mean(axis=0)
            covariance = np.cov(training, rowvar=False, ddof=1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1][: self.config.components]
            components = eigenvectors[:, order].T
            current = (returns[index] - mean) / scale
            reconstruction = reconstruct_from_components(current, pca_mean, components)
            residual = current - reconstruction
            if not np.isfinite(residual).all():
                raise PCAResidualError("PCA produced a nonfinite residual")
            if len(residual_history) >= rw:
                history = np.asarray(residual_history[-rw:])
                rmean = history.mean(axis=0)
                rscale = history.std(axis=0, ddof=1)
                if (rscale <= self.config.variance_epsilon).any():
                    raise PCAResidualError(
                        "prior residual variance is zero or near zero"
                    )
                zscores = (residual - rmean) / rscale
                if not np.isfinite(zscores).all():
                    raise PCAResidualError(
                        "residual normalization produced a nonfinite zscore"
                    )
                timestamp = panel.timestamps[index + 1]
                for column, instrument in enumerate(panel.instruments):
                    state = detectors[column].update(float(zscores[column]))
                    output.append(
                        ResidualObservation(
                            timestamp,
                            instrument,
                            float(returns[index, column]),
                            float(current[column]),
                            float(reconstruction[column]),
                            float(residual[column]),
                            float(zscores[column]),
                            state.active,
                            state.event_emitted,
                            state.shock_sign,
                            state.fade_direction,
                        )
                    )
            residual_history.append(residual.copy())
        return tuple(output)
