"""Minimal typed models for selecting an experiment variant."""

from dataclasses import dataclass
from types import MappingProxyType

type ParameterValue = str | int | float | bool


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the contract."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration-only description of one hypothesis-family variant."""

    instrument: str
    timeframe: str
    session: str
    strategy: str
    feature_model: str
    cost_model: str
    entry: dict[str, ParameterValue]
    exit: dict[str, ParameterValue]

    def __post_init__(self) -> None:
        for name in (
            "instrument",
            "timeframe",
            "session",
            "strategy",
            "feature_model",
            "cost_model",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{name} must be a non-empty string")

        for name in ("entry", "exit"):
            parameters = getattr(self, name)
            if not isinstance(parameters, dict):
                raise ConfigError(f"{name} must be a table")
            if not parameters:
                raise ConfigError(f"{name} must contain at least one parameter")
            for key, value in parameters.items():
                if not isinstance(key, str) or not key.strip():
                    raise ConfigError(
                        f"{name} parameter names must be non-empty strings"
                    )
                if not isinstance(value, str | int | float | bool):
                    raise ConfigError(f"{name}.{key} must be a scalar value")

            # Prevent callers from mutating a validated experiment after loading.
            object.__setattr__(self, name, MappingProxyType(dict(parameters)))
