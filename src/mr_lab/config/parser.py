"""TOML loading for experiment configurations."""

from pathlib import Path
from tomllib import TOMLDecodeError, load

from mr_lab.config.models import ConfigError, ExperimentConfig

_EXPECTED_KEYS = {
    "instrument",
    "timeframe",
    "session",
    "strategy",
    "feature_model",
    "cost_model",
    "entry",
    "exit",
}


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate one experiment configuration from TOML."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            raw = load(config_file)
    except (OSError, TOMLDecodeError) as error:
        message = f"could not load configuration {config_path}: {error}"
        raise ConfigError(message) from error

    unknown = raw.keys() - _EXPECTED_KEYS
    missing = _EXPECTED_KEYS - raw.keys()
    if unknown:
        raise ConfigError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"missing configuration fields: {', '.join(sorted(missing))}")

    return ExperimentConfig(**raw)
