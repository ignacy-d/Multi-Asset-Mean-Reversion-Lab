from pathlib import Path

import pytest

from mr_lab.config import ConfigError, load_config

EXAMPLE_CONFIG = Path(__file__).parents[1] / "configs" / "example.toml"


def test_example_configuration_loads() -> None:
    config = load_config(EXAMPLE_CONFIG)

    assert config.instrument == "EURUSD"
    assert config.timeframe == "M15"
    assert config.session == "london"
    assert config.strategy == "deviation_reversion"
    assert config.feature_model == "session_vwap_zscore"
    assert config.entry == {"threshold_sigma": 2.0}
    assert config.exit == {"holding_minutes": 60}
    assert config.cost_model == "baseline_fx"


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text('instrument = "EURUSD"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="missing configuration fields"):
        load_config(invalid)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        text.replace("\n[entry]", "\nsecret_override = true\n\n[entry]"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown configuration fields"):
        load_config(invalid)


def test_loaded_parameters_cannot_be_mutated() -> None:
    config = load_config(EXAMPLE_CONFIG)

    with pytest.raises(TypeError):
        config.entry["threshold_sigma"] = 1.5
