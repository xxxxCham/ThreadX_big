import pytest

from threadx.optimization.run import validate_cli_config
from threadx.config.errors import ConfigurationError


def _base_config():
    return {
        "run": {"iterations": 1, "seed": 0, "verbose": False},
        "params": {
            "timeframe": {"value": "1h"},
            "lookback_hours": {"values": [24, 48, 72]},
            "symbols": {"values": ["BTCUSDC", "ETHUSDC"]},
        },
        "scoring": {"primary": "sharpe"},
        "output": {"dir": "artifacts/sweeps/minimal", "format": "parquet"},
    }


def test_validate_cli_config_accepts_value_and_values(tmp_path):
    cfg = _base_config()
    # Doit passer sans lever
    validate_cli_config(cfg, str(tmp_path / "dummy.toml"))


def test_validate_cli_config_rejects_plain_list_for_param_block(tmp_path):
    cfg = _base_config()
    cfg["params"]["lookback_hours"] = [24, 48]  # non conforme

    with pytest.raises(ConfigurationError):
        validate_cli_config(cfg, str(tmp_path / "dummy.toml"))


def test_validate_cli_config_rejects_unknown_keys_in_param_block(tmp_path):
    cfg = _base_config()
    cfg["params"]["lookback_hours"] = {"grid": [24, 48]}  # clé non supportée

    with pytest.raises(ConfigurationError):
        validate_cli_config(cfg, str(tmp_path / "dummy.toml"))
