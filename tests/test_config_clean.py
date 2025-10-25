# ThreadX Configuration Tests - Core Suite
"""
Tests essentiels pour le système de configuration ThreadX
Version allégée focalisée sur les cas critiques
"""

import pytest
import tempfile
import os
from io import StringIO
from unittest.mock import patch

from threadx.config.settings import DEFAULT_SETTINGS
from threadx.config.loaders import (
    TOMLConfigLoader,
    load_settings,
    load_config_dict,
    print_config,
)
from threadx.config.errors import ConfigurationError

# Configurations de test
MINIMAL_VALID_CONFIG = """
[paths]
data_root = "./data"
logs = "./logs"

[gpu]
enable_gpu = true
devices = ["default"]

[performance]
max_workers = 4
cache_ttl_sec = 3600

[trading]
supported_timeframes = ["1h", "4h"]
default_timeframe = "1h"
"""

INVALID_TOML = """
[paths]
data_root = "./data"
logs = "./logs
"""  # Missing quote


class TestConfigFiles:
    """Tests de chargement et validation TOML"""

    def test_config_file_not_found(self):
        """Fichier de configuration introuvable"""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config_dict("nonexistent_file.toml")

    def test_invalid_toml_syntax(self):
        """TOML avec syntaxe invalide"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(INVALID_TOML)
            f.flush()

        try:
            with pytest.raises(ConfigurationError, match="Invalid TOML syntax"):
                load_config_dict(f.name)
        finally:
            os.unlink(f.name)

    def test_valid_config_loading(self):
        """Chargement configuration valide"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(MINIMAL_VALID_CONFIG)
            f.flush()

        try:
            config_data = load_config_dict(f.name)
            assert "paths" in config_data
            assert "gpu" in config_data
            assert config_data["paths"]["data_root"] == "./data"
        finally:
            os.unlink(f.name)


class TestRequiredSections:
    """Tests de validation des sections obligatoires"""

    def test_missing_required_sections(self):
        """Détection sections manquantes"""
        config_data = {
            "paths": {"data_root": "./data"}
            # Missing: gpu, performance, trading
        }

        # Simuler un loader avec données minimales
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("")  # Fichier vide
            f.flush()

        try:
            loader = TOMLConfigLoader(f.name)
            loader.config_data = config_data
            errors = loader.validate_config()
        finally:
            os.unlink(f.name)
        assert len(errors) >= 3
        assert any(
            "Missing required configuration section: gpu" in err for err in errors
        )
        assert any(
            "Missing required configuration section: performance" in err
            for err in errors
        )
        assert any(
            "Missing required configuration section: trading" in err for err in errors
        )


class TestPathValidation:
    """Tests de validation des chemins"""

    def test_absolute_paths_forbidden_when_disabled(self):
        """Chemins absolus interdits quand allow_absolute_paths=false"""
        config_data = {
            "paths": {"logs": "/tmp/absolute_path"},
            "security": {"allow_absolute_paths": False},
            "gpu": {"enable_gpu": True},
            "performance": {"max_workers": 4},
            "trading": {"supported_timeframes": ["1h"]},
        }

        # Simuler un loader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("")
            f.flush()

        try:
            loader = TOMLConfigLoader(f.name)
            loader.config_data = config_data
            errors = loader._validate_paths(check_only=True)
        finally:
            os.unlink(f.name)
        assert any("Absolute path not allowed" in err for err in errors)


class TestGPULoadBalance:
    """Tests de validation GPU load balance"""

    def test_gpu_load_balance_invalid_sum(self):
        """Load balance doit sommer à 1.0"""
        config_data = {
            "paths": {"data_root": "./data"},
            "performance": {"max_workers": 4},
            "trading": {"supported_timeframes": ["1h"]},
            "gpu": {"load_balance": {"GPU_A": 0.6, "GPU_B": 0.5}},  # Sum = 1.1
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader._validate_gpu_config(check_only=True)
        assert any("must sum to 1.0" in err for err in errors)


class TestCLIOverrides:
    """Tests des overrides de ligne de commande"""

    def test_cli_overrides_basic(self):
        """CLI overrides fonctionnent correctement"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(MINIMAL_VALID_CONFIG)
            f.flush()

        try:
            settings = load_settings(
                config_path=f.name,
                cli_args=["--data-root", "./custom_data", "--disable-gpu"],
            )

            assert settings.DATA_ROOT == "./custom_data"
            assert settings.ENABLE_GPU == False
        finally:
            os.unlink(f.name)


class TestSmokeTests:
    """Tests de fumée et d'intégration"""

    def test_print_config_no_crash(self):
        """print_config ne doit jamais crasher"""
        settings = DEFAULT_SETTINGS

        # Capturer stdout
        captured = StringIO()
        with patch("sys.stdout", captured):
            print_config(settings)  # Ne doit pas lever d'exception

        output = captured.getvalue()

        # Vérifier contenu clé présent
        assert (
            "Target Tasks/Min" in output
            or "GPU Enabled" in output
            or "Data Root" in output
        )
        assert len(output.strip()) > 0

    def test_default_settings_valid(self):
        """DEFAULT_SETTINGS est une configuration valide"""
        settings = DEFAULT_SETTINGS

        # Vérifications basiques
        assert hasattr(settings, "DATA_ROOT")
        assert hasattr(settings, "ENABLE_GPU")
        assert hasattr(settings, "MAX_WORKERS")
        assert hasattr(settings, "SUPPORTED_TF")

        # Types corrects
        assert isinstance(settings.DATA_ROOT, str)
        assert isinstance(settings.ENABLE_GPU, bool)
        assert isinstance(settings.MAX_WORKERS, int)
        assert isinstance(settings.SUPPORTED_TF, tuple)

    def test_settings_immutable(self):
        """Settings dataclass est frozen (immutable)"""
        settings = DEFAULT_SETTINGS

        # Tentative de modification doit échouer
        with pytest.raises(AttributeError):
            settings.DATA_ROOT = "./modified"  # type: ignore


# Fixtures
@pytest.fixture
def temp_config_file():
    """Fixture pour créer un fichier de config temporaire"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(MINIMAL_VALID_CONFIG)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def invalid_config_file():
    """Fixture pour créer un fichier de config invalide"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(INVALID_TOML)
        f.flush()
        yield f.name
    os.unlink(f.name)


# Tests avec fixtures
@pytest.mark.integration
def test_full_integration_with_real_config(temp_config_file):
    """Test d'intégration complet avec vrai fichier config"""
    settings = load_settings(config_path=temp_config_file, cli_args=[])
    assert settings is not None
    assert hasattr(settings, "DATA_ROOT")


def test_end_to_end_valid_config():
    """Test complet avec config valide"""
    config_content = """
    [paths]
    data_root = "./test_data"
    logs = "./test_logs"
    
    [gpu]
    enable_gpu = false
    devices = ["cpu"]
    
    [performance]
    max_workers = 2
    cache_ttl_sec = 300
    
    [trading]
    supported_timeframes = ["5m", "1h"]
    default_timeframe = "1h"
    
    [security]
    validate_paths = false
    allow_absolute_paths = true
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        f.flush()

    try:
        settings = load_settings(config_path=f.name, cli_args=[])

        # Vérifications
        assert settings.DATA_ROOT == "./test_data"
        assert settings.ENABLE_GPU == False
        assert settings.MAX_WORKERS == 2
        assert settings.CACHE_TTL_SEC == 300
        assert "5m" in settings.SUPPORTED_TF
        assert "1h" in settings.SUPPORTED_TF

    finally:
        os.unlink(f.name)
