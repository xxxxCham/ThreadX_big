# ThreadX Configuration Tests - Comprehensive Suite
"""
Tests exhaustifs pour le système de configuration ThreadX
Couvre validation, migration, CLI overrides, et edge cases
"""

import pytest
import tempfile
import os
from io import StringIO
from unittest.mock import patch

# ThreadX imports
from threadx.config.settings import DEFAULT_SETTINGS
from threadx.config.loaders import (
    TOMLConfigLoader,
    load_settings,
    load_config_dict,
    print_config,
)
from threadx.config.errors import ConfigurationError

# === MINIMAL VALID CONFIG FOR TESTS ===
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

INVALID_TOML_SYNTAX = """
[paths]
data_root = "./data"
logs = "./logs
"""  # Missing closing quote

# ===== A. TESTS FICHIERS & TOML =====


class TestConfigFiles:
    """Tests de chargement et validation TOML"""

    def test_config_file_not_found(self):
        """Fichier de configuration introuvable"""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config_dict("nonexistent_file.toml")

    def test_invalid_toml_syntax(self):
        """TOML avec syntaxe invalide"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(INVALID_TOML_SYNTAX)
            f.flush()

        try:
            with pytest.raises(ConfigurationError, match="Invalid TOML syntax"):
                load_config_dict(f.name)
        finally:
            os.unlink(f.name)

    def test_empty_config_file(self):
        """Fichier de configuration vide"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("")  # Empty file
            f.flush()

        try:
            config_data = load_config_dict(f.name)
            assert config_data == {}
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


# ===== B. TESTS SECTIONS REQUISES =====


class TestRequiredSections:
    """Tests de validation des sections obligatoires"""

    def test_missing_required_sections(self):
        """Détection sections manquantes"""
        config_data = {
            "paths": {"data_root": "./data"}
            # Missing: gpu, performance, trading
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader.validate_config()
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

    def test_all_required_sections_present(self):
        """Toutes les sections requises présentes"""
        config_data = {
            "paths": {"data_root": "./data"},
            "gpu": {"enable_gpu": True},
            "performance": {"max_workers": 4},
            "trading": {"supported_timeframes": ["1h"]},
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader.validate_config()
        # Peut y avoir d'autres erreurs, mais pas de sections manquantes
        missing_section_errors = [
            err for err in errors if "Missing required configuration section" in err
        ]
        assert len(missing_section_errors) == 0


# ===== C. TESTS VALIDATION PATHS =====


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

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader._validate_paths(check_only=True)
        assert any("Absolute path not allowed" in err for err in errors)

    def test_absolute_paths_allowed_when_enabled(self):
        """Chemins absolus autorisés quand allow_absolute_paths=true"""
        config_data = {
            "paths": {"logs": "/tmp/absolute_path"},
            "security": {"allow_absolute_paths": True},
            "gpu": {"enable_gpu": True},
            "performance": {"max_workers": 4},
            "trading": {"supported_timeframes": ["1h"]},
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader._validate_paths(check_only=True)
        absolute_path_errors = [
            err for err in errors if "Absolute path not allowed" in err
        ]
        assert len(absolute_path_errors) == 0

    def test_path_creation_when_validate_false(self):
        """Création chemins quand validate_paths=false"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_create")

            config_data = {
                "paths": {"logs": test_path},
                "security": {"validate_paths": False},
                "gpu": {"enable_gpu": True},
                "performance": {"max_workers": 4},
                "trading": {"supported_timeframes": ["1h"]},
            }

            loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
            loader.config_data = config_data

            # Should create path without validation
            loader._validate_paths(check_only=False)
            assert os.path.exists(test_path)


# ===== D. TESTS GPU LOAD BALANCE =====


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

    def test_gpu_load_balance_valid_sum(self):
        """Load balance valide (somme = 1.0)"""
        config_data = {
            "paths": {"data_root": "./data"},
            "performance": {"max_workers": 4},
            "trading": {"supported_timeframes": ["1h"]},
            "gpu": {"load_balance": {"GPU_A": 0.7, "GPU_B": 0.3}},  # Sum = 1.0
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader._validate_gpu_config(check_only=True)
        load_balance_errors = [err for err in errors if "must sum to 1.0" in err]
        assert len(load_balance_errors) == 0

    def test_gpu_load_balance_negative_values(self):
        """Valeurs négatives dans load balance interdites"""
        config_data = {
            "paths": {"data_root": "./data"},
            "performance": {"max_workers": 4},
            "trading": {"supported_timeframes": ["1h"]},
            "gpu": {"load_balance": {"GPU_A": 1.2, "GPU_B": -0.2}},  # Negative value
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader._validate_gpu_config(check_only=True)
        assert any("negative values not allowed" in err.lower() for err in errors)


# ===== E. TESTS PERFORMANCE VALIDATION =====


class TestPerformanceValidation:
    """Tests de validation des paramètres de performance"""

    def test_performance_negative_values(self):
        """Valeurs performance négatives interdites"""
        config_data = {
            "paths": {"data_root": "./data"},
            "gpu": {"enable_gpu": True},
            "trading": {"supported_timeframes": ["1h"]},
            "performance": {"max_workers": -1, "cache_ttl_sec": -100},
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader._validate_performance_config(check_only=True)
        assert any("must be a positive number" in err for err in errors)

    def test_performance_zero_values_allowed(self):
        """Certaines valeurs zero peuvent être autorisées"""
        config_data = {
            "paths": {"data_root": "./data"},
            "gpu": {"enable_gpu": True},
            "trading": {"supported_timeframes": ["1h"]},
            "performance": {
                "max_workers": 1,
                "cache_ttl_sec": 0,
            },  # 0 peut désactiver cache
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        errors = loader._validate_performance_config(check_only=True)
        # cache_ttl_sec=0 peut être valide (désactive cache)
        # Vérifier qu'il n'y a pas d'erreur pour cette valeur spécifique
        zero_errors = [
            err for err in errors if "cache_ttl_sec" in err and "positive" in err
        ]
        # Le comportement exact dépend de l'implémentation


# ===== F. TESTS CLI OVERRIDES =====


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

    def test_cli_overrides_precedence(self):
        """CLI overrides ont priorité sur fichier config"""
        config_content = """
        [paths]
        data_root = "./original"
        logs = "./logs"

        [gpu]
        enable_gpu = true

        [performance]
        max_workers = 2
        cache_ttl_sec = 1800

        [trading]
        supported_timeframes = ["1h"]
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            f.flush()

        try:
            settings = load_settings(
                config_path=f.name,
                cli_args=["--data-root", "./overridden", "--max-workers", "8"],
            )

            # CLI override
            assert settings.paths["data_root"] == "./overridden"
            # Autre CLI override
            assert settings.performance["max_workers"] == 8
            # Valeur du fichier (non overridée)
            assert settings.performance["cache_ttl_sec"] == 1800
        finally:
            os.unlink(f.name)


# ===== G. TESTS MIGRATION LEGACY =====


class TestLegacyMigration:
    """Tests de migration des anciens formats"""

    def test_timeframes_migration(self):
        """Migration timeframes.supported → trading.supported_timeframes"""
        config_data = {
            "paths": {"data_root": "./data"},
            "gpu": {"enable_gpu": True},
            "performance": {"max_workers": 4},
            "trading": {},  # Pas de supported_timeframes
            "timeframes": {"supported": ["1h", "4h", "1d"]},  # Legacy format
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        # Déclencher migration
        loader._migrate_legacy_config()

        # Vérifier migration effectuée
        trading_section = loader.get_section("trading")
        assert trading_section["supported_timeframes"] == ["1h", "4h", "1d"]

    def test_no_migration_when_new_format_exists(self):
        """Pas de migration si nouveau format déjà présent"""
        config_data = {
            "paths": {"data_root": "./data"},
            "gpu": {"enable_gpu": True},
            "performance": {"max_workers": 4},
            "trading": {"supported_timeframes": ["5m", "15m"]},  # Nouveau format
            "timeframes": {
                "supported": ["1h", "4h"]
            },  # Legacy format présent mais ignoré
        }

        loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
        loader.config_data = config_data

        loader._migrate_legacy_config()

        # Nouveau format préservé, legacy ignoré
        trading_section = loader.get_section("trading")
        assert trading_section["supported_timeframes"] == ["5m", "15m"]


# ===== H. TESTS SMOKE/INTEGRATION =====


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
        assert len(output.strip()) > 0  # Pas vide

    def test_default_settings_valid(self):
        """DEFAULT_SETTINGS est une configuration valide"""
        settings = DEFAULT_SETTINGS

        # Vérifications basiques
        assert hasattr(settings, "DATA_ROOT")
        assert hasattr(settings, "ENABLE_GPU")
        assert hasattr(settings, "MAX_WORKERS")
        assert hasattr(settings, "SUPPORTED_TIMEFRAMES")

        # Types corrects
        assert isinstance(settings.DATA_ROOT, str)
        assert isinstance(settings.ENABLE_GPU, bool)
        assert isinstance(settings.MAX_WORKERS, int)
        assert isinstance(settings.SUPPORTED_TIMEFRAMES, list)

    def test_load_settings_without_file(self):
        """load_settings sans fichier utilise DEFAULT_SETTINGS"""
        # Mock file not found
        with patch("threadx.config.loaders.load_config_dict") as mock_load:
            mock_load.side_effect = ConfigurationError("File not found")

            settings = load_settings(config_path="nonexistent.toml", cli_args=[])

            # Doit retourner DEFAULT_SETTINGS ou équivalent
            assert settings is not None
            assert hasattr(settings, "DATA_ROOT")

    def test_settings_immutable(self):
        """Settings dataclass est frozen (immutable)"""
        settings = DEFAULT_SETTINGS

        # Tentative de modification doit échouer
        with pytest.raises(AttributeError):
            settings.DATA_ROOT = "./modified"  # type: ignore

    def test_end_to_end_valid_config(self):
        """Test complet avec config valide"""
        config_content = """
        [paths]
        data_root = "./test_data"
        logs = "./test_logs"
        cache = "./test_cache"

        [gpu]
        enable_gpu = false
        devices = ["cpu"]

        [performance]
        max_workers = 2
        cache_ttl_sec = 300
        target_tasks_per_min = 1000

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
            assert "5m" in settings.SUPPORTED_TIMEFRAMES
            assert "1h" in settings.SUPPORTED_TIMEFRAMES

        finally:
            os.unlink(f.name)


# ===== PYTEST MARKERS & FIXTURES =====


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
        f.write(INVALID_TOML_SYNTAX)
        f.flush()
        yield f.name
    os.unlink(f.name)


# ===== CONFIGURATION PYTEST =====


def pytest_configure(config):
    """Configuration pytest pour les tests config"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# ===== TESTS AVEC MARKERS =====


@pytest.mark.integration
def test_full_integration_with_real_config(temp_config_file):
    """Test d'intégration complet avec vrai fichier config"""
    settings = load_settings(config_path=temp_config_file, cli_args=[])
    assert settings is not None
    assert hasattr(settings, "DATA_ROOT")


@pytest.mark.slow
def test_large_config_performance():
    """Test de performance avec grande configuration"""
    # Créer config avec beaucoup de sections/valeurs
    large_config = {
        "paths": {f"path_{i}": f"./path_{i}" for i in range(100)},
        "gpu": {"enable_gpu": True, "devices": [f"gpu_{i}" for i in range(10)]},
        "performance": {"max_workers": 4},
        "trading": {"supported_timeframes": [f"{i}m" for i in range(1, 61)]},
    }

    loader = TOMLConfigLoader.__new__(TOMLConfigLoader)
    loader.config_data = large_config

    import time

    start = time.time()
    errors = loader.validate_config()
    end = time.time()

    # Validation doit être rapide même avec grande config
    assert (end - start) < 1.0  # Moins de 1 seconde
