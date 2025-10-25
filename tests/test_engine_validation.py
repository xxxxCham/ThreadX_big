"""
Tests d'intégration pour validation dans BacktestEngine

Tests pour l'intégration du module validation dans BacktestEngine:
- Auto-configuration ValidationConfig
- Méthode run_backtest_with_validation()
- Fallback gracieux si module absent
- Logging automatique
- Alertes overfitting

Author: ThreadX Framework
Phase: Phase 2 Step 2.1 - Tests Intégration
"""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch

# Import du module à tester
from threadx.backtest.engine import BacktestEngine
from threadx.backtest.validation import ValidationConfig


# === Fixtures ===


@pytest.fixture
def sample_ohlcv_data():
    """Génère données OHLCV complètes pour BacktestEngine."""
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="1H", tz="UTC")
    np.random.seed(42)

    # Prix avec tendance
    close_prices = np.random.randn(len(dates)).cumsum() + 100

    data = pd.DataFrame(
        {
            "open": close_prices + np.random.randn(len(dates)) * 0.5,
            "high": close_prices + abs(np.random.randn(len(dates))) * 0.8,
            "low": close_prices - abs(np.random.randn(len(dates))) * 0.8,
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    # S'assurer high >= max(open, close) et low <= min(open, close)
    data["high"] = data[["open", "close", "high"]].max(axis=1)
    data["low"] = data[["open", "close", "low"]].min(axis=1)

    return data


@pytest.fixture
def sample_indicators(sample_ohlcv_data):
    """Génère indicateurs synthétiques Bollinger + ATR."""
    close = sample_ohlcv_data["close"]

    # Bollinger Bands simplifiés
    middle = close.rolling(20).mean().fillna(close)
    std = close.rolling(20).std().fillna(1.0)
    upper = middle + 2 * std
    lower = middle - 2 * std

    # ATR simplifié
    high = sample_ohlcv_data["high"]
    low = sample_ohlcv_data["low"]
    atr = (high - low).rolling(14).mean().fillna(2.0)

    return {"bollinger": (upper, middle, lower), "atr": atr.values}


@pytest.fixture
def sample_params():
    """Paramètres stratégie standard."""
    return {"entry_z": 2.0, "k_sl": 1.5, "leverage": 3}


# === Tests BacktestEngine Initialization ===


class TestBacktestEngineValidationInit:
    """Tests initialisation BacktestEngine avec validation."""

    def test_validation_auto_configured(self):
        """Test validation auto-configurée à l'init."""
        engine = BacktestEngine()

        # Devrait avoir validator et config
        assert hasattr(engine, "validator")
        assert hasattr(engine, "validation_config")

        # Si module disponible, devrait être non-None
        if engine.validator is not None:
            assert engine.validation_config is not None
            assert isinstance(engine.validation_config, ValidationConfig)

    def test_default_validation_config(self):
        """Test configuration par défaut."""
        engine = BacktestEngine()

        if engine.validation_config is not None:
            config = engine.validation_config
            assert config.method == "walk_forward"
            assert config.walk_forward_windows == 5
            assert config.purge_days == 1
            assert config.embargo_days == 1
            assert config.min_train_samples == 200
            assert config.min_test_samples == 50

    def test_engine_initialization_logging(self, caplog):
        """Test logging initialisation avec validation."""
        with caplog.at_level(logging.INFO):
            engine = BacktestEngine()

        # Chercher log validation
        log_messages = [record.message for record in caplog.records]

        # Devrait logger statut validation
        has_validation_log = any("Validation" in msg for msg in log_messages)
        assert has_validation_log


# === Tests run_backtest_with_validation ===


class TestRunBacktestWithValidation:
    """Tests pour méthode run_backtest_with_validation()."""

    def test_method_exists(self):
        """Test méthode existe."""
        engine = BacktestEngine()
        assert hasattr(engine, "run_backtest_with_validation")
        assert callable(engine.run_backtest_with_validation)

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_validation_basic_execution(
        self, sample_ohlcv_data, sample_indicators, sample_params
    ):
        """Test exécution basique validation."""
        engine = BacktestEngine()

        # Exécuter validation
        results = engine.run_backtest_with_validation(
            sample_ohlcv_data,
            sample_indicators,
            params=sample_params,
            symbol="BTCUSDC",
            timeframe="1h",
        )

        # Vérifier structure résultats
        assert isinstance(results, dict)
        assert "in_sample" in results
        assert "out_sample" in results
        assert "overfitting_ratio" in results
        assert "recommendation" in results
        assert "method" in results

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_validation_results_structure(
        self, sample_ohlcv_data, sample_indicators, sample_params
    ):
        """Test structure détaillée résultats."""
        engine = BacktestEngine()

        results = engine.run_backtest_with_validation(
            sample_ohlcv_data,
            sample_indicators,
            params=sample_params,
            symbol="BTCUSDC",
            timeframe="1h",
        )

        # Vérifier métriques in_sample
        assert "mean_sharpe_ratio" in results["in_sample"]
        assert "std_sharpe_ratio" in results["in_sample"]
        assert "mean_total_return" in results["in_sample"]
        assert "mean_max_drawdown" in results["in_sample"]

        # Vérifier métriques out_sample
        assert "mean_sharpe_ratio" in results["out_sample"]
        assert "std_sharpe_ratio" in results["out_sample"]

        # Vérifier overfitting ratio est nombre
        assert isinstance(results["overfitting_ratio"], (int, float))
        assert results["overfitting_ratio"] >= 0

        # Vérifier recommandation non vide
        assert len(results["recommendation"]) > 0

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_validation_with_custom_config(
        self, sample_ohlcv_data, sample_indicators, sample_params
    ):
        """Test validation avec config personnalisée."""
        engine = BacktestEngine()

        # Config train/test simple
        custom_config = ValidationConfig(
            method="train_test", train_ratio=0.8, purge_days=2
        )

        results = engine.run_backtest_with_validation(
            sample_ohlcv_data,
            sample_indicators,
            params=sample_params,
            symbol="BTCUSDC",
            timeframe="1h",
            validation_config=custom_config,
        )

        assert results["method"] == "train_test"

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_temporal_integrity_check(
        self, sample_ohlcv_data, sample_indicators, sample_params
    ):
        """Test vérification intégrité temporelle."""
        engine = BacktestEngine()

        # Données valides ne devraient pas lever erreur
        results = engine.run_backtest_with_validation(
            sample_ohlcv_data,
            sample_indicators,
            params=sample_params,
            symbol="BTCUSDC",
            timeframe="1h",
        )

        assert results is not None

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_temporal_integrity_failure(self, sample_indicators, sample_params):
        """Test échec intégrité temporelle."""
        engine = BacktestEngine()

        # Créer données avec duplicates
        dates = pd.date_range("2023-01-01", periods=100, freq="1H", tz="UTC")
        bad_data = pd.DataFrame(
            {
                "open": range(100),
                "high": range(100),
                "low": range(100),
                "close": range(100),
                "volume": [1000] * 100,
            },
            index=dates,
        )

        # Dupliquer première ligne
        bad_data = pd.concat([bad_data.iloc[[0]], bad_data])

        # Devrait lever ValueError
        with pytest.raises(ValueError, match="timestamps dupliqués"):
            engine.run_backtest_with_validation(
                bad_data,
                sample_indicators,
                params=sample_params,
                symbol="TEST",
                timeframe="1h",
            )


# === Tests Logging et Alertes ===


class TestValidationLogging:
    """Tests logging et alertes validation."""

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_validation_start_logging(
        self, sample_ohlcv_data, sample_indicators, sample_params, caplog
    ):
        """Test logging démarrage validation."""
        engine = BacktestEngine()

        with caplog.at_level(logging.INFO):
            results = engine.run_backtest_with_validation(
                sample_ohlcv_data,
                sample_indicators,
                params=sample_params,
                symbol="BTCUSDC",
                timeframe="1h",
            )

        # Chercher logs validation
        log_messages = [record.message for record in caplog.records]

        # Devrait logger démarrage
        has_start_log = any(
            "Démarrage backtest avec validation" in msg for msg in log_messages
        )
        assert has_start_log

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_validation_results_logging(
        self, sample_ohlcv_data, sample_indicators, sample_params, caplog
    ):
        """Test logging résultats validation."""
        engine = BacktestEngine()

        with caplog.at_level(logging.INFO):
            results = engine.run_backtest_with_validation(
                sample_ohlcv_data,
                sample_indicators,
                params=sample_params,
                symbol="BTCUSDC",
                timeframe="1h",
            )

        log_messages = [record.message for record in caplog.records]

        # Devrait logger métriques
        has_metrics_log = any("Sharpe" in msg for msg in log_messages)
        assert has_metrics_log


# === Tests Fallback et Error Handling ===


class TestValidationFallback:
    """Tests fallback si module validation absent."""

    def test_validation_unavailable_error(
        self, sample_ohlcv_data, sample_indicators, sample_params
    ):
        """Test erreur si validation non disponible."""
        engine = BacktestEngine()

        # Si validator est None, devrait lever ValueError
        if engine.validator is None:
            with pytest.raises(ValueError, match="Module validation non disponible"):
                engine.run_backtest_with_validation(
                    sample_ohlcv_data,
                    sample_indicators,
                    params=sample_params,
                    symbol="TEST",
                    timeframe="1h",
                )

    def test_standard_run_still_works(
        self, sample_ohlcv_data, sample_indicators, sample_params
    ):
        """Test méthode run() standard fonctionne toujours."""
        engine = BacktestEngine()

        # run() standard ne devrait pas être affecté
        result = engine.run(
            sample_ohlcv_data,
            sample_indicators,
            params=sample_params,
            symbol="BTCUSDC",
            timeframe="1h",
        )

        # Devrait retourner RunResult
        assert hasattr(result, "equity")
        assert hasattr(result, "returns")
        assert hasattr(result, "trades")


# === Tests Performance ===


class TestValidationPerformance:
    """Tests performance validation."""

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_validation_execution_time(
        self, sample_ohlcv_data, sample_indicators, sample_params
    ):
        """Test temps d'exécution raisonnable."""
        import time

        engine = BacktestEngine()

        start = time.time()
        results = engine.run_backtest_with_validation(
            sample_ohlcv_data,
            sample_indicators,
            params=sample_params,
            symbol="BTCUSDC",
            timeframe="1h",
        )
        duration = time.time() - start

        # Validation avec 5 windows ne devrait pas prendre > 60s
        assert duration < 60.0


# === Tests Cas d'Usage Réels ===


class TestRealWorldUsage:
    """Tests cas d'usage production."""

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_overfitting_detection_scenario(self, sample_ohlcv_data, sample_indicators):
        """Test scénario détection overfitting."""
        engine = BacktestEngine()

        # Paramètres potentiellement overfittés (beaucoup de params)
        overfit_params = {
            "entry_z": 2.347,
            "k_sl": 1.789,
            "leverage": 3.2,
            "risk_pct": 0.0234,
            "trail_k": 1.456,
        }

        results = engine.run_backtest_with_validation(
            sample_ohlcv_data,
            sample_indicators,
            params=overfit_params,
            symbol="BTCUSDC",
            timeframe="1h",
        )

        # Devrait avoir ratio et recommandation
        assert "overfitting_ratio" in results
        assert "recommendation" in results

    @pytest.mark.skipif(
        not hasattr(BacktestEngine(), "validator")
        or BacktestEngine().validator is None,
        reason="Module validation non disponible",
    )
    def test_strategy_comparison_workflow(self, sample_ohlcv_data, sample_indicators):
        """Test workflow comparaison stratégies."""
        engine = BacktestEngine()

        # Tester plusieurs stratégies
        strategies = [
            {"entry_z": 2.0, "k_sl": 1.5, "leverage": 3},
            {"entry_z": 2.5, "k_sl": 2.0, "leverage": 2},
        ]

        results = []
        for params in strategies:
            result = engine.run_backtest_with_validation(
                sample_ohlcv_data,
                sample_indicators,
                params=params,
                symbol="BTCUSDC",
                timeframe="1h",
            )
            results.append(
                {
                    "params": params,
                    "ratio": result["overfitting_ratio"],
                    "oos_sharpe": result["out_sample"]["mean_sharpe_ratio"],
                }
            )

        # Devrait avoir résultats pour chaque stratégie
        assert len(results) == len(strategies)

        # Trier par robustesse
        results_sorted = sorted(results, key=lambda x: x["ratio"])
        assert len(results_sorted) == len(strategies)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
