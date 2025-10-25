"""
Tests unitaires pour threadx.backtest.validation

Tests pour le module de validation anti-overfitting incluant:
- ValidationConfig dataclass
- BacktestValidator avec walk-forward et train/test split
- check_temporal_integrity()
- detect_lookahead_bias()
- Overfitting ratio calculation

Author: ThreadX Framework
Phase: Phase 2 Step 2.1 - Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Import du module à tester
from threadx.backtest.validation import (
    ValidationConfig,
    BacktestValidator,
    check_temporal_integrity,
    detect_lookahead_bias,
)


# === Fixtures ===


@pytest.fixture
def sample_data():
    """Génère données OHLCV synthétiques pour tests."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="1H", tz="UTC")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)).cumsum() + 100,
            "high": np.random.randn(len(dates)).cumsum() + 102,
            "low": np.random.randn(len(dates)).cumsum() + 98,
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    return data


@pytest.fixture
def valid_config():
    """Configuration par défaut valide."""
    return ValidationConfig(
        method="walk_forward",
        walk_forward_windows=5,
        purge_days=1,
        embargo_days=1,
        min_train_samples=100,
        min_test_samples=50,
    )


@pytest.fixture
def simple_backtest_func():
    """Fonction de backtest simplifiée pour tests."""

    def backtest(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        """Backtest simple retournant métriques aléatoires."""
        np.random.seed(42)
        return {
            "sharpe_ratio": np.random.uniform(0.5, 2.5),
            "total_return": np.random.uniform(-0.1, 0.5),
            "max_drawdown": np.random.uniform(-0.3, -0.05),
            "win_rate": np.random.uniform(0.4, 0.7),
            "profit_factor": np.random.uniform(0.8, 2.5),
        }

    return backtest


# === Tests ValidationConfig ===


class TestValidationConfig:
    """Tests pour ValidationConfig dataclass."""

    def test_config_default_values(self):
        """Test valeurs par défaut."""
        config = ValidationConfig()

        assert config.method == "walk_forward"
        assert config.train_ratio == 0.7
        assert config.test_ratio == 0.3
        assert config.walk_forward_windows == 5
        assert config.purge_days == 0
        assert config.embargo_days == 0
        assert config.min_train_samples == 100
        assert config.min_test_samples == 50

    def test_config_custom_values(self):
        """Test valeurs personnalisées."""
        config = ValidationConfig(
            method="train_test",
            train_ratio=0.8,
            test_ratio=0.2,  # train_ratio + test_ratio doit être <= 1.0
            purge_days=2,
            embargo_days=3,
            min_train_samples=200,
        )

        assert config.method == "train_test"
        assert config.train_ratio == 0.8
        assert config.test_ratio == 0.2
        assert config.purge_days == 2
        assert config.embargo_days == 3
        assert config.min_train_samples == 200

    def test_config_validation_method(self):
        """Test validation méthode."""
        # Méthodes valides
        for method in ["walk_forward", "train_test", "k_fold"]:
            config = ValidationConfig(method=method)
            assert config.method == method

        # Méthode invalide devrait lever ValueError
        with pytest.raises(ValueError, match="method doit être"):
            ValidationConfig(method="invalid")


# === Tests check_temporal_integrity ===


class TestCheckTemporalIntegrity:
    """Tests pour check_temporal_integrity()."""

    def test_valid_data(self, sample_data):
        """Test données valides."""
        # Ne devrait pas lever d'erreur
        check_temporal_integrity(sample_data)

    def test_non_datetime_index(self):
        """Test erreur si index non-datetime."""
        df = pd.DataFrame({"close": [1, 2, 3]}, index=[0, 1, 2])

        with pytest.raises(ValueError, match="Index doit être DatetimeIndex"):
            check_temporal_integrity(df)

    def test_future_data_detection(self):
        """Test détection données futures."""
        # Créer données avec timestamps futurs
        future_dates = pd.date_range("2030-01-01", "2030-12-31", freq="1H", tz="UTC")
        df_future = pd.DataFrame(
            {"close": range(len(future_dates))}, index=future_dates
        )

        with pytest.raises(ValueError, match="DONNÉES FUTURES DÉTECTÉES"):
            check_temporal_integrity(df_future)

    def test_duplicate_timestamps(self, sample_data):
        """Test détection duplicates."""
        # Dupliquer premier timestamp
        df_dup = sample_data.copy()
        df_dup = pd.concat([df_dup.iloc[[0]], df_dup])

        with pytest.raises(ValueError, match="TIMESTAMPS DUPLIQUÉS"):
            check_temporal_integrity(df_dup)

    def test_non_chronological_order(self, sample_data):
        """Test détection ordre non-chronologique."""
        # Inverser ordre
        df_reversed = sample_data.iloc[::-1].copy()

        with pytest.raises(ValueError, match="INDEX NON CHRONOLOGIQUE"):
            check_temporal_integrity(df_reversed)

    def test_large_temporal_gaps(self):
        """Test avertissement gaps temporels."""
        # Créer données avec gap de 60 jours
        dates1 = pd.date_range("2023-01-01", "2023-01-31", freq="1H", tz="UTC")
        dates2 = pd.date_range("2023-04-01", "2023-04-30", freq="1H", tz="UTC")
        dates = dates1.union(dates2)

        df_gaps = pd.DataFrame({"close": range(len(dates))}, index=dates)

        # Devrait logger warning mais pas lever erreur
        check_temporal_integrity(df_gaps)


# === Tests detect_lookahead_bias ===


class TestDetectLookaheadBias:
    """Tests pour detect_lookahead_bias()."""

    def test_valid_split(self, sample_data):
        """Test split valide (train < test)."""
        train = sample_data.iloc[:5000]
        test = sample_data.iloc[5100:]

        # Ne devrait pas lever d'erreur
        detect_lookahead_bias(train, test)

    def test_lookahead_bias_detection(self, sample_data):
        """Test détection overlap train/test."""
        train = sample_data.iloc[:5000]
        test = sample_data.iloc[4000:6000]  # Overlap!

        with pytest.raises(ValueError, match="LOOK-AHEAD BIAS DÉTECTÉ"):
            detect_lookahead_bias(train, test, raise_on_bias=True)

    def test_warning_mode(self, sample_data):
        """Test mode warning (pas d'erreur levée)."""
        train = sample_data.iloc[:5000]
        test = sample_data.iloc[4000:6000]  # Overlap

        # Mode warning ne devrait pas lever erreur
        has_bias = detect_lookahead_bias(train, test, raise_on_bias=False)
        assert has_bias is True


# === Tests BacktestValidator ===


class TestBacktestValidator:
    """Tests pour BacktestValidator."""

    def test_validator_initialization(self, valid_config):
        """Test initialisation validator."""
        validator = BacktestValidator(valid_config)

        assert validator.config == valid_config
        assert validator.config.method == "walk_forward"

    def test_walk_forward_split(self, sample_data, valid_config):
        """Test walk_forward_split génère bonnes fenêtres."""
        validator = BacktestValidator(valid_config)

        windows = list(validator.walk_forward_split(sample_data, n_windows=3))

        # Devrait avoir 3 fenêtres
        assert len(windows) == 3

        # Chaque fenêtre devrait avoir train et test
        for train, test in windows:
            assert isinstance(train, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)
            assert len(train) > 0
            assert len(test) > 0

            # Train doit être avant test (pas de look-ahead)
            assert train.index.max() < test.index.min()

    def test_walk_forward_with_purge(self, sample_data):
        """Test purge entre train et test."""
        config = ValidationConfig(purge_days=1)
        validator = BacktestValidator(config)

        windows = list(validator.walk_forward_split(sample_data, n_windows=2))

        for train, test in windows:
            # Gap entre train et test devrait être >= purge_days
            gap = test.index.min() - train.index.max()
            assert gap >= timedelta(days=1)

    def test_train_test_split(self, sample_data, valid_config):
        """Test train_test_split simple."""
        validator = BacktestValidator(valid_config)

        train, test = validator.train_test_split(sample_data)

        # Vérifier proportions approximatives (70/30)
        total_len = len(sample_data)
        assert abs(len(train) / total_len - 0.7) < 0.05
        assert abs(len(test) / total_len - 0.3) < 0.05

        # Train doit être avant test
        assert train.index.max() < test.index.min()

    def test_validate_backtest_walk_forward(
        self, sample_data, valid_config, simple_backtest_func
    ):
        """Test validate_backtest avec walk-forward."""
        validator = BacktestValidator(valid_config)

        results = validator.validate_backtest(
            backtest_func=simple_backtest_func,
            data=sample_data,
            params={"test": "param"},
        )

        # Vérifier structure résultats
        assert "in_sample" in results
        assert "out_sample" in results
        assert "overfitting_ratio" in results
        assert "recommendation" in results
        assert "method" in results
        assert results["method"] == "walk_forward"

        # Vérifier métriques in_sample
        assert "mean_sharpe_ratio" in results["in_sample"]
        assert "std_sharpe_ratio" in results["in_sample"]
        assert "mean_total_return" in results["in_sample"]

        # Vérifier overfitting ratio est un nombre
        assert isinstance(results["overfitting_ratio"], (int, float))
        assert results["overfitting_ratio"] > 0

    def test_validate_backtest_train_test(self, sample_data, simple_backtest_func):
        """Test validate_backtest avec train/test split."""
        config = ValidationConfig(method="train_test")
        validator = BacktestValidator(config)

        results = validator.validate_backtest(
            backtest_func=simple_backtest_func, data=sample_data, params={}
        )

        assert results["method"] == "train_test"
        assert "overfitting_ratio" in results

    def test_overfitting_ratio_calculation(self, sample_data, valid_config):
        """Test calcul ratio d'overfitting."""

        def fixed_backtest(data, params):
            """Backtest avec métriques fixes."""
            return {
                "sharpe_ratio": 2.0,
                "total_return": 0.3,
                "max_drawdown": -0.1,
                "win_rate": 0.6,
                "profit_factor": 2.0,
            }

        validator = BacktestValidator(valid_config)
        results = validator.validate_backtest(fixed_backtest, sample_data, {})

        # Ratio devrait être proche de 1.0 (performances identiques)
        assert abs(results["overfitting_ratio"] - 1.0) < 0.1

    def test_recommendation_excellent(self, sample_data, valid_config):
        """Test recommandation EXCELLENT (ratio < 1.2)."""

        def good_backtest(data, params):
            return {
                "sharpe_ratio": 1.5,
                "total_return": 0.25,
                "max_drawdown": -0.12,
                "win_rate": 0.58,
                "profit_factor": 1.8,
            }

        validator = BacktestValidator(valid_config)
        results = validator.validate_backtest(good_backtest, sample_data, {})

        if results["overfitting_ratio"] < 1.2:
            assert "EXCELLENT" in results["recommendation"]

    def test_recommendation_critical(self, sample_data, valid_config):
        """Test recommandation CRITIQUE (ratio > 2.0)."""
        call_count = {"count": 0}

        def overfitted_backtest(data, params):
            """Simule overfitting: IS bon, OOS mauvais."""
            call_count["count"] += 1
            # Premier appel (train) bon, second (test) mauvais
            if call_count["count"] % 2 == 1:
                return {
                    "sharpe_ratio": 3.0,
                    "total_return": 0.5,
                    "max_drawdown": -0.1,
                    "win_rate": 0.7,
                    "profit_factor": 2.5,
                }
            else:
                return {
                    "sharpe_ratio": 0.5,
                    "total_return": 0.05,
                    "max_drawdown": -0.3,
                    "win_rate": 0.45,
                    "profit_factor": 0.9,
                }

        validator = BacktestValidator(valid_config)
        results = validator.validate_backtest(overfitted_backtest, sample_data, {})

        # Devrait détecter overfitting
        assert results["overfitting_ratio"] > 1.0

    def test_insufficient_data(self, valid_config):
        """Test erreur si données insuffisantes."""
        # Données trop petites
        small_data = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2023-01-01", periods=3, freq="1H", tz="UTC"),
        )

        validator = BacktestValidator(valid_config)

        # Devrait lever erreur ou warning
        with pytest.raises((ValueError, IndexError)):
            list(validator.walk_forward_split(small_data, n_windows=5))


# === Tests d'Intégration ===


class TestValidationIntegration:
    """Tests d'intégration complets."""

    def test_full_validation_pipeline(self, sample_data):
        """Test pipeline complet de validation."""
        # 1. Vérifier intégrité
        check_temporal_integrity(sample_data)

        # 2. Créer config
        config = ValidationConfig(
            method="walk_forward", walk_forward_windows=3, purge_days=1
        )

        # 3. Créer validator
        validator = BacktestValidator(config)

        # 4. Définir backtest
        def my_backtest(data, params):
            return {
                "sharpe_ratio": np.random.uniform(1.0, 2.0),
                "total_return": np.random.uniform(0.1, 0.3),
                "max_drawdown": np.random.uniform(-0.2, -0.05),
                "win_rate": np.random.uniform(0.5, 0.65),
                "profit_factor": np.random.uniform(1.2, 2.0),
            }

        # 5. Exécuter validation
        results = validator.validate_backtest(my_backtest, sample_data, {})

        # 6. Vérifier résultats
        assert results["overfitting_ratio"] > 0
        assert len(results["recommendation"]) > 0
        assert results["method"] == "walk_forward"

    def test_validation_with_different_methods(self, sample_data, simple_backtest_func):
        """Test validation avec différentes méthodes."""
        methods = ["walk_forward", "train_test"]

        for method in methods:
            config = ValidationConfig(method=method)
            validator = BacktestValidator(config)

            results = validator.validate_backtest(simple_backtest_func, sample_data, {})

            assert results["method"] == method
            assert "overfitting_ratio" in results


# === Tests Edge Cases ===


class TestEdgeCases:
    """Tests cas limites."""

    def test_empty_dataframe(self, valid_config):
        """Test avec DataFrame vide."""
        empty_df = pd.DataFrame()

        with pytest.raises((ValueError, IndexError)):
            check_temporal_integrity(empty_df)

    def test_single_row(self, valid_config):
        """Test avec une seule ligne."""
        single_row = pd.DataFrame(
            {"close": [100]}, index=pd.DatetimeIndex(["2023-01-01"], tz="UTC")
        )

        validator = BacktestValidator(valid_config)

        with pytest.raises((ValueError, IndexError)):
            validator.train_test_split(single_row)

    def test_zero_sharpe_handling(self, sample_data, valid_config):
        """Test gestion Sharpe ratio nul."""

        def zero_sharpe_backtest(data, params):
            return {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.5,
                "profit_factor": 1.0,
            }

        validator = BacktestValidator(valid_config)
        results = validator.validate_backtest(zero_sharpe_backtest, sample_data, {})

        # Devrait gérer division par zéro gracieusement
        assert "overfitting_ratio" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
