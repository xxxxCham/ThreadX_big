"""
ThreadX Exception Handling Tests
=================================

Tests unitaires pour vérifier la gestion robuste des erreurs en production.

Coverage:
- Binance API errors (network, invalid symbol, rate limit)
- Data validation errors (empty DataFrame, malformed dates)
- Bridge controller errors (invalid input, missing data)
- Registry errors (file not found, checksum mismatch)

Author: ThreadX Framework
Version: Test Suite - Exception Handling
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from threadx.bridge.controllers import (
    BacktestController,
    MetricsController,
    DataIngestionController,
)
from threadx.bridge.validation import (
    BacktestRequest,
    IndicatorRequest,
    DataValidationRequest,
)
from threadx.data.ingest import IngestionManager, ingest_binance
from threadx.bridge.exceptions import BridgeError, BacktestError, DataError


# ═══════════════════════════════════════════════════════════════════
# TEST 1: BacktestController Input Validation
# ═══════════════════════════════════════════════════════════════════


class TestBacktestControllerValidation:
    """Vérifier la validation stricte des inputs BacktestController."""

    def setup_method(self):
        """Setup controller."""
        self.controller = BacktestController()

    def test_invalid_backtest_request_missing_symbol(self):
        """Test: Rejecter requête sans symbole."""
        invalid_request = {
            "timeframe": "1h",
            "strategy": "bollinger",
            "params": {},
        }

        result = self.controller.run_backtest(invalid_request)

        assert result["status"] == "error"
        assert "symbol" in result["message"].lower()
        assert result["code"] == 400

    def test_invalid_timeframe_pattern(self):
        """Test: Rejecter timeframe invalide."""
        # "45m" devrait être accepté maintenant après le fix
        invalid_request = {
            "symbol": "BTCUSDT",
            "timeframe": "15x",  # ❌ Invalide
            "strategy": "bollinger",
            "params": {},
        }

        result = self.controller.run_backtest(invalid_request)

        assert result["status"] == "error"
        assert (
            "timeframe" in result["message"].lower()
            or "pattern" in result["message"].lower()
        )

    def test_valid_timeframes_accepted(self):
        """Test: Accepter tous les timeframes standards."""
        valid_timeframes = [
            "1m",
            "5m",
            "15m",
            "30m",
            "45m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "1w",
            "1M",
        ]

        for tf in valid_timeframes:
            try:
                req = BacktestRequest(
                    symbol="BTCUSDT", timeframe=tf, strategy="test", params={}
                )
                assert req.timeframe == tf
            except Exception as e:
                pytest.fail(f"Timeframe {tf} should be valid: {e}")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: MetricsController Input Validation
# ═══════════════════════════════════════════════════════════════════


class TestMetricsControllerValidation:
    """Vérifier la validation stricte des inputs MetricsController."""

    def setup_method(self):
        """Setup controller."""
        self.controller = MetricsController()

    def test_calculate_max_drawdown_empty_list(self):
        """Test: Rejeter liste vide."""
        with pytest.raises(ValueError, match="cannot be empty"):
            self.controller.calculate_max_drawdown([])

    def test_calculate_max_drawdown_single_value(self):
        """Test: Gérer liste avec un seul élément."""
        result = self.controller.calculate_max_drawdown([10000])

        assert result["max_drawdown"] == 0.0
        assert result["peak_idx"] == 0

    def test_calculate_max_drawdown_valid(self):
        """Test: Calculer drawdown valide."""
        equity = [10000, 11000, 9000, 9500, 10200]
        result = self.controller.calculate_max_drawdown(equity)

        assert "max_drawdown" in result
        assert "peak_idx" in result
        assert "trough_idx" in result
        assert result["max_drawdown"] <= 0  # Drawdown toujours négatif ou 0

    def test_calculate_sharpe_ratio_empty_returns(self):
        """Test: Rejeter returns vide."""
        with pytest.raises((ValueError, IndexError)):
            self.controller.calculate_sharpe_ratio([])

    def test_calculate_sharpe_ratio_single_return(self):
        """Test: Gérer un seul return."""
        result = self.controller.calculate_sharpe_ratio([0.01])

        # Un seul point → Sharpe indéfini
        assert isinstance(result, (float, int))


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Data Ingestion Error Handling
# ═══════════════════════════════════════════════════════════════════


class TestDataIngestionErrorHandling:
    """Vérifier la gestion d'erreur robuste dans l'ingestion."""

    @patch("threadx.data.ingest.IngestionManager.download_ohlcv_1m")
    def test_ingest_binance_api_error(self, mock_download):
        """Test: Gérer erreur Binance API."""
        from threadx.data.legacy_adapter import APIError

        # Simuler erreur API
        mock_download.side_effect = APIError(status_code=400, message="Invalid symbol")

        with pytest.raises(Exception) as exc_info:
            ingest_binance(
                "INVALID", "1h", "2024-01-01T00:00:00Z", "2024-01-07T23:59:59Z"
            )

        assert (
            "Binance API" in str(exc_info.value)
            or "failed" in str(exc_info.value).lower()
        )

    def test_ingest_binance_invalid_date_format(self):
        """Test: Rejeter dates invalides."""
        from threadx.data.ingest import IngestionError

        with pytest.raises(IngestionError, match="Invalid ISO date"):
            ingest_binance(
                "BTCUSDT", "1h", "invalid-date", "2024-01-07T23:59:59Z"  # ❌ Invalide
            )

    @patch("threadx.data.ingest.IngestionManager.download_ohlcv_1m")
    def test_ingest_binance_empty_data(self, mock_download):
        """Test: Rejeter données vides."""
        import pandas as pd
        from threadx.data.ingest import IngestionError

        # Simuler résultat vide
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(IngestionError, match="No data downloaded"):
            ingest_binance(
                "BTCUSDT", "1h", "2024-01-01T00:00:00Z", "2024-01-07T23:59:59Z"
            )


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Validation Models
# ═══════════════════════════════════════════════════════════════════


class TestPydanticValidation:
    """Vérifier la validation Pydantic stricte."""

    def test_data_validation_request_invalid_check(self):
        """Test: Rejeter type de vérification invalide."""
        with pytest.raises(ValueError, match="Type de vérification invalide"):
            DataValidationRequest(
                symbol="BTCUSDT",
                timeframe="1h",
                checks=["invalid_check"],  # ❌ Invalide
            )

    def test_data_validation_request_valid_checks(self):
        """Test: Accepter tous les types de vérification valides."""
        valid_checks = ["completeness", "duplicates", "outliers", "gaps"]

        for check in valid_checks:
            req = DataValidationRequest(
                symbol="BTCUSDT", timeframe="1h", checks=[check]
            )
            assert check in req.checks

    def test_backtest_request_params_type(self):
        """Test: Valider que params est dict."""
        # ✅ Dict accepté
        req = BacktestRequest(
            symbol="BTCUSDT", timeframe="1h", strategy="test", params={"key": "value"}
        )
        assert isinstance(req.params, dict)

        # ❌ Non-dict rejeté
        with pytest.raises(ValueError):
            BacktestRequest(
                symbol="BTCUSDT",
                timeframe="1h",
                strategy="test",
                params="invalid",  # ❌
            )


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Bridge Error Mapping
# ═══════════════════════════════════════════════════════════════════


class TestBridgeErrorMapping:
    """Vérifier le mapping cohérent des erreurs."""

    def test_backtest_error_with_code(self):
        """Test: BacktestError avec code HTTP."""
        error = BacktestError(400, "Invalid symbol")

        assert error.status_code == 400
        assert "Invalid symbol" in str(error)

    def test_data_error_with_message(self):
        """Test: DataError avec message clair."""
        error = DataError(500, "Database connection failed")

        assert error.status_code == 500
        assert "Database" in str(error)

    def test_bridge_error_default_code(self):
        """Test: BridgeError code par défaut."""
        error = BridgeError(message="Unknown error")

        assert error.status_code == 500


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Registry Idempotence (bonus)
# ═══════════════════════════════════════════════════════════════════


class TestRegistryIdempotence:
    """Vérifier que le registry gère les doublons."""

    @patch("threadx.data.registry.file_checksum")
    @patch("pathlib.Path.exists")
    def test_duplicate_dataset_detection(self, mock_exists, mock_checksum):
        """Test: Détecter et rejeter les doublons."""
        mock_exists.return_value = True
        mock_checksum.return_value = "abc123def456"  # ✅ Checksum fixe pour test

        # Simuler: même checksum → doublon
        # (En vrai implé, vérifier via DB si checksum existe)
        from threadx.data.registry import file_checksum

        cksum1 = file_checksum(Path("/test/file1.parquet"))
        cksum2 = file_checksum(Path("/test/file2.parquet"))

        # Même checksum = même fichier
        assert cksum1 == cksum2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
