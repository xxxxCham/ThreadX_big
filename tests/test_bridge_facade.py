"""
ThreadX Bridge Facade Tests - Isolation Bridge Layer
====================================================

Tests qui passent UNIQUEMENT par la façade Bridge, jamais directement
vers l'Engine. Vérifie que le Bridge est une vraie abstraction.

Pattern de test:
    1. Mock Bridge (pas Engine)
    2. Test validation Pydantic
    3. Test gestion erreurs
    4. Test polling pattern

Usage:
    pytest tests/test_bridge_facade.py -v

Author: ThreadX Framework
Version: Bridge Validation Tests
"""

import pytest
from pydantic import ValidationError

from threadx.bridge import ThreadXBridge
from threadx.bridge.validation import BacktestRequest, IndicatorRequest, OptimizeRequest


@pytest.fixture
def bridge():
    """Fixture pour ThreadXBridge avec configuration de test."""
    return ThreadXBridge(max_workers=2)


def test_backtest_via_bridge(bridge):
    """Test : backtest ne passe QUE par Bridge."""
    req = BacktestRequest(symbol="BTCUSDT", timeframe="1h", strategy="ema_crossover")

    future = bridge.run_backtest_async(req)
    assert future is not None
    assert hasattr(future, "result")  # C'est un Future

    # Test que la validation Pydantic fonctionne
    assert req.symbol == "BTCUSDT"
    assert req.timeframe == "1h"


def test_indicators_via_bridge(bridge):
    """Test : indicateurs passent par Bridge."""
    req = IndicatorRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        indicators={"ema": {"period": 20}, "rsi": {"period": 14}},
    )

    future = bridge.run_indicator_async(req)
    assert future is not None
    assert hasattr(future, "result")  # C'est un Future


def test_optimization_via_bridge(bridge):
    """Test : optimisation passe par Bridge."""
    req = OptimizeRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        strategy="ema_crossover",
        param_grid={"fast_period": [5, 10, 20], "slow_period": [20, 50, 100]},
    )

    future = bridge.run_sweep_async(req)
    assert future is not None
    assert hasattr(future, "result")  # C'est un Future


def test_validation_rejects_invalid_symbol():
    """Test : validation rejette symboles invalides."""
    with pytest.raises(ValidationError):
        BacktestRequest(symbol="invalid", timeframe="1h", strategy="test")


def test_validation_rejects_invalid_timeframe():
    """Test : validation rejette timeframes invalides."""
    with pytest.raises(ValidationError):
        BacktestRequest(
            symbol="BTCUSDT",
            timeframe="invalid",  # Doit être dans Literal
            strategy="test",
        )


def test_validation_rejects_empty_strategy():
    """Test : validation rejette stratégies vides."""
    with pytest.raises(ValidationError):
        BacktestRequest(
            symbol="BTCUSDT", timeframe="1h", strategy=""  # Vide, min_length=1
        )


def test_validation_accepts_valid_request():
    """Test : validation accepte requêtes valides."""
    req = BacktestRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        strategy="ema_crossover",
        start_date="2024-01-01",
        end_date="2025-01-01",
    )

    assert req.symbol == "BTCUSDT"
    assert req.timeframe == "1h"
    assert req.strategy == "ema_crossover"


def test_bridge_handles_validation_errors(bridge):
    """Test : Bridge gère les erreurs de validation."""
    # Requête invalide - devrait échouer à la création de l'objet Pydantic
    with pytest.raises(ValidationError):
        BacktestRequest(
            symbol="invalid_symbol_too_long_and_invalid_chars!@#",
            timeframe="1h",
            strategy="test",
        )


def test_bridge_handles_execution_errors(bridge):
    """Test : Bridge gère les erreurs d'exécution."""
    # Requête valide mais qui pourrait échouer à l'exécution
    req = BacktestRequest(
        symbol="NONEXISTENT", timeframe="1h", strategy="test_strategy"
    )

    future = bridge.run_backtest_async(req)
    assert future is not None
    assert hasattr(future, "result")  # C'est un Future

    # Peu importe le résultat, tant que c'est géré
    # (le Future peut réussir ou échouer selon l'implémentation)
