"""
ThreadX Bridge - Orchestration Layer
====================================

Couche intermédiaire entre UI (Dash, CLI) et Engine (calculs purs).
Fournit API typée, synchrone, documentée pour tous composants ThreadX.

Architecture:
    UI Layer (Dash/CLI)
          ↓
    Bridge Layer (controllers + models) ← CETTE COUCHE
          ↓
    Engine Layer (backtest, indicators, optimization, data)

Modules:
    models: DataClasses Request/Result typées (PEP 604)
    controllers: Wrappers synchrones autour Engine modules
    exceptions: Hiérarchie erreurs Bridge

Usage Principal (CLI):
    >>> from threadx.bridge import BacktestController, BacktestRequest
    >>> req = BacktestRequest(
    ...     symbol='BTCUSDT',
    ...     timeframe='1h',
    ...     strategy='bollinger_reversion',
    ...     params={'period': 20, 'std': 2.0}
    ... )
    >>> controller = BacktestController()
    >>> result = controller.run_backtest(req)
    >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")

Usage Principal (Dash Callback):
    >>> from dash import callback, Input, Output
    >>> from threadx.bridge import BacktestController, BacktestRequest
    >>>
    >>> @callback(
    ...     Output('results-div', 'children'),
    ...     Input('run-button', 'n_clicks'),
    ...     State('symbol-input', 'value')
    ... )
    >>> def run_backtest_callback(n_clicks, symbol):
    ...     req = BacktestRequest(symbol=symbol, ...)
    ...     controller = BacktestController()
    ...     result = controller.run_backtest(req)
    ...     return format_results(result)

Principes Bridge:
    - Aucune logique métier (délègue à Engine)
    - Aucune dépendance UI (pas d'import dash/tkinter)
    - Type-safe (mypy --strict compatible)
    - Documenté (Google-style docstrings)
    - Synchrone (async géré par P3 ThreadXBridge)

Author: ThreadX Framework
Version: Prompt 3 - Async Coordinator
"""

# Models (Request/Result DataClasses)
from threadx.bridge.models import (
    BacktestRequest,
    BacktestResult,
    Configuration,
    DataRequest,
    DataValidationResult,
    IndicatorRequest,
    IndicatorResult,
    SweepRequest,
    SweepResult,
)

# Controllers (Orchestration)
from threadx.bridge.controllers import (
    BacktestController,
    DataController,
    IndicatorController,
    SweepController,
    MetricsController,
    DataIngestionController,
    DiversityPipelineController,
)

# Exceptions (Error Handling)
from threadx.bridge.exceptions import (
    BacktestError,
    BridgeError,
    ConfigurationError,
    DataError,
    IndicatorError,
    SweepError,
    ValidationError,
)

# Async Coordinator (Prompt 3)
from threadx.bridge.async_coordinator import ThreadXBridge

# Configuration Constants
from threadx.bridge.config import DEFAULT_SWEEP_CONFIG

# Public API exports
__all__ = [
    # Models
    "BacktestRequest",
    "BacktestResult",
    "IndicatorRequest",
    "IndicatorResult",
    "SweepRequest",
    "SweepResult",
    "DataRequest",
    "DataValidationResult",
    "Configuration",
    # Controllers
    "BacktestController",
    "IndicatorController",
    "SweepController",
    "DataController",
    "MetricsController",
    "DataIngestionController",
    "DiversityPipelineController",
    # Exceptions
    "BridgeError",
    "BacktestError",
    "IndicatorError",
    "SweepError",
    "DataError",
    "ConfigurationError",
    "ValidationError",
    # Async Coordinator (P3)
    "ThreadXBridge",
    # Configuration
    "DEFAULT_SWEEP_CONFIG",
]

# Version info
__version__ = "0.1.0"
__author__ = "ThreadX Framework"
__license__ = "MIT"
