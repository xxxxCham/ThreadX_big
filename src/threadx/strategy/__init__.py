"""
ThreadX Phase 4 - Strategy Layer
================================

Module de stratégies de trading avec gestion avancée du risque.

Modules:
- model.py : Types de données (Trade, RunStats, Strategy Protocol)
- bb_atr.py : Stratégie Bollinger Bands + ATR avec améliorations

Caractéristiques:
- Protocol Pattern pour extensibilité des stratégies
- Intégration native avec Phase 3 Indicators Layer
- Gestion positions avancée (trailing stops, risk sizing)
- Sérialisation JSON complète des résultats
- Backtest déterministe et reproductible

Améliorations vs TradXPro:
- Architecture modulaire et testable
- Paramètres typés avec validation
- Cache intelligent via IndicatorBank
- Filtrage min PnL et micro-optimisations
"""

from .model import (
    # Types de données
    Trade,
    TradeDict,
    RunStats,
    RunStatsDict,
    # Protocol interface
    Strategy,
    # JSON utilities
    ThreadXJSONEncoder,
    save_run_results,
    load_run_results,
    # Validation
    validate_ohlcv_dataframe,
    validate_strategy_params,
)

from .bb_atr import (
    # Paramètres et implémentation
    BBAtrParams,
    BBAtrStrategy,
    # Fonctions de convenance
    generate_signals,
    backtest,
    create_default_params,
)

__version__ = "4.0.0"

__all__ = [
    # Model exports
    "Trade",
    "TradeDict",
    "RunStats",
    "RunStatsDict",
    "Strategy",
    "ThreadXJSONEncoder",
    "save_run_results",
    "load_run_results",
    "validate_ohlcv_dataframe",
    "validate_strategy_params",
    # BB+ATR Strategy exports
    "BBAtrParams",
    "BBAtrStrategy",
    "generate_signals",
    "backtest",
    "create_default_params",
]
