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
    # Fonctions de convenance (préfixées pour éviter les conflits)
    generate_signals as bb_atr_generate_signals,
    backtest as bb_atr_backtest,
    create_default_params as bb_atr_create_default_params,
)

from .bollinger_dual import (
    # Paramètres et implémentation
    BollingerDualParams,
    BollingerDualStrategy,
    # Fonctions de convenance
    create_default_params as bollinger_dual_create_default_params,
)

from .amplitude_hunter import (
    # Paramètres et implémentation
    AmplitudeHunterParams,
    AmplitudeHunterStrategy,
    # Fonctions de convenance
    generate_signals as amplitude_hunter_generate_signals,
    backtest as amplitude_hunter_backtest,
    create_default_params as amplitude_hunter_create_default_params,
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
    "bb_atr_generate_signals",
    "bb_atr_backtest",
    "bb_atr_create_default_params",
    # Bollinger Dual Strategy exports
    "BollingerDualParams",
    "BollingerDualStrategy",
    "bollinger_dual_create_default_params",
    # AmplitudeHunter Strategy exports
    "AmplitudeHunterParams",
    "AmplitudeHunterStrategy",
    "amplitude_hunter_generate_signals",
    "amplitude_hunter_backtest",
    "amplitude_hunter_create_default_params",
]



