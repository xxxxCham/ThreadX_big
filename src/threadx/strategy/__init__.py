"""
ThreadX Phase 4 - Strategy Layer
================================

Module de stratégies de trading avec gestion avancée du risque.

Modules:
- model.py : Types de données (Trade, RunStats, Strategy Protocol)
- bollinger_dual.py : Stratégie Bollinger Bands double bande
- ma_crossover.py : Stratégie croisement de moyennes mobiles
- ema_cross.py : Stratégie croisement EMA
- atr_channel.py : Stratégie canal ATR

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

from .bollinger_dual import (
    # Paramètres et implémentation
    BollingerDualParams,
    BollingerDualStrategy,
)
from .bollinger_dual import (
    # Fonctions de convenance
    create_default_params as bollinger_dual_create_default_params,
)
from .ema_stochastic_scalp import (
    # EMA + Stochastic Scalping Strategy
    EMAStochScalpParams,
    EMAStochScalpStrategy,
)
from .ma_crossover import (
    # MA Crossover Strategy (test/validation)
    MACrossoverParams,
    MACrossoverStrategy,
)
from .ema_cross import (
    EMACrossParams,
    EMACrossStrategy,
)
from .atr_channel import (
    ATRChannelParams,
    ATRChannelStrategy,
)
from .model import (
    RunStats,
    RunStatsDict,
    # Protocol interface
    Strategy,
    # JSON utilities
    ThreadXJSONEncoder,
    # Types de données
    Trade,
    TradeDict,
    load_run_results,
    save_run_results,
    # Validation
    validate_ohlcv_dataframe,
    validate_strategy_params,
)
from .volume_profile_breakout import VolumeProfileBreakout
from .vwap_momentum_reversion import VWAPMomentumReversion

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
    # Bollinger Dual Strategy exports
    "BollingerDualParams",
    "BollingerDualStrategy",
    "bollinger_dual_create_default_params",
<<<<<<< HEAD
    # MA Crossover Strategy exports (test/validation)
    "MACrossoverParams",
    "MACrossoverStrategy",
    # EMA Cross Strategy exports
    "EMACrossParams",
    "EMACrossStrategy",
    # ATR Channel Strategy exports
    "ATRChannelParams",
    "ATRChannelStrategy",
=======
    # AmplitudeHunter Strategy exports
    "AmplitudeHunterParams",
    "AmplitudeHunterStrategy",
    "amplitude_hunter_generate_signals",
    "amplitude_hunter_backtest",
    "amplitude_hunter_create_default_params",
    # EMA + Stochastic Scalping Strategy exports
    "EMAStochScalpParams",
    "EMAStochScalpStrategy",
    # MA Crossover Strategy exports (test/validation)
    "MACrossoverParams",
    "MACrossoverStrategy",
    # Nouvelles Stratégies 2025
    "VolumeProfileBreakout",
    "VWAPMomentumReversion",
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
]
