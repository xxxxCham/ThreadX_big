"""
ThreadX Optimization Module
===========================

Module d'optimisation paramétrique unifié pour ThreadX.

Centralise tous les calculs via IndicatorBank (Phase 3) pour:
- éviter la duplication de code
- garantir la cohérence entre UI, optimisation et backtesting
- utiliser le cache GPU intelligent existant
- bénéficier de l'orchestration multi-GPU

Components:
- UnifiedOptimizationEngine: Moteur principal utilisant uniquement IndicatorBank
- ParametricOptimizationUI: Interface utilisateur intégrée
- Configuration et utilitaires

Author: ThreadX Framework
Version: Phase 10 - Unified Compute Engine
"""

from .engine import (
    DEFAULT_SWEEP_CONFIG,
    UnifiedOptimizationEngine,
    create_unified_engine,
)

# Presets for optimization
from .presets import (
    IndicatorRangePreset,
    StrategyPresetMapper,
    get_indicator_range,
    get_strategy_preset,
    list_available_indicators,
    load_all_presets,
)
from .ui import ParametricOptimizationUI, create_optimization_ui

__all__ = [
    # Engine
    "UnifiedOptimizationEngine",
    "create_unified_engine",
    "DEFAULT_SWEEP_CONFIG",
    # UI
    "ParametricOptimizationUI",
    "create_optimization_ui",
    # Presets
    "IndicatorRangePreset",
    "StrategyPresetMapper",
    "get_indicator_range",
    "get_strategy_preset",
    "list_available_indicators",
    "load_all_presets",
]

__version__ = "1.0.0"
__author__ = "ThreadX Framework"
