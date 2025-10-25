"""
ThreadX Optimization Module
===========================

Module d'optimisation paramétrique unifié pour ThreadX.

Centralise tous les calculs via IndicatorBank (Phase 3) pour:
- Éviter la duplication de code
- Garantir la cohérence entre UI, optimisation et backtesting
- Utiliser le cache GPU intelligent existant
- Bénéficier de l'orchestration multi-GPU

Components:
- UnifiedOptimizationEngine: Moteur principal utilisant uniquement IndicatorBank
- ParametricOptimizationUI: Interface utilisateur intégrée
- Configuration et utilitaires

Author: ThreadX Framework
Version: Phase 10 - Unified Compute Engine
"""

from .engine import (
    UnifiedOptimizationEngine,
    create_unified_engine,
    DEFAULT_SWEEP_CONFIG,
)
from .ui import ParametricOptimizationUI, create_optimization_ui

__all__ = [
    "UnifiedOptimizationEngine",
    "create_unified_engine",
    "ParametricOptimizationUI",
    "create_optimization_ui",
    "DEFAULT_SWEEP_CONFIG",
]

__version__ = "1.0.0"
__author__ = "ThreadX Framework"
