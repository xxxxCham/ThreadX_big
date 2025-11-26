"""
ThreadX UI - Strategy Registry (Re-export)
============================================

Ce fichier re-exporte le registry depuis threadx.strategy.registry
pour maintenir la compatibilité avec le code existant de l'UI.

DÉPRÉCIÉ: Préférer importer directement depuis threadx.strategy.registry

Migration:
    # Ancien (déprécié)
    from threadx.ui.strategy_registry import REGISTRY, base_params_for

    # Nouveau (recommandé)
    from threadx.strategy.registry import REGISTRY, base_params_for

Author: ThreadX Framework
Version: 2.0.0 - Refactored to re-export from strategy.registry
"""

from __future__ import annotations

# Re-export tout depuis le module core
from threadx.strategy.registry import (
    GLOBAL_PARAM_DEFAULT_OVERRIDES,
    REGISTRY,
    STRATEGY_PARAM_DEFAULT_OVERRIDES,
    SWEEP_PRESETS,
    base_params_for,
    get_ai_strategies,
    get_strategy_category,
    get_strategy_class,
    get_sweep_preset,
    indicator_specs_for,
    indicators_for,
    list_strategies,
    parameter_specs_for,
    register_ai_strategy,
    resolve_range,
    sync_ai_strategies_from_experimental,
    tunable_parameters_for,
    unregister_ai_strategy,
)

__all__ = [
    # Registry
    "REGISTRY",
    "SWEEP_PRESETS",
    "GLOBAL_PARAM_DEFAULT_OVERRIDES",
    "STRATEGY_PARAM_DEFAULT_OVERRIDES",
    # Accès specs
    "list_strategies",
    "parameter_specs_for",
    "indicator_specs_for",
    "base_params_for",
    "indicators_for",
    "tunable_parameters_for",
    "resolve_range",
    "get_sweep_preset",
    # Chargement classes
    "get_strategy_class",
    "get_strategy_category",
    # AI-Evolved
    "register_ai_strategy",
    "unregister_ai_strategy",
    "get_ai_strategies",
    "sync_ai_strategies_from_experimental",
]

# Note: Pas de définitions locales - tout est re-exporté depuis threadx.strategy.registry
