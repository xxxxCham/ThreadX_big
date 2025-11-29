"""
ThreadX UI - Strategy Registry (Re-export)
=========================================

Ce module re-exporte le registre principal des stratégies depuis
``threadx.strategy.registry`` pour maintenir la compatibilité des anciens
imports dans l'UI.

Aucune logique locale n'est définie ici : le registre source dans
``threadx.strategy.registry`` est l'unique référence pour les paramètres et
indicateurs. Toute modification doit donc être effectuée dans le module core.
"""

from __future__ import annotations

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
    "REGISTRY",
    "SWEEP_PRESETS",
    "GLOBAL_PARAM_DEFAULT_OVERRIDES",
    "STRATEGY_PARAM_DEFAULT_OVERRIDES",
    "list_strategies",
    "parameter_specs_for",
    "indicator_specs_for",
    "base_params_for",
    "indicators_for",
    "tunable_parameters_for",
    "resolve_range",
    "get_sweep_preset",
    "get_strategy_class",
    "get_strategy_category",
    "register_ai_strategy",
    "unregister_ai_strategy",
    "get_ai_strategies",
    "sync_ai_strategies_from_experimental",
]
