"""
ThreadX Optimization Presets
=============================

Module de gestion des pré-réglages d'optimisation pour les indicateurs techniques.

Ce module charge les plages "classiques" depuis indicator_ranges.toml et les mappe
automatiquement aux paramètres des stratégies pour faciliter l'optimisation.

Usage:
    >>> from threadx.optimization.presets import get_strategy_preset
    >>> preset = get_strategy_preset('AmplitudeHunter')
    >>> print(preset.get_ranges())
"""

from .ranges import (
    IndicatorRangePreset,
    StrategyPresetMapper,
    get_indicator_range,
    get_strategy_preset,
    list_available_indicators,
    load_all_presets,
)

__all__ = [
    "IndicatorRangePreset",
    "StrategyPresetMapper",
    "get_indicator_range",
    "get_strategy_preset",
    "list_available_indicators",
    "load_all_presets",
]
