"""
ThreadX - Strategy Registry (Core)
===================================

Registre central des stratégies et leurs spécifications de paramètres.

Ce module contient :
- REGISTRY: Mapping stratégie → métadonnées (params, indicateurs dérivés)
- Fonctions d'accès aux specs
- Gestion dynamique des stratégies AI-Evolved

Architecture :
- Les `params` sont la source de vérité unique
- Les `indicators` sont DÉRIVÉS automatiquement des params pertinents
- Pas de duplication entre indicators et params

Note: Ce fichier est dans threadx.strategy (core), pas threadx.ui,
      pour permettre l'utilisation en CLI/scripts sans dépendance UI.

Author: ThreadX Framework
Version: 2.0.0 - Refactoring: fusion indicators/params
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMÈTRES COMMUNS (SHARED SPECS)
# =============================================================================
# Définitions réutilisables pour éviter la duplication

_COMMON_RISK_PARAMS: dict[str, dict[str, Any]] = {
    "risk_per_trade": {
        "default": 0.02,
        "min": 0.005,
        "max": 0.05,
        "step": 0.005,
        "type": "float",
        "label": "Risque par Trade (fraction du capital)",
        "opt_range": (0.01, 0.03),
    },
    "max_hold_bars": {
        "default": 100,
        "min": 20,
        "max": 300,
        "step": 20,
        "type": "int",
        "label": "Durée Maximale en Position (barres)",
        "opt_range": (50, 200),
    },
    "leverage": {
        "default": 1.0,
        "min": 1.0,
        "max": 5.0,
        "step": 0.5,
        "type": "float",
        "label": "Levier (1.0 = sans levier)",
        "tunable": False,
    },
}

_COMMON_FEE_PARAMS: dict[str, dict[str, Any]] = {
    "fee_bps": {
        "default": 4.5,
        "min": 0.0,
        "max": 10.0,
        "step": 0.5,
        "type": "float",
        "label": "Frais (basis points)",
        "tunable": False,
    },
    "slippage_bps": {
        "default": 0.0,
        "min": 0.0,
        "max": 5.0,
        "step": 0.5,
        "type": "float",
        "label": "Slippage (basis points)",
        "tunable": False,
    },
}


def _merge_common_params(*param_dicts: dict) -> dict[str, Any]:
    """Fusionne plusieurs dicts de paramètres (deep copy)."""
    merged: dict[str, Any] = {}
    for d in param_dicts:
        for k, v in d.items():
            merged[k] = dict(v) if isinstance(v, dict) else v
    return merged


# =============================================================================
# REGISTRY PRINCIPAL
# =============================================================================
#
# Structure par stratégie:
# - category: "Core" | "AI-Evolved" | "Experimental"
# - params: specs complètes des paramètres (SOURCE DE VÉRITÉ)
# - indicator_mapping: mapping param → indicateur pour dérivation auto
# - (AI-Evolved) class_name, module_name, generated_at
#
# NOTE: Plus de section "indicators" séparée - dérivée des params
#

REGISTRY: dict[str, dict[str, Any]] = {
    # =========================================================================
    # MA_CROSSOVER - Moving Average Crossover
    # =========================================================================
    "MA_Crossover": {
        "category": "Core",
        "description": "Croisement de moyennes mobiles simples (SMA)",
        # Mapping: quel param alimente quel indicateur
        "indicator_mapping": {
            "sma_fast": {"window": "fast_period"},
            "sma_slow": {"window": "slow_period"},
        },
        "params": _merge_common_params(
            {
                # ── Indicateurs (Moving Averages) ──
                "fast_period": {
                    "default": 10,
                    "min": 5,
                    "max": 50,
                    "step": 5,
                    "type": "int",
                    "label": "Période SMA Rapide",
                    "opt_range": (5, 30),
                    "indicator": "sma_fast",  # Tag indicateur
                },
                "slow_period": {
                    "default": 30,
                    "min": 20,
                    "max": 100,
                    "step": 10,
                    "type": "int",
                    "label": "Période SMA Lente",
                    "opt_range": (20, 60),
                    "indicator": "sma_slow",
                },
                # ── Risk Management ──
                "stop_loss_pct": {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.5,
                    "type": "float",
                    "label": "Stop Loss % (fixe)",
                    "opt_range": (1.5, 3.0),
                },
                "take_profit_pct": {
                    "default": 4.0,
                    "min": 2.0,
                    "max": 10.0,
                    "step": 1.0,
                    "type": "float",
                    "label": "Take Profit % (fixe)",
                    "opt_range": (3.0, 6.0),
                },
            },
            _COMMON_RISK_PARAMS,
            _COMMON_FEE_PARAMS,
        ),
    },
    # =========================================================================
    # EMA_CROSS - Exponential Moving Average Crossover
    # =========================================================================
    "EMA_Cross": {
        "category": "Core",
        "description": "Croisement de moyennes mobiles exponentielles (EMA)",
        "indicator_mapping": {
            "ema_fast": {"window": "fast_window"},
            "ema_slow": {"window": "slow_window"},
        },
        "params": _merge_common_params(
            {
                # ── Indicateurs (EMA) ──
                "fast_window": {
                    "default": 12,
                    "min": 2,
                    "max": 50,
                    "step": 2,
                    "type": "int",
                    "label": "Fenêtre EMA rapide",
                    "opt_range": (8, 20),
                    "indicator": "ema_fast",
                },
                "slow_window": {
                    "default": 26,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "type": "int",
                    "label": "Fenêtre EMA lente",
                    "opt_range": (20, 50),
                    "indicator": "ema_slow",
                },
                # ── Risk Management ──
                "stop_loss_pct": {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.5,
                    "type": "float",
                    "label": "Stop Loss %",
                    "opt_range": (1.5, 3.0),
                },
                "take_profit_pct": {
                    "default": 4.0,
                    "min": 2.0,
                    "max": 10.0,
                    "step": 1.0,
                    "type": "float",
                    "label": "Take Profit %",
                    "opt_range": (3.0, 6.0),
                },
            },
            _COMMON_RISK_PARAMS,
            _COMMON_FEE_PARAMS,
        ),
    },
    # =========================================================================
    # ATR_CHANNEL - ATR-based Channel Breakout
    # =========================================================================
    "ATR_Channel": {
        "category": "Core",
        "description": "Breakout basé sur canal ATR",
        "indicator_mapping": {
            "atr": {"window": "atr_window", "mult": "atr_mult"},
        },
        "params": _merge_common_params(
            {
                # ── Indicateurs (ATR) ──
                "atr_window": {
                    "default": 14,
                    "min": 5,
                    "max": 50,
                    "step": 1,
                    "type": "int",
                    "label": "Période ATR",
                    "opt_range": (10, 21),
                    "indicator": "atr",
                },
                "atr_mult": {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.5,
                    "type": "float",
                    "label": "Multiplicateur ATR",
                    "opt_range": (1.5, 3.0),
                    "indicator": "atr",
                },
                # ── Risk Management ──
                "stop_loss_pct": {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.5,
                    "type": "float",
                    "label": "Stop Loss %",
                    "opt_range": (1.5, 3.5),
                },
                "take_profit_pct": {
                    "default": 5.0,
                    "min": 2.0,
                    "max": 12.0,
                    "step": 1.0,
                    "type": "float",
                    "label": "Take Profit %",
                    "opt_range": (3.0, 8.0),
                },
            },
            _COMMON_RISK_PARAMS,
            _COMMON_FEE_PARAMS,
        ),
    },
    # =========================================================================
    # BOLLINGER_DUAL - Bollinger Bands + ATR
    # =========================================================================
    "Bollinger_Dual": {
        "category": "Core",
        "description": "Stratégie Bollinger Bands avec stops ATR dynamiques",
        "indicator_mapping": {
            "bollinger": {"window": "bb_window", "std": "bb_std"},
            "atr": {"period": "atr_period"},
        },
        "params": _merge_common_params(
            {
                # ── Indicateurs (Bollinger) ──
                "bb_window": {
                    "default": 20,
                    "min": 10,
                    "max": 50,
                    "step": 5,
                    "type": "int",
                    "label": "Période Bollinger (SMA)",
                    "opt_range": (15, 30),
                    "indicator": "bollinger",
                },
                "bb_std": {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 3.0,
                    "step": 0.25,
                    "type": "float",
                    "label": "Sigma Bollinger (K)",
                    "opt_range": (1.5, 2.5),
                    "indicator": "bollinger",
                },
                # ── Indicateurs (ATR) ──
                "atr_period": {
                    "default": 14,
                    "min": 7,
                    "max": 21,
                    "step": 2,
                    "type": "int",
                    "label": "Période ATR",
                    "opt_range": (10, 18),
                    "indicator": "atr",
                },
                "atr_multiplier": {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.25,
                    "type": "float",
                    "label": "Multiplicateur ATR pour stops",
                    "opt_range": (1.0, 2.5),
                },
                # ── Logique d'entrée ──
                "entry_z": {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.25,
                    "type": "float",
                    "label": "Seuil Z-score entrée",
                    "opt_range": (0.8, 1.5),
                },
                # ── Filtres ──
                "min_pnl_pct": {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.05,
                    "step": 0.01,
                    "type": "float",
                    "label": "Filtre PnL Minimum (%)",
                    "opt_range": (0.0, 0.03),
                },
                "min_spacing_bars": {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 2,
                    "type": "int",
                    "label": "Espacement Minimum entre Trades",
                    "opt_range": (3, 10),
                },
                "ema_filter_period": {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 10,
                    "type": "int",
                    "label": "Période EMA Filtre Tendance (0 = désactivé)",
                    "opt_range": (0, 30),
                    "tunable": True,
                },
            },
            _COMMON_RISK_PARAMS,
            _COMMON_FEE_PARAMS,
        ),
    },
}


# =============================================================================
# PRÉRÉGLAGES GLOBAUX POUR LES SWEEPS
# =============================================================================

SWEEP_PRESETS: dict[str, dict[str, Any]] = {
    "max_hold_bars": {
        "min": 50,
        "max": 200,
        "n_values": 3,
        "step": 75,
        "reason": "Durée max avec 3 points (50/125/200) pour court/moyen/long.",
    },
    "risk_per_trade": {
        "min": 0.005,
        "max": 0.02,
        "n_values": 3,
        "step": 0.0075,
        "reason": "Bornes LLM (0.5%→2%) avec 3 points.",
    },
}


# =============================================================================
# OVERRIDES DE DÉFAUTS
# =============================================================================

GLOBAL_PARAM_DEFAULT_OVERRIDES: dict[str, Any] = {
    "risk_per_trade": 0.01,
    "max_hold_bars": 80,
}

STRATEGY_PARAM_DEFAULT_OVERRIDES: dict[str, dict[str, Any]] = {
    "MA_Crossover": {
        "max_hold_bars": 50,
    },
}


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def _scalar_default(spec: Any) -> Any:
    """Extrait la valeur par défaut d'une spec."""
    if isinstance(spec, dict):
        return spec.get("default")
    return spec


def _apply_overrides_to_params(
    params: dict[str, Any], overrides: dict[str, Any]
) -> None:
    """Applique des overrides aux defaults des params (in-place)."""
    for key, override in overrides.items():
        spec = params.get(key)
        if isinstance(spec, dict):
            new_default = override
            min_v = spec.get("min")
            max_v = spec.get("max")
            if isinstance(min_v, (int, float)) and new_default < min_v:
                new_default = min_v
            if isinstance(max_v, (int, float)) and new_default > max_v:
                new_default = max_v
            spec["default"] = new_default


# Appliquer overrides au chargement du module
for _definition in REGISTRY.values():
    _params = _definition.get("params", {})
    _apply_overrides_to_params(_params, GLOBAL_PARAM_DEFAULT_OVERRIDES)

for _strategy_name, _overrides in STRATEGY_PARAM_DEFAULT_OVERRIDES.items():
    if _strategy_name in REGISTRY:
        _params = REGISTRY[_strategy_name].get("params", {})
        _apply_overrides_to_params(_params, _overrides)


# =============================================================================
# API PUBLIQUE - ACCÈS AUX SPECS
# =============================================================================


def list_strategies(category: str | None = None) -> list[str]:
    """
    Retourne la liste des stratégies disponibles.

    Args:
        category: Filtre optionnel ("Core", "AI-Evolved", "Experimental")

    Returns:
        Liste des noms de stratégies
    """
    if category is None:
        return list(REGISTRY.keys())
    return [name for name, meta in REGISTRY.items() if meta.get("category") == category]


def parameter_specs_for(strategy: str) -> dict[str, Any]:
    """Retourne les specs complètes des paramètres pour une stratégie."""
    if strategy not in REGISTRY:
        raise KeyError(f"Strategy '{strategy}' not found in REGISTRY")
    result: dict[str, Any] = REGISTRY[strategy].get("params", {})
    return result


def indicator_specs_for(strategy: str) -> dict[str, Any]:
    """
    Dérive les specs indicateurs depuis les params (plus de duplication).

    Retourne un dict {indicator_name: {param: spec}} construit depuis
    les params ayant un tag 'indicator'.
    """
    if strategy not in REGISTRY:
        raise KeyError(f"Strategy '{strategy}' not found in REGISTRY")

    params = REGISTRY[strategy].get("params", {})
    mapping = REGISTRY[strategy].get("indicator_mapping", {})

    # Construire specs indicateurs depuis le mapping
    indicators: dict[str, dict[str, Any]] = {}

    for indicator_name, param_mapping in mapping.items():
        indicators[indicator_name] = {}
        for ind_param, strategy_param in param_mapping.items():
            if strategy_param in params:
                # Copier la spec du param vers l'indicateur
                indicators[indicator_name][ind_param] = dict(params[strategy_param])

    return indicators


def base_params_for(strategy: str) -> dict[str, Any]:
    """Retourne les valeurs par défaut des paramètres."""
    specs = parameter_specs_for(strategy)
    return {
        key: spec.get("default") if isinstance(spec, dict) else spec
        for key, spec in specs.items()
    }


def indicators_for(strategy: str) -> dict[str, dict[str, Any]]:
    """Retourne les valeurs par défaut des indicateurs."""
    specs = indicator_specs_for(strategy)
    defaults: dict[str, dict[str, Any]] = {}
    for indicator, params in specs.items():
        defaults[indicator] = {}
        for key, spec in params.items():
            defaults[indicator][key] = _scalar_default(spec)
    return defaults


def tunable_parameters_for(strategy: str) -> dict[str, dict[str, Any]]:
    """Retourne les paramètres optimisables (avec min/max/step)."""
    specs = parameter_specs_for(strategy)
    tunables: dict[str, dict[str, Any]] = {}

    for key, raw_spec in specs.items():
        if not isinstance(raw_spec, dict):
            continue

        # Exclure les non-tunables
        if raw_spec.get("tunable") is False:
            continue

        param_type = raw_spec.get("type")
        if param_type not in {"int", "float"}:
            continue

        # Vérifier qu'on a min/max
        if "min" not in raw_spec or "max" not in raw_spec:
            continue

        tunables[key] = dict(raw_spec)

    return tunables


def resolve_range(spec: dict[str, Any]) -> tuple[Any, Any]:
    """Retourne la plage (min, max) pour un paramètre."""
    opt_range = spec.get("opt_range")
    if isinstance(opt_range, (list, tuple)) and len(opt_range) == 2:
        return opt_range[0], opt_range[1]
    return spec.get("min", spec.get("default")), spec.get("max", spec.get("default"))


def get_sweep_preset(param_name: str) -> dict | None:
    """Récupère la config préréglée pour un paramètre de sweep."""
    preset = SWEEP_PRESETS.get(param_name)
    if not preset:
        return None

    return {
        "min": preset.get("min"),
        "max": preset.get("max"),
        "n_values": preset.get("n_values", 3),
        "step": preset.get("step"),
    }


# =============================================================================
# CHARGEMENT DYNAMIQUE DES CLASSES
# =============================================================================

# Mapping nom stratégie → (module, classe) pour les Core
_CORE_STRATEGY_MAPPING: dict[str, tuple[str, str]] = {
    "MA_Crossover": ("threadx.strategy.ma_crossover", "MACrossoverStrategy"),
    "Bollinger_Dual": ("threadx.strategy.bollinger_dual", "BollingerDualStrategy"),
    "EMA_Cross": ("threadx.strategy.ema_cross", "EMACrossStrategy"),
    "ATR_Channel": ("threadx.strategy.atr_channel", "ATRChannelStrategy"),
}


def get_strategy_class(strategy_name: str):
    """
    Charge dynamiquement une classe de stratégie par son nom.

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Classe de stratégie

    Raises:
        ValueError: Si stratégie non trouvée
        ImportError: Si module non importable
    """
    if strategy_name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(
            f'Strategy "{strategy_name}" not found. Available: {available}'
        )

    meta = REGISTRY[strategy_name]
    category = meta.get("category", "Core")

    if category == "Core":
        if strategy_name not in _CORE_STRATEGY_MAPPING:
            raise ValueError(
                f'Core strategy "{strategy_name}" not in _CORE_STRATEGY_MAPPING'
            )

        module_path, class_name = _CORE_STRATEGY_MAPPING[strategy_name]

        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f'Failed to import "{module_path}": {e}') from e
        except AttributeError as e:
            raise AttributeError(
                f'Class "{class_name}" not found in "{module_path}": {e}'
            ) from e

    elif category == "AI-Evolved":
        class_name_ai: str | None = meta.get("class_name")
        module_name = meta.get("module_name", strategy_name.lower())

        if not class_name_ai:
            raise ValueError(f'AI-Evolved "{strategy_name}" missing class_name')

        try:
            module = importlib.import_module(
                f"threadx.strategy.experimental.{module_name}"
            )
            return getattr(module, class_name_ai)
        except ImportError as e:
            raise ImportError(
                f'Failed to import "experimental.{module_name}": {e}'
            ) from e

    else:
        raise ValueError(f'Unknown category "{category}" for {strategy_name}')


def get_strategy_category(name: str) -> str | None:
    """Retourne la catégorie d'une stratégie."""
    meta = REGISTRY.get(name)
    return meta.get("category", "Core") if meta else None


# =============================================================================
# GESTION DYNAMIQUE DES STRATÉGIES AI-EVOLVED
# =============================================================================


def register_ai_strategy(
    name: str,
    class_name: str,
    module_name: str,
    params_schema: dict[str, Any],
    indicator_mapping: dict[str, dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Enregistre dynamiquement une stratégie AI-Evolved.

    Args:
        name: Nom de la stratégie (ex: "AI_MeanRev_v3")
        class_name: Nom de la classe Python
        module_name: Nom du module dans experimental/
        params_schema: Schema des paramètres
        indicator_mapping: Mapping param → indicateur (optionnel)
        metadata: Métadonnées additionnelles
    """
    if name in REGISTRY:
        logger.warning("⚠️  Stratégie '%s' déjà présente, écrasement", name)

    REGISTRY[name] = {
        "category": "AI-Evolved",
        "class_name": class_name,
        "module_name": module_name,
        "params": params_schema,
        "indicator_mapping": indicator_mapping or {},
        "generated_at": datetime.now().isoformat(),
        **(metadata or {}),
    }

    logger.info("✅ Stratégie AI enregistrée: %s → %s", name, class_name)


def unregister_ai_strategy(name: str) -> bool:
    """Supprime une stratégie AI-Evolved du REGISTRY."""
    if name not in REGISTRY:
        logger.warning("⚠️  Stratégie '%s' non trouvée", name)
        return False

    meta = REGISTRY[name]
    if meta.get("category") != "AI-Evolved":
        logger.error("❌ '%s' n'est pas AI-Evolved, impossible de supprimer", name)
        return False

    del REGISTRY[name]
    logger.info("✅ Stratégie AI supprimée: %s", name)
    return True


def get_ai_strategies() -> dict[str, dict[str, Any]]:
    """Retourne toutes les stratégies AI-Evolved."""
    return {
        name: meta
        for name, meta in REGISTRY.items()
        if meta.get("category") == "AI-Evolved"
    }


def sync_ai_strategies_from_experimental() -> int:
    """
    Synchronise le REGISTRY avec les stratégies dans experimental/.

    Returns:
        Nombre de nouvelles stratégies enregistrées
    """
    try:
        from threadx.strategy.experimental import (
            AI_STRATEGIES,
            get_ai_strategy_metadata,
            reload_ai_strategies,
        )
    except ImportError as e:
        logger.error("❌ Impossible d'importer experimental: %s", e)
        return 0

    reload_ai_strategies()
    new_count = 0

    for module_name, _ in AI_STRATEGIES.items():
        parts = module_name.split("_")
        registry_name = "_".join(p.capitalize() for p in parts)

        if registry_name in REGISTRY:
            continue

        meta = get_ai_strategy_metadata(module_name)
        if not meta:
            logger.warning("⚠️  Pas de métadonnées pour %s", module_name)
            continue

        register_ai_strategy(
            name=registry_name,
            class_name=meta["class_name"],
            module_name=module_name,
            params_schema={},
            metadata={"file": meta["file"], "auto_discovered": True},
        )
        new_count += 1

    if new_count > 0:
        logger.info("✅ %d nouvelles stratégies AI synchronisées", new_count)

    return new_count


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Registry
    "REGISTRY",
    "SWEEP_PRESETS",
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
