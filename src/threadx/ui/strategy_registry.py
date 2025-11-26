from __future__ import annotations

import importlib
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _scalar_default(spec: Any) -> Any:
    if isinstance(spec, dict):
        return spec.get("default")
    return spec


# =============================================================================
# PRÉRÉGLAGES GLOBAUX POUR LES SWEEPS
# =============================================================================
# Ces valeurs fixes sont utilisées par les pages Optimisation et Multi-LLM
# pour garantir une cohérence des tests

SWEEP_PRESETS = {
    "max_hold_bars": {
        "min": 50,
        "max": 200,
        "n_values": 4,
        "step": 50,
        "reason": "Durée max de maintien de position (50 à 200 barres). Évite les positions zombies tout en testant plusieurs horizons.",
    },
    "risk_per_trade": {
        "min": 0.01,
        "max": 0.03,
        "n_values": 5,
        "step": 0.005,
        "reason": "Risque entre 1% et 3% par trade (standard money management conservateur).",
    },
    "feeder_aggr": {
        "min": 8,
        "max": 24,
        "n_values": 5,
        "step": 4,
        "reason": "Pipeline CPU étudié (8 à 24 workers) pour booster le pré-calcul GPU des indicateurs.",
    },
}


def get_sweep_preset(param_name: str) -> dict | None:
    """
    Récupère la configuration préréglée pour un paramètre de sweep.

    Args:
        param_name: Nom du paramètre (ex: "max_hold_bars")

    Returns:
        dict avec {"min": val, "max": val, "n_values": n, "step": pas}
        ou None si pas de préréglage
    """
    preset = SWEEP_PRESETS.get(param_name)
    if not preset:
        return None

    # Compatibilité avec l'ancien format à valeur unique
    if "value" in preset:
        val = preset["value"]
        return {"min": val, "max": val, "n_values": 1, "step": None}

    min_val = preset.get("min")
    max_val = preset.get("max")
    if min_val is None or max_val is None:
        return None

    return {
        "min": min_val,
        "max": max_val,
        "n_values": preset.get("n_values", 3),
        "step": preset.get("step"),
    }


# =============================================================================
# MAPPING STRATEGIE → MÉTADONNÉES INDICATEURS + PARAMÈTRES
# =============================================================================
#
# Chaque stratégie a les métadonnées suivantes:
# - category: "Core" | "AI-Evolved" | "Experimental"
# - indicators: specs des indicateurs requis
# - params: specs des paramètres de stratégie
# - (AI-Evolved uniquement) class_name, module_name, generated_at
#
REGISTRY: dict[str, dict[str, Any]] = {
    "MA_Crossover": {
        "category": "Core",  # Stratégie manuelle stable
        "indicators": {
            "sma": {
                "window": {
                    "default": 10,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                    "type": "int",
                    "label": "Période SMA",
                }
            }
        },
        "params": {
            # Moving Averages
            "fast_period": {
                "default": 10,
                "min": 5,
                "max": 50,
                "step": 5,
                "type": "int",
                "label": "Période SMA Rapide",
                "opt_range": (5, 30),
            },
            "slow_period": {
                "default": 30,
                "min": 20,
                "max": 100,
                "step": 10,
                "type": "int",
                "label": "Période SMA Lente",
                "opt_range": (20, 60),
            },
            # Risk Management (SIMPLE et CLAIR)
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
            "risk_per_trade": {
                "default": 0.01,
                "min": 0.005,
                "max": 0.03,
                "step": 0.005,
                "type": "float",
                "label": "Risque par Trade (fraction du capital)",
                "opt_range": (0.01, 0.02),
            },
            # Position Management
            "leverage": {
                "default": 1.0,
                "min": 1.0,
                "max": 5.0,
                "step": 0.5,
                "type": "float",
                "label": "Levier (1.0 = sans levier)",
                "opt_range": (1.0, 2.0),
                "tunable": False,  # Non optimisable par défaut pour validation
            },
            "max_hold_bars": {
                "default": 100,
                "min": 20,
                "max": 300,
                "step": 20,
                "type": "int",
                "label": "Durée Maximale en Position (barres)",
                "opt_range": (50, 150),
            },
            # Frais
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
        },
    },
    "EMA_Cross": {
        "category": "Core",
        "indicators": {
            "ema_fast": {
                "window": {
                    "default": 12,
                    "min": 2,
                    "max": 120,
                    "step": 1,
                    "type": "int",
                    "label": "EMA rapide",
                }
            },
            "ema_slow": {
                "window": {
                    "default": 26,
                    "min": 5,
                    "max": 200,
                    "step": 1,
                    "type": "int",
                    "label": "EMA lente",
                }
            },
        },
        "params": {
            "fast_window": {
                "default": 12,
                "min": 2,
                "max": 120,
                "step": 1,
                "type": "int",
                "label": "Fenêtre EMA rapide",
            },
            "slow_window": {
                "default": 26,
                "min": 5,
                "max": 200,
                "step": 1,
                "type": "int",
                "label": "Fenêtre EMA lente",
            },
        },
    },
    "ATR_Channel": {
        "category": "Core",
        "indicators": {
            "atr": {
                "window": {
                    "default": 14,
                    "min": 2,
                    "max": 100,
                    "step": 1,
                    "type": "int",
                    "label": "Période ATR",
                },
                "mult": {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "type": "float",
                    "label": "Multiplicateur",
                },
            }
        },
        "params": {
            "atr_window": {
                "default": 14,
                "min": 2,
                "max": 100,
                "step": 1,
                "type": "int",
                "label": "Fenêtre ATR",
            },
            "atr_mult": {
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "step": 0.1,
                "type": "float",
                "label": "Multiplicateur ATR",
            },
        },
    },
    "Bollinger_Dual": {
        "category": "Core",
        "indicators": {
            "bollinger": {
                "window": {
                    "default": 20,
                    "min": 5,
                    "max": 120,
                    "step": 1,
                    "type": "int",
                    "label": "Période Bollinger",
                },
                "std": {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "type": "float",
                    "label": "Sigma (σ)",
                },
            },
            "atr": {
                "period": {
                    "default": 14,
                    "min": 5,
                    "max": 50,
                    "step": 1,
                    "type": "int",
                    "label": "Période ATR",
                },
            },
        },
        "params": {
            "bb_window": {
                "default": 20,
                "min": 10,
                "max": 50,
                "step": 10,  # Ajusté pour réduire combinaisons (5 valeurs)
                "type": "int",
                "label": "Période Bollinger (SMA)",
                "opt_range": (10, 50),
            },
            "bb_std": {
                "default": 2.0,
                "min": 1.5,
                "max": 3.0,
                "step": 0.5,  # Ajusté pour réduire combinaisons (3 valeurs)
                "type": "float",
                "label": "Sigma Bollinger (K)",
                "opt_range": (1.5, 3.0),
            },
            "entry_z": {
                "default": 1.0,
                "min": 0.8,
                "max": 2.0,
                "step": 0.25,  # 4.8 combinaisons
                "type": "float",
                "label": "Seuil Z-score entrée",
                "opt_range": (0.8, 2.0),
            },
            "atr_period": {
                "default": 14,
                "min": 7,
                "max": 21,
                "step": 4,  # Combinaisons: 4 (7, 11, 15, 19)
                "type": "int",
                "label": "Période ATR",
                "opt_range": (7, 21),
            },
            "atr_multiplier": {
                "default": 1.5,
                "min": 1.0,
                "max": 3.0,
                "step": 0.5,  # Combinaisons: 4 (1.0, 1.5, 2.0, 2.5, 3.0)
                "type": "float",
                "label": "Multiplicateur ATR pour stops",
                "opt_range": (1.0, 3.0),
            },
            "risk_per_trade": {
                "default": 0.02,
                "min": 0.015,
                "max": 0.03,
                "step": 0.005,  # Combinaisons: 3 (0.015, 0.02, 0.025, 0.03)
                "type": "float",
                "label": "Risque par Trade (fraction du capital)",
                "opt_range": (0.015, 0.03),
            },
            "min_pnl_pct": {
                "default": 0.0,
                "min": 0.0,
                "max": 0.05,
                "step": 0.02,  # Combinaisons: 2.5 (0.0, 0.02, 0.04)
                "type": "float",
                "label": "Filtre PnL Minimum (%)",
                "opt_range": (0.0, 0.05),
            },
            "max_hold_bars": {
                "default": 100,
                "min": 50,
                "max": 200,
                "step": 40,  # Combinaisons: 4 (50, 90, 130, 170)
                "type": "int",
                "label": "Durée Maximale en Position (barres)",
                "opt_range": (50, 200),
            },
            "min_spacing_bars": {
                "default": 10,
                "min": 3,
                "max": 20,
                "step": 5,  # Combinaisons: 4 (3, 8, 13, 18)
                "type": "int",
                "label": "Espacement Minimum entre Trades",
                "opt_range": (3, 20),
            },
            "ema_filter_period": {
                "default": 0,
                "min": 0,
                "max": 50,
                "step": 10,  # Combinaisons: 6 (0, 10, 20, 30, 40, 50)
                "type": "int",
                "label": "Période EMA Filtre Tendance (0 = désactivé)",
                "opt_range": (0, 50),
            },
        },
    },
}

GLOBAL_PARAM_DEFAULT_OVERRIDES: dict[str, Any] = {
    "risk_per_trade": 0.005,
    "min_pnl_pct": 0.02,
    "spacing_bars": 1,
    "max_hold_bars": 40,
}

STRATEGY_PARAM_DEFAULT_OVERRIDES: dict[str, dict[str, Any]] = {
    "MA_Crossover": {
        "max_hold_bars": 20,
    }
}


def _apply_overrides_to_params(
    params: dict[str, Any], overrides: dict[str, Any]
) -> None:
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


for definition in REGISTRY.values():
    params = definition.get("params", {})
    _apply_overrides_to_params(params, GLOBAL_PARAM_DEFAULT_OVERRIDES)

for strategy_name, overrides in STRATEGY_PARAM_DEFAULT_OVERRIDES.items():
    if strategy_name in REGISTRY:
        params = REGISTRY[strategy_name].get("params", {})
        _apply_overrides_to_params(params, overrides)


def list_strategies(category: str | None = None) -> list[str]:
    """
    Retourne la liste ordonnée des stratégies disponibles.

    Args:
        category: Filtre optionnel ("Core", "AI-Evolved", "Experimental")
                 Si None, retourne toutes les stratégies

    Returns:
        list[str] des noms de stratégies
    """
    if category is None:
        return list(REGISTRY.keys())

    return [
        name
        for name, meta in REGISTRY.items()
        if meta.get("category") == category
    ]


def indicator_specs_for(strategy: str) -> dict[str, Any]:
    return REGISTRY[strategy].get("indicators", {})


def parameter_specs_for(strategy: str) -> dict[str, Any]:
    return REGISTRY[strategy].get("params", {})


def _extract_indicator_defaults(
    specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    defaults: dict[str, dict[str, Any]] = {}
    for indicator, params in specs.items():
        defaults[indicator] = {}
        for key, spec in params.items():
            defaults[indicator][key] = _scalar_default(spec)
    return defaults


def _extract_param_defaults(specs: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, spec in specs.items():
        if isinstance(spec, dict):
            defaults[key] = spec.get("default")
        else:
            defaults[key] = spec
    return defaults


def indicators_for(strategy: str) -> dict[str, dict[str, Any]]:
    """Retourne uniquement les valeurs par défaut des indicateurs."""
    return _extract_indicator_defaults(indicator_specs_for(strategy))


def base_params_for(strategy: str) -> dict[str, Any]:
    """Retourne les paramètres par défaut pour la stratégie."""
    return _extract_param_defaults(parameter_specs_for(strategy))


def tunable_parameters_for(strategy: str) -> dict[str, dict[str, Any]]:
    """Paramètres numériques pouvant être optimisés (min/max/step)."""
    specs = parameter_specs_for(strategy)
    tunables: dict[str, dict[str, Any]] = {}
    for key, raw_spec in specs.items():
        if isinstance(raw_spec, dict):
            param_type = raw_spec.get("type")
            default = raw_spec.get("default")
            # Exclure les paramètres non-optimisables (tunable: False)
            if raw_spec.get("tunable") is False:
                continue
        else:
            default = raw_spec
            param_type = (
                "float"
                if isinstance(raw_spec, float)
                else "int" if isinstance(raw_spec, int) else None
            )
            raw_spec = {"default": raw_spec}

        if param_type in {"int", "float"} or isinstance(default, (int, float)):
            spec = dict(raw_spec)
            spec.setdefault(
                "type", param_type or ("float" if isinstance(default, float) else "int")
            )
            tunables[key] = spec

    return tunables


def resolve_range(spec: dict[str, Any]) -> tuple[Any, Any]:
    """Renvoie la plage par défaut (min/max) pour un paramètre."""
    opt_range = spec.get("opt_range")
    if isinstance(opt_range, (list, tuple)) and len(opt_range) == 2:
        return opt_range[0], opt_range[1]
    return spec.get("min", spec.get("default")), spec.get("max", spec.get("default"))


# =============================================================================
# FONCTION UTILITAIRE POUR CHARGER LES CLASSES DE STRATÉGIES
# =============================================================================


def get_strategy_class(strategy_name: str):
    """
    Charge dynamiquement une classe de stratégie par son nom.

    Gère à la fois stratégies Core (import standard) et AI-Evolved
    (import dynamique depuis experimental/).

    Args:
        strategy_name: Nom de la stratégie (ex: 'MA_Crossover', 'ai_meanrev_v3')

    Returns:
        Classe de stratégie

    Raises:
        ValueError: Si la stratégie n'existe pas
        ImportError: Si le module de stratégie ne peut pas être importé

    Examples:
        >>> # Stratégie Core
        >>> strategy_class = get_strategy_class('MA_Crossover')
        >>> strategy = strategy_class()

        >>> # Stratégie AI-Evolved
        >>> ai_strategy_class = get_strategy_class('AI_MeanRev_v3')
        >>> ai_strategy = ai_strategy_class()
    """
    # Vérifier si stratégie existe dans REGISTRY
    if strategy_name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(
            f'Strategy "{strategy_name}" not found. '
            f"Available strategies: {available}"
        )

    meta = REGISTRY[strategy_name]
    category = meta.get("category", "Core")

    # Stratégies Core: import standard
    if category == "Core":
        # Mapping nom de stratégie → (module, classe)
        strategy_mapping = {
            "MA_Crossover": ("threadx.strategy.ma_crossover", "MACrossoverStrategy"),
            "Bollinger_Dual": ("threadx.strategy.bollinger_dual", "BollingerDualStrategy"),
            "EMA_Cross": ("threadx.strategy.ema_cross", "EMACrossStrategy"),
            "ATR_Channel": ("threadx.strategy.atr_channel", "ATRChannelStrategy"),
        }

        if strategy_name not in strategy_mapping:
            raise ValueError(
                f'Core strategy "{strategy_name}" mapping not found. '
                f'Add to strategy_mapping in get_strategy_class()'
            )

        module_path, class_name = strategy_mapping[strategy_name]

        try:
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            return strategy_class
        except ImportError as e:
            raise ImportError(
                f'Failed to import strategy module "{module_path}": {e}'
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f'Strategy class "{class_name}" not found in module "{module_path}": {e}'
            ) from e

    # Stratégies AI-Evolved: import dynamique depuis experimental/
    elif category == "AI-Evolved":
        class_name = meta.get("class_name")
        module_name = meta.get("module_name", strategy_name.lower())

        if not class_name:
            raise ValueError(
                f'AI-Evolved strategy "{strategy_name}" missing class_name in metadata'
            )

        try:
            module = importlib.import_module(
                f"threadx.strategy.experimental.{module_name}"
            )
            strategy_class = getattr(module, class_name)
            return strategy_class
        except ImportError as e:
            raise ImportError(
                f'Failed to import AI strategy module "experimental.{module_name}": {e}'
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f'Strategy class "{class_name}" not found in experimental.{module_name}: {e}'
            ) from e

    else:
        raise ValueError(
            f'Unknown strategy category "{category}" for {strategy_name}'
        )


# =============================================================================
# GESTION DYNAMIQUE DES STRATÉGIES AI-EVOLVED
# =============================================================================


def register_ai_strategy(
    name: str,
    class_name: str,
    module_name: str,
    params_schema: dict[str, Any],
    indicators_schema: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Enregistre dynamiquement une stratégie AI-Evolved dans le REGISTRY.

    Appelé par Critic après validation pour rendre la stratégie
    disponible dans l'UI et les outils d'optimisation.

    Args:
        name: Nom de la stratégie dans le registry (ex: "AI_MeanRev_v3")
        class_name: Nom de la classe Python (ex: "AIMeanRevV3Strategy")
        module_name: Nom du module dans experimental/ (ex: "ai_meanrev_v3")
        params_schema: Schema des paramètres (format REGISTRY)
        indicators_schema: Schema des indicateurs (format REGISTRY)
        metadata: Métadonnées optionnelles (generated_at, base_strategy, etc.)

    Example:
        >>> register_ai_strategy(
        ...     name="AI_MeanRev_v3",
        ...     class_name="AIMeanRevV3Strategy",
        ...     module_name="ai_meanrev_v3",
        ...     params_schema={
        ...         "bb_period": {"default": 20, "min": 10, "max": 50, ...},
        ...         "stop_loss_pct": {"default": 2.0, "min": 1.0, "max": 5.0, ...},
        ...     },
        ...     indicators_schema={
        ...         "bollinger": {"period": {...}, "std": {...}},
        ...         "atr": {"period": {...}},
        ...     },
        ...     metadata={
        ...         "base_strategy": "Bollinger_Dual",
        ...         "objective": "improve_sharpe",
        ...         "sharpe_improvement": 0.3,
        ...     }
        ... )
    """
    if name in REGISTRY:
        logger.warning(f"⚠️  Stratégie '{name}' déjà dans REGISTRY, écrasement")

    REGISTRY[name] = {
        "category": "AI-Evolved",
        "class_name": class_name,
        "module_name": module_name,
        "indicators": indicators_schema,
        "params": params_schema,
        "generated_at": datetime.now().isoformat(),
        **(metadata or {}),
    }

    logger.info(f"✅ Stratégie AI enregistrée: {name} → {class_name}")


def unregister_ai_strategy(name: str) -> bool:
    """
    Supprime une stratégie AI-Evolved du REGISTRY.

    Args:
        name: Nom de la stratégie à supprimer

    Returns:
        True si supprimée, False si non trouvée

    Example:
        >>> unregister_ai_strategy("AI_BadStrategy_v1")
    """
    if name not in REGISTRY:
        logger.warning(f"⚠️  Stratégie '{name}' non trouvée dans REGISTRY")
        return False

    meta = REGISTRY[name]
    if meta.get("category") != "AI-Evolved":
        logger.error(
            f"❌ Impossible de supprimer '{name}': catégorie={meta.get('category')} (seulement AI-Evolved autorisé)"
        )
        return False

    del REGISTRY[name]
    logger.info(f"✅ Stratégie AI supprimée du registry: {name}")
    return True


def get_ai_strategies() -> dict[str, dict[str, Any]]:
    """
    Retourne toutes les stratégies AI-Evolved enregistrées.

    Returns:
        dict[name, metadata] des stratégies AI-Evolved uniquement
    """
    return {
        name: meta
        for name, meta in REGISTRY.items()
        if meta.get("category") == "AI-Evolved"
    }


def get_strategy_category(name: str) -> str | None:
    """
    Retourne la catégorie d'une stratégie.

    Args:
        name: Nom de la stratégie

    Returns:
        "Core" | "AI-Evolved" | "Experimental" | None si non trouvée
    """
    meta = REGISTRY.get(name)
    if not meta:
        return None
    return meta.get("category", "Core")


def sync_ai_strategies_from_experimental() -> int:
    """
    Synchronise le REGISTRY avec les stratégies dans experimental/.

    Scanne strategy/experimental/ et enregistre toutes les stratégies
    AI découvertes qui ne sont pas déjà dans le REGISTRY.

    Returns:
        Nombre de nouvelles stratégies enregistrées

    Example:
        >>> # CodeWriter a généré une nouvelle stratégie
        >>> count = sync_ai_strategies_from_experimental()
        >>> print(f"{count} nouvelles stratégies ajoutées")
    """
    try:
        from threadx.strategy.experimental import (
            AI_STRATEGIES,
            get_ai_strategy_metadata,
            reload_ai_strategies,
        )
    except ImportError as e:
        logger.error(f"❌ Impossible d'importer experimental: {e}")
        return 0

    # Force reload pour détecter nouvelles stratégies
    reload_ai_strategies()

    new_count = 0

    for module_name, _ in AI_STRATEGIES.items():
        # Générer nom pour registry
        # ex: "ai_meanrev_v3" → "AI_MeanRev_v3"
        parts = module_name.split("_")
        registry_name = "_".join(p.capitalize() for p in parts)

        # Vérifier si déjà enregistrée
        if registry_name in REGISTRY:
            continue

        # Récupérer métadonnées
        meta = get_ai_strategy_metadata(module_name)
        if not meta:
            logger.warning(f"⚠️  Pas de métadonnées pour {module_name}, skip")
            continue

        # Enregistrer avec schemas par défaut
        # (à améliorer: parser le code pour extraire schemas automatiquement)
        register_ai_strategy(
            name=registry_name,
            class_name=meta["class_name"],
            module_name=module_name,
            params_schema={},  # TODO: auto-extraction depuis Params class
            indicators_schema={},  # TODO: auto-extraction depuis docstring
            metadata={
                "file": meta["file"],
                "auto_discovered": True,
            },
        )

        new_count += 1

    if new_count > 0:
        logger.info(f"✅ {new_count} nouvelles stratégies AI synchronisées")

    return new_count


