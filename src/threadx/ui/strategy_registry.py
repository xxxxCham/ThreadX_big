from __future__ import annotations

from typing import Any


def _scalar_default(spec: Any) -> Any:
    if isinstance(spec, dict):
        return spec.get("default")
    return spec


# Mapping Strategie → métadonnées indicateurs + paramètres
REGISTRY: dict[str, dict[str, Any]] = {
    "Bollinger_Breakout": {
        "indicators": {
            "bollinger": {
                "period": {
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
            "bb_period": {
                "default": 20,
                "min": 10,
                "max": 50,
                "step": 10,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "int",
                "label": "Période Bollinger (SMA)",
                "opt_range": (10, 50),  # Plage classique du tableau : 10 → 50
            },
            "bb_std": {
                "default": 2.0,
                "min": 1.5,
                "max": 3.0,
                "step": 0.5,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "float",
                "label": "Sigma Bollinger (K)",
                "opt_range": (1.5, 3.0),  # Plage classique du tableau : 1.5 → 3.0
            },
            "entry_z": {
                "default": 1.0,
                "min": 0.5,
                "max": 2.5,
                "step": 0.25,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "float",
                "label": "Seuil Z-score entrée",
                "opt_range": (0.8, 2.0),
            },
            "entry_logic": {
                "default": "AND",
                "options": ["AND", "OR"],
                "type": "select",
                "label": "Logique d'entrée (AND/OR)",
                "tunable": False,  # Non-optimisable (enum, pas de plage numérique)
            },
            "atr_period": {
                "default": 14,
                "min": 7,
                "max": 21,
                "step": 4,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "int",
                "label": "Période ATR",
                "opt_range": (7, 21),  # Plage classique du tableau : 7 → 21
            },
            "atr_multiplier": {
                "default": 1.5,
                "min": 1.0,
                "max": 3.0,
                "step": 0.5,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "float",
                "label": "Multiplicateur ATR pour stops",
                "opt_range": (1.0, 3.0),  # Plage classique : 1.0 → 3.0 (pas de 0.25)
            },
            "trailing_stop": {
                "default": True,
                "type": "bool",
                "label": "Activer Trailing Stop ATR",
                "tunable": False,  # Non-optimisable (booléen, pas de plage)
            },
            "risk_per_trade": {
                "default": 0.02,
                "min": 0.005,
                "max": 0.1,
                "step": 0.005,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "float",
                "label": "Risque par Trade (fraction du capital)",
                "opt_range": (
                    0.015,
                    0.03,
                ),  # Plage d'optimisation : 1.5% → 3% (pas de 0.25%)
            },
            "min_pnl_pct": {
                "default": 0.0,  # FIX: 0.0 = désactivé (0.01% filtrait TOUS les trades)
                "min": 0.0,
                "max": 0.5,
                "step": 0.02,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "float",
                "label": "Filtre PnL Minimum (%)",
                "opt_range": (0.0, 0.05),  # Plage d'optimisation : 0% → 5%
            },
            "leverage": {
                "default": 1.0,
                "min": 1.0,
                "max": 150.0,
                "step": 1.0,
                "type": "float",
                "label": "Levier (1.0 = sans levier, OPTIONNEL)",
                "tunable": False,  # Non optimisable par défaut
            },
            "max_hold_bars": {
                "default": 72,
                "min": 10,
                "max": 500,
                "step": 40,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "int",
                "label": "Durée Maximale en Position (barres)",
                "opt_range": (30, 200),
            },
            "spacing_bars": {
                "default": 6,
                "min": 1,
                "max": 50,
                "step": 5,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "int",
                "label": "Espacement Minimum entre Trades",
                "opt_range": (3, 20),
            },
            "trend_period": {
                "default": 0,
                "min": 0,
                "max": 100,
                "step": 10,  # Augmenté pour réduire combinaisons (ajustable via slider sensibilité)
                "type": "int",
                "label": "Période EMA Filtre Tendance (0 = désactivé)",
                "opt_range": (0, 50),  # Plage d'optimisation : 0 (désactivé) → 50
            },
        },
    },
    "EMA_Cross": {
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
    "AmplitudeHunter": {
        "indicators": {
            "bollinger": {
                "period": {
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
            "macd": {
                "fast": {
                    "default": 12,
                    "min": 5,
                    "max": 30,
                    "step": 1,
                    "type": "int",
                    "label": "MACD Rapide",
                },
                "slow": {
                    "default": 26,
                    "min": 10,
                    "max": 60,
                    "step": 1,
                    "type": "int",
                    "label": "MACD Lente",
                },
                "signal": {
                    "default": 9,
                    "min": 3,
                    "max": 20,
                    "step": 1,
                    "type": "int",
                    "label": "MACD Signal",
                },
            },
        },
        "params": {
            # Bollinger Bands
            "bb_period": {
                "default": 20,
                "min": 10,
                "max": 50,
                "step": 10,
                "type": "int",
                "label": "Période Bollinger",
                "opt_range": (10, 50),
            },
            "bb_std": {
                "default": 2.0,
                "min": 1.5,
                "max": 3.0,
                "step": 0.5,
                "type": "float",
                "label": "Sigma Bollinger",
                "opt_range": (1.5, 3.0),
            },
            # Filtre de régime
            "bbwidth_percentile_threshold": {
                "default": 50.0,
                "min": 30.0,
                "max": 70.0,
                "step": 10.0,
                "type": "float",
                "label": "BBWidth Percentile (seuil)",
                "opt_range": (30.0, 70.0),
            },
            "bbwidth_lookback": {
                "default": 100,
                "min": 50,
                "max": 200,
                "step": 50,
                "type": "int",
                "label": "BBWidth Lookback",
                "opt_range": (50, 200),
                "tunable": False,
            },
            "volume_zscore_threshold": {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 0.25,
                "type": "float",
                "label": "Volume Z-Score (seuil)",
                "opt_range": (0.0, 1.5),
            },
            "volume_lookback": {
                "default": 50,
                "min": 20,
                "max": 100,
                "step": 20,
                "type": "int",
                "label": "Volume Lookback",
                "opt_range": (20, 100),
                "tunable": False,
            },
            "use_adx_filter": {
                "default": False,
                "type": "bool",
                "label": "Activer Filtre ADX",
                "tunable": False,
            },
            "adx_threshold": {
                "default": 15.0,
                "min": 10.0,
                "max": 30.0,
                "step": 5.0,
                "type": "float",
                "label": "ADX Seuil",
                "opt_range": (10.0, 30.0),
                "tunable": False,
            },
            "adx_period": {
                "default": 14,
                "min": 7,
                "max": 21,
                "step": 7,
                "type": "int",
                "label": "ADX Période",
                "opt_range": (7, 21),
                "tunable": False,
            },
            # Setup Spring → Drive
            "spring_lookback": {
                "default": 20,
                "min": 10,
                "max": 30,
                "step": 10,
                "type": "int",
                "label": "Spring Lookback (barres)",
                "opt_range": (10, 30),
            },
            "pb_entry_threshold_min": {
                "default": 0.2,
                "min": 0.1,
                "max": 0.5,
                "step": 0.1,
                "type": "float",
                "label": "%B Entrée Min",
                "opt_range": (0.2, 0.5),
            },
            "pb_entry_threshold_max": {
                "default": 0.5,
                "min": 0.3,
                "max": 0.7,
                "step": 0.1,
                "type": "float",
                "label": "%B Entrée Max",
                "opt_range": (0.35, 0.65),
            },
            "macd_fast": {
                "default": 12,
                "min": 8,
                "max": 20,
                "step": 4,
                "type": "int",
                "label": "MACD Rapide",
                "opt_range": (8, 20),
                "tunable": False,
            },
            "macd_slow": {
                "default": 26,
                "min": 20,
                "max": 40,
                "step": 10,
                "type": "int",
                "label": "MACD Lente",
                "opt_range": (20, 40),
                "tunable": False,
            },
            "macd_signal": {
                "default": 9,
                "min": 5,
                "max": 15,
                "step": 5,
                "type": "int",
                "label": "MACD Signal",
                "opt_range": (5, 15),
                "tunable": False,
            },
            # Score d'Amplitude
            "amplitude_score_threshold": {
                "default": 0.6,
                "min": 0.4,
                "max": 0.75,
                "step": 0.05,
                "type": "float",
                "label": "Score Amplitude (seuil)",
                "opt_range": (0.4, 0.75),
            },
            "amplitude_w1_bbwidth": {
                "default": 0.3,
                "min": 0.1,
                "max": 0.5,
                "step": 0.05,
                "type": "float",
                "label": "Poids BBWidth",
                "opt_range": (0.2, 0.4),
                "tunable": False,
            },
            "amplitude_w2_pb": {
                "default": 0.2,
                "min": 0.1,
                "max": 0.4,
                "step": 0.05,
                "type": "float",
                "label": "Poids %B",
                "opt_range": (0.1, 0.3),
                "tunable": False,
            },
            "amplitude_w3_macd_slope": {
                "default": 0.3,
                "min": 0.1,
                "max": 0.5,
                "step": 0.05,
                "type": "float",
                "label": "Poids Pente MACD",
                "opt_range": (0.2, 0.4),
                "tunable": False,
            },
            "amplitude_w4_volume": {
                "default": 0.2,
                "min": 0.1,
                "max": 0.4,
                "step": 0.05,
                "type": "float",
                "label": "Poids Volume",
                "opt_range": (0.1, 0.3),
                "tunable": False,
            },
            # Pyramiding
            "pyramiding_enabled": {
                "default": False,
                "type": "bool",
                "label": "Activer Pyramiding",
                "tunable": False,
            },
            "pyramiding_max_adds": {
                "default": 1,
                "min": 1,
                "max": 2,
                "step": 1,
                "type": "int",
                "label": "Nombre Max d'Adds (1 ou 2)",
                "opt_range": (1, 2),
                "tunable": False,
            },
            # Stops et Trailing
            "atr_period": {
                "default": 14,
                "min": 7,
                "max": 21,
                "step": 7,
                "type": "int",
                "label": "Période ATR",
                "opt_range": (7, 21),
            },
            "sl_atr_multiplier": {
                "default": 2.0,
                "min": 1.5,
                "max": 3.0,
                "step": 0.5,
                "type": "float",
                "label": "Mult ATR pour SL Initial",
                "opt_range": (1.5, 3.0),
            },
            "sl_min_pct": {
                "default": 0.37,
                "min": 0.2,
                "max": 0.6,
                "step": 0.1,
                "type": "float",
                "label": "SL Min % (médiane-basse)",
                "opt_range": (0.3, 0.5),
            },
            "short_stop_pct": {
                "default": 0.37,
                "min": 0.2,
                "max": 0.6,
                "step": 0.1,
                "type": "float",
                "label": "Stop Loss SHORT % (fixe)",
                "opt_range": (0.3, 0.5),
            },
            "trailing_activation_pb_threshold": {
                "default": 1.0,
                "min": 0.8,
                "max": 1.2,
                "step": 0.1,
                "type": "float",
                "label": "%B Activation Trailing",
                "opt_range": (0.8, 1.2),
            },
            "trailing_activation_gain_r": {
                "default": 1.0,
                "min": 0.8,
                "max": 1.5,
                "step": 0.1,
                "type": "float",
                "label": "Gain R Activation Trailing",
                "opt_range": (0.8, 1.5),
            },
            "trailing_type": {
                "default": "chandelier",
                "options": ["chandelier", "pb_floor", "macd_fade"],
                "type": "select",
                "label": "Type Trailing Stop",
                "tunable": False,
            },
            "trailing_chandelier_atr_mult": {
                "default": 2.5,
                "min": 1.5,
                "max": 3.5,
                "step": 0.5,
                "type": "float",
                "label": "Mult ATR Chandelier",
                "opt_range": (2.0, 3.0),
            },
            "trailing_pb_floor": {
                "default": 0.5,
                "min": 0.3,
                "max": 0.7,
                "step": 0.1,
                "type": "float",
                "label": "%B Floor (sortie)",
                "opt_range": (0.4, 0.6),
                "tunable": False,
            },
            # Cible BIP
            "use_bip_target": {
                "default": True,
                "type": "bool",
                "label": "Activer Cible BIP",
                "tunable": False,
            },
            "bip_partial_exit_pct": {
                "default": 0.5,
                "min": 0.3,
                "max": 0.7,
                "step": 0.1,
                "type": "float",
                "label": "% Sortie Partielle BIP",
                "opt_range": (0.3, 0.7),
                "tunable": False,
            },
            # Risk Management
            "risk_per_trade": {
                "default": 0.02,
                "min": 0.01,
                "max": 0.05,
                "step": 0.005,
                "type": "float",
                "label": "Risque par Trade (fraction)",
                "opt_range": (0.015, 0.03),
            },
            "max_hold_bars": {
                "default": 100,
                "min": 50,
                "max": 300,
                "step": 50,
                "type": "int",
                "label": "Durée Max Position (barres)",
                "opt_range": (50, 200),
            },
            "leverage": {
                "default": 1.0,
                "min": 1.0,
                "max": 150.0,
                "step": 1.0,
                "type": "float",
                "label": "Levier",
                "tunable": False,
            },
        },
    },
}


def list_strategies() -> list[str]:
    """Retourne la liste ordonnée des stratégies disponibles."""
    return list(REGISTRY.keys())


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
