from __future__ import annotations
from typing import Any, Dict, List, Tuple


def _scalar_default(spec: Any) -> Any:
    if isinstance(spec, dict):
        return spec.get("default")
    return spec


# Mapping Strategie → métadonnées indicateurs + paramètres
REGISTRY: Dict[str, Dict[str, Any]] = {
    "Bollinger_Breakout": {
        "indicators": {
            "bollinger": {
                "window": {
                    "default": 20,
                    "min": 5,
                    "max": 120,
                    "step": 1,
                    "type": "int",
                    "label": "Période SMA",
                },
                "std": {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.05,
                    "type": "float",
                    "label": "Sigma (σ)",
                    "opt_range": (1.5, 3.5),
                },
                "bandwidth_threshold": {
                    "default": 0.04,
                    "min": 0.01,
                    "max": 0.2,
                    "step": 0.005,
                    "type": "float",
                    "label": "Seuil largeur bande (ratio)",
                    "opt_range": (0.02, 0.12),
                },
                "price_gap_pct": {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "type": "float",
                    "label": "Écart médiane (%)",
                    "opt_range": (0.2, 2.5),
                },
                "signal_window": {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "type": "int",
                    "label": "Lissage signal (barres)",
                    "opt_range": (2, 15),
                },
            },
            "volatilité": {
                "bandwidth_ma": {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "type": "int",
                    "label": "Lissage largeur bandes",
                },
            },
        },
        "params": {
            "window": {
                "default": 20,
                "min": 5,
                "max": 120,
                "step": 1,
                "type": "int",
                "label": "Période Bollinger",
                "opt_range": (10, 60),
            },
            "std": {
                "default": 2.0,
                "min": 1.0,
                "max": 4.0,
                "step": 0.05,
                "type": "float",
                "label": "Sigma Bollinger",
                "opt_range": (1.5, 3.5),
            },
            "bandwidth_threshold": {
                "default": 0.04,
                "min": 0.0,
                "max": 0.2,
                "step": 0.005,
                "type": "float",
                "label": "Seuil largeur (ratio)",
                "opt_range": (0.02, 0.12),
            },
            "price_gap_pct": {
                "default": 0.5,
                "min": 0.0,
                "max": 5.0,
                "step": 0.05,
                "type": "float",
                "label": "Écart médiane (%)",
                "opt_range": (0.2, 2.5),
            },
            "signal_window": {
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
                "type": "int",
                "label": "Lissage signal (barres)",
                "opt_range": (2, 15),
            },
            "confirm_breakout": {
                "default": True,
                "type": "bool",
                "label": "Confirmer par cassure opposée",
            },
            "use_bandwidth_filter": {
                "default": True,
                "type": "bool",
                "label": "Activer filtre largeur",
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
}


def list_strategies() -> List[str]:
    """Retourne la liste ordonnée des stratégies disponibles."""
    return list(REGISTRY.keys())


def indicator_specs_for(strategy: str) -> Dict[str, Any]:
    return REGISTRY[strategy].get("indicators", {})


def parameter_specs_for(strategy: str) -> Dict[str, Any]:
    return REGISTRY[strategy].get("params", {})


def _extract_indicator_defaults(specs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    defaults: Dict[str, Dict[str, Any]] = {}
    for indicator, params in specs.items():
        defaults[indicator] = {}
        for key, spec in params.items():
            defaults[indicator][key] = _scalar_default(spec)
    return defaults


def _extract_param_defaults(specs: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for key, spec in specs.items():
        if isinstance(spec, dict):
            defaults[key] = spec.get("default")
        else:
            defaults[key] = spec
    return defaults


def indicators_for(strategy: str) -> Dict[str, Dict[str, Any]]:
    """Retourne uniquement les valeurs par défaut des indicateurs."""
    return _extract_indicator_defaults(indicator_specs_for(strategy))


def base_params_for(strategy: str) -> Dict[str, Any]:
    """Retourne les paramètres par défaut pour la stratégie."""
    return _extract_param_defaults(parameter_specs_for(strategy))


def tunable_parameters_for(strategy: str) -> Dict[str, Dict[str, Any]]:
    """Paramètres numériques pouvant être optimisés (min/max/step)."""
    specs = parameter_specs_for(strategy)
    tunables: Dict[str, Dict[str, Any]] = {}
    for key, raw_spec in specs.items():
        if isinstance(raw_spec, dict):
            param_type = raw_spec.get("type")
            default = raw_spec.get("default")
        else:
            default = raw_spec
            param_type = "float" if isinstance(raw_spec, float) else "int" if isinstance(raw_spec, int) else None
            raw_spec = {"default": raw_spec}

        if param_type in {"int", "float"} or isinstance(default, (int, float)):
            spec = dict(raw_spec)
            spec.setdefault("type", param_type or ("float" if isinstance(default, float) else "int"))
            tunables[key] = spec

    return tunables


def resolve_range(spec: Dict[str, Any]) -> Tuple[Any, Any]:
    """Renvoie la plage par défaut (min/max) pour un paramètre."""
    opt_range = spec.get("opt_range")
    if isinstance(opt_range, (list, tuple)) and len(opt_range) == 2:
        return opt_range[0], opt_range[1]
    return spec.get("min", spec.get("default")), spec.get("max", spec.get("default"))




