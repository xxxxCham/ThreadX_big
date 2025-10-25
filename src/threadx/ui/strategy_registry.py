from __future__ import annotations
from typing import Dict, Any, List

# Mapping Strategie  Indicateurs + parametres par defaut
REGISTRY: Dict[str, Dict[str, Any]] = {
    "Bollinger_Breakout": {
        "indicators": {
            "bollinger": {"window": 20, "std": 2.0},
            "rsi": {"window": 14},
        },
        "params": {"confirm_breakout": True},
    },
    "EMA_Cross": {
        "indicators": {
            "ema_fast": {"window": 12},
            "ema_slow": {"window": 26},
        },
        "params": {},
    },
    "ATR_Channel": {"indicators": {"atr": {"window": 14, "mult": 2.0}}, "params": {}},
}


def list_strategies() -> List[str]:
    return list(REGISTRY.keys())


def indicators_for(strategy: str) -> Dict[str, Any]:
    return REGISTRY[strategy]["indicators"]


def base_params_for(strategy: str) -> Dict[str, Any]:
    return REGISTRY[strategy]["params"].copy()



