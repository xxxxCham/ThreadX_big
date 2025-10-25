# src/threadx/optimization/scenarios.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import itertools
import random

__all__ = [
    "ScenarioSpec",
    "generate_param_grid",
    "generate_monte_carlo",
]


@dataclass(frozen=True)
class ScenarioSpec:
    type: str
    params: Dict[str, Any]
    seed: int = 42
    n_scenarios: int = 100
    sampler: str = "grid"
    constraints: List[Any] = field(default_factory=list)


def _normalize_param(values: Any) -> List[Any]:
    """
    Normalise un paramètre vers une liste de candidats.
    Accepte :
      - scalaire -> [scalaire]
      - liste/tuple/set -> list(...)
      - dict avec 'value' -> [value]
      - dict avec 'values' -> list(values)
      - dict avec 'grid' -> list(grid)
    """
    if isinstance(values, dict):
        if "value" in values:
            return [values["value"]]
        if "values" in values:
            return list(values["values"])
        if "grid" in values:
            return list(values["grid"])
        raise ValueError(f"Format invalide pour paramètre: {values}")
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


def generate_param_grid(grid_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construit le produit cartésien des paramètres.
    grid_spec est un dict { param_name: <format supporté par _normalize_param> }.
    Retourne une liste de combinaisons (dict).
    """
    keys = list(grid_spec.keys())
    choices = [_normalize_param(grid_spec[k]) for k in keys]

    combos: List[Dict[str, Any]] = []
    for prod in itertools.product(*choices):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def generate_monte_carlo(
    spec: Dict[str, Any], n: int, seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Tire au hasard n combinaisons à partir de spec (mêmes formats que _normalize_param).
    """
    rnd = random.Random(seed)
    pools = {k: _normalize_param(v) for k, v in spec.items()}
    keys = list(pools.keys())

    combos: List[Dict[str, Any]] = []
    for _ in range(n):
        combos.append({k: rnd.choice(pools[k]) for k in keys})
    return combos


if __name__ == "__main__":
    # Test simple pour valider la génération de grille
    test_grid = {
        "timeframe": {"value": "1h"},
        "lookback_hours": {"values": [24, 48, 72]},
        "symbols": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],
    }
    combos = generate_param_grid(test_grid)
    print(f"Grille initiale: {len(combos)} combinaisons")
    for i, combo in enumerate(combos[:3], 1):
        print(f"  Combo {i}: {combo}")
