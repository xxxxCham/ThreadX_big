"""
ThreadX Experimental Strategies - AI-Generated Code

⚠️ **ATTENTION: Stratégies Expérimentales**

Ce module contient des stratégies de trading générées automatiquement par
le système Multi-LLM Quant Research Lab.

## Workflow de Génération

1. **CodeWriter Agent** génère le code Python ici (.py files)
2. **Critic Agent** compile, teste et valide la stratégie
3. **Si approuvé** → Promotion vers `strategy/` + enregistrement dans registry
4. **Si rejeté** → Reste ici pour debugging / amélioration

## ⚠️ Avertissements

- Ces stratégies sont **EXPÉRIMENTALES**
- Elles n'ont **PAS** été revues manuellement par un humain
- **NE PAS** utiliser en production sans validation approfondie
- Les performances passées ne garantissent pas les performances futures
- Les backtests peuvent contenir des biais d'overfitting

## Structure des Fichiers

Chaque stratégie générée doit suivre ce pattern:

```
ai_<nom>_v<version>.py     # ex: ai_meanrev_v3.py
```

Et contenir:
- Classe `<Nom>Params` pour paramètres
- Classe `<Nom>Strategy` implémentant Protocol Strategy
- Méthode `run(data, indicators, params) -> RunStats`

## Utilisation

### Auto-Discovery (Automatique)

```python
from threadx.strategy.experimental import AI_STRATEGIES

# Toutes les stratégies AI sont chargées automatiquement
for name, cls in AI_STRATEGIES.items():
    print(f"{name}: {cls}")
```

### Import Manuel (Debugging)

```python
from threadx.strategy.experimental.ai_meanrev_v3 import (
    AIMeanRevV3Params,
    AIMeanRevV3Strategy,
)

# Utilisation
params = AIMeanRevV3Params(...)
strategy = AIMeanRevV3Strategy()
result = strategy.run(data, indicators, params)
```

## Critères de Validation (Critic Agent)

Pour qu'une stratégie soit promue, elle doit passer:

### Critères Techniques
- ✅ Compilation Python sans erreur
- ✅ Import dynamique fonctionne
- ✅ Classe Strategy conforme au Protocol
- ✅ Méthode `run()` retourne RunStats valide

### Critères de Performance (sur données out-of-sample)
- ✅ Sharpe Ratio ≥ 0.5
- ✅ Max Drawdown ≤ 30%
- ✅ Nombre de trades ≥ 10 (robustesse statistique)
- ✅ Win Rate ≥ 35%

### Critères de Qualité Code (LLM Review optionnel)
- ✅ Pas de bibliothèques exotiques (NumPy/Pandas uniquement)
- ✅ Gestion risque présente (stop-loss, position sizing)
- ✅ Code déterministe (seed fixe si aléatoire)
- ✅ Commentaires expliquant la logique

## Exemples de Stratégies

### Exemple 1: Mean Reversion avec Bollinger

```python
# ai_meanrev_v1.py
from threadx.strategy.model import Strategy, RunStats, Trade
import numpy as np
import pandas as pd

class AIMeanRevV1Params:
    def __init__(self, bb_period=20, bb_std=2.0, stop_loss_pct=2.0):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stop_loss_pct = stop_loss_pct

class AIMeanRevV1Strategy:
    def run(self, data, indicators, params):
        # Logic here
        trades = []
        # ... trading logic ...
        return RunStats(trades=trades, equity=...)
```

### Exemple 2: Trend Following avec EMA

```python
# ai_trend_v1.py
from threadx.strategy.model import Strategy, RunStats, Trade
import numpy as np
import pandas as pd

class AITrendV1Params:
    def __init__(self, fast_ema=12, slow_ema=26, atr_period=14):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period

class AITrendV1Strategy:
    def run(self, data, indicators, params):
        # Logic here
        trades = []
        # ... trading logic ...
        return RunStats(trades=trades, equity=...)
```

## Maintenance

### Nettoyage

Stratégies rejetées ou obsolètes peuvent être supprimées manuellement:

```bash
# Supprimer une stratégie rejetée
rm src/threadx/strategy/experimental/ai_badstrategy_v1.py

# Nettoyer toutes les stratégies non promues (danger!)
# rm src/threadx/strategy/experimental/ai_*.py
```

### Promotion Manuelle

Si vous voulez promouvoir manuellement une stratégie:

```bash
# 1. Copier vers strategy/
cp src/threadx/strategy/experimental/ai_meanrev_v3.py src/threadx/strategy/

# 2. Ajouter à strategy/__init__.py
# 3. Enregistrer dans ui/strategy_registry.py
```

---

**Version**: 1.0
**Créé**: 2025-11-25
**Documentation**: docs/QUANT_LAB_FEASIBILITY.md
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache des stratégies AI découvertes
_AI_STRATEGIES: dict[str, type] = {}
_AI_STRATEGIES_METADATA: dict[str, dict[str, Any]] = {}


def _discover_ai_strategies() -> dict[str, type]:
    """
    Scanne experimental/ et charge dynamiquement toutes les stratégies AI.

    Returns:
        dict[module_name, StrategyClass]

    Notes:
        - Cherche tous fichiers ai_*.py
        - Import chaque module
        - Détecte classes se terminant par "Strategy"
        - Ignore les erreurs d'import (logs warning)
    """
    strategies = {}
    metadata = {}

    current_dir = Path(__file__).parent

    # Chercher tous fichiers ai_*.py
    ai_files = list(current_dir.glob("ai_*.py"))

    if not ai_files:
        logger.debug("Aucune stratégie AI trouvée dans experimental/")
        return {}

    logger.info(f"🔍 Scanning {len(ai_files)} stratégies AI potentielles...")

    for py_file in ai_files:
        module_name = py_file.stem

        try:
            # Import dynamique
            mod = importlib.import_module(
                f".{module_name}", package="threadx.strategy.experimental"
            )

            # Chercher classes *Strategy
            strategy_class = None
            params_class = None

            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)

                # Ignorer imports et builtins
                if not isinstance(attr, type):
                    continue

                if attr.__module__ != mod.__name__:
                    continue  # Classe importée, pas définie ici

                if attr_name.endswith("Strategy"):
                    strategy_class = attr

                elif attr_name.endswith("Params"):
                    params_class = attr

            if strategy_class:
                strategies[module_name] = strategy_class

                # Métadonnées (optionnelles)
                metadata[module_name] = {
                    "class_name": strategy_class.__name__,
                    "params_class": params_class.__name__ if params_class else None,
                    "module": mod.__name__,
                    "file": str(py_file),
                    "docstring": strategy_class.__doc__,
                }

                logger.info(f"  ✅ {module_name} → {strategy_class.__name__}")

            else:
                logger.warning(
                    f"  ⚠️  {module_name}: Aucune classe *Strategy trouvée"
                )

        except Exception as e:
            logger.warning(f"  ❌ {module_name}: Erreur import - {e}")
            continue

    logger.info(f"✅ {len(strategies)} stratégies AI chargées")

    return strategies, metadata


# Chargement automatique au import
AI_STRATEGIES: dict[str, type]
_temp_strategies, _AI_STRATEGIES_METADATA = _discover_ai_strategies()
AI_STRATEGIES = _temp_strategies


def get_ai_strategy_metadata(module_name: str) -> dict[str, Any] | None:
    """
    Récupère les métadonnées d'une stratégie AI.

    Args:
        module_name: Nom du module (ex: "ai_meanrev_v3")

    Returns:
        dict avec {class_name, params_class, module, file, docstring}
        ou None si non trouvé
    """
    return _AI_STRATEGIES_METADATA.get(module_name)


def list_ai_strategies() -> list[str]:
    """
    Liste les noms de toutes les stratégies AI disponibles.

    Returns:
        list[str] des noms de modules (ex: ["ai_meanrev_v3", "ai_trend_v1"])
    """
    return list(AI_STRATEGIES.keys())


def reload_ai_strategies() -> None:
    """
    Force le rechargement de toutes les stratégies AI.

    Utile après génération d'une nouvelle stratégie par CodeWriter.
    """
    global AI_STRATEGIES, _AI_STRATEGIES_METADATA

    logger.info("🔄 Rechargement des stratégies AI...")

    # Invalider cache importlib
    import sys

    for module_name in list(sys.modules.keys()):
        if "threadx.strategy.experimental.ai_" in module_name:
            del sys.modules[module_name]

    # Re-scanner
    AI_STRATEGIES, _AI_STRATEGIES_METADATA = _discover_ai_strategies()


__all__ = [
    "AI_STRATEGIES",
    "get_ai_strategy_metadata",
    "list_ai_strategies",
    "reload_ai_strategies",
]
