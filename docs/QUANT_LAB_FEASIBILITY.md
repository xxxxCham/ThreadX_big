# Quant Research Lab Autonome - Analyse de Faisabilité

> **Date**: 2025-11-25
> **Version**: 1.0
> **Statut**: ✅ FAISABLE avec architecture existante

---

## 🎯 Vision du Projet

**Objectif**: Créer un laboratoire autonome de recherche quantitative qui génère, teste, critique et fait évoluer automatiquement des stratégies de trading en utilisant des LLMs.

**Workflow cible**:
```
Stratégie Base → Sweep → Analyst → Strategist → CodeWriter → Critic →
    ↑                                                              ↓
    └────────────────────── Boucle d'évolution ←─────────────────┘
```

---

## ✅ Architecture Existante - Forces

### 1. Moteur de Backtest Robuste

**Fichiers**: `backtest/engine.py` (1240 lignes)

**Capacités**:
- ✅ Multi-GPU (RTX 5080 80% + RTX 2060 20%)
- ✅ Validation anti-overfitting (walk-forward, Monte Carlo)
- ✅ Métriques complètes (Sharpe, DD, Win Rate, etc.)
- ✅ Support Strategy Protocol

**Code clé**:
```python
class BacktestEngine:
    def run(self, data, indicators, params, use_gpu=True):
        # Déterministe (seed=42), reproductible, logs structurés
        # → Retourne RunResult avec equity, trades, metrics
```

**✅ Réutilisable tel quel** pour tester les stratégies générées.

---

### 2. Système Multi-Agents LLM

**Fichiers**:
- `llm/agents/base_agent.py` (140 lignes) - Classe abstraite
- `llm/agents/analyst.py` - Analyse de résultats
- `llm/agents/strategist.py` - Génération de propositions

**Capacités**:
- ✅ Timeout + retries automatiques
- ✅ Logging structuré avec contexte agent
- ✅ Métriques de performance (latence, tokens)
- ✅ Validation des réponses JSON

**Code clé**:
```python
class BaseAgent(ABC):
    def __init__(self, name, model, timeout=60.0):
        self.client = LLMClient(model, timeout)

    def _call_llm(self, prompt, system=None):
        # Gestion erreurs, métriques, retries

    @abstractmethod
    def run(self, **kwargs):
        # Implémentation spécifique par agent
```

**✅ Pattern éprouvé** - CodeWriter et Critic hériteront simplement de BaseAgent.

---

### 3. Registry de Stratégies Centralisé

**Fichiers**:
- `ui/strategy_registry.py` - Mapping stratégie → métadonnées
- `strategy/__init__.py` - Exports centralisés

**Capacités**:
- ✅ Métadonnées indicateurs + paramètres
- ✅ Ranges d'optimisation (min/max/step)
- ✅ Validation de types
- ✅ Support dynamique (importlib)

**Structure actuelle**:
```python
REGISTRY = {
    "MA_Crossover": {
        "indicators": {...},
        "params": {...}
    },
    "EMA_Cross": {...},
    "Bollinger_Dual": {...},
    # ... stratégies Core
}
```

**✅ Extension facile** vers:
```python
REGISTRY = {
    # Stratégies Core (manuelles)
    "MA_Crossover": {"category": "Core", ...},

    # Stratégies AI-Evolved (générées)
    "AI_MeanReversion_v3": {"category": "AI-Evolved", ...},
    "AI_TrendFollower_v1": {"category": "AI-Evolved", ...},
}
```

---

### 4. Orchestration LLM Existante

**Fichier**: `ui/page_llm_optimizer.py` (2189 lignes)

**Pipeline actuel**:
```python
def run_multi_llm_optimization():
    # 1. Sweep GPU (100-1000 configs)
    sweep_results = execute_sweep(...)

    # 2. Analyst - Analyse patterns
    analysis = analyst_agent.run(sweep_results)

    # 3. Strategist - Propositions params
    proposals = strategist_agent.run(analysis)

    # 4. Tests propositions
    test_results = test_proposals(proposals)

    # 5. Rapport + indexation
    report = create_report(...)
```

**✅ Infrastructure complète** - Il suffit d'ajouter:
```python
# Étape 5bis: CodeWriter génère stratégie
code = codewriter_agent.run(analysis, proposals, test_results)

# Étape 6: Critic valide
validation = critic_agent.run(code)

# Étape 7: Si OK → promouvoir + relancer boucle
if validation.status == "APPROVED":
    promote_strategy(code)
    # Reboucle avec nouvelle stratégie comme base
```

---

### 5. Data Access Unifié

**Fichiers**: `data_access/` + caching automatique

**Capacités**:
- ✅ CSV local, InfluxDB, MongoDB
- ✅ Cache TTL intelligent
- ✅ Validation OHLCV automatique

**✅ Données prêtes** pour backtests Critic.

---

## 🆕 Briques à Ajouter - Plan Détaillé

### Brique 1: Dossier Expérimental

**Nouveau**: `src/threadx/strategy/experimental/`

**Contenu**:
```
experimental/
├── __init__.py           # Exports dynamiques
├── ai_meanrev_v1.py      # Stratégie générée par AI
├── ai_trend_v2.py        # Autre stratégie générée
└── README.md             # Documentation workflow
```

**`__init__.py`**:
```python
"""
Stratégies générées automatiquement par le Quant Lab AI.

⚠️ ATTENTION:
- Ces stratégies sont EXPÉRIMENTALES
- Elles doivent passer validation Critic avant promotion
- Ne PAS utiliser en production sans revue humaine

Workflow:
1. CodeWriter génère .py ici
2. Critic teste et valide
3. Si approuvé → copie vers strategy/ + enregistrement registry
"""

# Auto-discovery des stratégies AI
import importlib
from pathlib import Path

_AI_STRATEGIES = {}

def _discover_ai_strategies():
    """Scanne experimental/ et charge dynamiquement."""
    current_dir = Path(__file__).parent
    for py_file in current_dir.glob("ai_*.py"):
        module_name = py_file.stem
        try:
            mod = importlib.import_module(f".{module_name}", package=__package__)
            # Chercher classes *Strategy
            for attr in dir(mod):
                if attr.endswith("Strategy"):
                    _AI_STRATEGIES[module_name] = getattr(mod, attr)
        except Exception as e:
            print(f"⚠️  Erreur chargement {module_name}: {e}")

    return _AI_STRATEGIES

# Chargement automatique
AI_STRATEGIES = _discover_ai_strategies()

__all__ = ["AI_STRATEGIES"]
```

**Complexité**: ⭐⭐☆☆☆ (2/5) - Simple dossier + __init__.py

---

### Brique 2: Agent CodeWriter

**Nouveau**: `src/threadx/llm/agents/codewriter.py`

**Responsabilités**:
1. Prendre en entrée analyse + propositions + métriques baseline
2. Générer du code Python de stratégie complet
3. Écrire dans `strategy/experimental/`
4. Retourner métadonnées (filename, class_name, explanation)

**Signature**:
```python
class CodeWriter(BaseAgent):
    """Génère du code de stratégie à partir d'analyses LLM."""

    def run(
        self,
        task: str,  # "improve_sharpe", "reduce_drawdown", "new_strategy"
        base_strategy: str,  # Nom stratégie de départ
        analysis: dict,  # Résultat Analyst
        proposals: dict,  # Résultat Strategist
        failed_metrics: dict,  # Métriques actuelles insatisfaisantes
        ideas: list[str] | None = None,  # Idées additionnelles
    ) -> dict:
        """
        Returns:
            {
                "status": "success" | "error",
                "filename": "ai_meanrev_v3.py",
                "class_name": "AIMeanRevV3Strategy",
                "code": "...",  # Code Python complet
                "explanation": "Cette stratégie améliore...",
                "error": "..." if status=error
            }
        """
```

**Prompt système**:
```python
SYSTEM_PROMPT = """Tu es un expert en stratégies de trading quantitatives.

TÂCHE: Générer du code Python de stratégie ThreadX.

CONTRAINTES STRICTES:
1. Hériter de la classe Strategy Protocol
2. Utiliser UNIQUEMENT NumPy/Pandas (pas de bibliothèques exotiques)
3. Calculer indicateurs via IndicatorBank (Bollinger, ATR, SMA, EMA, RSI)
4. Implémenter méthode run(data, indicators, params) → RunStats
5. Gestion risque OBLIGATOIRE (stop-loss, position sizing)
6. Code déterministe et reproductible

FORMAT DE SORTIE: JSON strict
{
    "filename": "ai_strategy_v1.py",
    "class_name": "AIStrategyV1",
    "code": "# Code Python complet...",
    "explanation": "Cette stratégie utilise..."
}

EXEMPLE DE STRUCTURE:
```python
from threadx.strategy.model import Strategy, RunStats, Trade
import numpy as np
import pandas as pd

class MyStrategyParams:
    def __init__(self, param1=10, param2=2.0):
        self.param1 = param1
        self.param2 = param2

class MyStrategy:
    def run(self, data, indicators, params):
        # Logique de trading
        # ...
        return RunStats(trades=[...], equity=...)
```
"""
```

**Complexité**: ⭐⭐⭐☆☆ (3/5) - Pattern existant + prompt engineering

---

### Brique 3: Agent Critic

**Nouveau**: `src/threadx/llm/agents/critic.py`

**Responsabilités**:
1. Compiler le code (py_compile + import dynamique)
2. Lancer backtests rapides (sample de données)
3. Appliquer critères de validation durs
4. Optionnel: demander avis LLM sur qualité architecture

**Signature**:
```python
class Critic(BaseAgent):
    """Valide et teste les stratégies générées."""

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.30,
        min_trades: int = 10,
        min_win_rate: float = 0.35,
        use_llm_review: bool = True,
    ):
        super().__init__(name="Critic", model=model)
        self.criteria = {
            "min_sharpe": min_sharpe,
            "max_drawdown": max_drawdown,
            "min_trades": min_trades,
            "min_win_rate": min_win_rate,
        }
        self.use_llm_review = use_llm_review

    def run(
        self,
        code_path: str,  # Chemin vers .py généré
        test_data: pd.DataFrame,  # Données de test
        baseline_metrics: dict,  # Métriques stratégie de base
    ) -> dict:
        """
        Returns:
            {
                "status": "APPROVED" | "REJECTED" | "ERROR",
                "compilation": {"success": bool, "error": str},
                "backtest_metrics": {
                    "sharpe_ratio": float,
                    "max_drawdown": float,
                    "win_rate": float,
                    "total_trades": int,
                },
                "criteria_met": {
                    "min_sharpe": bool,
                    "max_drawdown": bool,
                    ...
                },
                "llm_review": {
                    "quality_score": float,  # 0-10
                    "strengths": [str],
                    "weaknesses": [str],
                    "recommendation": "approve" | "reject" | "revise"
                },
                "improvement_vs_baseline": {
                    "sharpe_delta": float,
                    "dd_delta": float,
                },
                "reason": str  # Explication finale
            }
        """
```

**Workflow interne**:
```python
def run(self, code_path, test_data, baseline_metrics):
    # 1. Compilation
    try:
        spec = importlib.util.spec_from_file_location("temp_strategy", code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        return {"status": "ERROR", "compilation": {"success": False, "error": str(e)}}

    # 2. Import classe Strategy
    strategy_class = None
    for attr in dir(module):
        if attr.endswith("Strategy"):
            strategy_class = getattr(module, attr)
            break

    if not strategy_class:
        return {"status": "ERROR", "reason": "No Strategy class found"}

    # 3. Backtest rapide
    engine = BacktestEngine(use_multi_gpu=False)  # CPU pour rapidité
    result = engine.run(data=test_data, strategy_class=strategy_class, params={...})

    # 4. Validation critères durs
    metrics = {
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        # ...
    }

    criteria_met = {
        "min_sharpe": metrics["sharpe_ratio"] >= self.criteria["min_sharpe"],
        "max_drawdown": metrics["max_drawdown"] <= self.criteria["max_drawdown"],
        # ...
    }

    # 5. Si tous critères OK → LLM review (optionnel)
    if all(criteria_met.values()) and self.use_llm_review:
        llm_review = self._llm_architecture_review(code_path)
    else:
        llm_review = None

    # 6. Décision finale
    if all(criteria_met.values()):
        if llm_review and llm_review["recommendation"] == "approve":
            status = "APPROVED"
        elif not llm_review:
            status = "APPROVED"
        else:
            status = "REJECTED"  # LLM a trouvé un problème
    else:
        status = "REJECTED"

    return {
        "status": status,
        "compilation": {"success": True},
        "backtest_metrics": metrics,
        "criteria_met": criteria_met,
        "llm_review": llm_review,
        "reason": self._generate_reason(criteria_met, llm_review)
    }
```

**Complexité**: ⭐⭐⭐⭐☆ (4/5) - Import dynamique + backtests + validation

---

### Brique 4: Script d'Orchestration

**Nouveau**: `tools/run_evolution_loop.py`

**Responsabilités**:
- Orchestrer boucle complète
- Logger toutes les générations
- Gérer rollback si échec
- Tracking des métriques par génération

**Structure**:
```python
#!/usr/bin/env python
"""
Boucle d'évolution autonome de stratégies.

Usage:
    python tools/run_evolution_loop.py \\
        --base-strategy MA_Crossover \\
        --symbol BTCUSDC \\
        --timeframe 1h \\
        --max-generations 10 \\
        --min-improvement 0.1

Résultats: results/ai_evolution/
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.strategist import Strategist
from threadx.llm.agents.codewriter import CodeWriter
from threadx.llm.agents.critic import Critic
from threadx.optimization.engine import OptimizationEngine
from threadx.data_access import get_cached_ohlcv

def run_evolution_loop(
    base_strategy: str,
    symbol: str,
    timeframe: str,
    max_generations: int = 10,
    min_improvement: float = 0.1,
):
    """Boucle d'évolution complète."""

    # Setup
    output_dir = Path(f"results/ai_evolution/{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Agents
    analyst = Analyst(model="deepseek-r1:70b")
    strategist = Strategist(model="deepseek-r1:70b")
    codewriter = CodeWriter(model="deepseek-r1:70b")
    critic = Critic(
        model="deepseek-r1:8b",
        min_sharpe=0.5,
        max_drawdown=0.30,
        use_llm_review=True
    )

    # Données
    df_train = get_cached_ohlcv(symbol, timeframe, "2023-01-01", "2023-06-30")
    df_test = get_cached_ohlcv(symbol, timeframe, "2023-07-01", "2023-12-31")

    # État
    current_strategy = base_strategy
    best_sharpe = -999
    generation = 0

    while generation < max_generations:
        logger.info(f"\\n{'='*60}")
        logger.info(f"GÉNÉRATION {generation}: {current_strategy}")
        logger.info(f"{'='*60}\\n")

        # 1. Sweep
        sweep_results = run_sweep(current_strategy, df_train)

        # 2. Analyst
        analysis = analyst.run(sweep_results=sweep_results)

        # 3. Strategist
        proposals = strategist.run(analysis=analysis, baseline=sweep_results[0])

        # 4. Tests propositions
        test_results = test_proposals(proposals, df_train)

        # 5. CodeWriter génère nouvelle stratégie
        code_result = codewriter.run(
            task="improve_sharpe",
            base_strategy=current_strategy,
            analysis=analysis,
            proposals=proposals,
            failed_metrics={
                "current_sharpe": sweep_results[0]["sharpe_ratio"],
                "target_sharpe": best_sharpe + min_improvement
            }
        )

        if code_result["status"] != "success":
            logger.error(f"CodeWriter failed: {code_result.get('error')}")
            break

        # 6. Sauvegarder code
        code_path = output_dir / code_result["filename"]
        with open(code_path, "w") as f:
            f.write(code_result["code"])

        # 7. Critic valide
        validation = critic.run(
            code_path=str(code_path),
            test_data=df_test,  # Walk-forward: test sur période différente
            baseline_metrics={"sharpe_ratio": best_sharpe}
        )

        # 8. Sauvegarde résultats
        gen_report = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "code_result": code_result,
            "validation": validation,
        }
        with open(output_dir / f"gen_{generation:03d}.json", "w") as f:
            json.dump(gen_report, f, indent=2)

        # 9. Décision
        if validation["status"] == "APPROVED":
            new_sharpe = validation["backtest_metrics"]["sharpe_ratio"]

            if new_sharpe > best_sharpe + min_improvement:
                logger.info(f"✅ PROMOTION: Sharpe {best_sharpe:.3f} → {new_sharpe:.3f}")

                # Promouvoir stratégie
                promote_strategy(code_path, code_result["class_name"])

                # Update état
                current_strategy = code_result["class_name"]
                best_sharpe = new_sharpe
                generation += 1
            else:
                logger.warning(f"⚠️  Amélioration insuffisante: {new_sharpe:.3f} vs target {best_sharpe + min_improvement:.3f}")
                break
        else:
            logger.error(f"❌ REJET: {validation['reason']}")
            break

    logger.info(f"\\n{'='*60}")
    logger.info(f"Évolution terminée après {generation} générations")
    logger.info(f"Meilleur Sharpe: {best_sharpe:.3f}")
    logger.info(f"Stratégie finale: {current_strategy}")
    logger.info(f"{'='*60}")

def promote_strategy(code_path: Path, class_name: str):
    """Copie stratégie vers strategy/ et enregistre dans registry."""
    from shutil import copy2

    # Copier vers strategy/
    dest = Path("src/threadx/strategy") / code_path.name
    copy2(code_path, dest)

    # TODO: Auto-register dans strategy_registry.py
    # (nécessite parser + modifier le REGISTRY dynamiquement)

    logger.info(f"✅ Stratégie promue: {dest}")
```

**Complexité**: ⭐⭐⭐⭐⭐ (5/5) - Orchestration complète + gestion état

---

### Brique 5: Adaptation Registry

**Modification**: `src/threadx/ui/strategy_registry.py`

**Changements**:
```python
# Ajouter champ "category" dans REGISTRY
REGISTRY = {
    "MA_Crossover": {
        "category": "Core",  # ← NOUVEAU
        "indicators": {...},
        "params": {...},
    },
    # Stratégies AI-Evolved chargées dynamiquement
}

def get_strategies_by_category(category: str = "all") -> dict:
    """
    Filtre stratégies par catégorie.

    Args:
        category: "Core", "AI-Evolved", "all"

    Returns:
        dict des stratégies filtrées
    """
    if category == "all":
        return REGISTRY

    return {
        name: meta
        for name, meta in REGISTRY.items()
        if meta.get("category") == category
    }

def register_ai_strategy(
    name: str,
    class_name: str,
    params_schema: dict,
    indicators_schema: dict,
):
    """
    Enregistre dynamiquement une stratégie AI-Evolved.

    Utilisé par Critic après validation pour rendre
    la stratégie disponible dans l'UI.
    """
    REGISTRY[name] = {
        "category": "AI-Evolved",
        "class_name": class_name,
        "params": params_schema,
        "indicators": indicators_schema,
        "generated_at": datetime.now().isoformat(),
    }

    logger.info(f"✅ Stratégie AI enregistrée: {name}")

def list_strategies() -> list[str]:
    """Liste tous noms de stratégies (Core + AI-Evolved)."""
    return list(REGISTRY.keys())

def get_strategy_class(name: str):
    """
    Récupère classe Strategy par nom.

    Gère à la fois stratégies Core (import standard)
    et AI-Evolved (import dynamique depuis experimental/).
    """
    meta = REGISTRY.get(name)
    if not meta:
        return None

    category = meta.get("category", "Core")

    if category == "Core":
        # Import standard
        from threadx.strategy import (
            MACrossoverStrategy,
            BollingerDualStrategy,
            # ...
        )
        mapping = {
            "MA_Crossover": MACrossoverStrategy,
            "Bollinger_Dual": BollingerDualStrategy,
            # ...
        }
        return mapping.get(name)

    elif category == "AI-Evolved":
        # Import dynamique
        class_name = meta.get("class_name")
        module_name = meta.get("module_name", name.lower())

        try:
            mod = importlib.import_module(
                f"threadx.strategy.experimental.{module_name}"
            )
            return getattr(mod, class_name)
        except Exception as e:
            logger.error(f"Erreur import AI strategy {name}: {e}")
            return None

    return None
```

**Complexité**: ⭐⭐⭐☆☆ (3/5) - Modifications ciblées

---

## 🎯 Plan d'Implémentation - Roadmap

### Phase 1: Fondations (Étape 1-2)

**Objectif**: Infrastructure de base

**Tâches**:
1. ✅ Créer `strategy/experimental/__init__.py`
2. ✅ Adapter `strategy_registry.py` pour catégorie AI-Evolved
3. ✅ Tests unitaires registry (Core + AI-Evolved)

**Durée estimée**: 2-3h

**Validation**:
- Registry charge à la fois stratégies Core et experimental/
- `list_strategies()` retourne les deux catégories
- `get_strategy_class()` fonctionne pour les deux

---

### Phase 2: Agents IA (Étape 3-4)

**Objectif**: CodeWriter + Critic fonctionnels

**Tâches**:
1. Implémenter `llm/agents/codewriter.py`
2. Prompts système pour génération de code
3. Validation syntaxe + format JSON
4. Implémenter `llm/agents/critic.py`
5. Compilation + import dynamique
6. Backtests rapides + critères validation
7. LLM review (optionnel)

**Durée estimée**: 8-10h

**Validation**:
- CodeWriter génère code Python valide
- Critic compile + teste sans crash
- Critères de validation fonctionnent

---

### Phase 3: Orchestration (Étape 5)

**Objectif**: Boucle d'évolution complète

**Tâches**:
1. Script `tools/run_evolution_loop.py`
2. Logging génération par génération
3. Promotion automatique si approuvé
4. Gestion rollback si échec
5. Sauvegarde résultats JSON

**Durée estimée**: 6-8h

**Validation**:
- Boucle tourne pendant N générations
- Métriques s'améliorent ou stagnent de façon tracée
- Stratégies promues apparaissent dans registry

---

### Phase 4: UI & Monitoring (Optionnel)

**Objectif**: Interface Streamlit pour visualiser évolutions

**Tâches**:
1. Nouvel onglet dans `page_llm_optimizer.py`
2. Vue historique générations
3. Graphiques Sharpe/DD par génération
4. Relancer manuellement une stratégie AI

**Durée estimée**: 4-6h

---

## ⚠️ Risques et Limitations

### Risque 1: Qualité du Code Généré

**Problème**: LLM peut générer code avec bugs subtils

**Mitigation**:
- Critic compile + teste AVANT promotion
- Critères durs (min_sharpe, max_dd)
- LLM review pour qualité architecture
- Whitelist de bibliothèques (NumPy/Pandas uniquement)

---

### Risque 2: Overfitting des Stratégies

**Problème**: Stratégies optimisées sur train explosent sur test

**Mitigation**:
- Walk-forward validation (train 2023 H1, test 2023 H2)
- Critic teste sur **période différente** du sweep
- Critère min_trades pour robustesse statistique
- Monte Carlo optionnel pour shuffling temporel

---

### Risque 3: Divergence / Dégénérescence

**Problème**: Boucle génère stratégies de plus en plus complexes et instables

**Mitigation**:
- Limite max_generations (ex: 10)
- Critère min_improvement (ex: 0.1 Sharpe) pour stopper si stagnation
- Complexity penalty dans LLM review (favoriser simplicité)
- Rollback vers best_strategy si régression

---

### Risque 4: Coût LLM

**Problème**: Boucle d'évolution consomme beaucoup de tokens

**Mitigation**:
- Utiliser modèles locaux (Ollama) → coût nul
- CodeWriter/Strategist: modèles 70B (qualité)
- Critic LLM review: modèle 8B (rapidité)
- Caching des réponses LLM identiques

---

## 📊 Métriques de Succès

### Critère 1: Amélioration Sharpe

**Objectif**: +0.3 Sharpe minimum sur 5-10 générations

**Baseline**: MA_Crossover = 0.5
**Target**: AI_Strategy_v5 = 0.8+

---

### Critère 2: Robustesse

**Objectif**: Win Rate > 40%, Max DD < 25%

**Validation**: Backtest sur période out-of-sample

---

### Critère 3: Autonomie

**Objectif**: Boucle tourne sans intervention humaine

**Validation**:
- 0 crash sur 10 générations
- Logs structurés pour debugging
- Rollback automatique si échec

---

## ✅ Conclusion de Faisabilité

### Verdict: **FAISABLE** ✅

**Raisons**:
1. ✅ Architecture ThreadX déjà robuste (BacktestEngine, Multi-LLM)
2. ✅ Pattern BaseAgent éprouvé pour nouveaux agents
3. ✅ Registry extensible facilement
4. ✅ Données + validation déjà en place

**Complexité globale**: ⭐⭐⭐⭐☆ (4/5)

**Durée estimée totale**:
- Phase 1-3 (MVP): **20-25h**
- Phase 4 (UI): **+4-6h**
- **Total**: 24-31h

**Recommandation**:
1. Commencer par **Phase 1** (fondations)
2. Tester **Phase 2** avec génération manuelle
3. Puis **Phase 3** (boucle autonome)
4. **Phase 4** optionnelle selon besoins

---

**Prochaine étape**: Implémenter Phase 1 - Créer `strategy/experimental/` + adapter registry.

**Rapport généré**: 2025-11-25
**Auteur**: Claude Code Agent
**Version**: 1.0
