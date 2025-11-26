# Système Contexte Intelligent LLM - Guide Utilisateur
## Gestion Complète Historiques, Tokens, Stratégies

**Version**: 1.0  
**Status**: ✅ Implémenté et Testé  
**Date**: 2024-11-21

---

## 🎯 Vue d'Ensemble

Le **ContextManager** fournit aux agents LLM (Analyst, Strategist, Critic) un **contexte intelligent complet** pour prendre des décisions éclairées lors de l'optimisation autonome de stratégies trading.

### Problèmes Résolus

✅ **"Comment les LLM savent quels historiques sont disponibles ?"**  
→ `DataInventory` scanne automatiquement dossier `data/` et catalogue tous tokens (dates, timeframes, qualité)

✅ **"Comment gérer tokens qui évoluent dans le temps ?"**  
→ `TokenAvailability.end_date` détecte tokens delistés (ex: FTT post-FTX)

✅ **"Comment détecter données invalides et re-rechercher ?"**  
→ Validation multi-niveaux (pre-flight + runtime + quality scoring) + fallback automatique

✅ **"Comment les agents connaissent stratégies existantes ?"**  
→ `StrategyRegistry` avec versioning complet (arbre évolution, performances)

✅ **"Comment gérer modifications/création nouvelles stratégies ?"**  
→ Immutabilité + versioning automatique (jamais écraser, toujours créer version)

---

## 📦 Composants Principaux

### 1. `ContextManager`
Orchestrateur global (inventaire + registry).

**Fichier**: `src/threadx/llm/context_manager.py`

**Fonctionnalités**:
- Scanne automatiquement dossier `data/` au démarrage
- Charge registry stratégies depuis `exports/strategy_registry.json`
- Fournit contexte complet pour agents LLM
- Valide requêtes optimisation (token, période, stratégie)

**Utilisation**:
```python
from threadx.llm.context_manager import ContextManager
from datetime import datetime

# Initialiser
context_manager = ContextManager(
    data_dir=Path("./data"),
    registry_path=Path("./exports/strategy_registry.json")
)

# Valider requête optimisation
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m",
    strategy_name="ma_crossover"
)

if valid:
    # Lancer optimisation avec ctx fourni aux agents
    full_context = context_manager.get_full_context("ma_crossover")
else:
    print(f"❌ {msg}")
```

### 2. `DataInventory`
Inventaire complet historiques disponibles.

**Attributs Clés**:
- `tokens: dict[str, TokenAvailability]` - Disponibilité par token
- `global_start_date` - 2024-01-01 (date début globale)
- `global_end_date` - Aujourd'hui (date fin globale)

**Méthodes**:
- `get_available_tokens(start, end, timeframe)` → Liste tokens valides
- `validate_token_period(symbol, start, end, tf)` → (bool, message)
- `to_llm_context()` → Dict JSON pour agents

**Exemple Contexte LLM**:
```json
{
  "global_period": {
    "start": "2024-01-01",
    "end": "2024-11-21",
    "description": "Historiques disponibles du 1 jan 2024 à aujourd'hui"
  },
  "total_tokens": 5,
  "tokens": {
    "BTCUSDC": {
      "available_since": "2024-01-01",
      "available_until": "today",
      "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
      "quality_score": "98%",
      "total_bars": 500000,
      "gaps": 10000
    }
  },
  "recommendations": [
    "Tokens haute qualité (≥95%): BTCUSDC, ETHUSDC, SOLUSDC"
  ]
}
```

### 3. `StrategyRegistry`
Catalogue stratégies avec versioning.

**Attributs Clés**:
- `strategies: dict[str, list[StrategyVersion]]` - Versions par stratégie
- `registry_path` - Fichier JSON persistant

**Méthodes**:
- `add_strategy(version)` - Ajoute version + save()
- `get_latest_version(name)` → Dernière version
- `get_best_version(name, metric)` → Meilleure version selon métrique
- `get_evolution_tree(name)` → Arbre parent→children

**Exemple Arbre Évolution**:
```
ma_crossover
│
├── v1.0 (human, 2024-01-01)
│   params: {short: 10, long: 50}
│   performance: null
│
├── optimized_2024-02-15 (optimizer)
│   params: {short: 12, long: 48}
│   parent: v1.0
│   performance: {sharpe: 1.65, tier_s: 65}
│
└── optimized_2024-03-20 (optimizer)
    params: {short: 14, long: 45}
    parent: optimized_2024-02-15
    performance: {sharpe: 1.85, tier_s: 72}  ← BEST
```

**Utilisation**:
```python
from threadx.llm.context_manager import StrategyVersion

# Ajouter version
registry.add_strategy(StrategyVersion(
    name="ma_crossover",
    version="v1.0",
    params={"short_period": 10, "long_period": 50},
    created_by="human"
))

# Requêtes
latest = registry.get_latest_version("ma_crossover")
best = registry.get_best_version("ma_crossover", "sharpe_ratio")
tree = registry.get_evolution_tree("ma_crossover")
```

### 4. `PromptEnricher`
Enrichissement prompts agents avec contexte.

**Fichier**: `src/threadx/llm/prompt_enricher.py`

**Méthodes**:
- `enrich_analyst_prompt()` - Contexte + résultat backtest
- `enrich_strategist_prompt()` - Contexte + diagnostic + params actuels
- `enrich_critic_prompt()` - Contexte + propositions à valider

**Utilisation**:
```python
from threadx.llm.prompt_enricher import PromptEnricher

# Analyst
prompt = PromptEnricher.enrich_analyst_prompt(
    base_prompt="Analyze backtest result",
    context_manager=context_manager,
    strategy_name="ma_crossover",
    backtest_result={"sharpe_ratio": 1.75},
    memory=memory
)
diagnosis = analyst.analyze(prompt)

# Strategist
prompt = PromptEnricher.enrich_strategist_prompt(
    base_prompt="Propose optimizations",
    context_manager=context_manager,
    strategy_name="ma_crossover",
    current_params={"short": 10, "long": 50},
    analyst_diagnosis=diagnosis,
    memory=memory
)
proposals = strategist.propose(prompt)
```

---

## 🔄 Workflow Complet

### Séquence Optimisation Autonome

```
1. INITIALIZATION
   └─> ContextManager.__init__()
       ├─> Scan data/ directory → DataInventory
       └─> Load strategy_registry.json → StrategyRegistry

2. PRE-FLIGHT VALIDATION
   └─> validate_optimization_request(symbol, dates, timeframe, strategy)
       ├─> Check token availability
       ├─> Check data quality (gaps, etc)
       ├─> Check strategy exists
       └─> Generate alternatives if invalid

3. ITERATION LOOP (until convergence)
   │
   ├─> STEP 1: Backtest
   │   └─> backtest_engine.run(strategy, params, data)
   │       └─> If exception: propose alternative token
   │
   ├─> STEP 2: Analyst Diagnosis
   │   └─> prompt = PromptEnricher.enrich_analyst_prompt(...)
   │       └─> Contains: inventory, registry, memory, result
   │   └─> diagnosis = analyst.analyze(prompt)
   │
   ├─> STEP 3: Strategist Proposals
   │   └─> prompt = PromptEnricher.enrich_strategist_prompt(...)
   │       └─> Contains: inventory, registry, diagnosis, memory
   │   └─> proposals = strategist.propose(prompt)
   │   └─> FOR EACH proposal:
   │       └─> registry.add_strategy(StrategyVersion(...))
   │
   ├─> STEP 4: Critic Validation
   │   └─> prompt = PromptEnricher.enrich_critic_prompt(...)
   │   └─> validated = critic.validate(prompt)
   │       └─> Filter overfitting, invalid tokens, risks
   │
   ├─> STEP 5: Parallel Backtests
   │   └─> FOR EACH validated_proposal:
   │       └─> result = backtest_engine.run(...)
   │
   ├─> STEP 6: Select Best
   │   └─> best_idx = argmax(scores)
   │
   └─> STEP 7: Update Registry
       └─> registry.add_strategy(StrategyVersion(
               name=strategy_name,
               version=f"iter_{iteration}_optimized",
               params=best_params,
               performance=metrics,
               tier_s_score=score,
               created_by="optimizer",
               parent_version=previous_version
           ))
       └─> registry.save()
```

---

## 🛡️ Validation Multi-Niveaux

### 1. Pre-Flight Validation
**Avant backtest** (détection problèmes avant calcul).

```python
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m",
    strategy_name="ma_crossover"
)

# Si invalid:
# - msg contient raison précise (token manquant, période hors dispo, etc)
# - ctx contient alternatives (tokens haute qualité, périodes valides)
```

**Vérifications**:
- ✅ Token existe dans inventaire
- ✅ Timeframe disponible pour token
- ✅ Période couverte par données (start ≥ token.start_date, end ≤ token.end_date)
- ✅ Qualité données suffisante (≥80%)
- ✅ Stratégie existe dans registry (warning si non)

### 2. Runtime Error Handling
**Pendant backtest** (exception catching).

```python
try:
    result = backtest_engine.run(strategy, params, data, gpu_id)
except Exception as e:
    logger.error(f"Backtest failed: {e}")
    
    # Proposer alternative token
    alternatives = context_manager.inventory.get_available_tokens(...)
    next_token = sorted(alternatives, key=lambda s: quality[s], reverse=True)[0]
    
    logger.info(f"💡 Retry with {next_token}")
```

### 3. Quality Monitoring
**Post-backtest** (validation résultat).

```python
if result.sharpe_ratio is None or pd.isna(result.sharpe_ratio):
    # Détection cause
    if result.trades == 0:
        issue = "No trades executed (strategy inactive)"
    elif result.equity_curve is None:
        issue = "Empty equity curve (data gap?)"
    
    # Enrichir contexte pour prochaine itération
    memory.add_issue({"iteration": iteration, "issue": issue})
```

---

## 🔄 Gestion Tokens Évolutifs

### Scénario 1: Token Delisted
**Exemple**: FTT (delisted après crash FTX, Nov 2022).

```python
inventory.add_token(TokenAvailability(
    symbol="FTTUSDC",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 11, 15),  # ⚠️ Delisting
    timeframes=["15m", "1h"],
    data_quality=0.92
))

# Validation détecte problème
valid, msg = inventory.validate_token_period(
    symbol="FTTUSDC",
    start_date=datetime(2024, 1, 1),  # ❌ Après delisting
    end_date=datetime(2024, 11, 21)
)
# valid=False
# msg="FTTUSDC disponible jusqu'à 2022-11-15. Fin demandée: 2024-11-21"
```

### Scénario 2: Token Nouvellement Listé
**Exemple**: ARB (Arbitrum, listé Mars 2023).

```python
inventory.add_token(TokenAvailability(
    symbol="ARBUSDC",
    start_date=datetime(2023, 3, 23),  # Listing date
    end_date=None,  # Actif
    data_quality=0.96
))

# Contexte LLM contient note
{
  "ARBUSDC": {
    "available_since": "2023-03-23",
    "note": "Recently listed, limited historical data"
  }
}
```

### Scénario 3: Gaps Massifs Détectés
**Exemple**: Binance maintenance → gaps 6h.

```python
# Qualité <80% → Validation échoue
token.data_quality = 0.78  # 78% (gaps 22%)

valid, msg = inventory.validate_token_period(...)
# valid=False
# msg="BTCUSDC qualité données insuffisante: 78%. Gaps détectés: 110000"

# Proposer alternatives
alternatives = inventory.get_available_tokens(...)  # Tokens quality ≥95%
```

---

## 📦 Versioning Stratégies

### Règles Immutabilité

**1. Jamais écraser stratégie existante**
```python
# ❌ INTERDIT
strategy.params = new_params  # Perd version précédente

# ✅ BON
registry.add_strategy(StrategyVersion(
    name="ma_crossover",
    version=f"optimized_{datetime.now().date()}",
    params=new_params,
    parent_version=current_version.version
))
```

**2. Chaque itération optimisation = nouvelle version**
```python
# Orchestrator run() loop
best_params = select_best_proposal(proposals)

registry.add_strategy(StrategyVersion(
    name=config.strategy_name,
    version=f"iter_{iteration}_optimized",
    params=best_params,
    performance=backtest_result.metrics,
    tier_s_score=tier_s_score,
    created_by="optimizer",
    parent_version=f"iter_{iteration-1}_optimized"
))
```

**3. Nouvelle stratégie = nouveau nom**
```python
# Strategist propose stratégie innovante
registry.add_strategy(StrategyVersion(
    name="bollinger_rsi_fusion",  # ✅ Nouveau nom
    version="v1.0",
    params={...},
    created_by="strategist"
))
```

### Persistence Disk

**Format JSON** (`exports/strategy_registry.json`):
```json
{
  "ma_crossover": [
    {
      "name": "ma_crossover",
      "version": "v1.0",
      "params": {"short_period": 10, "long_period": 50},
      "performance": null,
      "created_at": "2024-01-01T10:00:00",
      "created_by": "human",
      "parent_version": null
    },
    {
      "name": "ma_crossover",
      "version": "optimized_2024-03-20",
      "params": {"short_period": 14, "long_period": 45},
      "performance": {"sharpe_ratio": 1.85},
      "tier_s_score": 72,
      "created_at": "2024-03-20T15:30:00",
      "created_by": "optimizer",
      "parent_version": "optimized_2024-02-15"
    }
  ]
}
```

**Opérations**:
- Auto-save après chaque `add_strategy()`
- `registry.load()` au démarrage orchestrator
- Git-friendly (JSON diffable, human-readable)

---

## 💡 Exemples Utilisation

### Exemple 1: Optimisation Standard

```python
from threadx.llm.orchestrator import OptimizationOrchestrator, OptimizationConfig
from threadx.llm.context_manager import ContextManager
from pathlib import Path
from datetime import datetime
import pandas as pd

# 1. Charger données
data = pd.read_parquet("data/BTCUSDC_15m_2024-01-01_2024-11-21.parquet")

# 2. Initialiser context manager
context_manager = ContextManager(
    data_dir=Path("./data"),
    registry_path=Path("./exports/strategy_registry.json")
)

# 3. Validation pré-optimisation
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m",
    strategy_name="ma_crossover"
)

if not valid:
    print(f"❌ {msg}")
    exit(1)

print(f"✅ {msg}")

# 4. Créer config optimisation
config = OptimizationConfig(
    strategy_name="ma_crossover",
    initial_params={"short_period": 10, "long_period": 50},
    target_sharpe=2.0,
    max_iterations=20
)

# 5. Lancer orchestrator
orchestrator = OptimizationOrchestrator(
    config=config,
    data=data,
    analyst_model="deepseek-r1:70b",
    strategist_model="gpt-oss:20b"
)

result = orchestrator.run()

# 6. Résultats
print(f"Best Sharpe: {result['best_sharpe']:.2f}")
print(f"Best Params: {result['best_params']}")

# 7. Vérifier registry
latest = context_manager.registry.get_latest_version("ma_crossover")
print(f"Latest version: {latest.version}")
print(f"Tier S: {latest.tier_s_score}/100")
```

### Exemple 2: Gestion Token Invalide

```python
# Token avec gaps massifs
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="SHIBAINU",  # Problématique
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="1m"
)

# valid=False
# msg="SHIBAINU qualité données insuffisante: 65%. Gaps: 350000"

# Récupérer alternatives
alternatives = ctx['data_inventory']['recommendations']
# ["Tokens haute qualité (≥95%): BTCUSDC, ETHUSDC, SOLUSDC"]

# Retry avec alternative
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="BTCUSDC",  # ✅ Alternative
    ...
)
```

### Exemple 3: Création Nouvelle Stratégie

```python
from threadx.llm.context_manager import StrategyVersion
from datetime import datetime

# Strategist génère stratégie innovante
new_strategy = StrategyVersion(
    name="rsi_macd_fusion",  # Nouveau nom
    version="v1.0",
    params={
        "rsi_period": 14,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26
    },
    created_by="strategist",
    description="Fusion RSI + MACD avec divergences"
)

# Ajouter à registry
context_manager.registry.add_strategy(new_strategy)

# Backtest initial
result = backtest_engine.run(
    strategy_name="rsi_macd_fusion",
    params=new_strategy.params,
    data=data
)

# Update performance
new_strategy.performance = {
    "sharpe_ratio": result.sharpe_ratio,
    "sortino_ratio": result.sortino_ratio
}
new_strategy.tier_s_score = result.tier_s_score

context_manager.registry.save()
```

---

## 🧪 Testing

### Lancer Tests POC

```bash
cd /workspaces/ThreadX_big
PYTHONPATH=/workspaces/ThreadX_big/src:$PYTHONPATH python tools/poc_context_manager.py
```

**Tests Inclus**:
1. ✅ ContextManager - Inventaire + Registry
2. ✅ Validation Pre-Flight - Token + Période
3. ✅ Prompt Enrichment - Contexte Agents
4. ✅ Strategy Versioning - Évolution
5. ✅ Error Handling - Données Invalides
6. ✅ Full Integration - Workflow Complet

**Résultat Attendu**:
```
============================================================
✅ ALL TESTS PASSED
============================================================

📁 Registry saved to: ./exports/strategy_registry_poc.json
```

### Inspecter Registry Généré

```bash
cat exports/strategy_registry_poc.json | jq '.ma_crossover[] | {version, params, performance, created_by}'
```

**Sortie Exemple**:
```json
{
  "version": "v1.0",
  "params": {"short_period": 10, "long_period": 50},
  "performance": null,
  "created_by": "human"
}
{
  "version": "optimized_2024-03-20",
  "params": {"short_period": 14, "long_period": 45},
  "performance": {"sharpe_ratio": 1.85},
  "created_by": "optimizer"
}
```

---

## 📚 Documentation Connexe

### Fichiers Implémentés
- `src/threadx/llm/context_manager.py` - ContextManager + DataInventory + StrategyRegistry (700+ lignes)
- `src/threadx/llm/prompt_enricher.py` - Enrichissement prompts agents (400+ lignes)
- `tools/poc_context_manager.py` - POC test complet (500+ lignes)

### Architecture Docs
- `ARCHITECTURE_CONTEXT_INTELLIGENT.md` - Design complet système (2000+ lignes)
- `ARCHITECTURE_MULTI_LLM.md` - Architecture globale agents
- `POC_MULTI_LLM_AGENT.md` - POC agents LLM

### Guides Utilisateur
- `README_MULTI_LLM.md` - Guide utilisation multi-agent
- `GUIDE_ORCHESTRATOR_UI.md` - Interface Streamlit supervision

---

## 🚀 Extensions Futures

### 1. Multi-Token Optimization
Optimiser stratégie sur plusieurs tokens simultanément.

```python
config = OptimizationConfig(
    strategy_name="ma_crossover",
    tokens=["BTCUSDC", "ETHUSDC", "SOLUSDC"],  # Multi-token
    ...
)

# Registry stocke best_params par token
```

### 2. Walk-Forward Validation
Validation croisée temporelle pour robustesse.

```python
splits = inventory.generate_walk_forward_splits(
    symbol="BTCUSDC",
    train_months=3,
    test_months=1
)
# [
#   {"train": (2024-01-01, 2024-03-31), "test": (2024-04-01, 2024-04-30)},
#   ...
# ]
```

### 3. Auto-Cleaning Données
Détecter + corriger gaps automatiquement.

```python
gaps = inventory.detect_gaps("BTCUSDC", timeframe="15m")
filled_data = inventory.fill_gaps(data, method="forward_fill")
```

### 4. Collaborative Registry
Partager stratégies entre utilisateurs (cloud).

```python
registry.upload_strategy(
    strategy_name="ma_crossover",
    version="optimized_2024-11-21",
    visibility="public"
)

community_strategies = registry.search_community(
    tags=["ma", "high_sharpe"],
    min_tier_s=70
)
```

---

## 📝 Notes Importantes

### Conventions Naming Versions
- `v1.0`, `v2.0` → Versions humaines (major changes)
- `optimized_YYYY-MM-DD` → Versions optimizer (date-based)
- `proposal_iter{N}_{i}` → Propositions intermédiaires
- `iter_{N}_optimized` → Meilleure proposition itération N

### Dates ISO 8601
Toutes dates au format ISO : `2024-11-21T15:30:00`

### Quality Score Threshold
Qualité données <80% → Validation échoue (trop gaps)

### Registry Auto-Save
Chaque `add_strategy()` déclenche `save()` automatique.

---

**Auteur**: ThreadX Framework  
**Contact**: System Multi-Agent Optimization  
**License**: Internal Use  
**Version**: 1.0 - Production Ready
