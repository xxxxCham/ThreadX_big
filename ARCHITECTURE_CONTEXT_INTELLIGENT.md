# Architecture Contexte Intelligent LLM - ThreadX
## Design System Gestion Historiques, Tokens, Stratégies

**Version**: 1.0  
**Date**: 2024-11-21  
**Status**: ✅ Implémenté

---

## 📋 Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Problématiques Résolues](#problématiques-résolues)
3. [Architecture Système](#architecture-système)
4. [Composants](#composants)
5. [Workflow Orchestration](#workflow-orchestration)
6. [Gestion Tokens Évolutifs](#gestion-tokens-évolutifs)
7. [Versioning Stratégies](#versioning-stratégies)
8. [Détection Données Invalides](#détection-données-invalides)
9. [Exemples Utilisation](#exemples-utilisation)
10. [Extensions Futures](#extensions-futures)

---

## 🎯 Vue d'Ensemble

### Objectif
Fournir aux agents LLM (Analyst, Strategist, Critic) un **contexte intelligent et structuré** leur permettant de :

1. **Connaître historiques disponibles** (tokens, dates, qualité données)
2. **Accéder registry stratégies** (versions, performances, évolution)
3. **Détecter données invalides** automatiquement (gaps, tokens manquants)
4. **Gérer évolution tokens** dans le temps (listings/delistings)
5. **Créer/modifier stratégies** avec versioning automatique

### Principe Fondamental
> **"Les agents LLM doivent avoir accès au même contexte qu'un trader humain"**

Un trader sait :
- Quels tokens sont tradables (et depuis quand)
- Quelles stratégies ont déjà été testées (et leurs performances)
- Quelles données sont fiables (qualité, gaps)
- Comment éviter overfitting (validation croisée, walk-forward)

→ Les agents LLM doivent avoir **exactement les mêmes informations**.

---

## ❓ Problématiques Résolues

### 1. **"Comment les LLM savent quels historiques sont disponibles ?"**
**Problème** : Agents proposent params pour tokens inexistants ou périodes sans données.

**Solution** : `DataInventory` scanne dossier `data/` et catalogue :
- Tokens disponibles (BTCUSDC, ETHUSDC, etc)
- Dates disponibilité (start_date → end_date)
- Timeframes (1m, 15m, 1h, etc)
- Qualité données (score 0-1, nombre gaps)

**Résultat** : Agents reçoivent inventaire JSON avant chaque requête.

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
  }
}
```

### 2. **"Comment gérer tokens qui évoluent dans le temps ?"**
**Problème** : Certains tokens listés/delistés (ex: FTT disparu en Nov 2022).

**Solution** : `TokenAvailability.end_date` :
- `None` = token actif aujourd'hui
- `datetime(2022, 11, 15)` = delisting FTX

**Workflow Validation** :
```python
valid, msg = inventory.validate_token_period(
    symbol="FTTUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m"
)
# valid=False, msg="FTTUSDC disponible jusqu'à 2022-11-15. Fin demandée: 2024-11-21"
```

**Résultat** : Agents avertis des tokens obsolètes + alternatives proposées.

### 3. **"Comment détecter données invalides et re-rechercher ?"**
**Problème** : Backtest échoue car données corrompues/gaps massifs.

**Solution** : Validation multi-niveaux :

#### Niveau 1: Pre-Flight Validation (avant backtest)
```python
valid, msg, context = context_manager.validate_optimization_request(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m",
    strategy_name="ma_crossover"
)
# Si invalid → Agents reçoivent message d'erreur + alternatives
```

#### Niveau 2: Quality Scoring (inventaire)
```python
token.data_quality = 1.0 - (gaps / total_bars)
# Qualité < 80% → Warning dans contexte LLM
```

#### Niveau 3: Fallback Strategy (orchestrateur)
Si backtest échoue (exception, NaN, données vides) :
1. Logger erreur avec détails (symbol, période, exception)
2. Callback UI : `log_callback(iteration, "❌ Données invalides BTCUSDC", "ERROR")`
3. Analyst reçoit contexte enrichi : "Previous iteration failed due to data issues"
4. Strategist propose **alternative token** (ETHUSDC) ou **période différente**

**Résultat** : Boucle auto-corrective, pas de blocage sur données invalides.

### 4. **"Comment les agents connaissent stratégies existantes ?"**
**Problème** : Strategist re-crée stratégie déjà testée (perte temps).

**Solution** : `StrategyRegistry` avec versioning :

```python
# Version 1: Humain crée stratégie initiale
registry.add_strategy(StrategyVersion(
    name="ma_crossover",
    version="v1.0",
    params={"short_period": 10, "long_period": 50},
    created_by="human"
))

# Version 2: Optimizer améliore
registry.add_strategy(StrategyVersion(
    name="ma_crossover",
    version="optimized_2024-11-21",
    params={"short_period": 12, "long_period": 48},
    performance={"sharpe_ratio": 1.85, "sortino_ratio": 2.6},
    tier_s_score=72,
    created_by="optimizer",
    parent_version="v1.0"
))

# Agents reçoivent arbre évolution
tree = registry.get_evolution_tree("ma_crossover")
```

**Arbre Évolution** (JSON fourni aux LLM) :
```json
{
  "v1.0": {
    "params": {"short_period": 10, "long_period": 50},
    "performance": null,
    "parent": null,
    "created_by": "human"
  },
  "optimized_2024-11-21": {
    "params": {"short_period": 12, "long_period": 48},
    "performance": {"sharpe_ratio": 1.85},
    "tier_s_score": 72,
    "parent": "v1.0",
    "created_by": "optimizer"
  }
}
```

**Résultat** : Strategist voit historique complet, évite doublons, crée versions incrémentales.

### 5. **"Comment gérer modifications/création nouvelles stratégies ?"**
**Problème** : Orchestrateur modifie stratégie en place → perte versions précédentes.

**Solution** : **Immutabilité + Versioning** :

#### Règle 1: Jamais écraser stratégie existante
```python
# ❌ INTERDIT
strategy.params = new_params  # Perd version précédente

# ✅ BON
registry.add_strategy(StrategyVersion(
    name="ma_crossover",
    version=f"optimized_{datetime.now().date()}",
    params=new_params,
    parent_version=current_version.version  # Lien parent
))
```

#### Règle 2: Chaque itération optimisation = nouvelle version
```python
# Orchestrator run() loop - Itération 5
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

#### Règle 3: Nouvelle stratégie = nouveau nom
```python
# Strategist propose stratégie complètement nouvelle
registry.add_strategy(StrategyVersion(
    name="bollinger_rsi_fusion",  # Nouveau nom
    version="v1.0",
    params={...},
    created_by="strategist",
    description="Fusion Bollinger Bands + RSI divergence"
))
```

**Résultat** : Historique complet, rollback possible, A/B testing facile.

---

## 🏗️ Architecture Système

### Schéma Global

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           ContextManager                            │   │
│  │  ┌──────────────────┐  ┌─────────────────────┐     │   │
│  │  │  DataInventory   │  │  StrategyRegistry   │     │   │
│  │  │                  │  │                     │     │   │
│  │  │ - Tokens         │  │ - Strategies        │     │   │
│  │  │ - Availability   │  │ - Versions          │     │   │
│  │  │ - Quality        │  │ - Performance       │     │   │
│  │  └──────────────────┘  └─────────────────────┘     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           PromptEnricher                            │   │
│  │  enrich_analyst_prompt()                            │   │
│  │  enrich_strategist_prompt()                         │   │
│  │  enrich_critic_prompt()                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│          ┌───────────────┼───────────────┐                 │
│          ▼               ▼               ▼                 │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│    │ Analyst │    │Strategist│    │ Critic  │              │
│    │  LLM    │    │   LLM    │    │  LLM    │              │
│    └─────────┘    └─────────┘    └─────────┘              │
│          │               │               │                 │
│          └───────────────┼───────────────┘                 │
│                          ▼                                  │
│                  Backtest Engine                            │
└─────────────────────────────────────────────────────────────┘
```

### Flux de Données

1. **Initialization** (démarrage orchestrator)
   ```
   ContextManager.__init__()
   → DataInventory._scan_data_directory()
   → StrategyRegistry.load()
   ```

2. **Pre-Flight Validation** (avant optimisation)
   ```
   validate_optimization_request(symbol, dates, timeframe, strategy)
   → validate_token_period()
   → get_latest_version(strategy)
   → return (valid, message, full_context)
   ```

3. **Agent Invocation** (chaque itération)
   ```
   # Analyst
   prompt = PromptEnricher.enrich_analyst_prompt(
       base_prompt,
       context_manager,
       strategy_name,
       backtest_result,
       memory
   )
   diagnosis = analyst.analyze(prompt)

   # Strategist
   prompt = PromptEnricher.enrich_strategist_prompt(
       base_prompt,
       context_manager,
       strategy_name,
       current_params,
       diagnosis,
       memory
   )
   proposals = strategist.propose(prompt)

   # Critic
   prompt = PromptEnricher.enrich_critic_prompt(
       base_prompt,
       context_manager,
       strategy_name,
       proposals,
       memory
   )
   validated = critic.validate(prompt)
   ```

4. **Registry Update** (fin itération)
   ```
   registry.add_strategy(StrategyVersion(
       name=strategy_name,
       version=f"iter_{iteration}",
       params=best_params,
       performance=metrics,
       tier_s_score=score,
       created_by="optimizer",
       parent_version=previous_version
   ))
   registry.save()  # Persist to disk
   ```

---

## 🧩 Composants

### 1. `DataInventory`
**Rôle** : Inventaire complet historiques disponibles.

**Attributs** :
- `tokens: dict[str, TokenAvailability]` - Disponibilité par token
- `global_start_date: datetime` - Date début globale (1 jan 2024)
- `global_end_date: datetime` - Date fin globale (aujourd'hui)

**Méthodes Clés** :
- `get_available_tokens(start, end, timeframe)` → Liste tokens valides
- `validate_token_period(symbol, start, end, tf)` → (bool, message)
- `to_llm_context()` → Dict JSON pour agents

**Exemple** :
```python
inventory = DataInventory()
inventory.add_token(TokenAvailability(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=None,  # Actif
    timeframes=["1m", "15m", "1h"],
    data_quality=0.98,
    total_bars=500000,
    gaps_detected=10000
))

available = inventory.get_available_tokens(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m"
)
# ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
```

### 2. `StrategyRegistry`
**Rôle** : Catalogue stratégies avec versioning.

**Attributs** :
- `strategies: dict[str, list[StrategyVersion]]` - Versions par stratégie
- `registry_path: Path` - Fichier JSON persistant

**Méthodes Clés** :
- `add_strategy(version)` - Ajoute version + save()
- `get_latest_version(name)` → Dernière version
- `get_best_version(name, metric)` → Meilleure version selon métrique
- `get_evolution_tree(name)` → Arbre parent→children
- `save()` / `load()` - Persistence disque

**Exemple** :
```python
registry = StrategyRegistry()
registry.add_strategy(StrategyVersion(
    name="ma_crossover",
    version="v1.0",
    params={"short": 10, "long": 50},
    created_by="human"
))

latest = registry.get_latest_version("ma_crossover")
# version="v1.0", params={...}

best = registry.get_best_version("ma_crossover", metric="sharpe_ratio")
# version avec sharpe_ratio max
```

### 3. `ContextManager`
**Rôle** : Orchestrateur contexte global (inventory + registry).

**Attributs** :
- `inventory: DataInventory` - Inventaire données
- `registry: StrategyRegistry` - Registry stratégies

**Méthodes Clés** :
- `get_full_context(strategy_name)` → Dict contexte complet
- `validate_optimization_request(...)` → Pre-flight validation
- `_scan_data_directory()` - Scanner automatique dossier data/

**Exemple** :
```python
context_manager = ContextManager(
    data_dir=Path("./data"),
    registry_path=Path("./exports/strategy_registry.json")
)

# Validation avant optimisation
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m",
    strategy_name="ma_crossover"
)

if valid:
    # Lancer optimisation avec ctx fourni aux agents
    ...
else:
    print(f"❌ {msg}")
```

### 4. `PromptEnricher`
**Rôle** : Enrichissement prompts agents avec contexte.

**Méthodes** :
- `enrich_analyst_prompt(base, ctx_mgr, strategy, result, memory)`
- `enrich_strategist_prompt(base, ctx_mgr, strategy, params, diagnosis, memory)`
- `enrich_critic_prompt(base, ctx_mgr, strategy, proposals, memory)`

**Template Prompt Enrichi** :
```markdown
# CONTEXTE GLOBAL DISPONIBLE

## Données Disponibles
- Période globale: 2024-01-01 → 2024-11-21
- Tokens: 5 disponibles
- BTCUSDC: 2024-01-01 → today, Quality 98%, Timeframes: 1m, 15m, 1h

## Stratégie Analysée
- Nom: ma_crossover
- Versions: 3
- Meilleure version: v2.1 (Sharpe 1.85, Tier S 72/100)

## Historique Optimisation
- Itération 3: Sharpe 1.75, Analyst 7/10
- Itération 4: Sharpe 1.82, Analyst 8/10
- ⚠️ Convergence détectée (pas d'amélioration)

---

# RÉSULTAT BACKTEST À ANALYSER
```json
{
  "sharpe_ratio": 1.65,
  "sortino_ratio": 2.4,
  "max_drawdown": -12.3,
  "tier_s_score": 68
}
```

---

[BASE PROMPT ORIGINAL]
```

---

## 🔄 Workflow Orchestration

### Séquence Complète Optimisation

```
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 0: INITIALIZATION                                     │
├─────────────────────────────────────────────────────────────┤
│ 1. ContextManager.__init__()                                │
│    → Scan data/ directory                                   │
│    → Load strategy registry                                 │
│ 2. validate_optimization_request()                          │
│    → Check token availability                               │
│    → Check strategy exists                                  │
│    → Generate alternatives if invalid                       │
│ 3. Generate full_context for agents                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 1: BACKTEST INITIAL                                   │
├─────────────────────────────────────────────────────────────┤
│ params = registry.get_latest_version(strategy).params       │
│ result = backtest_engine.run(strategy, params, data)        │
│ → If exception: log error, propose alternative token        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 2: ANALYST DIAGNOSIS                                  │
├─────────────────────────────────────────────────────────────┤
│ prompt = PromptEnricher.enrich_analyst_prompt(              │
│     base_prompt,                                            │
│     context_manager,                                        │
│     strategy_name,                                          │
│     result,                                                 │
│     memory                                                  │
│ )                                                           │
│ → Prompt contient: inventory, registry, memory, result      │
│ diagnosis = analyst.analyze(prompt)                         │
│ → Score 0-10 + recommandations                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 3: STRATEGIST PROPOSALS                               │
├─────────────────────────────────────────────────────────────┤
│ prompt = PromptEnricher.enrich_strategist_prompt(           │
│     base_prompt,                                            │
│     context_manager,                                        │
│     strategy_name,                                          │
│     current_params,                                         │
│     diagnosis,                                              │
│     memory                                                  │
│ )                                                           │
│ → Prompt contient: inventory, registry, diagnosis, memory   │
│ proposals = strategist.propose(prompt)                      │
│ → 3-5 configurations candidates                            │
│                                                             │
│ FOR EACH proposal:                                          │
│   registry.add_strategy(StrategyVersion(                    │
│       name=strategy_name,                                   │
│       version=f"proposal_{iteration}_{i}",                  │
│       params=proposal["params"],                            │
│       created_by="strategist",                              │
│       parent_version=current_version                        │
│   ))                                                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 4: CRITIC VALIDATION                                  │
├─────────────────────────────────────────────────────────────┤
│ prompt = PromptEnricher.enrich_critic_prompt(               │
│     base_prompt,                                            │
│     context_manager,                                        │
│     strategy_name,                                          │
│     proposals,                                              │
│     memory                                                  │
│ )                                                           │
│ → Prompt contient: inventory, registry, proposals, memory   │
│ validated = critic.validate(prompt)                         │
│ → Filtre overfitting, tokens invalides, risques            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 5: PARALLEL BACKTESTS                                 │
├─────────────────────────────────────────────────────────────┤
│ FOR EACH validated_proposal:                                │
│   result = backtest_engine.run(strategy, proposal, data)    │
│   → If exception: log error, skip proposal                  │
│   scores.append(result.sharpe_ratio)                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 6: SELECT BEST                                        │
├─────────────────────────────────────────────────────────────┤
│ best_idx = argmax(scores)                                   │
│ best_params = validated[best_idx]["params"]                 │
│ best_result = results[best_idx]                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 7: UPDATE REGISTRY                                    │
├─────────────────────────────────────────────────────────────┤
│ registry.add_strategy(StrategyVersion(                      │
│     name=strategy_name,                                     │
│     version=f"iter_{iteration}_optimized",                  │
│     params=best_params,                                     │
│     performance=best_result.metrics,                        │
│     tier_s_score=best_result.tier_s_score,                  │
│     created_by="optimizer",                                 │
│     parent_version=current_version                          │
│ ))                                                          │
│ registry.save()  # Persist to disk                          │
│                                                             │
│ memory.add_iteration({                                      │
│     "iteration": iteration,                                 │
│     "sharpe_ratio": best_result.sharpe,                     │
│     "params": best_params                                   │
│ })                                                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                 Loop back to ÉTAPE 1
            (until convergence or max_iterations)
```

---

## 🔄 Gestion Tokens Évolutifs

### Scénario 1: Token Delisted

**Contexte** : FTT delisted après crash FTX (Nov 2022).

**Configuration Inventory** :
```python
inventory.add_token(TokenAvailability(
    symbol="FTTUSDC",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 11, 15),  # Delisting
    timeframes=["15m", "1h"],
    data_quality=0.92,
    total_bars=300000
))
```

**Validation Request** :
```python
valid, msg = inventory.validate_token_period(
    symbol="FTTUSDC",
    start_date=datetime(2024, 1, 1),  # ❌ Après delisting
    end_date=datetime(2024, 11, 21),
    timeframe="15m"
)
# valid=False
# msg="FTTUSDC disponible jusqu'à 2022-11-15. Fin demandée: 2024-11-21"
```

**Action Orchestrator** :
1. Log error: `❌ FTTUSDC not available for period`
2. Get alternatives: `inventory.get_available_tokens(...)` → `["BTCUSDC", "ETHUSDC"]`
3. Callback UI: `log_callback(iteration, "Proposing BTCUSDC instead", "WARNING")`
4. Strategist reçoit contexte enrichi: "FTT unavailable, alternatives: BTC, ETH"
5. Retry avec token alternatif

### Scénario 2: Token Nouvellement Listé

**Contexte** : ARB (Arbitrum) listé Mars 2023.

**Configuration Inventory** :
```python
inventory.add_token(TokenAvailability(
    symbol="ARBUSDC",
    start_date=datetime(2023, 3, 23),  # Listing date
    end_date=None,  # Actif
    timeframes=["1m", "15m", "1h"],
    data_quality=0.96,
    total_bars=150000
))
```

**Validation Request** :
```python
valid, msg = inventory.validate_token_period(
    symbol="ARBUSDC",
    start_date=datetime(2024, 1, 1),  # ✅ Après listing
    end_date=datetime(2024, 11, 21),
    timeframe="15m"
)
# valid=True, msg="OK"
```

**Contexte LLM** :
```json
{
  "tokens": {
    "ARBUSDC": {
      "available_since": "2023-03-23",
      "available_until": "today",
      "note": "Recently listed, limited historical data"
    }
  }
}
```

### Scénario 3: Gaps Massifs Détectés

**Contexte** : Binance maintenance → gaps 6h.

**Configuration Inventory** :
```python
inventory.add_token(TokenAvailability(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=None,
    timeframes=["1m", "15m"],
    data_quality=0.78,  # ⚠️ <80% threshold
    total_bars=500000,
    gaps_detected=110000  # 22% gaps
))
```

**Validation Request** :
```python
valid, msg = inventory.validate_token_period(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m"
)
# valid=False
# msg="BTCUSDC qualité données insuffisante: 78%. Gaps détectés: 110000"
```

**Action** :
1. Proposer alternatives haute qualité (quality ≥95%)
2. Ou proposer période différente (subset sans gaps)
3. Ou proposer timeframe supérieur (1h moins gaps que 15m)

---

## 📦 Versioning Stratégies

### Arbre Évolution Exemple

```
ma_crossover
│
├── v1.0 (human, 2024-01-01)
│   params: {short: 10, long: 50}
│   performance: null
│
├── optimized_2024-02-15 (optimizer, 2024-02-15)
│   params: {short: 12, long: 48}
│   parent: v1.0
│   performance: {sharpe: 1.65, tier_s: 65}
│
├── optimized_2024-03-20 (optimizer, 2024-03-20)
│   params: {short: 14, long: 45}
│   parent: optimized_2024-02-15
│   performance: {sharpe: 1.85, tier_s: 72}  ← BEST
│
└── experimental_bollinger_fusion (strategist, 2024-04-10)
    params: {short: 12, long: 48, bb_period: 20}
    parent: optimized_2024-03-20
    performance: {sharpe: 1.42, tier_s: 58}  ← Failed experiment
```

### Requêtes Registry

```python
# Latest version
latest = registry.get_latest_version("ma_crossover")
# → experimental_bollinger_fusion

# Best version (sharpe)
best = registry.get_best_version("ma_crossover", "sharpe_ratio")
# → optimized_2024-03-20 (sharpe 1.85)

# Evolution tree
tree = registry.get_evolution_tree("ma_crossover")
# → Dict parent→children avec params, performance

# Rollback to version
rollback_params = registry.strategies["ma_crossover"][1].params
# → {short: 12, long: 48} (optimized_2024-02-15)
```

### Persistence Disk

**Format JSON** (`exports/strategy_registry.json`) :
```json
{
  "ma_crossover": [
    {
      "name": "ma_crossover",
      "version": "v1.0",
      "params": {"short_period": 10, "long_period": 50},
      "performance": null,
      "tier_s_score": null,
      "created_at": "2024-01-01T10:00:00",
      "created_by": "human",
      "parent_version": null,
      "description": "Initial implementation",
      "status": "active"
    },
    {
      "name": "ma_crossover",
      "version": "optimized_2024-03-20",
      "params": {"short_period": 14, "long_period": 45},
      "performance": {"sharpe_ratio": 1.85, "sortino_ratio": 2.8},
      "tier_s_score": 72,
      "created_at": "2024-03-20T15:30:00",
      "created_by": "optimizer",
      "parent_version": "optimized_2024-02-15",
      "description": "Improved risk-adjusted returns",
      "status": "active"
    }
  ],
  "bollinger_dual": [
    ...
  ]
}
```

**Opérations** :
- `registry.save()` : Écrit JSON sur disque
- `registry.load()` : Charge depuis disque (init orchestrator)
- Auto-save après chaque `add_strategy()`

---

## 🛡️ Détection Données Invalides

### 1. Pre-Flight Validation

**Avant backtest** (orchestrator `run()` init) :
```python
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="15m",
    strategy_name="ma_crossover"
)

if not valid:
    logger.error(f"Validation failed: {msg}")
    # Proposer alternatives
    alternatives = ctx['data_inventory']['recommendations']
    for alt in alternatives:
        logger.info(f"💡 {alt}")
    return {"error": msg, "alternatives": alternatives}
```

### 2. Runtime Error Handling

**Pendant backtest** (exception catching) :
```python
try:
    result = backtest_engine.run(
        strategy_name=strategy,
        params=params,
        data=data,
        gpu_id=gpu_id
    )
except Exception as e:
    logger.error(f"Backtest failed: {e}")
    
    # Log détails
    error_context = {
        "symbol": symbol,
        "period": f"{start_date} → {end_date}",
        "timeframe": timeframe,
        "exception": str(e),
        "traceback": traceback.format_exc()
    }
    
    # Callback UI
    if self.log_callback:
        self.log_callback(
            iteration,
            f"❌ Backtest failed: {str(e)[:100]}",
            "ERROR"
        )
    
    # Re-validation avec contexte enrichi
    valid, msg, ctx = context_manager.validate_optimization_request(...)
    
    # Proposer alternative token
    alternatives = ctx['data_inventory']['tokens']
    next_token = sorted(
        alternatives.keys(),
        key=lambda s: alternatives[s]['quality_score'],
        reverse=True
    )[0]
    
    logger.info(f"💡 Retry with {next_token} (quality {alternatives[next_token]['quality_score']})")
```

### 3. Quality Monitoring

**Post-backtest** (validation résultat) :
```python
if result.sharpe_ratio is None or pd.isna(result.sharpe_ratio):
    logger.warning("Invalid result: sharpe_ratio is NaN")
    
    # Détecter cause
    if result.trades == 0:
        issue = "No trades executed (strategy inactive)"
    elif result.equity_curve is None:
        issue = "Empty equity curve (data gap?)"
    else:
        issue = "Unknown issue"
    
    # Enrichir contexte pour prochaine itération
    memory.add_issue({
        "iteration": iteration,
        "issue": issue,
        "params": params
    })
    
    # Analyst reçoit historique issues
    analyst_prompt += f"\n\n⚠️ Previous iteration failed: {issue}"
```

---

## 💡 Exemples Utilisation

### Exemple 1: Optimisation Standard

```python
from threadx.llm.orchestrator import OptimizationOrchestrator, OptimizationConfig
from threadx.llm.context_manager import ContextManager
import pandas as pd
from datetime import datetime

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
print(f"📊 Context: {len(ctx['data_inventory']['tokens'])} tokens available")

# 4. Créer config optimisation
config = OptimizationConfig(
    strategy_name="ma_crossover",
    initial_params={"short_period": 10, "long_period": 50},
    target_sharpe=2.0,
    max_iterations=20,
    convergence_threshold=3,
    proposals_per_iteration=3,
    export_dir=Path("./exports")
)

# 5. Lancer orchestrator
orchestrator = OptimizationOrchestrator(
    config=config,
    data=data,
    analyst_model="deepseek-r1:70b",
    strategist_model="gpt-oss:20b",
    critic_model="deepseek-r1:70b",
    gpu_id=0
)

result = orchestrator.run()

# 6. Afficher résultats
print(f"Best Sharpe: {result['best_sharpe']:.2f}")
print(f"Best Params: {result['best_params']}")
print(f"Iterations: {result['total_iterations']}")

# 7. Vérifier registry
latest = context_manager.registry.get_latest_version("ma_crossover")
print(f"Latest version: {latest.version}")
print(f"Tier S: {latest.tier_s_score}/100")
```

### Exemple 2: Gestion Token Invalide

```python
# Cas: Token avec gaps massifs

# 1. Validation détecte problème
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="SHIBAINU",  # Token problématique
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="1m",  # Timeframe granulaire → plus gaps
    strategy_name="ma_crossover"
)

# valid=False
# msg="SHIBAINU qualité données insuffisante: 65%. Gaps détectés: 350000"

# 2. Récupérer alternatives
alternatives = ctx['data_inventory']['recommendations']
# ["Tokens haute qualité (≥95%): BTCUSDC, ETHUSDC, SOLUSDC"]

# 3. Retry avec token alternatif
valid, msg, ctx = context_manager.validate_optimization_request(
    symbol="BTCUSDC",  # Alternative
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    timeframe="1m",
    strategy_name="ma_crossover"
)

# valid=True
# msg="✅ Token: OK. Quality 98%."
```

### Exemple 3: Création Nouvelle Stratégie

```python
# Strategist propose nouvelle stratégie (pas optimisation existante)

from threadx.llm.context_manager import StrategyVersion
from datetime import datetime

# 1. Strategist génère stratégie innovante
new_strategy = StrategyVersion(
    name="rsi_macd_fusion",  # Nouveau nom
    version="v1.0",
    params={
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9
    },
    created_by="strategist",
    description="Fusion RSI + MACD avec divergences"
)

# 2. Ajouter à registry
context_manager.registry.add_strategy(new_strategy)

# 3. Backtest initial
result = backtest_engine.run(
    strategy_name="rsi_macd_fusion",
    params=new_strategy.params,
    data=data
)

# 4. Update version avec performance
new_strategy.performance = {
    "sharpe_ratio": result.sharpe_ratio,
    "sortino_ratio": result.sortino_ratio,
    "max_drawdown": result.max_drawdown
}
new_strategy.tier_s_score = result.tier_s_score

context_manager.registry.save()

print(f"✅ New strategy created: rsi_macd_fusion v1.0")
print(f"   Sharpe: {result.sharpe_ratio:.2f}")
print(f"   Tier S: {result.tier_s_score}/100")
```

---

## 🚀 Extensions Futures

### 1. Multi-Token Optimization
**Objectif** : Optimiser stratégie sur plusieurs tokens simultanément.

**Implémentation** :
```python
config = OptimizationConfig(
    strategy_name="ma_crossover",
    tokens=["BTCUSDC", "ETHUSDC", "SOLUSDC"],  # Multi-token
    ...
)

# Orchestrator backteste chaque token
# Registry stocke best_params par token
registry.add_strategy(StrategyVersion(
    name="ma_crossover",
    version="multi_token_optimized_2024-11-21",
    params={
        "BTCUSDC": {"short": 12, "long": 48},
        "ETHUSDC": {"short": 10, "long": 50},
        "SOLUSDC": {"short": 15, "long": 45}
    },
    ...
))
```

### 2. Walk-Forward Validation
**Objectif** : Valider robustesse avec validation croisée temporelle.

**Implémentation** :
```python
# DataInventory génère splits automatiques
splits = inventory.generate_walk_forward_splits(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 11, 21),
    train_months=3,
    test_months=1
)

# [
#   {"train": (2024-01-01, 2024-03-31), "test": (2024-04-01, 2024-04-30)},
#   {"train": (2024-02-01, 2024-04-30), "test": (2024-05-01, 2024-05-31)},
#   ...
# ]

# Orchestrator backteste chaque split
# Registry stocke performance par split
```

### 3. Auto-Cleaning Données
**Objectif** : Détecter + corriger gaps automatiquement.

**Implémentation** :
```python
# DataInventory détecte gaps
gaps = inventory.detect_gaps("BTCUSDC", timeframe="15m")

# Auto-fill strategies
filled_data = inventory.fill_gaps(
    data,
    method="forward_fill"  # ou "interpolate", "fetch_from_api"
)

# Re-calculer quality score
token.data_quality = 1.0 - (new_gaps / total_bars)
```

### 4. Multi-Timeframe Optimization
**Objectif** : Optimiser stratégie sur plusieurs timeframes.

**Implémentation** :
```python
config = OptimizationConfig(
    strategy_name="ma_crossover",
    timeframes=["15m", "1h", "4h"],  # Multi-timeframe
    ...
)

# Orchestrator backteste chaque timeframe
# Registry stocke best_params par timeframe
```

### 5. Collaborative Registry
**Objectif** : Partager stratégies entre utilisateurs (cloud).

**Implémentation** :
```python
# Upload strategy to cloud
registry.upload_strategy(
    strategy_name="ma_crossover",
    version="optimized_2024-11-21",
    visibility="public"  # ou "private"
)

# Download community strategies
community_strategies = registry.search_community(
    tags=["ma", "high_sharpe"],
    min_tier_s=70
)
```

---

## 📚 Références

### Fichiers Implémentés
- `src/threadx/llm/context_manager.py` - ContextManager + DataInventory + StrategyRegistry
- `src/threadx/llm/prompt_enricher.py` - Enrichissement prompts agents
- `src/threadx/llm/orchestrator.py` - Intégration ContextManager (TODO)

### Documentation Connexe
- `ARCHITECTURE_MULTI_LLM.md` - Architecture globale agents
- `POC_MULTI_LLM_AGENT.md` - POC agents LLM
- `README_MULTI_LLM.md` - Guide utilisation système multi-agent

### Standards
- **Versioning** : [Semantic Versioning 2.0](https://semver.org/)
- **Registry Format** : JSON (compatible Git, diffable, human-readable)
- **Dates** : ISO 8601 (2024-11-21T15:30:00)

---

**Auteur** : ThreadX Framework  
**Date Création** : 2024-11-21  
**Dernière Mise à Jour** : 2024-11-21  
**Status** : ✅ Implémenté (ContextManager, PromptEnricher ready)
