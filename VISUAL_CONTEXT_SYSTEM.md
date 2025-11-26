# Architecture Contexte Intelligent - Diagrammes & Workflows
## Visualisation Système Gestion Historiques, Tokens, Stratégies

**Version**: 1.0  
**Date**: 2024-11-21

---

## 📊 Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR AUTONOME                        │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     CONTEXT MANAGER                           │ │
│  │                                                               │ │
│  │  ┌─────────────────────┐      ┌──────────────────────────┐   │ │
│  │  │  DATA INVENTORY     │      │  STRATEGY REGISTRY       │   │ │
│  │  │                     │      │                          │   │ │
│  │  │  Tokens:            │      │  Strategies:             │   │ │
│  │  │  - BTCUSDC ✅       │      │  - ma_crossover (3 vers) │   │ │
│  │  │  - ETHUSDC ✅       │      │  - bollinger (2 vers)    │   │ │
│  │  │  - FTTUSDC ❌ 2022  │      │                          │   │ │
│  │  │                     │      │  Operations:             │   │ │
│  │  │  Operations:        │      │  - get_latest_version()  │   │ │
│  │  │  - validate_token() │      │  - get_best_version()    │   │ │
│  │  │  - get_available()  │      │  - get_evolution_tree()  │   │ │
│  │  │  - to_llm_context() │      │  - add_strategy()        │   │ │
│  │  └─────────────────────┘      │  - save() / load()       │   │ │
│  │                                └──────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                               │
│                                    ▼                               │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     PROMPT ENRICHER                           │ │
│  │                                                               │ │
│  │  enrich_analyst_prompt()      → Add: inventory + registry    │ │
│  │  enrich_strategist_prompt()   → Add: diagnosis + memory      │ │
│  │  enrich_critic_prompt()       → Add: proposals + constraints │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                               │
│                    ┌───────────────┼───────────────┐              │
│                    ▼               ▼               ▼              │
│          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│          │   ANALYST    │  │  STRATEGIST  │  │    CRITIC    │    │
│          │     LLM      │  │     LLM      │  │     LLM      │    │
│          │ (diagnosis)  │  │ (proposals)  │  │ (validation) │    │
│          └──────────────┘  └──────────────┘  └──────────────┘    │
│                    │               │               │              │
│                    └───────────────┼───────────────┘              │
│                                    ▼                               │
│                          BACKTEST ENGINE (GPU)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Workflow Validation Pre-Flight

```
┌──────────────────────────────────────────────────────────────┐
│ USER REQUEST                                                 │
│ Optimize "ma_crossover" on BTCUSDC 15m (2024-01-01 → today) │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ CONTEXT MANAGER: validate_optimization_request()            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │ CHECK 1: Token Availability                 │           │
│  │ ✅ BTCUSDC in inventory                     │           │
│  │ ✅ Available since 2024-01-01               │           │
│  │ ✅ Still active (end_date = None)           │           │
│  └──────────────────────────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ CHECK 2: Timeframe Availability              │           │
│  │ ✅ 15m in ["1m", "5m", "15m", "1h", "4h"]   │           │
│  └──────────────────────────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ CHECK 3: Period Coverage                     │           │
│  │ ✅ start_date (2024-01-01) ≥ token.start     │           │
│  │ ✅ end_date (today) ≤ token.end              │           │
│  └──────────────────────────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ CHECK 4: Data Quality                        │           │
│  │ ✅ quality_score = 98% (≥80% threshold)     │           │
│  │ ✅ gaps = 10k / 500k bars (2%)              │           │
│  └──────────────────────────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ CHECK 5: Strategy Exists                     │           │
│  │ ✅ ma_crossover found in registry           │           │
│  │ ✅ Latest version: v2.1                     │           │
│  │    (Sharpe 1.85, Tier S 72/100)             │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ RESULT                                                       │
├──────────────────────────────────────────────────────────────┤
│ valid = True                                                 │
│ message = "✅ Token: OK\n✅ Strategy: ma_crossover loaded"  │
│ context = {                                                  │
│   "data_inventory": {...},  ← Full token catalog            │
│   "strategy_registry": {...},  ← Full version tree          │
│   "system_info": {...}                                       │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  LAUNCH OPTIMIZATION ✅
```

---

## ❌ Workflow Validation Échec (Token Invalide)

```
┌──────────────────────────────────────────────────────────────┐
│ USER REQUEST                                                 │
│ Optimize "ma_crossover" on FTTUSDC 15m (2024-01-01 → today) │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ CONTEXT MANAGER: validate_optimization_request()            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────┐           │
│  │ CHECK 1: Token Availability                 │           │
│  │ ✅ FTTUSDC in inventory                     │           │
│  │ ✅ Available since 2021-01-01               │           │
│  │ ❌ end_date = 2022-11-15 (DELISTED)        │           │
│  └──────────────────────────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ CHECK 3: Period Coverage                     │           │
│  │ ❌ end_date (2024-11-21) > token.end        │           │
│  │    (2022-11-15)                              │           │
│  │                                              │           │
│  │ ERROR: "FTTUSDC disponible jusqu'à          │           │
│  │         2022-11-15. Fin demandée:           │           │
│  │         2024-11-21"                          │           │
│  └──────────────────────────────────────────────┘           │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────┐           │
│  │ GENERATE ALTERNATIVES                        │           │
│  │ get_available_tokens(2024-01-01, today, 15m)│           │
│  │ → ["BTCUSDC", "ETHUSDC", "SOLUSDC"]         │           │
│  └──────────────────────────────────────────────┘           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ RESULT                                                       │
├──────────────────────────────────────────────────────────────┤
│ valid = False                                                │
│ message = "❌ Token: FTTUSDC disponible jusqu'à 2022-11-15."│
│ context = {                                                  │
│   "recommendations": [                                       │
│     "💡 Alternative tokens: BTCUSDC, ETHUSDC, SOLUSDC"     │
│   ]                                                          │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  PROPOSE RETRY WITH BTCUSDC
```

---

## 🌳 Arbre Évolution Stratégie

```
ma_crossover
│
├─ v1.0 (2024-01-01, human)
│  │  params: {short: 10, long: 50}
│  │  performance: null
│  │  status: active
│  │
│  ├─ optimized_2024-02-15 (optimizer)
│  │  │  params: {short: 12, long: 48}
│  │  │  performance: {sharpe: 1.65, tier_s: 65}
│  │  │  parent: v1.0
│  │  │
│  │  └─ optimized_2024-03-20 (optimizer) ⭐ BEST
│  │     │  params: {short: 14, long: 45}
│  │     │  performance: {sharpe: 1.85, tier_s: 72}
│  │     │  parent: optimized_2024-02-15
│  │     │
│  │     ├─ iter_5_optimized (optimizer)
│  │     │  params: {short: 13, long: 46}
│  │     │  performance: {sharpe: 1.88, tier_s: 75}
│  │     │  parent: optimized_2024-03-20
│  │     │
│  │     └─ experimental_multi_tf (strategist) ❌ Failed
│  │        params: {short: 14, long: 45, tf2: "1h"}
│  │        performance: {sharpe: 1.12, tier_s: 42}
│  │        parent: optimized_2024-03-20
│  │
│  └─ proposal_iter1_3 (strategist)
│     params: {short: 16, long: 42}
│     performance: null  (not tested yet)
│     parent: v1.0
│
└─ [8 versions totales]

QUERIES:
  latest = registry.get_latest_version("ma_crossover")
    → iter_5_optimized (most recent created_at)

  best = registry.get_best_version("ma_crossover", "sharpe_ratio")
    → iter_5_optimized (sharpe 1.88)

  tree = registry.get_evolution_tree("ma_crossover")
    → Dict parent→children avec performances
```

---

## 🔄 Workflow Itération Optimisation Complète

```
┌────────────────────────────────────────────────────────────────┐
│ ITERATION N (ex: N=5)                                          │
└────────────────────────────────────────────────────────────────┘
│
│ CURRENT STATE:
│ - best_params = {short: 14, long: 45}
│ - best_sharpe = 1.85
│ - convergence_counter = 1
│
├─ STEP 1: BACKTEST CURRENT ──────────────────────────────────┐
│  backtest_engine.run(ma_crossover, best_params, data)       │
│  → result = {sharpe: 1.85, tier_s: 72}                      │
│  LOG: "✅ Baseline: Sharpe 1.85"                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: ANALYST DIAGNOSIS                                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────┐                 │
│  │ PROMPT ENRICHMENT                      │                 │
│  │ base: "Analyze backtest result..."     │                 │
│  │                                        │                 │
│  │ + DATA INVENTORY (5 tokens, quality)  │                 │
│  │ + STRATEGY REGISTRY (3 versions)      │                 │
│  │ + MEMORY (last 3 iterations)          │                 │
│  │ + RESULT (sharpe 1.85, tier_s 72)     │                 │
│  │                                        │                 │
│  │ → enriched_prompt (1500 chars)        │                 │
│  └────────────────────────────────────────┘                 │
│                       │                                      │
│                       ▼                                      │
│  analyst.analyze(enriched_prompt)                           │
│  → diagnosis = {                                            │
│      "score": 7/10,                                         │
│      "summary": "Good risk-adjusted returns...",            │
│      "issues": ["Max DD could be lower"],                  │
│      "recommendations": ["Tighten stop loss"]              │
│    }                                                        │
│                                                              │
│  LOG: "📊 Analyst Score: 7/10"                              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: STRATEGIST PROPOSALS                                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────┐                 │
│  │ PROMPT ENRICHMENT                      │                 │
│  │ base: "Propose 3 optimizations..."     │                 │
│  │                                        │                 │
│  │ + DATA INVENTORY                       │                 │
│  │ + STRATEGY REGISTRY (evolution tree)   │                 │
│  │ + CURRENT PARAMS {short: 14, long: 45} │                 │
│  │ + ANALYST DIAGNOSIS (score 7/10)       │                 │
│  │ + MEMORY                               │                 │
│  │                                        │                 │
│  │ → enriched_prompt (1800 chars)        │                 │
│  └────────────────────────────────────────┘                 │
│                       │                                      │
│                       ▼                                      │
│  strategist.propose(enriched_prompt)                        │
│  → proposals = [                                            │
│      {"short": 13, "long": 46},  # Conservative             │
│      {"short": 15, "long": 43},  # Aggressive               │
│      {"short": 14, "long": 47}   # Moderate                 │
│    ]                                                        │
│                                                              │
│  FOR EACH proposal:                                         │
│    registry.add_strategy(StrategyVersion(                   │
│        name="ma_crossover",                                 │
│        version=f"proposal_iter5_{i}",                       │
│        params=proposal,                                     │
│        created_by="strategist",                             │
│        parent_version="optimized_2024-03-20"                │
│    ))                                                       │
│                                                              │
│  LOG: "💡 3 proposals generated"                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: CRITIC VALIDATION                                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  critic.validate(proposals)                                 │
│  → validated = [                                            │
│      {"short": 13, "long": 46},  ✅ Conservative OK         │
│      {"short": 15, "long": 43},  ❌ Too aggressive          │
│      {"short": 14, "long": 47}   ✅ Moderate OK             │
│    ]                                                        │
│                                                              │
│  LOG: "✅ 2/3 proposals validated"                          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: PARALLEL BACKTESTS                                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  FOR EACH validated_proposal:                               │
│    result = backtest_engine.run(...)                        │
│                                                              │
│  results = [                                                │
│    {params: {short: 13, long: 46}, sharpe: 1.82},           │
│    {params: {short: 14, long: 47}, sharpe: 1.88}  ⭐ BEST   │
│  ]                                                          │
│                                                              │
│  LOG: "⚡ 2 backtests completed"                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: SELECT BEST                                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  best_idx = argmax([1.82, 1.88])  → 1                       │
│  best_params = {short: 14, long: 47}                        │
│  best_sharpe = 1.88                                         │
│                                                              │
│  IMPROVEMENT = 1.88 - 1.85 = +0.03 ✅                       │
│                                                              │
│  LOG: "🎯 New best: Sharpe 1.88 (+0.03)"                    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 7: UPDATE REGISTRY + MEMORY                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  registry.add_strategy(StrategyVersion(                     │
│      name="ma_crossover",                                   │
│      version="iter_5_optimized",                            │
│      params={short: 14, long: 47},                          │
│      performance={sharpe: 1.88, tier_s: 75},                │
│      created_by="optimizer",                                │
│      parent_version="optimized_2024-03-20"                  │
│  ))                                                         │
│  registry.save()                                            │
│                                                              │
│  memory.add_iteration({                                     │
│      iteration: 5,                                          │
│      sharpe: 1.88,                                          │
│      params: {short: 14, long: 47},                         │
│      improvement: +0.03                                     │
│  })                                                         │
│                                                              │
│  convergence_counter = 0  (reset, improvement detected)     │
│                                                              │
│  LOG: "💾 Registry updated: iter_5_optimized"               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  CONTINUE TO ITERATION 6
```

---

## 📊 Token Availability Timeline

```
Timeline: 2020 ──────────── 2024 ──────────── 2028 →

BTCUSDC   ████████████████████████████████████████████  (active)
          └─ 2024-01-01                          now

ETHUSDC   ████████████████████████████████████████████  (active)
          └─ 2024-01-01                          now

FTTUSDC   ████████████████████╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳  (delisted)
          └─ 2021-01-01    2022-11-15

ARBUSDC   ╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳████████████████████████████  (new)
                      └─ 2023-03-23              now

SHIBAINU  ████████████████████████████████████████████  (low quality)
          └─ 2024-01-01                          now
          Quality: 65% ⚠️ (gaps 35%)


VALIDATION EXAMPLES:

1. Request: BTCUSDC (2024-01-01 → 2024-11-21)
   ✅ VALID - Token active, period covered

2. Request: FTTUSDC (2024-01-01 → 2024-11-21)
   ❌ INVALID - Token delisted 2022-11-15, period exceeds

3. Request: ARBUSDC (2022-01-01 → 2024-11-21)
   ❌ INVALID - Token only available since 2023-03-23

4. Request: SHIBAINU (2024-01-01 → 2024-11-21)
   ❌ INVALID - Quality 65% < 80% threshold

5. Request: ETHUSDC (2024-06-01 → 2024-11-21)
   ✅ VALID - Token active, period covered, quality 97%
```

---

## 🎨 Contexte LLM Structure

```json
{
  "data_inventory": {
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
        "gaps": 10000,
        "last_update": "2024-11-20"
      },
      "ETHUSDC": {...},
      "FTTUSDC": {
        "available_since": "2021-01-01",
        "available_until": "2022-11-15",  ← DELISTED
        "timeframes": ["15m", "1h"],
        "quality_score": "92%",
        "note": "⚠️ Delisted after FTX collapse"
      }
    },
    "recommendations": [
      "Tokens haute qualité (≥95%): BTCUSDC, ETHUSDC, SOLUSDC",
      "5 tokens mis à jour derniers 30 jours",
      "⚠️ 1 tokens avec gaps >10: SHIBAINU"
    ]
  },

  "strategy_registry": {
    "strategy_name": "ma_crossover",
    "total_versions": 3,
    "latest_version": {
      "version": "optimized_2024-03-20",
      "params": {"short_period": 14, "long_period": 45},
      "performance": {"sharpe_ratio": 1.85, "sortino_ratio": 2.8},
      "tier_s_score": 72,
      "created_at": "2024-03-20",
      "created_by": "optimizer"
    },
    "best_version": {
      "version": "optimized_2024-03-20",
      "params": {"short_period": 14, "long_period": 45},
      "performance": {"sharpe_ratio": 1.85},
      "tier_s_score": 72
    },
    "evolution_tree": {
      "v1.0": {
        "params": {"short_period": 10, "long_period": 50},
        "performance": null,
        "parent": null,
        "created_by": "human"
      },
      "optimized_2024-02-15": {
        "params": {"short_period": 12, "long_period": 48},
        "performance": {"sharpe_ratio": 1.65},
        "parent": "v1.0",
        "created_by": "optimizer"
      },
      "optimized_2024-03-20": {
        "params": {"short_period": 14, "long_period": 45},
        "performance": {"sharpe_ratio": 1.85},
        "tier_s_score": 72,
        "parent": "optimized_2024-02-15",
        "created_by": "optimizer"
      }
    }
  },

  "system_info": {
    "current_date": "2024-11-21",
    "data_range": "2024-01-01 to today",
    "total_tokens_tracked": 5,
    "total_strategies": 1
  }
}
```

Ce contexte est injecté dans CHAQUE prompt agent (Analyst, Strategist, Critic) via `PromptEnricher`.

---

**Auteur**: ThreadX Framework  
**Version**: 1.0 - Production Ready  
**Date**: 2024-11-21
