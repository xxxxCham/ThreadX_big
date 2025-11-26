# Quant Research Lab - Guide d'Utilisation V1

**Système autonome de génération et validation de stratégies de trading.**

---

## 🎯 Vue d'ensemble

Le Quant Research Lab permet de:
1. Analyser les résultats d'optimisation d'une stratégie baseline
2. Proposer des modifications paramétriques et architecturales
3. Générer automatiquement du code Python de stratégie améliorée
4. Valider syntaxe, performance, et critères quantitatifs
5. Décider de la promotion automatiquement

**Workflow**: Sweep → Analyst → Strategist → CodeWriter → Critic → Promotion

---

## 📋 Prérequis

1. **ThreadX installé** avec dépendances:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama en cours d'exécution** avec modèle deepseek-r1:
   ```bash
   ollama pull deepseek-r1:32b
   ollama pull deepseek-r1:8b
   ollama serve
   ```

3. **Données OHLCV** disponibles via data_access:
   ```python
   from threadx.data_access.data_loader import load_ohlcv
   data = load_ohlcv("BTCUSDC", "15m", start="2023-07-01", end="2023-12-31")
   ```

---

## 🚀 Utilisation

### Étape 1: Générer Baseline Sweep

Exécuter un sweep d'optimisation sur votre stratégie de base:

```python
from threadx.optimization.engine import OptimizationEngine
from threadx.data_access.data_loader import load_ohlcv

# Charger données
data = load_ohlcv("BTCUSDC", "15m", start="2023-07-01", end="2023-12-31")

# Optimiser
engine = OptimizationEngine(strategy_name="Bollinger_Dual")
results = engine.run_sweep(
    data=data,
    param_ranges={
        "bb_period": [10, 20, 30, 40],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
    },
    metric="sharpe_ratio"
)

# Sauvegarder résultats
import json
with open("results/sweep_bollinger_20251126.json", "w") as f:
    json.dump(results, f, indent=2)
```

**Format attendu du JSON**:
```json
{
  "baseline": {
    "params": {"bb_period": 20, "bb_std": 2.0},
    "sharpe_ratio": 0.45,
    "max_drawdown": -25.0,
    "total_trades": 120,
    "win_rate": 38.5
  },
  "results": [
    {
      "params": {"bb_period": 30, "bb_std": 2.5},
      "sharpe_ratio": 0.62,
      "max_drawdown": -18.0,
      "total_trades": 95,
      "win_rate": 42.1
    },
    ...
  ]
}
```

---

### Étape 2: Lancer Evolution Loop

```bash
cd tools
python run_evolution_loop.py \
    --base-strategy Bollinger_Dual \
    --sweep-results ../results/sweep_bollinger_20251126.json \
    --task improve_sharpe \
    --generation 1 \
    --debug
```

**Paramètres**:
- `--base-strategy`: Nom de la stratégie (`Bollinger_Dual`, `MA_Crossover`, etc.)
- `--sweep-results`: Path vers JSON des résultats de sweep
- `--task`: Objectif (`improve_sharpe`, `reduce_drawdown`, `increase_winrate`)
- `--generation`: Numéro de génération (pour tracking)
- `--debug`: Active logs détaillés

---

### Étape 3: Analyser Résultats

Le script génère un rapport JSON dans `results/ai_evolution/`:

```bash
cat results/ai_evolution/20251126_143052_Bollinger_Dual_gen1.json
```

**Exemple de rapport**:
```json
{
  "generation_num": 1,
  "base_strategy": "Bollinger_Dual",
  "task": "improve_sharpe",
  "status": "approved",
  "steps": {
    "analyst": {
      "status": "success",
      "duration_sec": 12.3,
      "patterns_found": 5
    },
    "strategist": {
      "status": "success",
      "proposals_count": 3
    },
    "codewriter": {
      "status": "success",
      "filename": "ai_bollinger_dual_v1.py",
      "filepath": "src/threadx/strategy/experimental/ai_bollinger_dual_v1.py"
    },
    "critic": {
      "status": "approved",
      "recommendation": "APPROVE",
      "test_quantitative": {
        "best_sharpe": 0.68,
        "worst_drawdown": -19.5,
        "min_trades": 87
      }
    }
  }
}
```

---

### Étape 4: Tester Stratégie Générée

Si `status == "approved"`, tester manuellement la stratégie:

```python
from threadx.strategy.experimental import reload_ai_strategies, AI_STRATEGIES

# Recharger stratégies AI
reload_ai_strategies()

# Importer stratégie générée
strategy_module = AI_STRATEGIES["ai_bollinger_dual_v1"]

# Backtest
from threadx.backtest.engine import BacktestEngine
from threadx.data_access.data_loader import load_ohlcv

data = load_ohlcv("BTCUSDC", "15m", start="2024-01-01", end="2024-06-30")

engine = BacktestEngine()
result = engine.run(
    data=data,
    strategy_class=strategy_module,
    params={"bb_period": 30, "bb_std": 2.5}  # Params optimaux
)

print(f"Sharpe: {result.sharpe_ratio:.3f}")
print(f"Max DD: {result.max_drawdown:.2%}")
print(f"Trades: {result.total_trades}")
```

---

### Étape 5: Promotion Manuelle (V1)

Si la stratégie est satisfaisante après review humaine:

```bash
# 1. Copier vers strategy/
cp src/threadx/strategy/experimental/ai_bollinger_dual_v1.py \
   src/threadx/strategy/

# 2. Ajouter import dans strategy/__init__.py
echo "from .ai_bollinger_dual_v1 import AIBollingerDualV1Strategy" \
     >> src/threadx/strategy/__init__.py

# 3. Enregistrer dans ui/strategy_registry.py
```

**Enregistrement dans registry**:
```python
# Dans ui/strategy_registry.py
from threadx.ui.strategy_registry import register_ai_strategy

register_ai_strategy(
    name="AI_Bollinger_Dual_V1",
    class_name="AIBollingerDualV1Strategy",
    module_name="ai_bollinger_dual_v1",
    params_schema={
        "bb_period": {"type": "int", "min": 10, "max": 50, "default": 30},
        "bb_std": {"type": "float", "min": 1.0, "max": 4.0, "default": 2.5},
        # ...
    },
    indicators_schema={
        "bollinger": {"period_param": "bb_period", "std_param": "bb_std"},
        "atr": {"period": 14},
    },
    metadata={
        "base_strategy": "Bollinger_Dual",
        "generation": 1,
        "sharpe_improvement": 0.23,  # 0.68 - 0.45
    }
)
```

---

## 📊 Critères de Validation

Le Critic applique ces seuils minimaux:

| Métrique | Seuil Minimal | Rationale |
|----------|---------------|-----------|
| **Sharpe Ratio** | ≥ 0.5 | Performance ajustée au risque acceptable |
| **Max Drawdown** | ≥ -30% | Risque de perte contrôlé |
| **Total Trades** | ≥ 10 | Robustesse statistique minimale |
| **Win Rate** | ≥ 35% | Qualité des signaux |

**Personnalisation**:
```python
from threadx.llm.agents.critic import ValidationCriteria

criteria = ValidationCriteria(
    min_sharpe=0.8,           # Plus strict
    max_drawdown_pct=-20.0,   # Plus strict
    min_trades=30,            # Plus strict
    min_win_rate_pct=40.0,    # Plus strict
)

critic = Critic(criteria=criteria)
```

---

## 🔐 Sécurité & Bonnes Pratiques

### Contraintes CodeWriter

Le CodeWriter ne peut que:
- ✅ Utiliser IndicatorBank pour calculs techniques
- ✅ Modifier paramètres existants
- ✅ Ajouter filtres/conditions
- ✅ Utiliser NumPy/Pandas

Le CodeWriter **NE PEUT PAS**:
- ❌ Modifier `data_access`, `backtest/engine`, `indicators/bank`
- ❌ Utiliser bibliothèques exotiques (TA-Lib, etc.)
- ❌ Créer stratégies non-déterministes (sans seed)
- ❌ Ignorer la gestion de risque (stop-loss obligatoire)

### Workflow Sécurisé

```
CodeWriter génère → src/threadx/strategy/experimental/
                         ↓
                    Critic valide
                         ↓
                  Tests passent?
                    ↙        ↘
                 OUI         NON
                  ↓           ↓
            Humain Review   Debugging
                  ↓
          Promotion manuelle
                  ↓
         src/threadx/strategy/ + Registry
```

### Revue Humaine Obligatoire

Avant promotion, vérifier:
1. ✅ **Logique trading** fait sens économiquement
2. ✅ **Gestion risque** présente (stop-loss, position sizing)
3. ✅ **Pas de biais** d'overfitting évidents
4. ✅ **Code lisible** et commenté
5. ✅ **Tests out-of-sample** sur période différente

---

## 🐛 Troubleshooting

### Erreur: "Fichier sweep non trouvé"
```bash
# Vérifier path
ls -la results/sweep_*.json

# Utiliser path absolu
python run_evolution_loop.py \
    --sweep-results /full/path/to/sweep.json
```

### Erreur: "Module threadx not found"
```bash
# Ajouter src/ au PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Ou utiliser depuis racine projet
cd /path/to/ThreadX_big
python tools/run_evolution_loop.py ...
```

### Erreur: "Ollama connection refused"
```bash
# Vérifier Ollama
ollama list

# Démarrer service
ollama serve

# Tester connexion
curl http://localhost:11434/api/tags
```

### Stratégie rejetée par Critic

Vérifier les raisons dans le rapport:
```python
import json
with open("results/ai_evolution/latest.json") as f:
    report = json.load(f)

if report["status"] == "rejected":
    print("Raisons:", report.get("rejection_reasons", []))
    print("Erreurs Critic:", report["steps"]["critic"]["result"]["errors"])
```

**Raisons communes**:
- Sharpe < 0.5 → Paramètres trop conservateurs ou stratégie faible
- Trades < 10 → Conditions d'entrée trop strictes
- Syntaxe invalide → Bug dans CodeWriter (à investiguer)

---

## 📈 Exemples d'Utilisation

### Exemple 1: Améliorer Sharpe de Bollinger Dual

```bash
# 1. Sweep baseline
python -c "
from threadx.optimization.engine import OptimizationEngine
from threadx.data_access.data_loader import load_ohlcv
import json

data = load_ohlcv('BTCUSDC', '15m', start='2023-07-01', end='2023-12-31')
engine = OptimizationEngine(strategy_name='Bollinger_Dual')
results = engine.run_sweep(data, {'bb_period': [10,20,30,40], 'bb_std': [1.5,2.0,2.5,3.0]})

with open('results/sweep_bb.json', 'w') as f:
    json.dump(results, f)
"

# 2. Evolution Loop
python tools/run_evolution_loop.py \
    --base-strategy Bollinger_Dual \
    --sweep-results results/sweep_bb.json \
    --task improve_sharpe \
    --generation 1

# 3. Review résultats
cat results/ai_evolution/*.json | jq '.status'
```

### Exemple 2: Réduire Drawdown de MA Crossover

```bash
python tools/run_evolution_loop.py \
    --base-strategy MA_Crossover \
    --sweep-results results/sweep_ma.json \
    --task reduce_drawdown \
    --generation 1 \
    --debug
```

---

## 📚 Ressources

- **Architecture complète**: [docs/QUANT_LAB_FEASIBILITY.md](../docs/QUANT_LAB_FEASIBILITY.md)
- **Phase 1 (Fondations)**: [Commit 9b5d3e1](https://github.com/.../commit/9b5d3e1)
- **Phase 2A (CodeWriter)**: [Commit a2c118c](https://github.com/.../commit/a2c118c)
- **Phase 2B + 3 (Critic + Orchestration)**: [QUANT_LAB_PHASE2B_PHASE3.md](../QUANT_LAB_PHASE2B_PHASE3.md)

---

## 🔮 Roadmap V2

- [ ] Intégrer vraie BacktestEngine dans Critic (remplacer mock)
- [ ] Walk-forward validation multi-périodes
- [ ] Boucle évolutionnaire multi-générations avec feedback
- [ ] Promotion automatique si approuvé
- [ ] UI Streamlit pour monitoring live
- [ ] LLM code review optionnel (qualité architecture)

---

**Version**: 1.0
**Créé**: 2025-11-26
**Maintenu par**: Quant Research Lab AI
