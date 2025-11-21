# 🤖 Multi-LLM Optimizer + Realistic Execution - Guide Complet

Système d'optimisation automatique de stratégies de trading utilisant 2 agents LLM collaboratifs + Moteur d'exécution réaliste.

---

## 📋 Vue d'Ensemble

Ce POC démontre un système multi-LLM capable d'analyser des résultats de backtests et de proposer automatiquement des améliorations de paramètres, **avec simulation réaliste des frictions de trading**.

**Architecture**:
- **Analyst Agent** (deepseek-r1:70b): Analyse quantitative, identification de patterns
- **Strategist Agent** (gpt-oss:20b): Génération créative de propositions
- **BacktestEngine** (GPU): Validation automatique des propositions
- **RealisticExecutor** (Nouveau!): Frictions réalistes (spread, slippage, latence, rejets)

**Workflow**:
```
Sweep GPU (24 configs)
    ↓
Analyst → Patterns + Recommandations
    ↓
Strategist → 3 Propositions créatives
    ↓
Tests GPU → Validation performances
    ↓
Rapport + Visualisation
```

**Temps d'exécution**: ~2-5 minutes total (selon GPU + vitesse LLM)

---

## 🎯 Frictions Trading Réalistes (TOUS Timeframes)

### Pourquoi c'est crucial ?

**Problème** : Les backtests classiques avec `fee_rate` fixe sont **trop optimistes**.

| Friction Ignorée | Impact Réel | Exemple 1 BTC @ $50k |
|------------------|-------------|----------------------|
| Spread bid/ask | 0.01-0.05% | $5-$25 par trade |
| Slippage variable | 0.03-0.15% | $15-$75 |
| Latence (50-500ms) | 0.005-0.02% | $2.50-$10 |
| Rejets d'ordres | 2-10% trades | -$18.75 par trade perdu |
| Maker vs Taker | 2x frais | $10 → $20 |

**Résultat** : Un backtest à **+45%** peut devenir **+18%** en live trading ! 😱

### Système Intégré

Le moteur `RealisticExecutor` est **intégré directement dans `engine.py`** (pas de fichier séparé).

```python
from threadx.backtest import RealisticExecutor

# Auto-ajustement selon timeframe
executor = RealisticExecutor(
    timeframe="1m",      # Impact x2 (scalping)
    # timeframe="1h",    # Impact x0.6 (swing)
    symbol="BTCUSDT",
    exchange="BINANCE"
)

result = executor.execute_order(
    side="BUY",
    intended_price=50000.0,
    quantity=0.5,
    order_type="MARKET",
    current_volatility=0.015,    # ATR 1.5%
    current_volume_ratio=0.8     # 80% volume normal
)

if result.success:
    print(f"Prix voulu : ${result.intended_price:,.2f}")
    print(f"Prix réel  : ${result.executed_price:,.2f}")
    print(f"Slippage   : {result.slippage_pct:.4f}%")
    print(f"Frais      : ${result.total_fees:.2f}")
    print(f"Latence    : {result.latency_ms:.0f}ms")
else:
    print(f"REJETÉ : {result.rejection_reason}")
```

### Paramètres par Timeframe

| Timeframe | Spread | Slippage | Latence | Rejets | Impact Total |
|-----------|--------|----------|---------|--------|--------------|
| **1 minute** | 0.02% | 0.05% | 150ms | 5-10% | **0.11%** (CRITIQUE) |
| **5 minutes** | 0.015% | 0.03% | 100ms | 3-5% | **0.075%** (IMPORTANT) |
| **15 minutes** | 0.01% | 0.02% | <500ms | 1-3% | **0.05%** (MODÉRÉ) |
| **1 heure** | 0.01% | 0.01% | <500ms | <1% | **0.04%** (FAIBLE) |

### Configuration Exchanges

```python
from threadx.backtest import EXCHANGE_CONFIGS

# Pré-configurations disponibles
BINANCE = EXCHANGE_CONFIGS["BINANCE"]
# → maker_fee: 0.02%, taker_fee: 0.04%
# → spread: 0.01-0.05%, latence: 50-500ms

BINANCE_FUTURES = EXCHANGE_CONFIGS["BINANCE_FUTURES"]
# → Meilleure liquidité (spread 0.005-0.03%)

BYBIT = EXCHANGE_CONFIGS["BYBIT"]
# → Plus cher (taker_fee: 0.055%, rejets: 5%)
```

### Version Numba Ultra-Rapide

Pour backtests sur millions de barres, utilisez la version optimisée :

```python
from threadx.backtest import apply_realistic_execution_numba

# Dans backtest loop (JIT-compiled)
success, exec_price, filled_qty, fees = apply_realistic_execution_numba(
    intended_price=50000.0,
    side=1,  # 1=BUY, -1=SELL
    quantity=0.5,
    spread_bps=2.0,
    slippage_bps=5.0,
    fee_bps=4.0,  # Taker fees
    rejection_prob=0.02,
    random_seed=np.random.random()
)
```

### Impact sur Stratégies

**Scalping 1 minute** :
- 100 trades/jour × 365 jours = 36,500 trades
- Frictions : 0.11% par trade × 36,500 = **228% du capital** en frais ! 😱
- Return brut nécessaire : **+250%** juste pour break-even

**Swing 1 heure** :
- 2 trades/jour × 365 jours = 730 trades
- Frictions : 0.04% par trade × 730 = **7.3% du capital**
- Return brut nécessaire : **+15%** pour +7% net (raisonnable)

**Conclusion** : Les frictions TUENT le scalping haute fréquence, favorisent le swing trading.

---

## ⚙️ Prérequis

### 1. Ollama + Modèles LLM

Installer Ollama:
```bash
# Windows: Télécharger depuis https://ollama.ai
# Linux/Mac:
curl -fsSL https://ollama.ai/install.sh | sh
```

Lancer serveur Ollama:
```bash
ollama serve
```

Télécharger modèles (dans un autre terminal):
```bash
ollama pull deepseek-r1:70b   # ~40GB - Analyse quantitative
ollama pull gpt-oss:20b       # ~12GB - Propositions créatives
```

Vérifier modèles disponibles:
```bash
ollama list
```

### 2. Environnement Python

ThreadX requiert Python 3.12+ avec GPU support (optionnel mais recommandé).

Activer environnement:
```powershell
# PowerShell
.\activate_threadx.ps1
```

Vérifier packages:
```bash
pip list | grep -E "ollama|scipy|numpy|pandas"
```

### 3. GPU (Optionnel)

Le POC fonctionne sur CPU mais **5-10x plus rapide sur GPU**.

Vérifier GPU disponible:
```python
import cupy as cp
print(cp.cuda.Device(0).compute_capability)  # Doit afficher version CUDA
```

Si erreur: ThreadX utilisera NumPy (CPU) automatiquement.

---

## 🚀 Exécution Rapide

### Option 1: Notebook Jupyter (Recommandé)

```bash
# Lancer Jupyter
jupyter notebook notebooks/multi_llm_optimizer.ipynb
```

Exécuter cellules **dans l'ordre**:
1. **Section 1**: Configuration environnement
2. **Section 2**: Définition paramètres
3. **Section 3**: Validation données
4. **Section 4**: Sweep initial (24 configs, ~30s GPU)
5. **Section 5**: Analyse Analyst (~30-60s)
6. **Section 6**: Propositions Strategist (~20-40s)
7. **Section 7**: Tests automatiques (~10s)
8. **Section 8**: Visualisation + Rapport

**Temps total**: 2-5 minutes

### Option 2: Script Python

```python
# TODO: Créer version script standalone
# python scripts/run_multi_llm_poc.py --strategy MA_Crossover --n-proposals 3
```

---

## 📊 Résultats Attendus

### Outputs du Notebook

1. **Tableau Sweep Initial** (Section 4):
   ```
   Top 3 Sharpe:
      short_period  long_period  sharpe_ratio  max_drawdown
   0            15           30         1.823        -0.156
   1            10           50         1.742        -0.189
   2            20           30         1.698        -0.142
   ```

2. **Analyse Analyst** (Section 5):
   ```
   PATTERNS IDENTIFIÉS:
   1. short_period < 15 dans 4/5 top configs
   2. long_period entre 30-50 optimal
   3. use_ema=False légèrement supérieur
   
   RECOMMANDATIONS:
   1. Tester short_period=12 avec long_period=35
   2. Explorer zone short_period < 10
   3. Réduire long_period pour limiter lag
   ```

3. **Propositions Strategist** (Section 6):
   ```
   PROPOSITION 1: Conservative
      short_period: 10 → 12
      long_period: 30 → 35
      Rationale: Réduit drawdown observé...
   
   PROPOSITION 2: Aggressive
      short_period: 10 → 8
      long_period: 30 → 40
      Rationale: Exploite pattern short < 10...
   
   PROPOSITION 3: Exploratoire
      short_period: 10 → 18
      long_period: 30 → 25
      Rationale: Teste zone peu explorée...
   ```

4. **Comparaison Résultats** (Section 8):
   ```
   MEILLEURE CONFIG: Aggressive
      Sharpe: 1.912 (+0.089)
      Return: 42.3% (+8.1%)
      Drawdown: -14.2%
      
      💡 Amélioration Sharpe: +4.9%
   ```

5. **Visualisation**: `multi_llm_comparison.png` (3 graphiques bars)

---

## 🔧 Configuration Avancée

### Modifier Stratégie

```python
# Dans Section 2 du notebook
STRATEGY_NAME = "Bollinger_Breakout"  # Au lieu de MA_Crossover

BASELINE_PARAMS = {
    "period": 20,
    "num_std": 2.0,
    # ...
}

PARAM_SPECS = {
    "period": {"min": 10, "max": 50, "step": 5, "type": int},
    # ...
}
```

### Ajuster Sweep

```python
# Plus de configs pour meilleure analyse
SWEEP_CONFIG = {
    "short_period": [5, 8, 10, 12, 15, 20],  # 6 valeurs
    "long_period": [25, 30, 40, 50, 70],     # 5 valeurs
    "use_ema": [False, True],                # 2 valeurs
}
# Total: 6 * 5 * 2 = 60 configs (~1 min GPU)
```

### Changer Modèles LLM

```python
# Analyst plus rapide (moins précis)
analyst = Analyst(model="qwen3-vl:30b", debug=False)

# Strategist plus créatif
strategist = Strategist(model="gpt-oss:120b", debug=False)
```

Modèles disponibles ThreadX:
- `deepseek-r1:70b` (70B params, analyse profonde)
- `gpt-oss:120b` (120B params, très créatif, lent)
- `gpt-oss:20b` (20B params, bon compromis)
- `gemma3:27b` (27B params, rapide)
- `qwen3-vl:30b` (30B params, multimodal)

---

## 🐛 Troubleshooting

### Erreur: "Connection refused to Ollama"

**Cause**: Ollama serveur pas lancé.

**Solution**:
```bash
# Terminal 1: Lancer serveur
ollama serve

# Terminal 2: Vérifier status
curl http://localhost:11434/api/tags
```

### Erreur: "Model not found: deepseek-r1:70b"

**Cause**: Modèle pas téléchargé.

**Solution**:
```bash
ollama pull deepseek-r1:70b
ollama pull gpt-oss:20b
```

### Erreur: "CUDA out of memory"

**Cause**: GPU mémoire insuffisante pour modèle 70B.

**Solution**:
```python
# Utiliser modèle plus petit
analyst = Analyst(model="gemma3:27b")  # Au lieu de deepseek-r1:70b
```

Ou activer quantization Ollama (réduit VRAM):
```bash
ollama run deepseek-r1:70b --quantize q4_0  # 4-bit quantization
```

### Sweep très lent (>5 min)

**Cause**: CPU uniquement (pas de GPU).

**Solution**:
- Réduire configs de sweep (12 au lieu de 24)
- Installer CuPy pour GPU: `pip install cupy-cuda12x`
- Vérifier GPU dispo: `nvidia-smi`

### LLM timeout après 60s

**Cause**: Modèle trop lent ou serveur surchargé.

**Solution**:
```python
# Augmenter timeout
analyst = Analyst(model="deepseek-r1:70b", timeout=120.0)  # 2 min au lieu de 1 min
```

---

## 📈 Prochaines Étapes

### Niveau 2: Semi-Automatique (2-3 semaines)

1. **Orchestrateur de boucle**:
   - Itérations automatiques (N rounds)
   - Meilleure config devient nouvelle baseline
   - Arrêt si Sharpe converge

2. **Historique des runs**:
   - SQLite database pour tracer propositions
   - Analyse meta-learning (quels patterns fonctionnent)

3. **UI Streamlit**:
   - Dashboard visualisation temps réel
   - Contrôle manuel (pause/resume/abort)

### Niveau 3: Production Complète (6-8 semaines)

1. **Débat multi-agents**:
   - 3+ agents (Analyst, Strategist, Risk Manager)
   - Rounds de discussion (pro/con chaque proposition)
   - Vote consensuel

2. **Adaptive Sweep**:
   - LLM génère les ranges de sweep
   - Bayesian optimization guidée par LLM

3. **Walk-Forward Validation**:
   - Tests out-of-sample automatiques
   - Détection overfitting
   - Robustness scoring

---

## 📚 Ressources

- **Documentation ThreadX**: `COMPLETE_CODEBASE_SURVEY.md`
- **Architecture Multi-LLM**: `ARCHITECTURE_MULTI_LLM.md`
- **Use Cases détaillés**: `POC_MULTI_LLM_AGENT.md`
- **Code Agents**: `src/threadx/llm/agents/`

---

## 🤝 Support

**Problèmes courants**: Voir section Troubleshooting ci-dessus

**Questions/Bugs**: Ouvrir issue GitHub ou consulter docs ThreadX

**Performances LLM**: Vérifier `ollama logs` pour diagnostics

---

**Version**: 1.0.0 (POC Option A)  
**Dernière MAJ**: 2025-01-XX  
**Auteur**: ThreadX Team
