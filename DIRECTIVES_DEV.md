# 📋 DIRECTIVES DE DÉVELOPPEMENT ThreadX

**Fichier centralisé d'instructions pour tous les LLM et développeurs.**

*Dernière mise à jour: 21 novembre 2025*

---

## 🎯 PRINCIPE FONDAMENTAL

> **Consolidation plutôt que dispersion.**
> 
> Préférer ajouter fonctions à fichiers existants plutôt que créer nouveaux fichiers.
> Performance et qualité restent prioritaires.

---

## ✨ RÉSUMÉ EXÉCUTIF

### À FAIRE
- ✅ Ajouter fonctions aux modules existants
- ✅ Un seul fichier doc par sujet
- ✅ Type hints + docstrings obligatoires
- ✅ Logging avec logger, pas print()

### À ÉVITER
- ❌ Créer nouveaux fichiers sans raison
- ❌ Multiplier fichiers markdown
- ❌ Disperser code similaire
- ❌ Négliger qualité code

### Exceptions (Créer fichier SI)
1. **Performance impactée** → Nouvelle séparation OK
2. **Module > 2000 lignes** → Split OK
3. **Langage différent** (Go vs Python) → Dossier séparé OK

---

## 📐 ARCHITECTURE GÉNÉRALE

### Structure Répertoires
```
/workspaces/ThreadX_big/
├── DIRECTIVES_DEV.md       # CE FICHIER - À jour obligatoire
├── README.md               # Documentation principale
├── .llmrc                  # Instructions LLM
├── requirements.txt        # Dépendances
│
├── src/threadx/            # Code principal
│   ├── backtest/           # Engine + métriques + frictions
│   ├── indicators/         # EMA, Stochastique, Bollinger
│   ├── strategy/           # Stratégies (MA, EMA/Stoch, etc)
│   └── utils/              # Logging, helpers
│
└── tools/                  # Outils séparés
    └── netdata-bridge/     # Pont Netdata MCP (Go) - SÉPARÉ
```

### Responsabilités Modules

| Module | Rôle |
|--------|------|
| `backtest/engine.py` | Moteur backtest + RealisticExecutor + frictions |
| `indicators/` | Indicateurs techniques (tous les .py) |
| `strategy/` | Stratégies trading (tous les .py) |
| `utils/log.py` | Logging centralisé |

---

## 📝 CONVENTIONS NOMMAGE

### Python
```python
# Classes - PascalCase
class MACrossover:
class ExecutionResult:

# Fonctions - snake_case
def execute_realistic_order():
def calculate_sharpe_ratio():

# Constantes - UPPERCASE
EXCHANGE_CONFIGS = {}
TIMEFRAME_MULTIPLIERS = {}

# Variables - snake_case
local_variable = 42
result_list = []
```

### Go
```go
// Exports - PascalCase
type MCPBridge struct {}
func (b *MCPBridge) Run() {}

// Private - camelCase
func (b *MCPBridge) dial() {}
```

---

## 🔧 STACK TECHNOLOGIQUE

### Versions Requises
```
Python:   3.11+ MINIMUM
NumPy:    2.0.2 (EXACTEMENT - <2.1 pour numba)
Pandas:   2.2.0+
Numba:    0.60.0 (JIT backtest loops)
```

**CRITIQUE:** NumPy <2.1 pour compatibilité Numba. Sinon backtests crash.

---

## 📊 FRICTIONS RÉALISTES - SPECS

### Où Intégrer
**Location:** `src/threadx/backtest/engine.py`

**Classes Requises:**
```python
@dataclass
class ExecutionResult:
    success: bool
    intended_price: float
    executed_price: float
    slippage_pct: float
    total_fees: float
    latency_ms: float
    rejection_reason: str | None

class RealisticExecutor:
    def execute_order(...) -> ExecutionResult:
        pass
```

### Timeframe Multipliers
```python
TIMEFRAME_MULTIPLIERS = {
    "1m": 2.0,    # Impact x2
    "5m": 1.4,
    "15m": 1.0,   # Référence
    "1h": 0.6,
    "1d": 0.2
}
```

### Exchanges
```python
EXCHANGE_CONFIGS = {
    "BINANCE": {
        "maker_fee": 2.0,    # bps
        "taker_fee": 4.0,    # bps
        "min_spread": 1.0,   # bps
        "rejection_rate": 0.02  # 2%
    },
    "BYBIT": {...},
    "BINANCE_FUTURES": {...}
}
```

---

## 🌐 NETDATA MCP BRIDGE

### Qu'est-ce que c'est
- Langage: **Go 1.16+**
- Localisation: `/tools/netdata-bridge/`
- Rôle: WebSocket MCP → Netdata monitoring
- Status: Outil développement (pas intégré trading)

### Installation
```bash
cd /tools/netdata-bridge
chmod +x build.sh
./build.sh
```

### Usage
```bash
./nd-mcp ws://localhost:19999/mcp
```

### Intérêt
**Monitoring temps réel pour que Claude devienne DevOps assistant:**
- Auto-diagnostic performance (CPU, RAM, I/O)
- Optimisation data-driven (workers selon ressources)
- Alertes proactives (OOM, swap thrashing)
- Analyse post-mortem backtests

---

## 🤖 ARCHITECTURE MULTI-AGENTS LLM (ThreadX v2.0)

### Vision Générale
**Système d'agents autonomes pour optimisation stratégies de trading.**

Principe: Combiner puissance backtest GPU + intelligence LLM locale pour auto-amélioration itérative.

### Composants Existants (ThreadX v2.0)

| Module | Rôle | Performance |
|--------|------|-------------|
| `backtest/engine.py` | BacktestEngine GPU (CuPy) | 715 tests/sec |
| `backtest/performance.py` | Métriques (Sharpe, Sortino, etc) | GPU-accelerated |
| `optimization/engine.py` | SweepRunner multi-thread | ~50 workers |
| `llm/client.py` | LLMClient (Ollama) | Async + retries |
| `strategy/registry.py` | 6 stratégies prêtes | Params tunables |

### Agents LLM Spécialisés

#### 🕵️ Agent Analyste (Analyst)
**Rôle:** Analyste quantitatif virtuel

**Input:**
- Résultats backtest/sweep (RunResult)
- Métriques performance (Sharpe, drawdown, etc)

**Output:**
```json
{
  "score_global": 6.5,
  "forces": ["Sharpe 1.2", "Win rate 65%"],
  "faiblesses": ["Drawdown 18%", "Trades insuffisants"],
  "hypotheses": ["Période courte trop volatile"],
  "anomalies": ["Config #23 suspecte (overfitting?)"]
}
```

**Modèle:** deepseek-r1 70B (température basse, rigueur)

**Location:** `src/threadx/llm/agents/analyst.py`

#### 💡 Agent Stratège (Strategist)
**Rôle:** Générateur solutions créatives

**Input:**
- Analyse Analyst
- Params actuels stratégie
- Historique itérations (mémoire)
- Contraintes registry (min/max)

**Output:**
```json
{
  "propositions": [
    {
      "id": 1,
      "type": "ajustement_params",
      "modifications": {"fast_period": 14, "stop_loss_pct": 1.5},
      "rationale": "Augmenter fast_period réduit faux signaux",
      "impact_estime": "Drawdown -5%, Sharpe +0.3"
    },
    {...}
  ]
}
```

**Modèle:** gpt-oss 20B (température élevée, créativité)

**Location:** `src/threadx/llm/agents/strategist.py`

#### 🔍 Agent Critique (Critic)
**Rôle:** Validateur/filtre propositions

**Input:**
- Propositions Strategist
- Analyse Analyst
- Heuristiques trading

**Output:**
```json
{
  "propositions_validees": [1, 3],
  "propositions_rejetees": [
    {"id": 2, "raison": "Métriques irréalistes (overfitting)"}
  ],
  "scores_confiance": [0.85, 0.72, 0.45]
}
```

**Modèle:** deepseek-r1 70B (température basse, rigueur)

**Location:** `src/threadx/llm/agents/critic.py`

### Orchestrateur - Boucle Optimisation Autonome

**Location:** `src/threadx/llm/orchestrator.py`

**Workflow (7 étapes):**

```
1. Backtest initial
   └─ run_backtest_gpu() → RunResult → Score qualité

2. Analyse Analyst
   └─ analyze_results() → Diagnostic JSON

3. Génération propositions
   └─ Strategist.generate_proposals(N=3) → Liste configs

4. Validation Critic
   └─ Critic.validate_proposals() → Filtrage

5. Backtests parallèles
   └─ SweepRunner multi-GPU → Scores propositions

6. Sélection meilleure
   └─ Compare scores → Mise à jour params

7. Mise à jour mémoire
   └─ OptimizationMemory (5 dernières itérations)

LOOP jusqu'à:
  - Convergence (3 cycles sans amélioration)
  - Score cible atteint (ex: Sharpe > 2.0)
  - Max itérations
```

### Mémoire Système

**OptimizationMemory:**
```python
class OptimizationMemory:
    def __init__(self, max_history: int = 5):
        self.iterations: list[dict] = []
    
    def add_iteration(self, params, score, analysis):
        """Journalise config + résultats."""
    
    def get_recent_configs(self) -> list[dict]:
        """Évite reproposer configs testées."""
```

**Location:** `src/threadx/llm/memory.py`

### Extensions Avancées

#### 🤝 Système Débat Multi-Agents
**Concept:** Dialogue structuré multi-tours avant décision

```
Round 1: Analyst présente faits
Round 2: Strategist propose plan
Round 3: Critic contre-argumente
Round N: Consensus via vote pondéré
```

**Location:** `src/threadx/llm/debate_system.py`

**Bénéfice:** +30% fiabilité décisions complexes

#### 🔄 Optimisation Adaptative
**Concept:** Sweep guidé par LLM (vs grid search aveugle)

```
1. Sweep sparse (25% espace params)
2. Analyst identifie zones prometteuses
3. Sweep dense sur zones ciblées
4. Itérer exploration ↔ raffinement
```

**Bénéfice:** -70% backtests nécessaires pour optimum

**Location:** `src/threadx/optimization/adaptive_sweep.py`

### Principes Directeurs

1. **Réutilisation:** Moteur existant (pas recoder backtest)
2. **LLM = Analyse uniquement:** Pas génération code Python risqué
3. **Autonomie locale:** Ollama (rapidité + confidentialité)
4. **Parallélisation:** Multi-GPU + appels LLM concurrents
5. **Mémoire:** Historique pour éviter répétitions

### Workflow Intégration

**AVANT de coder agents:**
- [ ] Consulter DIRECTIVES_DEV.md
- [ ] Code dans `src/threadx/llm/agents/`
- [ ] Hériter de BaseAgent (logs, retry)
- [ ] Tests unitaires (coverage >80%)
- [ ] Type hints + docstrings complètes

**Exemple intégration:**
```python
# src/threadx/llm/agents/analyst.py
from threadx.llm.client import LLMClient
from threadx.llm.agents.base import BaseAgent

class Analyst(BaseAgent):
    def __init__(self, model: str = "deepseek-r1:70b"):
        super().__init__(model, temperature=0.2)
    
    def analyze_sweep_results(
        self, 
        results: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyse résultats sweep optimisation.
        
        Args:
            results: DataFrame avec params + métriques
            
        Returns:
            {
                "score_global": float,
                "top_configs": list[dict],
                "patterns": list[str],
                "anomalies": list[str]
            }
        """
        # Implementation avec LLMClient
        pass
```

### Performance Estimée

| Métrique | Sans Agents | Avec Agents |
|----------|-------------|-------------|
| Temps convergence | 50-100 sweeps | 15-30 sweeps |
| Qualité optimum | Sharpe ~1.5 | Sharpe ~2.0+ |
| Overfitting risk | 40% | 15% (Critic) |
| Backtests nécessaires | 5000+ | 1500 (adaptif) |

### Implémentation Actuelle (v2.0)

**✅ COMPOSANTS OPÉRATIONNELS:**

| Fichier | Lignes | Status | Description |
|---------|--------|--------|-------------|
| `llm/orchestrator.py` | 650+ | ✅ PROD | Boucle autonome 7 étapes, convergence auto |
| `llm/agents/base_agent.py` | 248 | ✅ PROD | Classe abstraite (retry, timeout, logs) |
| `llm/agents/analyst.py` | 293 | ✅ PROD | Analyse quantitative backtests |
| `llm/agents/strategist.py` | ~300 | ✅ PROD | Génération propositions créatives |
| `llm/agents/critic.py` | ~250 | ✅ PROD | Validation overfitting/risques |
| `llm/memory.py` | 257 | ✅ PROD | Historique itérations (évite redondances) |
| `llm/adapters.py` | 380+ | ✅ PROD | Connecteurs RunResult ↔ JSON LLM |
| `llm/prompts.py` | 700+ | ✅ PROD | Templates structurés agents + orchestration |
| `tools/poc_orchestrator.py` | 290 | ✅ POC | Script test workflow autonome complet |

**📋 USAGE POC:**
```bash
# Terminal
cd /workspaces/ThreadX_big
python tools/poc_orchestrator.py

# Prérequis:
# - Ollama: deepseek-r1:70b, gpt-oss:20b
# - GPU disponible
# - Durée: 10-15 min (5 itérations)
```

**🎯 WORKFLOW AUTONOME (7 ÉTAPES):**
```
1. Backtest initial → RunResult (BacktestEngine GPU)
2. Analyse → Analyst (diagnostic JSON structuré)
3. Propositions → Strategist (3 configs candidates)
4. Validation → Critic (filtrage overfitting)
5. Backtests parallèles → SweepRunner (scores)
6. Sélection → Meilleure config
7. Mémoire → OptimizationMemory (tracking)

LOOP jusqu'à:
  - Convergence (2-3 cycles stagnation)
  - Target Sharpe atteint
  - Max iterations
```

**⚙️ FONCTIONNALITÉS PRÊTES:**
- Coordination 3 agents LLM
- Convergence automatique
- Mémoire évite repropositions
- Export résultats JSON
- Graphique convergence (matplotlib)
- Frictions réalistes intégrées
- Logs structurés

**🚀 PROCHAINES ÉTAPES (Extensions):**
- [ ] Débat multi-agents (debate_system.py)
- [ ] Sweeps adaptatifs (adaptive_sweep.py)
- [ ] Interface Streamlit orchestrateur
- [ ] Tests unitaires orchestrateur
- [ ] Documentation complète API

---

## ✔️ CHECKLIST AVANT RÉPONSE

Pour chaque modification:
- [ ] Consulté DIRECTIVES_DEV.md
- [ ] Code dans module existant (pas nouveau fichier)
- [ ] Type hints complets
- [ ] Docstrings (Google/NumPy style)
- [ ] Tests (coverage >80%)
- [ ] Logging: logger.info(), pas print()
- [ ] Respecte conventions nommage

---

## 🚫 ERREURS COURANTES

```python
# ❌ MAUVAIS: Nouveau fichier
# src/threadx/backtest/realistic_execution.py

# ✅ BON: Ajouter à engine.py
def execute_realistic_order(...): pass

# ❌ MAUVAIS: Sans type hints
def calculate(prices):
    return prices.mean()

# ✅ BON: Avec types
def calculate(prices: np.ndarray) -> float:
    """Description."""
    return prices.mean()

# ❌ MAUVAIS: Print
print("Processing...")

# ✅ BON: Logger
logger.info("Processing...")
```

---

## 📚 FICHIERS CRITIQUES

| Fichier | Rôle |
|---------|------|
| DIRECTIVES_DEV.md | À jour obligatoire |
| README.md | Documentation principale |
| .llmrc | Instructions LLM |
| requirements.txt | Dépendances |

**Ne pas créer de copies!**

---

## 📞 EN CAS DE DOUTE

**Réponse standard:**

> Selon DIRECTIVES_DEV.md [Section X]:
> - [Règle]
> - Donc je [Action]

**Exemple:**
> Selon DIRECTIVES_DEV.md "Règles Consolidation":
> - Préférer ajouter à fichiers existants
> - Donc j'ajoute execute_realistic_order() à engine.py

---

**Document centralisé pour tous. À jour = meilleur code.**

*Last Updated: 21 novembre 2025*
