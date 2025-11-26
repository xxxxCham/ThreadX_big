# 🤖 Multi-LLM Optimizer - Guide d'Exécution

Système d'optimisation automatique de stratégies de trading utilisant 2 agents LLM collaboratifs.

---

## 📋 Vue d'Ensemble

Ce POC démontre un système multi-LLM capable d'analyser des résultats de backtests et de proposer automatiquement des améliorations de paramètres.

**Architecture**:
- **Analyst Agent** (deepseek-r1:70b): Analyse quantitative, identification de patterns
- **Strategist Agent** (gpt-oss:20b): Génération créative de propositions
- **BacktestEngine** (GPU): Validation automatique des propositions

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
