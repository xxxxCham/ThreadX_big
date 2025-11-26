<<<<<<< HEAD
# ThreadX v2.0 - Trading Quantitatif Haute Performance

> Plateforme de backtesting et optimisation de stratégies de trading avec support GPU multi-device et analyse LLM

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%2B%20CuPy-green.svg)](https://cupy.dev/)
[![Multi-GPU](https://img.shields.io/badge/Multi--GPU-NCCL-orange.svg)](https://developer.nvidia.com/nccl)

---

## 🎯 Fonctionnalités Principales

### 🚀 **Optimisation GPU Haute Performance**
- **Multi-GPU** : Répartition automatique selon VRAM (ex: RTX 5080 66% + RTX 2060 34%)
- **Cache GPU intelligent** : IndicatorBank avec TTL et checksums
- **ProcessPool** : Parallélisation multi-core avec workers adaptatifs

### 🤖 **Analyse LLM Intégrée**
- **Multi-LLM Optimizer** : Génération automatique de stratégies via Ollama
- **Agents spécialisés** : Analyst (données) + Strategist (propositions)
- **Workflow complet** : Sweep GPU → Analyse LLM → Test propositions

### 📊 **Backtesting Robuste**
- **4 stratégies intégrées** : Bollinger Dual, MA Crossover, EMA Cross, ATR Channel
- **Métriques complètes** : Sharpe, drawdown, win rate, profit factor
- **Données réelles** : Support Binance OHLCV avec cache local

### 🎨 **Interface Streamlit Moderne**
- **4 pages unifiées** : Config, Backtest, LLM Optimizer, Monitor
- **Temps réel** : Monitoring système (CPU, RAM, GPU, disque)
- **Multi-GPU UI** : Détection automatique et configuration dynamique

---

## 📁 Structure du Projet

```
ThreadX_big/
├── src/threadx/           # Code source principal
│   ├── backtest/          # Moteur de backtest
│   ├── data_access/       # Accès données Binance
│   ├── gpu/               # Gestion GPU & Multi-GPU
│   ├── indicators/        # Indicateurs techniques (GPU)
│   ├── llm/               # Agents LLM & Ollama
│   ├── optimization/      # Sweep paramétrique
│   ├── strategy/          # Stratégies de trading
│   └── ui/                # Interface Streamlit
│
├── docs/                  # 📚 Documentation complète
│   ├── guides/            # Guides utilisateur
│   ├── fixes/             # Corrections bugs
│   ├── architecture/      # Architecture système
│   ├── diagnostics/       # Rapports performance
│   └── archives/          # Documentation historique
│
├── cache/                 # Cache indicateurs GPU
├── CSV/                   # Données OHLCV locales
├── notebooks/             # Notebooks Jupyter
└── requirements.txt       # Dépendances Python
```

---

## 🚀 Installation Rapide

### Prérequis

- **Python 3.12+**
- **NVIDIA GPU** avec CUDA 11.8+ (optionnel mais recommandé)
- **Ollama** (optionnel, pour fonctionnalités LLM)

### Installation

```bash
# 1. Cloner le dépôt
git clone <repo-url>
cd ThreadX_big

# 2. Créer environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Configurer GPU (si disponible)
# CuPy sera automatiquement détecté
```

### Lancement Application

```bash
# Lancer l'interface Streamlit
streamlit run src/threadx/streamlit_app.py

# Ou utiliser le raccourci Windows
src\threadx\TreadX.bat
```

**Application accessible sur** : http://localhost:8501

---

## 📚 Documentation

### 📖 Guides Utilisateur

| Document | Description |
|----------|-------------|
| [Guide LLM Optimizer](docs/guides/GUIDE_UTILISATION_LLM_OPTIMIZER.md) | Utilisation complète du Multi-LLM Optimizer |
| [Quick Start LLM](docs/guides/QUICKSTART_LLM.md) | Démarrage rapide optimisation LLM |
| [Fonctions Système](docs/guides/FONCTIONS_SYSTEME.md) | Arrêt & Redémarrage de l'application |
| [Lancer UI MA Crossover](docs/guides/LANCER_UI_MA_CROSSOVER.md) | Test stratégie MA Crossover |

### 🔧 Corrections & Fixes

| Document | Description |
|----------|-------------|
| [Fix Pickle ProcessPool](docs/fixes/FIX_PICKLE_PROCESSPOOL.md) | Résolution erreur pickle Windows |
| [Fix Ollama](docs/fixes/OLLAMA_FIXES.md) | Corrections Ollama Manager |
| [Fix Cache Streamlit](docs/fixes/FIX_CACHE_STREAMLIT.md) | Optimisation cache Streamlit |
| [Fix MA Crossover](docs/fixes/FIX_FINAL_MA_CROSSOVER.md) | Corrections stratégie MA Crossover |

### 🏗️ Architecture

| Document | Description |
|----------|-------------|
| [Architecture Multi-LLM](docs/architecture/ARCHITECTURE_MULTI_LLM.md) | Architecture système LLM |
| [Multi-GPU Status](docs/architecture/MULTI_GPU_STATUS.md) | État et configuration Multi-GPU |
| [POC Multi-LLM](docs/architecture/POC_MULTI_LLM_AGENT.md) | Proof of Concept agents LLM |
| [Intégration MA Crossover](docs/architecture/INTEGRATION_MA_CROSSOVER.md) | Intégration stratégie MA Crossover |

### 📊 Diagnostics & Rapports

| Document | Description |
|----------|-------------|
| [Analyse Performance](docs/diagnostics/ANALYSE_PERFORMANCE_COMPLETE.md) | Analyse complète des performances |
| [Diagnostic Ralentissements](docs/diagnostics/DIAGNOSTIC_RALENTISSEMENTS.md) | Investigation ralentissements |
| [Rapport Optimisations](docs/diagnostics/RAPPORT_OPTIMISATIONS_FINAL.md) | Bilan des optimisations finales |
| [Plan Optimisations P0](docs/diagnostics/PLAN_OPTIMISATIONS_P0.md) | Priorités d'optimisation |

### 📦 Archives

Documentation historique et de référence dans [docs/archives/](docs/archives/)

---

## 🎮 Utilisation Rapide

### 1. **Charger des Données**

Page **Configuration** :
- Sélectionner symbole (ex: BTCUSDC)
- Choisir timeframe (ex: 15m)
- Définir période (ex: 2024-12-01 → 2025-01-31)
- Cliquer **"Charger les données"**

### 2. **Backtest Simple**

Page **Backtest & Optimisation** :
- Choisir stratégie (ex: Bollinger Dual)
- Configurer paramètres
- Lancer backtest
- Analyser résultats (courbe equity, trades, métriques)

### 3. **Optimisation GPU**

Page **Backtest & Optimisation** :
- Activer **"Optimisation paramétrique"**
- Définir plages de paramètres
- Choisir mode : Grid (exhaustif) ou Monte Carlo (échantillonnage)
- Lancer sweep GPU
- Sélectionner meilleure config

### 4. **Optimisation Multi-LLM** 🤖

Page **LLM Optimizer** :
- Configurer sweep initial (plages larges)
- Choisir modèles LLM (Analyst + Strategist)
- Lancer workflow complet :
  1. **Sweep GPU** : Test configs initiales
  2. **Analyst** : Analyse des résultats
  3. **Strategist** : Propositions améliorées
  4. **Test final** : Validation propositions
- Comparer baseline vs propositions LLM

---

## 🖥️ Multi-GPU

### Détection Automatique

Au démarrage, ThreadX détecte automatiquement les GPUs :

```
============================================================
💎 MULTI-GPU DÉTECTÉ : 2 GPUs
============================================================
   GPU 0: NVIDIA GeForce RTX 5080
      └─ 15.9 GB VRAM | CC 12.0
   GPU 1: NVIDIA GeForce RTX 2060 SUPER
      └─ 8.0 GB VRAM | CC 7.5
============================================================
Multi-GPU Manager initialisé: 2 GPU(s), NCCL=activé
💎 Multi-GPU optimal: 5080 (66%) + 2060 (34%)
```

### Configuration

**Sidebar** > **🖥️ GPU & Calcul** > **"Activer Multi-GPU"**

- ✅ **Activé** : Répartition automatique selon VRAM
- ❌ **Désactivé** : GPU principal uniquement

---

## ⚙️ Configuration Avancée

### Variables d'Environnement

```bash
# Feeder aggr (taille pipeline ProcessPool)
THREADX_FEEDER_AGGR=16  # Défaut: 10

# Désactiver logs Streamlit
THREADX_SILENCE_LOGS=1

# GPUs visibles (ordre PCI)
CUDA_VISIBLE_DEVICES=0,1
CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### Fichiers Configuration

- **`paths.toml`** : Chemins données/cache
- **`pyproject.toml`** : Config Python/ruff/pytest
- **`cspell.yml`** : Dictionnaire spell check

---

## 🔧 Actions Système

**Sidebar** > **🔧 Actions Système**

### 🔄 **Redémarrer**
- Réinitialise tout (cache, GPU, session)
- Équivalent à un premier démarrage
- Vide VRAM + RAM + cache fichiers

### 🛑 **Arrêter**
- Nettoie mémoire et ferme l'application
- Libère ressources GPU/RAM proprement

---

## 🐛 Dépannage

### Problème : Erreur CUDA out of memory

**Solution** :
1. **Redémarrer** l'application (Sidebar > 🔄 Redémarrer)
2. Réduire `max_workers` dans Config Performance
3. Désactiver Multi-GPU si un GPU est utilisé ailleurs

### Problème : Ollama ne répond pas

**Solution** :
1. Sidebar > **🧹 Nettoyage Complet**
2. Si persiste : **🔄 Redémarrer**
3. Vérifier Ollama : `ollama list` dans terminal

### Problème : Cache incohérent

**Solution** :
1. **🔄 Redémarrer** (vide cache fichiers)
2. Si persiste : Supprimer manuellement `cache/indicators/`

### Problème : Multi-GPU non détecté

**Solution** :
1. Vérifier GPUs : `nvidia-smi` dans terminal
2. Vérifier CUDA_VISIBLE_DEVICES
3. Relancer Streamlit

---

## 🏆 Performances

### Benchmarks (RTX 5080 + RTX 2060)

| Tâche | Sans GPU | GPU Single | Multi-GPU | Gain |
|-------|----------|-----------|-----------|------|
| **Calcul 100 indicateurs** | 45s | 2.3s | 1.5s | **30x** |
| **Sweep 1000 configs** | 18min | 48s | 32s | **34x** |
| **Optimisation LLM (full)** | N/A | 12min | 8min | **1.5x** |

**Config** : 30 workers, feeder_aggr=16, ProcessPool

---

## 🤝 Contribution

### Workflow Git

```bash
# Créer branche feature
git checkout -b feature/ma-nouvelle-feature

# Commit avec format
git commit -m "feat(module): Description courte

- Détail 1
- Détail 2

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# Push et créer PR
git push -u origin feature/ma-nouvelle-feature
```

### Conventions

- **Typage** : Type hints obligatoires
- **Docstrings** : Format Google
- **Logs** : Via `threadx.utils.log.get_logger()`
- **Tests** : pytest (à venir)

---

## 📜 Licence

Propriétaire - © 2025 ThreadX Framework

---

## 📞 Support

- **Documentation** : [docs/](docs/)
- **Issues** : [GitHub Issues](https://github.com/your-org/threadx/issues)
- **Email** : support@threadx.dev (fictif)

---

## 🎉 Remerciements

- **CuPy** : Calculs GPU haute performance
- **Streamlit** : Framework UI moderne
- **Ollama** : Runtime LLM local
- **NCCL** : Synchronisation multi-GPU

---

**ThreadX v2.0** - Trading Quantitatif Nouvelle Génération 🚀
=======
# ThreadX - Framework Trading Algorithmique

**Framework Python professionnel pour backtesting, optimisation et déploiement de stratégies trading.**

Version: **2025.11.21**  
Python: **3.11+**  
License: Propriétaire

---

## 🚨 IMPORTANT

**👉 Lire obligatoirement:** [`DIRECTIVES_DEV.md`](DIRECTIVES_DEV.md)

Ce fichier centralise **TOUTES** les instructions:
- ✅ Règles consolidation code
- ✅ Architecture générale
- ✅ Conventions nommage
- ✅ Stack technologique
- ✅ Info Netdata MCP Bridge
- ✅ Checklist qualité code

---

## 🎯 QUICKSTART

### Installation
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

### Premier Backtest
```python
from src.threadx.strategy.ma_crossover import MACrossover, MACrossoverParams
import pandas as pd

strategy = MACrossover()
equity, stats = strategy.backtest(
    df=df_1min,
    params=MACrossoverParams(fast_period=10, slow_period=30),
    initial_capital=10000.0
)
print(f"Return: {stats['total_return']:.2f}%")
```

---

## 📊 FRICTIONS RÉALISTES

Backtests SANS frictions = +200% trop optimistes!

**Solution:** `RealisticExecutor` dans `src/threadx/backtest/engine.py`

```python
executor = RealisticExecutor(timeframe="1m", symbol="BTCUSDT")
result = executor.execute_order(
    side="BUY",
    intended_price=50000.0,
    quantity=0.5,
    current_volatility=0.015
)
```

---

## 🌐 NETDATA MCP BRIDGE

Outil monitoring en Go (SÉPARÉ du trading):

```bash
cd tools/netdata-bridge
./build.sh
./nd-mcp ws://localhost:19999/mcp
```

---

## 📚 DOCUMENTATION

- **DIRECTIVES_DEV.md** ← À LIRE EN PRIORITÉ
- **src/threadx/** → Docstrings dans le code
- **tools/** → Scripts développement
- **tests/** → Exemples usage

---

**Avant de coder, lire** [`DIRECTIVES_DEV.md`](DIRECTIVES_DEV.md)
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
