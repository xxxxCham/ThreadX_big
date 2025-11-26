# 📘 DICTIONNAIRE DU CODE - ThreadX

> **Document de référence** : Architecture et organisation du projet ThreadX  
> **Date de création** : 22/11/2025  
> **Objectif** : Guide complet de tous les dossiers, fichiers et leur utilité

---

## 📂 RACINE (`/threadx`)

### Fichiers principaux

| Fichier | Utilité |
|---------|---------|
| `__init__.py` | Point d'entrée du package Python ThreadX |
| `streamlit_app.py` | **APPLICATION PRINCIPALE** - Interface Streamlit avec navigation multi-pages |
| `config.py` | Configuration globale du projet (chemins, constantes) |
| `data_access.py` | Accès centralisé aux données de marché (crypto) |
| `requirements.txt` | Liste des dépendances Python avec versions épinglées |
| `LANCER_STREAMLIT.bat` | Script de lancement de l'application (active venv + lance Streamlit) |
| `LANCER_STREAMLIT_DEBUG.bat` | Version debug avec logs activés |
| `STRUCTURE_PROJET.md` | Documentation de l'architecture du projet |
| `threadx.code-workspace` | Configuration du workspace VS Code |

---

## 🎯 DOSSIER `/backtest` - Moteur de Backtest

**Rôle** : Simulation de stratégies de trading sur données historiques

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module backtest |
| `engine.py` | **MOTEUR PRINCIPAL** - Exécute les backtests, gère positions/ordres |
| `performance.py` | Calcul des métriques de performance (Sharpe, drawdown, win rate) |
| `validation.py` | Validation des résultats et détection d'anomalies |

**Interactions** :
- Utilisé par `/optimization` pour tester les configs
- Appelé depuis `/ui/page_backtest_optimization.py`

---

## ⚙️ DOSSIER `/configuration` - Gestion de Configuration

**Rôle** : Chargement, validation et gestion des configs stratégies

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `settings.py` | Définition des paramètres de stratégie (dataclasses) |
| `loaders.py` | Chargement depuis JSON/TOML |
| `errors.py` | Exceptions personnalisées pour la config |

**Format** : Utilise Pydantic pour validation stricte des types

---

## 💾 DOSSIER `/data` - Accès aux Données

**Rôle** : Stockage et normalisation des données de marché

| Fichier/Dossier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `normalize.py` | Normalisation des données (OHLCV) |
| `schemas.py` | Schémas Pydantic pour validation |
| `crypto_data_parquet/` | **15 000+ fichiers Parquet** - Données crypto (BTC, ETH, etc.) par timeframe |
| `arbo_data_folder.txt` | Documentation de l'arborescence des données |

**Format Parquet** : Optimisé pour lecture rapide avec Pandas/Polars

---

## 🧪 DOSSIER `/dataset` - Validation de Datasets

| Fichier | Description |
|---------|-------------|
| `validate.py` | Validation de la qualité des données (trous, outliers) |

---

## 🖥️ DOSSIER `/gpu` - **GESTION GPU** ⚡

**Rôle** : Configuration multi-GPU, allocation, monitoring CUDA

| Fichier | Description | Importance |
|---------|-------------|------------|
| `__init__.py` | Exports du module | ⭐ |
| `device_manager.py` | **GESTIONNAIRE PRINCIPAL** - Sélection et allocation des GPUs | ⭐⭐⭐ |
| `multi_gpu.py` | Répartition de charge multi-GPU (CUDA) | ⭐⭐⭐ |
| `profile_persistence.py` | Sauvegarde des profils GPU | ⭐⭐ |
| `vector_checks.py` | Vérification vectorielle GPU (performance) | ⭐⭐ |

**🔧 FICHIERS CRITIQUES POUR VOTRE CONFIGURATION RTX 5080/2060** :
- `device_manager.py` : Définit quelle carte utiliser
- `multi_gpu.py` : Gère la répartition entre 5080 (principale) et 2060 (secondaire)

---

## 📊 DOSSIER `/indicators` - Indicateurs Techniques

**Rôle** : Calcul d'indicateurs TA (Bollinger, ATR, etc.)

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `bank.py` | Bibliothèque centralisée d'indicateurs |
| `bollinger.py` | Bandes de Bollinger |
| `xatr.py` | Average True Range (ATR) |
| `gpu_integration.py` | **Accélération GPU** pour calculs vectoriels |
| `indicators_cache/` | Cache des indicateurs pré-calculés (optimisation) |

**Performance** : Les calculs lourds utilisent CUDA via `gpu_integration.py`

---

## 🤖 DOSSIER `/llm` - Intégration IA (LLM)

**Rôle** : Agents IA pour analyse et génération de stratégies

| Fichier/Dossier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `client.py` | Client Ollama (modèles LLM locaux) |
| `prompts.py` | Templates de prompts pour les agents |
| `interpreters.py` | Interprétation des réponses LLM en configs |
| `/agents/` | **Agents IA spécialisés** |
| `agents/base_agent.py` | Classe de base pour agents |
| `agents/analyst.py` | Agent d'analyse de marché |
| `agents/strategist.py` | Agent de création de stratégies |

**Note** : Communique avec modèles Ollama (Llama, Mistral, etc.)

---

## 🔍 DOSSIER `/optimization` - Optimisation de Paramètres

**Rôle** : Recherche automatique des meilleurs paramètres (Grid Search, Monte Carlo)

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `engine.py` | Moteur d'optimisation principal |
| `multi_sweep.py` | Balayage multi-paramètres |
| `parallel_sweep_manager.py` | **Parallélisation multi-GPU** pour sweeps |
| `pruning.py` | Élagage de résultats (garde top N) |
| `reporting.py` | Génération de rapports d'optimisation |
| `scenarios.py` | Gestion de scénarios de marché |
| `ui.py` | Helpers UI pour affichage des résultats |

### Sous-dossiers

| Dossier | Description |
|---------|-------------|
| `/presets/` | Présets de ranges de paramètres |
| `/templates/` | Templates d'optimiseurs (Grid, Monte Carlo) |

**🔧 GPU intensif** : `parallel_sweep_manager.py` répartit les calculs entre GPUs

---

## 📈 DOSSIER `/profiling` - Analyse de Performance

| Fichier | Description |
|---------|-------------|
| `performance_analyzer.py` | Profilage CPU/GPU, détection de goulots |

---

## 🎲 DOSSIER `/strategy` - Stratégies de Trading

**Rôle** : Implémentation des stratégies (logique long/short)

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `model.py` | Classe de base `StrategyBase` |
| `bb_atr.py` | Stratégie Bollinger + ATR |
| `bollinger_dual.py` | Stratégie Bollinger dual-band |
| `ma_crossover.py` | Stratégie Moving Average Crossover |
| `amplitude_hunter.py` | Stratégie chasseur de volatilité |
| `_archive/gpu_examples.py` | Exemples GPU (obsolète mais référence) |

**Pattern** : Toutes héritent de `StrategyBase` (see `model.py`)

---

## 🧪 DOSSIER `/testing` - Tests Unitaires

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `mocks.py` | Mocks pour tests (données, GPUs, etc.) |

---

## 🎨 DOSSIER `/ui` - Interface Streamlit

**Rôle** : Pages de l'application web

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `page_config_strategy.py` | Page de configuration de stratégie |
| `page_backtest_optimization.py` | **PAGE PRINCIPALE** - Backtest + Optimisation + Monte Carlo |
| `page_llm_optimizer.py` | Page d'optimisation via LLM |
| `backtest_bridge.py` | Pont entre UI et moteur backtest |
| `fast_sweep.py` | Optimisation rapide (UI simplifiée) |
| `strategy_registry.py` | Registre centralisé des stratégies |
| `styles.py` | Styles CSS custom pour Streamlit |
| `system_monitor.py` | Monitoring système (CPU/GPU/RAM) en temps réel |

### Sous-dossier `/components/`

| Fichier | Description |
|---------|-------------|
| `charts.py` | Composants de graphiques (Plotly) |
| `config.py` | Composants de configuration (formulaires) |
| `metrics.py` | Affichage de métriques (KPIs) |

---

## 🛠️ DOSSIER `/utils` - Utilitaires

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `common_imports.py` | Imports centralisés (évite répétitions) |
| `log.py` | Configuration du logging (niveaux, formats) |
| `timing.py` | Décorateurs de timing (profiling simple) |
| `cache.py` | Gestion de cache (memoization) |
| `xp.py` | Expérimentation (fonctionnalités en test) |

---

## 📉 DOSSIER `/visualization` - Visualisations

| Fichier | Description |
|---------|-------------|
| `__init__.py` | Exports du module |
| `backtest_charts.py` | Graphiques de backtest (equity curve, drawdown) |

---

## 🔑 FICHIERS CLÉS POUR CONFIGURATION GPU

### **Priorité 1** : Configuration RTX 5080 + RTX 2060

| Fichier | Action requise |
|---------|----------------|
| `gpu/device_manager.py` | Définir RTX 5080 comme `CUDA:0` (device par défaut) |
| `gpu/multi_gpu.py` | Configurer ratio 5080/2060 (ex: 70/30) |
| `llm/client.py` | S'assurer que les modèles LLM utilisent la bonne carte |
| `optimization/parallel_sweep_manager.py` | Répartir les workers entre GPUs NVIDIA |

### **Priorité 2** : Exclusion AMD Radeon

| Fichier | Action requise |
|---------|----------------|
| `gpu/device_manager.py` | Blacklist de la Radeon (GPU #2 AMD) pour calculs |
| `config.py` | Variable d'environnement pour forcer GPUs NVIDIA uniquement |

---

## 📊 STATISTIQUES DU PROJET

- **Total fichiers Python** : ~75 fichiers
- **Lignes de code** : ~15 000 lignes (estimé)
- **Modules principaux** : 12 modules
- **Pages UI** : 3 pages principales
- **Stratégies implémentées** : 5 stratégies

---

## 🚀 FLUX D'EXÉCUTION TYPIQUE

```
1. LANCER_STREAMLIT.bat
   ↓
2. streamlit_app.py (navigation)
   ↓
3. ui/page_backtest_optimization.py
   ↓
4. backtest/engine.py ← strategy/*.py
   ↓
5. indicators/*.py (calculs GPU via gpu/)
   ↓
6. optimization/parallel_sweep_manager.py (multi-GPU)
   ↓
7. visualization/backtest_charts.py (affichage)
```

---

## ⚠️ ZONES GPU-INTENSIVES (Concernées par votre config)

1. **`indicators/gpu_integration.py`** - Calculs vectoriels CUDA
2. **`optimization/parallel_sweep_manager.py`** - Parallélisation sweeps
3. **`llm/client.py`** - Modèles LLM (peuvent utiliser GPU)
4. **`gpu/multi_gpu.py`** - Répartition de charge

---

## 🎯 PROCHAINES ÉTAPES (Config GPU)

1. ✅ **Bug `_render_config_history` corrigé**
2. ⏳ **Installation PyTorch CUDA 12.1 en cours**
3. 🔧 **À faire** :
   - Modifier `gpu/device_manager.py` pour prioriser RTX 5080
   - Configurer `gpu/multi_gpu.py` pour ratio 5080/2060
   - Blacklist AMD Radeon pour calculs (affichage seulement)
   - Tester avec `nvidia-smi` la répartition

---

**📝 Note** : Ce dictionnaire sera mis à jour au fil de l'évolution du projet.
