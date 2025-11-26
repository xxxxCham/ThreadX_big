# 📚 Documentation ThreadX v2.0

> Index complet de la documentation technique et utilisateur

---

## 🗂️ Organisation

```
docs/
├── guides/           📖 Guides utilisateur
├── fixes/            🔧 Corrections bugs
├── architecture/     🏗️ Architecture système
├── diagnostics/      📊 Rapports performance
└── archives/         📦 Documentation historique
```

---

## 📖 Guides Utilisateur

### Multi-LLM Optimizer

- **[Guide Complet LLM Optimizer](guides/GUIDE_UTILISATION_LLM_OPTIMIZER.md)**
  - Workflow complet Multi-LLM
  - Configuration Analyst + Strategist
  - Interprétation résultats
  - Dépannage

- **[Quick Start LLM](guides/QUICKSTART_LLM.md)**
  - Démarrage rapide (5 min)
  - Premier run LLM
  - Configuration minimale

### Fonctionnalités Système

- **[Fonctions Système](guides/FONCTIONS_SYSTEME.md)**
  - 🔄 Redémarrage application
  - 🛑 Arrêt propre
  - Nettoyage mémoire/cache

- **[Lancer UI MA Crossover](guides/LANCER_UI_MA_CROSSOVER.md)**
  - Test stratégie MA Crossover
  - Validation backtest

---

## 🔧 Corrections & Fixes

### Bugs Majeurs Résolus

- **[Fix Pickle ProcessPool](fixes/FIX_PICKLE_PROCESSPOOL.md)**
  - Erreur : `Can't pickle <function _init_process_globals>`
  - Solution : Désactivation `initializer` Windows
  - Détection OS automatique

- **[Fix Ollama Manager](fixes/OLLAMA_FIXES.md)**
  - Erreur : `UnicodeDecodeError` + `NoneType`
  - Solution : Encoding UTF-8 + checks None
  - Désactivation reset auto

- **[Fix Cache Streamlit](fixes/FIX_CACHE_STREAMLIT.md)**
  - Optimisation cache Streamlit
  - Éviter rechargements inutiles

### Stratégies

- **[Fix Final MA Crossover](fixes/FIX_FINAL_MA_CROSSOVER.md)**
  - Corrections stratégie MA Crossover
  - Validation complète

- **[Fix Optimisation MA Crossover](fixes/FIX_MA_CROSSOVER_OPTIMIZATION.md)**
  - Optimisation paramètres
  - Tests performance

---

## 🏗️ Architecture

### Architecture Système

- **[Architecture Multi-LLM](architecture/ARCHITECTURE_MULTI_LLM.md)**
  - Agents Analyst + Strategist
  - Workflow complet
  - Intégration Ollama

- **[POC Multi-LLM Agent](architecture/POC_MULTI_LLM_AGENT.md)**
  - Proof of Concept initial
  - Tests agents LLM
  - Validation architecture

### Configuration GPU

- **[Multi-GPU Status](architecture/MULTI_GPU_STATUS.md)**
  - État détection Multi-GPU
  - Configuration RTX 5080 + RTX 2060
  - Balance automatique 66%/34%
  - NCCL activé

### Intégrations

- **[Intégration MA Crossover](architecture/INTEGRATION_MA_CROSSOVER.md)**
  - Ajout stratégie MA Crossover
  - Intégration UI
  - Tests validation

- **[Test Stratégie MA Crossover](architecture/STRATEGIE_MA_CROSSOVER_TEST.md)**
  - Plan de test complet
  - Résultats validation

---

## 📊 Diagnostics & Performance

### Analyses Performance

- **[Analyse Performance Complète](diagnostics/ANALYSE_PERFORMANCE_COMPLETE.md)**
  - Benchmark CPU vs GPU
  - Analyse Multi-GPU
  - Bottlenecks identifiés

- **[Diagnostic Ralentissements](diagnostics/DIAGNOSTIC_RALENTISSEMENTS.md)**
  - Investigation ralentissements UI
  - Solutions appliquées

- **[Diagnostic Chute Perfs](diagnostics/DIAGNOSTIC_CHUTE_PERFS.md)**
  - Analyse dégradation performance
  - Root cause analysis

### Rapports Optimisation

- **[Rapport Optimisations Final](diagnostics/RAPPORT_OPTIMISATIONS_FINAL.md)**
  - Bilan toutes optimisations
  - Gains performance mesurés
  - Recommandations futures

- **[Rapport Optimisation Workers](diagnostics/RAPPORT_OPTIMISATION_WORKERS.md)**
  - Optimisation ProcessPool
  - Workers adaptatifs
  - Feeder aggr

### Plans Action

- **[Plan Optimisations P0](diagnostics/PLAN_OPTIMISATIONS_P0.md)**
  - Priorités optimisation
  - Roadmap P0/P1/P2

---

## 📦 Archives

Documentation historique et référence complète.

- **[Complete Codebase Survey](archives/COMPLETE_CODEBASE_SURVEY.md)**
  - État complet du code
  - Inventaire modules
  - Architecture globale

- **[Index Documentation LLM](archives/INDEX_DOCUMENTATION_LLM.md)**
  - Index historique docs LLM
  - Références croisées

- **[README Multi-LLM](archives/README_MULTI_LLM.md)**
  - Documentation initiale Multi-LLM
  - Premiers tests

- **[Résumé Final Intégration LLM](archives/RESUME_FINAL_INTEGRATION_LLM.md)**
  - Synthèse intégration LLM
  - État final phase 1

- **[Synthèse Visuelle LLM](archives/SYNTHESE_VISUELLE_LLM.md)**
  - Diagrammes workflow
  - Schémas architecture

---

## 🔍 Recherche Rapide

### Par Sujet

| Sujet | Documents |
|-------|-----------|
| **Multi-GPU** | [Multi-GPU Status](architecture/MULTI_GPU_STATUS.md), [Analyse Perf](diagnostics/ANALYSE_PERFORMANCE_COMPLETE.md) |
| **LLM** | [Guide LLM](guides/GUIDE_UTILISATION_LLM_OPTIMIZER.md), [Architecture Multi-LLM](architecture/ARCHITECTURE_MULTI_LLM.md) |
| **Bugs** | [Fix Pickle](fixes/FIX_PICKLE_PROCESSPOOL.md), [Fix Ollama](fixes/OLLAMA_FIXES.md) |
| **Performance** | [Analyse Complète](diagnostics/ANALYSE_PERFORMANCE_COMPLETE.md), [Rapport Final](diagnostics/RAPPORT_OPTIMISATIONS_FINAL.md) |
| **Stratégies** | [Fix MA Crossover](fixes/FIX_FINAL_MA_CROSSOVER.md), [Test MA Crossover](architecture/STRATEGIE_MA_CROSSOVER_TEST.md) |

### Par Type

| Type | Dossier |
|------|---------|
| **📖 Tutoriels** | [guides/](guides/) |
| **🔧 Dépannage** | [fixes/](fixes/) |
| **🏗️ Conception** | [architecture/](architecture/) |
| **📊 Benchmarks** | [diagnostics/](diagnostics/) |
| **📦 Historique** | [archives/](archives/) |

---

## 🆕 Dernières Mises à Jour

### 2025-11-24

- ✅ **[Fonctions Système](guides/FONCTIONS_SYSTEME.md)** - Ajout Redémarrage/Arrêt
- ✅ **[Fix Pickle ProcessPool](fixes/FIX_PICKLE_PROCESSPOOL.md)** - Résolution Windows
- ✅ **[Fix Ollama](fixes/OLLAMA_FIXES.md)** - Corrections encoding + reset
- ✅ **[Multi-GPU Status](architecture/MULTI_GPU_STATUS.md)** - Validation 2 GPUs

### 2025-11-23

- ✅ **[Rapport Optimisations Final](diagnostics/RAPPORT_OPTIMISATIONS_FINAL.md)** - Bilan complet
- ✅ **[Guide LLM Optimizer](guides/GUIDE_UTILISATION_LLM_OPTIMIZER.md)** - Version finale

---

## 💡 Comment Utiliser Cette Documentation

### Nouveau Utilisateur

1. Lire [README principal](../README.md)
2. Suivre [Quick Start LLM](guides/QUICKSTART_LLM.md)
3. Consulter [Guide LLM Optimizer](guides/GUIDE_UTILISATION_LLM_OPTIMIZER.md)

### Développeur

1. Consulter [Architecture Multi-LLM](architecture/ARCHITECTURE_MULTI_LLM.md)
2. Lire [Complete Codebase Survey](archives/COMPLETE_CODEBASE_SURVEY.md)
3. Examiner [Rapports Performance](diagnostics/)

### Dépannage

1. Identifier le problème
2. Chercher dans [fixes/](fixes/)
3. Si non résolu : [diagnostics/](diagnostics/)
4. Consulter [Fonctions Système](guides/FONCTIONS_SYSTEME.md) pour reset

---

## 📞 Besoin d'Aide ?

**Documentation manquante ou obsolète ?**

1. Consulter [README principal](../README.md)
2. Vérifier [archives/](archives/) (docs historiques)
3. Créer une issue GitHub

---

**Documentation ThreadX v2.0** - Dernière mise à jour : 2025-11-24
