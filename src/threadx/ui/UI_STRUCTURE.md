# 🏗️ Structure UI - ThreadX v2.0

## 📊 Vue d'ensemble

L'interface Streamlit v2.0 est organisée en **2 pages principales** et **6 modules de support**.

```
src/threadx/ui/
├── 📄 Page Modules (Actifs)
│   ├── page_config_strategy.py      ✅ Configuration & Stratégie
│   └── page_backtest_optimization.py ✅ Backtest & Optimisation
│
├── 🔧 Support Modules
│   ├── strategy_registry.py          (Registre de stratégies)
│   ├── backtest_bridge.py            (Interface avec le backtest)
│   ├── fast_sweep.py                 (Optimisations sweep rapides)
│   └── system_monitor.py             (Monitoring système)
│
└── 📦 Archive Legacy
    └── _legacy_v1/                   (Pages fusionnées de v1)
        ├── page_selection_token.py
        ├── page_strategy_indicators.py
        ├── page_backtest_results.py
        └── README.md
```

## 📱 Pages Actives

### 1. **page_config_strategy.py**
**Rôle**: Configuration & Stratégie (Page 1)

```python
# Fusion de deux anciennes pages v1:
# - page_selection_token.py → Sélection des données
# - page_strategy_indicators.py → Configuration de stratégie

# Fonctionnalités:
- Sélection du symbole (BTC, préréglé)
- Sélection du timeframe (15m, préréglé)
- Plage de dates (Dec 1 2024 - Jan 31 2025, préréglée)
- Chargement et validation des données OHLCV
- Sélection de la stratégie (Bollinger_Breakout)
- Configuration des paramètres de stratégie
- Aperçu des données chargées
```

### 2. **page_backtest_optimization.py**
**Rôle**: Backtest & Optimisation (Page 2)

```python
# Fusion de deux anciennes fonctionnalités v1:
# - page_backtest_results.py → Affichage des résultats
# - Optimisation sweep + Monte-Carlo

# Fonctionnalités:
- Onglet 1: Sweep
  * Configuration des plages de paramètres
  * Sliders de sensibilité (granularité)
  * Calcul du nombre de combinaisons
  * Validation (≤100K optimal, ≤3M max)
  * Barre de progression avec vitesse
  * Affichage des résultats et export CSV

- Onglet 2: Monte-Carlo
  * Plages de paramètres aléatoires
  * Nombre de scénarios configurables
  * Seed pour reproductibilité
  * Barre de progression
  * Résultats tabulés
  * Export CSV
```

## 🔧 Modules de Support

### `strategy_registry.py`
Registre centralisé des stratégies et leurs paramètres
```python
- Bollinger_Breakout (stratégie active)
- EMA_Cross (disponible)
- ATR_Channel (disponible)

- Paramètres non-tunable: entry_logic, trailing_stop, leverage
- Paramètres tunable: 10 paramètres optimisables
```

### `backtest_bridge.py`
Interface pour exécuter les backtests
```python
- run_backtest() - Backtest simple
- run_backtest_gpu() - Backtest GPU
- BacktestResult - Classe de résultats
```

### `fast_sweep.py`
Optimisations pour les sweeps rapides
```python
- Caching des indicateurs
- Exécution parallèle
- GPU acceleration
```

### `system_monitor.py`
Monitoring des ressources système
```python
- CPU/GPU usage
- Memory tracking
- Performance metrics
```

## 📦 Archive Legacy (_legacy_v1/)

Restes de l'ancienne architecture v1 conservés pour référence:
- **page_selection_token.py** (169 lignes) → Code fusionné dans page_config_strategy.py
- **page_strategy_indicators.py** (202 lignes) → Code fusionné dans page_config_strategy.py
- **page_backtest_results.py** (451 lignes) → Code fusionné dans page_backtest_optimization.py

**Total code archivé: 822 lignes**

## 🎯 Architecture v2.0 - Avantages

✅ **Consolidation**: 5 pages → 2 pages
✅ **Clarté**: Interface simplifiée et intuitive
✅ **Performance**: Meilleur partage des ressources
✅ **Maintenance**: Moins de code à maintenir
✅ **Testabilité**: Structure modulaire

## 📊 Comparaison v1 → v2.0

| Métrique | v1 | v2.0 |
|----------|-----|------|
| Pages UI | 5 | 2 |
| Fichiers orphelins | - | 0 |
| Code actif | ~2000 lignes | ~1800 lignes |
| Complexité UI | Élevée | Optimale |
| Facilité navigation | Moyenne | Excellent |

## 🚀 Points d'Intégration

### Avec streamlit_app.py
```python
from threadx.ui.page_config_strategy import main as config_page_main
from threadx.ui.page_backtest_optimization import main as backtest_page_main

PAGE_RENDERERS = {
    "config": config_page_main,      # Page 1
    "backtest": backtest_page_main   # Page 2
}
```

## 📝 Note

Cette structure est définitive pour v2.0. L'archive _legacy_v1/ peut être supprimée une fois que v2.0 est stable en production.

---
**Dernière mise à jour**: 2025-10-31
**Version**: 2.0.0
