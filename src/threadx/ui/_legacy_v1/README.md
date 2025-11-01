# ThreadX v1 - Archive des Pages Obsolètes

Ce dossier contient les pages de l'ancienne architecture v1 qui ont été **fusionnées** dans la nouvelle architecture v2.0.

## 📦 Fichiers Archivés

### 1. `page_selection_token.py` (169 lignes)
Ancienne page pour la **sélection des tokens/symboles, timeframes et dates**.
- Fonctionnalités: découverte dynamique des tokens, validation des données
- **Fusionné dans**: `page_config_strategy.py` (Page "Configuration & Stratégie" v2.0)

### 2. `page_strategy_indicators.py` (202 lignes)
Ancienne page pour la **sélection et configuration de la stratégie**.
- Fonctionnalités: sélection de stratégie, configuration des indicateurs et paramètres
- **Fusionné dans**: `page_config_strategy.py` (Page "Configuration & Stratégie" v2.0)

### 3. `page_backtest_results.py` (451 lignes)
Ancienne page pour l'**affichage des résultats de backtest**.
- Fonctionnalités: graphiques de prix, tableaux de trades, métriques
- **Fusionné dans**: `page_backtest_optimization.py` (Page "Backtest & Optimisation" v2.0)

## 🏗️ Architecture v2.0 (Actuelle)

### Pages Actives:
1. **`page_config_strategy.py`** - Configuration & Stratégie
   - Remplace: page_selection_token.py + page_strategy_indicators.py

2. **`page_backtest_optimization.py`** - Backtest & Optimisation
   - Remplace: page_backtest_results.py + optimisation

## 📊 Stats

| Item | v1 | v2.0 |
|------|-----|------|
| **Nombre de pages** | 5 | 2 |
| **Lignes de code orphelin** | 822 | 0 |
| **Complexité UI** | Élevée | Optimisée |

## ⚠️ Utilité de l'Archive

- **Référence**: Si vous devez récupérer du code spécifique de v1
- **Historique**: Git conserve la version complète
- **Nettoyage**: N'impacte pas l'application actuelle

## 🗑️ Suppression

Si vous êtes sûr que v2.0 est complètement stable, ce dossier peut être supprimé.

---
**Date d'archivage**: 2025-10-31
**Raison**: Refactorisation architectural v1 → v2.0
