---
description: Plan pour corriger le système Multi-LLM
---

# 🔧 Plan de correction du système Multi-LLM

## 📋 Problèmes identifiés

### ✅ 1. Module LLM - FONCTIONNEL
- ✅ `threadx.llm.client.LLMClient` existe et fonctionne
- ✅ `threadx.llm.agents.analyst.Analyst` existe et fonctionne  
- ✅ `threadx.llm.agents.strategist.Strategist` existe et fonctionne
- ✅ `threadx.llm.agents.base_agent.BaseAgent` existe et fonctionne
- ✅ Tous les imports LLM fonctionnent correctement

### ❌ 2. Fonction manquante - PROBLÈME PRINCIPAL
**Fichier**: `threadx.ui.strategy_registry.py`
**Problème**: La fonction `get_strategy_class()` n'existe pas
**Impact**: Le notebook ne peut pas charger les classes de stratégies
**Solution**: Créer la fonction `get_strategy_class()` dans `strategy_registry.py`

### 📁 3. Structure des stratégies
- Dossier: `D:\ThreadX_big\src\threadx\strategy\`
- Stratégies disponibles:
  - `ma_crossover.py` (utilisée dans le notebook)
  - `bb_atr.py`
  - `bollinger_dual.py`
  - `amplitude_hunter.py`

## 🎯 Actions à réaliser

### Action 1: Créer `get_strategy_class()` dans `strategy_registry.py`
**Fichier**: `D:\ThreadX_big\src\threadx\ui\strategy_registry.py`
**Description**: Ajouter une fonction qui charge dynamiquement les classes de stratégies

```python
def get_strategy_class(strategy_name: str):
    """
    Charge dynamiquement une classe de stratégie par son nom.
    
    Args:
        strategy_name: Nom de la stratégie (ex: "MA_Crossover", "Bollinger_Breakout")
    
    Returns:
        Classe de stratégie
    
    Raises:
        ValueError: Si la stratégie n'existe pas
    """
    # Mapping nom → module
    strategy_modules = {
        "MA_Crossover": "threadx.strategy.ma_crossover",
        "Bollinger_Breakout": "threadx.strategy.bb_atr",
        "Bollinger_Dual": "threadx.strategy.bollinger_dual",
        "Amplitude_Hunter": "threadx.strategy.amplitude_hunter",
    }
    
    if strategy_name not in strategy_modules:
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {list(strategy_modules.keys())}")
    
    module_path = strategy_modules[strategy_name]
    module = __import__(module_path, fromlist=[''])
    
    # Trouver la classe de stratégie dans le module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and hasattr(attr, 'generate_signals'):
            return attr
    
    raise ValueError(f"No strategy class found in {module_path}")
```

### Action 2: Vérifier la structure de `ma_crossover.py`
**Fichier**: `D:\ThreadX_big\src\threadx\strategy\ma_crossover.py`
**Vérifications**:
- La classe existe et a une méthode `generate_signals()`
- Les paramètres attendus correspondent au notebook (`short_period`, `long_period`, `use_ema`)

### Action 3: Tester le système complet
**Script de test**: Créer `test_multi_llm_full.py` qui:
1. Importe tous les modules nécessaires
2. Charge la stratégie MA_Crossover
3. Crée des données synthétiques
4. Exécute un backtest simple
5. Teste l'agent Analyst (sans appel LLM réel)
6. Teste l'agent Strategist (sans appel LLM réel)

### Action 4: Corriger le notebook si nécessaire
**Fichier**: `D:\ThreadX_big\notebooks\multi_llm_optimizer.ipynb`
**Corrections potentielles**:
- Vérifier que les imports sont corrects
- Ajouter des try/except pour gérer les erreurs gracieusement
- Ajouter des validations avant les appels LLM

## 🚀 Ordre d'exécution

1. ✅ **Diagnostic complet** (FAIT)
2. 🔧 **Créer `get_strategy_class()`** (EN COURS)
3. 🔍 **Vérifier `ma_crossover.py`**
4. ✅ **Tester imports et intégration**
5. 🎯 **Tester le notebook complet**

## 📊 Résultats attendus

Après corrections:
- ✅ Tous les imports du notebook fonctionnent
- ✅ Le sweep initial s'exécute correctement
- ✅ L'agent Analyst peut analyser les résultats
- ✅ L'agent Strategist peut générer des propositions
- ✅ Les backtests des propositions fonctionnent
- ✅ Les visualisations sont générées

## ⚠️ Notes importantes

- Le système LLM lui-même est **FONCTIONNEL**
- Le problème est uniquement dans la **fonction utilitaire manquante**
- Aucune modification du code LLM n'est nécessaire
- Les agents sont correctement implémentés
