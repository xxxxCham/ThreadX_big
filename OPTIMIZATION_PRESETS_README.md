# Système de Pré-Réglages d'Optimisation ThreadX

## Vue d'ensemble

Le **système de presets d'optimisation** fournit des plages de paramètres "classiques" pré-configurées pour faciliter l'optimisation des stratégies de trading. Basé sur le tableau de référence des backtests professionnels, il automatise le mapping des indicateurs techniques aux paramètres de stratégies.

## Architecture

```
threadx/
├── optimization/
│   └── presets/
│       ├── __init__.py                    # Exports publics
│       ├── ranges.py                      # Classes de gestion
│       └── indicator_ranges.toml          # Configuration des plages
```

### Respect de l'architecture Moteur|Bridge|Interface

- **Moteur** : `ranges.py` - Logique de chargement et mapping
- **Bridge** : Méthodes statiques dans les stratégies (ex: `AmplitudeHunterStrategy.get_optimization_ranges()`)
- **Interface** : Import via `threadx.optimization.presets` pour l'UI

## Fichier de Configuration

### indicator_ranges.toml

Définit les plages pour 58+ indicateurs techniques :

```toml
[bollinger.period]
min = 10
max = 50
step = 1
default = 20
description = "Période de la moyenne mobile (SMA)"

[bollinger.std_dev]
min = 1.5
max = 3.0
step = 0.1
default = 2.0
description = "Multiplicateur d'écart-type (K)"

[macd.fast_period]
min = 8
max = 15
step = 1
default = 12
description = "MACD EMA rapide"

[amplitude_hunter.amplitude_score_threshold]
min = 0.4
max = 0.8
step = 0.05
default = 0.6
description = "Score d'amplitude minimum"
```

### Types de Paramètres Supportés

1. **Numeric** : `min`, `max`, `step`, `default`
2. **Categorical** : `values` (liste de choix)
3. **Boolean** : `type = "boolean"`
4. **Fixed** : `value` (valeur fixe, non optimisable)

## Utilisation

### 1. Récupérer un Preset Spécifique

```python
from threadx.optimization.presets import get_indicator_range

bb_period = get_indicator_range('bollinger.period')
print(bb_period.get_range())  # (10, 50)
print(bb_period.step)  # 1
print(bb_period.default)  # 20
print(bb_period.get_grid_values())  # [10, 11, 12, ..., 50]
```

### 2. Lister Tous les Indicateurs Disponibles

```python
from threadx.optimization.presets import list_available_indicators

indicators = list_available_indicators()
# ['adx.period', 'bollinger.period', 'macd.fast_period', ...]
print(f"{len(indicators)} indicateurs disponibles")
```

### 3. Utiliser le Mapper pour une Stratégie

```python
from threadx.optimization.presets import get_strategy_preset

# Créer le mapper pour AmplitudeHunter
mapper = get_strategy_preset('AmplitudeHunter')

# Récupérer les plages d'optimisation (pour Monte Carlo)
ranges = mapper.get_optimization_ranges()
# {'bb_period': (10, 50), 'bb_std': (1.5, 3.0), ...}

# Récupérer les grilles de valeurs (pour Grid Search)
grid = mapper.get_grid_parameters()
# {'bb_period': [10, 11, 12, ..., 50], 'bb_std': [1.5, 1.6, ..., 3.0], ...}

# Récupérer les valeurs par défaut
defaults = mapper.get_default_parameters()
# {'bb_period': 20, 'bb_std': 2.0, ...}
```

### 4. Intégration Directe dans les Stratégies

Chaque stratégie expose 3 méthodes statiques :

```python
from threadx.strategy import AmplitudeHunterStrategy

# Plages pour Monte Carlo
ranges = AmplitudeHunterStrategy.get_optimization_ranges()
# {'bb_period': (10, 50), 'amplitude_score_threshold': (0.4, 0.8), ...}

# Grilles pour Grid Search
grid = AmplitudeHunterStrategy.get_optimization_grid()
# {'bb_period': [10, 11, ..., 50], 'pyramiding_max_adds': [0, 1, 2], ...}

# Defaults recommandés
defaults = AmplitudeHunterStrategy.get_default_optimization_params()
# {'bb_period': 20, 'bb_std': 2.0, ...}
```

## Exemple Complet: Monte Carlo avec Presets

```python
from threadx.strategy import AmplitudeHunterStrategy, AmplitudeHunterParams
from threadx.optimization import MonteCarloOptimizer
import pandas as pd

# Charger données
df = pd.read_parquet('BTCUSDT_1h.parquet')
df.index = pd.to_datetime(df.index, utc=True)

# Stratégie
strategy = AmplitudeHunterStrategy("BTCUSDT", "1h")

# Paramètres de base (non optimisés)
base_params = AmplitudeHunterParams(
    pyramiding_enabled=True,
    use_bip_target=True,
).to_dict()

# Récupérer les plages d'optimisation depuis les presets
param_ranges = strategy.get_optimization_ranges()

# Sélectionner un sous-ensemble de paramètres à optimiser
optimization_ranges = {
    'bb_period': param_ranges['bb_period'],
    'amplitude_score_threshold': param_ranges['amplitude_score_threshold'],
    'sl_atr_multiplier': param_ranges['sl_atr_multiplier'],
    'trailing_chandelier_atr_mult': param_ranges['trailing_chandelier_atr_mult'],
}

# Fonction objectif (Sharpe Ratio)
def objective(params):
    test_params = {**base_params, **params}
    _, stats = strategy.backtest(df, test_params, 10000.0)
    return stats.sharpe_ratio if stats.sharpe_ratio else 0.0

# Optimisation Monte Carlo
optimizer = MonteCarloOptimizer(
    param_ranges=optimization_ranges,
    objective_fn=objective,
    n_trials=100,
    maximize=True,
    seed=42
)

result = optimizer.optimize()

print(f"\n🏆 Meilleurs paramètres:")
for key, value in result.best_params.items():
    print(f"  {key}: {value}")
print(f"\n📊 Meilleur Sharpe: {result.best_score:.3f}")
```

## Exemple: Grid Search avec Presets

```python
from threadx.optimization import GridSearchOptimizer

# Récupérer toutes les grilles
full_grid = strategy.get_optimization_grid()

# Sélectionner un sous-ensemble (grid complet serait trop large)
test_grid = {
    'bb_period': full_grid['bb_period'][::5],  # Échantillonner tous les 5
    'amplitude_score_threshold': full_grid['amplitude_score_threshold'],
    'pyramiding_max_adds': full_grid['pyramiding_max_adds'],  # [0, 1, 2]
}

# Total de combinaisons
total = 1
for values in test_grid.values():
    total *= len(values)
print(f"Total de combinaisons: {total}")

# Grid search (exemple simplifié)
results = []
for bb_period in test_grid['bb_period']:
    for amp_score in test_grid['amplitude_score_threshold']:
        for max_adds in test_grid['pyramiding_max_adds']:
            params = {
                **base_params,
                'bb_period': bb_period,
                'amplitude_score_threshold': amp_score,
                'pyramiding_max_adds': max_adds,
            }
            _, stats = strategy.backtest(df, params, 10000.0)
            results.append({
                'bb_period': bb_period,
                'amplitude_score_threshold': amp_score,
                'pyramiding_max_adds': max_adds,
                'sharpe': stats.sharpe_ratio or 0.0,
                'total_pnl_pct': stats.total_pnl_pct,
            })

# Trier par Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)
print("\nTop 5 configurations:")
for i, r in enumerate(results[:5], 1):
    print(f"{i}. Sharpe={r['sharpe']:.3f}, ROI={r['total_pnl_pct']:.2f}%, "
          f"BB={r['bb_period']}, Score={r['amplitude_score_threshold']:.2f}, Adds={r['pyramiding_max_adds']}")
```

## Ajout de Nouveaux Indicateurs

### 1. Modifier indicator_ranges.toml

```toml
[custom_indicator.param_name]
min = 5
max = 25
step = 1
default = 10
description = "Description du paramètre"
```

### 2. Ajouter le Mapping dans ranges.py

Modifier la fonction `get_strategy_preset()` :

```python
def get_strategy_preset(strategy_name: str) -> StrategyPresetMapper:
    mapper = StrategyPresetMapper(strategy_name)

    if strategy_name.lower() == "my_new_strategy":
        mapper.add_mappings({
            "my_param": "custom_indicator.param_name",
            "another_param": "bollinger.period",  # Réutilisation
        })

    return mapper
```

### 3. Ajouter les Méthodes à la Stratégie

```python
class MyNewStrategy:
    @staticmethod
    def get_optimization_ranges() -> Dict[str, Tuple[float, float]]:
        try:
            from threadx.optimization.presets import get_strategy_preset
            mapper = get_strategy_preset("MyNewStrategy")
            return mapper.get_optimization_ranges()
        except ImportError:
            # Fallback
            return {"my_param": (5, 25)}

    # Idem pour get_optimization_grid() et get_default_optimization_params()
```

## Stratégies Pré-Configurées

### AmplitudeHunter

**21 paramètres mappés** :

| Paramètre Stratégie | Indicateur Preset | Range |
|---------------------|-------------------|-------|
| `bb_period` | `bollinger.period` | 10-50 |
| `bb_std` | `bollinger.std_dev` | 1.5-3.0 |
| `macd_fast` | `macd.fast_period` | 8-15 |
| `macd_slow` | `macd.slow_period` | 20-35 |
| `macd_signal` | `macd.signal_period` | 6-12 |
| `adx_period` | `adx.period` | 7-20 |
| `adx_threshold` | `adx.trend_threshold` | 20-30 |
| `atr_period` | `atr.period` | 7-21 |
| `sl_atr_multiplier` | `atr.stop_multiplier` | 1.0-3.0 |
| `amplitude_score_threshold` | `amplitude_hunter.amplitude_score_threshold` | 0.4-0.8 |
| ... | ... | ... |

### BBAtr

**5 paramètres mappés** :

- `bb_period` → `bollinger.period`
- `bb_std` → `bollinger.std_dev`
- `atr_period` → `atr.period`
- `atr_multiplier` → `atr.stop_multiplier`
- `risk_per_trade` → `amplitude_hunter.risk_per_trade`

### BollingerDual

**4 paramètres mappés** :

- `bb_period` → `bollinger.period`
- `bb_std` → `bollinger.std_dev`
- `ma_window` → `sma.short_period`
- `risk_per_trade` → `amplitude_hunter.risk_per_trade`

## Filtrage Intelligent

Le système filtre automatiquement :

1. **Indicateurs non mappés** : Si un indicateur n'existe pas dans `indicator_ranges.toml`, le mapping est ignoré avec un warning
2. **Paramètres incompatibles** : Si une stratégie n'utilise pas un indicateur, il n'apparaîtra pas dans les ranges

Exemple :
```python
mapper = StrategyPresetMapper("MyStrategy")
mapper.add_mapping("unknown_param", "non_existent_indicator")
# ⚠️ Warning: Indicateur 'non_existent_indicator' non trouvé, mapping ignoré

ranges = mapper.get_optimization_ranges()
# 'unknown_param' n'apparaîtra PAS dans ranges
```

## Intégration avec l'UI Streamlit

Le bridge expose automatiquement les presets à l'UI :

```python
# Dans l'UI Streamlit
from threadx.strategy import AmplitudeHunterStrategy

# Récupérer les presets pour affichage
ranges = AmplitudeHunterStrategy.get_optimization_ranges()
defaults = AmplitudeHunterStrategy.get_default_optimization_params()

# Construire les sliders automatiquement
for param, (min_val, max_val) in ranges.items():
    default = defaults.get(param, min_val)
    value = st.slider(
        param,
        min_value=min_val,
        max_value=max_val,
        value=default,
        help=f"Range recommandé: [{min_val}, {max_val}]"
    )
```

## Performance

- **Chargement TOML** : ~5-10ms pour 58 indicateurs
- **Mapping stratégie** : Instantané (<1ms)
- **Génération grille** : ~1-5ms par paramètre
- **Cache** : Les presets sont rechargés à chaque appel (léger, pas de cache nécessaire)

## Avantages

✅ **Centralisé** : Une seule source de vérité pour les plages
✅ **Maintenable** : Modification simple du TOML
✅ **Type-safe** : Validation automatique des types
✅ **Extensible** : Ajout facile de nouveaux indicateurs
✅ **Flexible** : Mapping custom par stratégie
✅ **Documenté** : Descriptions intégrées
✅ **Architecture propre** : Respect Moteur|Bridge|Interface

## Limitations et Considérations

1. **Pas de cache** : Les presets sont rechargés à chaque utilisation (acceptable vu la rapidité)
2. **Mapping manuel** : Chaque stratégie doit définir ses mappings dans `get_strategy_preset()`
3. **TOML requis** : Nécessite le fichier `indicator_ranges.toml` présent

## Troubleshooting

### Erreur: "Fichier de presets non trouvé"

Vérifier que `src/threadx/optimization/presets/indicator_ranges.toml` existe.

### Warning: "Indicateur non trouvé"

L'indicateur référencé dans le mapping n'existe pas dans le TOML. Ajouter l'entrée ou corriger le nom.

### Plages vides retournées

Vérifier que les mappings sont bien définis dans `get_strategy_preset()` pour la stratégie concernée.

## Fichiers Créés/Modifiés

```
✨ NOUVEAUX FICHIERS:
├── src/threadx/optimization/presets/
│   ├── __init__.py
│   ├── ranges.py
│   └── indicator_ranges.toml

📝 FICHIERS MODIFIÉS:
├── src/threadx/optimization/__init__.py (exports ajoutés)
├── src/threadx/strategy/amplitude_hunter.py (3 méthodes statiques ajoutées)
```

---

**Documentation complète du système de pré-réglages d'optimisation ThreadX**
**Version**: 1.0.0
**Date**: Octobre 2025
