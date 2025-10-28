# AmplitudeHunter - Stratégie BB Amplitude Rider

## Vue d'ensemble

**AmplitudeHunter** (aussi appelée "BB Amplitude Rider" ou "BB-AR") est une stratégie de trading algorithmique avancée conçue pour capturer l'amplitude complète des mouvements de prix à travers les Bollinger Bands.

**Concept clé** : Capturer l'amplitude complète d'un swing Bollinger Bands (basse → médiane → haute → extension) et laisser courir les positions au-delà de la bande opposée lorsque le momentum le permet.

## Caractéristiques Principales

### 1. Filtre de Régime Multi-Critères
- **BBWidth percentile** : Trade uniquement quand il y a assez de volatilité
- **Volume z-score** : Confirme l'énergie du marché
- **ADX optionnel** : Détecte les contextes propices aux extensions

### 2. Setup "Spring → Drive"
- **Spring** : Détection d'un point bas de survente (ou haut de surachat)
- **Drive** : Impulsion MACD confirmant le retournement
- **Validation %B** : Franchissement de seuils de %B pour timing optimal

### 3. Score d'Amplitude
Calcul d'un score composite pour moduler l'agressivité :
```
Score = w1·BBWidth_pct + w2·|%B| + w3·pente_MACD + w4·volume_zscore
```

### 4. Pyramiding Intelligent
- **Add #1** : Quand le prix franchit la bande opposée ET MACD s'intensifie
- **Add #2** : Sur pullback tenu à la médiane avec MACD favorable

### 5. Gestion Avancée des Stops
- **SL initial** : `max(swing_low - k·ATR, pct·(médiane-basse))`
- **Trailing conditionnel** : Activé quand %B > 1 OU gain ≥ 1R
- **3 types de trailing** :
  - Chandelier : `médiane - k·ATR`
  - %B floor : Sortir si %B < 0.5 après extension
  - MACD fade : Sortie sur retournement MACD
- **Stop fixe SHORT** : 37% au-dessus du prix d'entrée (protection catastrophe)

### 6. Cible BIP (Bollinger Implied Price)
- **BIP** = `médiane + (médiane - basse)`
- Sortie partielle à BIP (50% par défaut)
- Runner sous trailing pour capturer les extensions

## Installation et Import

```python
from threadx.strategy import (
    AmplitudeHunterParams,
    AmplitudeHunterStrategy,
    amplitude_hunter_generate_signals,
    amplitude_hunter_backtest,
    amplitude_hunter_create_default_params,
)
```

## Utilisation Basique

### Configuration Simple

```python
import pandas as pd
from threadx.strategy import AmplitudeHunterParams, AmplitudeHunterStrategy

# Charger vos données OHLCV
df = pd.read_csv('BTCUSDT_1h.csv', index_col='timestamp', parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)

# Créer la stratégie
strategy = AmplitudeHunterStrategy(symbol="BTCUSDT", timeframe="1h")

# Paramètres par défaut
params = AmplitudeHunterParams().to_dict()

# Générer les signaux
signals_df = strategy.generate_signals(df, params)

# Exécuter le backtest
equity_curve, run_stats = strategy.backtest(
    df,
    params,
    initial_capital=10000.0,
    fee_bps=4.5
)

# Afficher les résultats
print(f"ROI: {run_stats.total_pnl_pct:.2f}%")
print(f"Sharpe Ratio: {run_stats.sharpe_ratio:.2f}")
print(f"Win Rate: {run_stats.win_rate_pct:.1f}%")
print(f"Max Drawdown: {run_stats.max_drawdown_pct:.2f}%")
```

### Configuration Avancée

```python
# Configuration personnalisée avec pyramiding et trailing agressif
params = AmplitudeHunterParams(
    # Bollinger Bands
    bb_period=20,
    bb_std=2.0,

    # Filtre de régime
    bbwidth_percentile_threshold=50,  # 50ème percentile
    volume_zscore_threshold=0.5,
    use_adx_filter=True,
    adx_threshold=20,

    # Score d'Amplitude (seuil plus élevé = plus sélectif)
    amplitude_score_threshold=0.75,

    # Pyramiding activé
    pyramiding_enabled=True,
    pyramiding_max_adds=2,  # 2 adds maximum

    # Stops et trailing
    trailing_type="chandelier",  # ou "pb_floor" ou "macd_fade"
    trailing_activation_pb_threshold=1.0,
    trailing_activation_gain_r=1.0,

    # Cible BIP
    use_bip_target=True,
    bip_partial_exit_pct=0.5,  # 50% de sortie partielle

    # Risk management
    risk_per_trade=0.02,  # 2% du capital par trade
    max_hold_bars=100,
)

equity_curve, run_stats = strategy.backtest(df, params.to_dict(), 10000.0)
```

## Paramètres Configurables

### Bollinger Bands
- `bb_period` (int, défaut: 20) : Période des Bollinger Bands
- `bb_std` (float, défaut: 2.0) : Multiplicateur d'écart-type

### Filtre de Régime
- `bbwidth_percentile_threshold` (float, défaut: 50) : Seuil percentile BBWidth (30-70 recommandé)
- `bbwidth_lookback` (int, défaut: 100) : Période lookback pour percentile
- `volume_zscore_threshold` (float, défaut: 0.5) : Seuil z-score volume
- `volume_lookback` (int, défaut: 50) : Période lookback volume
- `use_adx_filter` (bool, défaut: False) : Activer filtre ADX
- `adx_threshold` (float, défaut: 15) : Seuil ADX minimum

### Setup Spring → Drive
- `spring_lookback` (int, défaut: 20) : Lookback pour détecter spring
- `pb_entry_threshold_min` (float, défaut: 0.2) : %B minimum pour entrée
- `pb_entry_threshold_max` (float, défaut: 0.5) : %B maximum pour entrée
- `macd_fast` (int, défaut: 12) : Période MACD rapide
- `macd_slow` (int, défaut: 26) : Période MACD lente
- `macd_signal` (int, défaut: 9) : Période signal MACD

### Score d'Amplitude
- `amplitude_score_threshold` (float, défaut: 0.6) : Score minimum pour trade
- `amplitude_w1_bbwidth` (float, défaut: 0.3) : Poids BBWidth
- `amplitude_w2_pb` (float, défaut: 0.2) : Poids |%B|
- `amplitude_w3_macd_slope` (float, défaut: 0.3) : Poids pente MACD
- `amplitude_w4_volume` (float, défaut: 0.2) : Poids volume

### Pyramiding
- `pyramiding_enabled` (bool, défaut: False) : Activer pyramiding
- `pyramiding_max_adds` (int, défaut: 1) : Nombre max d'adds (1 ou 2)

### Stops et Trailing
- `atr_period` (int, défaut: 14) : Période ATR
- `sl_atr_multiplier` (float, défaut: 2.0) : Multiplicateur ATR pour SL
- `sl_min_pct` (float, défaut: 0.37) : SL minimum en % (médiane-basse)
- `short_stop_pct` (float, défaut: 0.37) : Stop fixe SHORT (37%)
- `trailing_type` (str, défaut: "chandelier") : Type de trailing
- `trailing_activation_pb_threshold` (float, défaut: 1.0) : %B pour activer
- `trailing_activation_gain_r` (float, défaut: 1.0) : Gain en R pour activer

### Cible BIP
- `use_bip_target` (bool, défaut: True) : Utiliser cible BIP
- `bip_partial_exit_pct` (float, défaut: 0.5) : % sortie partielle

### Risk Management
- `risk_per_trade` (float, défaut: 0.02) : Risque par trade (2%)
- `max_hold_bars` (int, défaut: 100) : Durée max position
- `leverage` (float, défaut: 1.0) : Effet de levier

## Optimisation des Paramètres

### Grille d'Optimisation Recommandée

```python
from threadx.optimization import MonteCarloOptimizer

# Plages de paramètres à tester
param_ranges = {
    'bbwidth_percentile_threshold': (30, 70),
    'spring_lookback': (10, 20),
    'pb_entry_threshold_min': (0.2, 0.5),
    'sl_atr_multiplier': (1.5, 3.0),
    'trailing_activation_gain_r': (0.8, 1.5),
    'amplitude_score_threshold': (0.4, 0.75),
}

# Fonction objectif
def objective_fn(test_params):
    _, stats = strategy.backtest(df, test_params, 10000)
    # Optimiser le Sharpe Ratio
    return stats.sharpe_ratio or 0.0

# Optimisation Monte Carlo
optimizer = MonteCarloOptimizer(
    param_ranges=param_ranges,
    objective_fn=objective_fn,
    n_trials=100,
    maximize=True,
    seed=42
)

result = optimizer.optimize(max_iterations=100)
print(f"Meilleurs paramètres: {result.best_params}")
print(f"Meilleur Sharpe: {result.best_score:.2f}")
```

## Indicateurs Calculés

La stratégie calcule et expose les indicateurs suivants dans `signals_df` :

- `bb_upper`, `bb_middle`, `bb_lower` : Bandes de Bollinger
- `percent_b` : Position du prix dans les bandes ([0, 1])
- `bb_width` : Largeur des bandes
- `bbwidth_percentile` : Percentile du BBWidth
- `macd_line`, `macd_signal`, `macd_hist` : MACD complet
- `volume_zscore` : Z-score du volume
- `adx` : Average Directional Index (si activé)
- `amplitude_score` : Score d'amplitude composite
- `atr` : Average True Range
- `signal` : Signal de trading ("ENTER_LONG", "ENTER_SHORT", "HOLD")

## Intégration GPU

La stratégie est compatible avec l'accélération GPU via le système IndicatorBank de ThreadX :

```python
# Les indicateurs BB et ATR utilisent automatiquement le GPU si disponible
# Pas de configuration spéciale requise
strategy = AmplitudeHunterStrategy(symbol="BTCUSDT", timeframe="1h")
# Le calcul des indicateurs sera accéléré sur GPU si CUDA est détecté
```

## Exemple Complet : Monte Carlo avec GPU

```python
from threadx.strategy import AmplitudeHunterStrategy, AmplitudeHunterParams
from threadx.optimization import MonteCarloOptimizer
import pandas as pd

# Charger données
df = pd.read_parquet('BTCUSDT_1h.parquet')
df.index = pd.to_datetime(df.index, utc=True)

# Stratégie
strategy = AmplitudeHunterStrategy("BTCUSDT", "1h")

# Paramètres de base
base_params = AmplitudeHunterParams(
    pyramiding_enabled=True,
    pyramiding_max_adds=2,
    use_bip_target=True,
).to_dict()

# Plages d'optimisation
param_ranges = {
    'bbwidth_percentile_threshold': (30, 70),
    'amplitude_score_threshold': (0.4, 0.75),
    'sl_atr_multiplier': (1.5, 3.0),
    'trailing_chandelier_atr_mult': (2.0, 3.5),
}

# Fonction objectif (Sharpe Ratio)
def objective(params):
    test_params = {**base_params, **params}
    _, stats = strategy.backtest(df, test_params, 10000.0)
    return stats.sharpe_ratio if stats.sharpe_ratio else 0.0

# Optimisation Monte Carlo (100 itérations)
optimizer = MonteCarloOptimizer(
    param_ranges=param_ranges,
    objective_fn=objective,
    n_trials=100,
    maximize=True,
    seed=42
)

result = optimizer.optimize()

print(f"\n🏆 Meilleurs paramètres trouvés:")
for key, value in result.best_params.items():
    print(f"  {key}: {value}")

print(f"\n📊 Meilleur Sharpe Ratio: {result.best_score:.3f}")

# Backtest final avec les meilleurs paramètres
final_params = {**base_params, **result.best_params}
equity_curve, run_stats = strategy.backtest(df, final_params, 10000.0)

print(f"\n📈 Résultats optimisés:")
print(f"  ROI: {run_stats.total_pnl_pct:.2f}%")
print(f"  Win Rate: {run_stats.win_rate_pct:.1f}%")
print(f"  Total Trades: {run_stats.total_trades}")
print(f"  Max Drawdown: {run_stats.max_drawdown_pct:.2f}%")
```

## Critères de Succès Attendus

Lors de l'optimisation et du backtesting, chercher à obtenir :

- **Expectancy ↑** sans explosion du max drawdown
- **Gain médian par trade ↑** grâce aux extensions capturées
- **Moins de sorties prématurées** en tendance forte
- **Robustesse cross-assets** via le Score d'Amplitude comme filtre

## Architecture Technique

### Intégration ThreadX

La stratégie suit l'architecture modulaire de ThreadX :

```
AmplitudeHunter
├── Paramètres (AmplitudeHunterParams)
│   ├── Validation automatique (__post_init__)
│   └── Sérialisation (to_dict/from_dict)
│
├── Indicateurs (via IndicatorBank)
│   ├── Bollinger Bands (GPU-accelerated)
│   ├── ATR (GPU-accelerated)
│   ├── MACD (calculé localement)
│   ├── ADX (calculé localement)
│   └── Score d'Amplitude (composite)
│
├── Génération de signaux (generate_signals)
│   ├── Filtre de régime multi-critères
│   ├── Détection Spring → Drive
│   └── Validation Score d'Amplitude
│
└── Backtest (backtest)
    ├── Position sizing ATR-based
    ├── Pyramiding intelligent (2 adds max)
    ├── Stops multiples (initial, trailing, fixe SHORT)
    ├── Cible BIP avec sortie partielle
    └── Calcul RunStats complet
```

### Compatibilité

- ✅ ThreadX Phase 3+ (IndicatorBank)
- ✅ GPU acceleration (CUDA optionnel)
- ✅ Multi-GPU distribution
- ✅ Monte Carlo optimization
- ✅ Walk-forward analysis
- ✅ UI Streamlit integration

## Notes et Recommandations

### Performance
- La stratégie génère moins de signaux que des stratégies simples (par design)
- Les filtres stricts assurent une meilleure qualité des trades
- Le pyramiding améliore les gains sur les bons setups
- Le trailing conditionnel préserve les profits en volatilité

### Tuning
- **Marchés tendanciels** : Augmenter `amplitude_score_threshold`, activer ADX
- **Marchés range** : Réduire `bbwidth_percentile_threshold`, désactiver pyramiding
- **Haute volatilité** : Augmenter `sl_atr_multiplier`, utiliser trailing "chandelier"
- **Basse volatilité** : Réduire seuils, utiliser trailing "%B floor"

### Debugging
Activer le logging détaillé :

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Voir tous les signaux générés et les décisions de trading
```

## Auteur et Licence

- **Auteur** : Claude (Anthropic)
- **Nom de code** : AmplitudeHunter (BB-AR v2.2)
- **Version** : 1.0.0
- **Date** : Octobre 2025
- **Licence** : Propriétaire ThreadX

---

**Bon trading ! 🚀**
