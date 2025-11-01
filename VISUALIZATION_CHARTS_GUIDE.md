# 📊 Génération de Graphiques Backtest - ThreadX

Guide d'utilisation du module `visualization` pour générer graphiques interactifs Plotly.

## 🎯 Installation

```bash
pip install plotly kaleido
```

## 📖 Usage de base

### 1. Graphique simple (1 timeframe)

```python
from threadx.visualization import generate_backtest_chart
import pandas as pd

# Après avoir effectué un backtest optimisé
best_combo = {'bb_window': 20, 'bb_num_std': 2.0, 'atr_window': 14}

# Génération graphique
chart_path = generate_backtest_chart(
    results_df=best_results,  # DataFrame avec equity, entry_price, exit_price, position
    ohlcv_data=ohlcv,         # DataFrame avec open, high, low, close, volume
    best_combo=best_combo,
    symbol='BTCUSDC',
    timeframe='1h',
    output_path='charts/backtest_BTCUSDC_1h.html',
    show_browser=True  # Ouvre automatiquement dans navigateur
)

print(f"✅ Graphique généré: {chart_path}")
```

### 2. Graphique multi-timeframes

```python
from threadx.visualization import generate_multi_timeframe_chart

# Résultats pour plusieurs timeframes
results_dict = {
    '1h': results_1h,
    '4h': results_4h,
    '1d': results_1d
}

ohlcv_dict = {
    '1h': ohlcv_1h,
    '4h': ohlcv_4h,
    '1d': ohlcv_1d
}

best_combos = {
    '1h': {'bb_window': 20, 'bb_num_std': 2.0},
    '4h': {'bb_window': 40, 'bb_num_std': 2.5},
    '1d': {'bb_window': 100, 'bb_num_std': 3.0}
}

chart_path = generate_multi_timeframe_chart(
    results_dict=results_dict,
    ohlcv_dict=ohlcv_dict,
    best_combos=best_combos,
    symbol='BTCUSDC',
    output_path='charts/multi_tf_BTCUSDC.html',
    show_browser=True
)
```

## 🔍 Structure DataFrame results_df

Le DataFrame de résultats doit contenir:

```python
# Colonnes requises:
- timestamp (index): pd.DatetimeIndex
- position: int (1=long, -1=short, 0=flat)
- equity: float (valeur portefeuille en $)
- entry_price: float (NaN si pas d'entrée)
- exit_price: float (NaN si pas de sortie)

# Exemple:
                        position  equity  entry_price  exit_price
timestamp
2024-01-01 00:00:00           0   10000          NaN         NaN
2024-01-01 01:00:00           1   10000     42150.50         NaN
2024-01-01 02:00:00           1   10250          NaN         NaN
2024-01-01 03:00:00           0   10250          NaN    42380.20
```

## 🎨 Features du graphique

### Sous-graphique 1: Prix & Signaux (60% hauteur)
- **Candlesticks**: OHLC avec couleurs vert/rouge
- **Bollinger Bands**: 3 lignes (sup/mid/inf) + zone semi-transparente
- **Marqueurs d'entrée**: ▲ vert sur prix d'entrée
- **Marqueurs de sortie**: ▼ rouge sur prix de sortie

### Sous-graphique 2: Courbe d'Équité (20% hauteur)
- **Ligne bleue**: Évolution capital
- **Zone remplie**: Sous la courbe en bleu transparent
- **Ligne pointillée**: Capital initial (référence)

### Sous-graphique 3: Position (20% hauteur)
- **Barres colorées**:
  - Vert: Long position
  - Rouge: Short position
  - Gris: Flat (pas de position)

## 🚀 Intégration dans Sweep Runner

```python
from threadx.optimization.engine import SweepRunner
from threadx.visualization import generate_backtest_chart

# Exécution sweep
runner = SweepRunner(
    symbol='BTCUSDC',
    timeframe='1h',
    max_workers=30,
    batch_size=2000,
    use_cache=True
)

results_df = runner.run_grid(
    param_grid={'bb_window': [10, 20, 30], 'bb_num_std': [1.5, 2.0, 2.5]},
    top_n=5
)

# Récupération meilleur combo
best_combo = results_df.iloc[0]['params']  # Meilleure combo
best_results = runner.get_backtest_results(best_combo)  # Détails backtest

# Génération graphique
chart_path = generate_backtest_chart(
    results_df=best_results,
    ohlcv_data=runner.ohlcv_data,  # Données OHLCV utilisées
    best_combo=best_combo,
    symbol='BTCUSDC',
    timeframe='1h',
    output_path=f'charts/best_{runner.symbol}_{runner.timeframe}.html',
    show_browser=True
)
```

## 📝 Format HTML interactif

Le graphique HTML généré permet:
- **Zoom**: Cliquer-glisser sur axe X
- **Pan**: Shift + cliquer-glisser
- **Hover**: Affiche valeurs au survol
- **Reset**: Double-clic pour réinitialiser vue
- **Export**: Bouton caméra pour PNG

## 🎯 Exemple complet end-to-end

```python
from threadx.data.binance_loader import BinanceLoader
from threadx.optimization.engine import SweepRunner
from threadx.visualization import generate_backtest_chart

# 1. Chargement données
loader = BinanceLoader()
ohlcv = loader.load('BTCUSDC', '1h', days=30)

# 2. Sweep optimisation
runner = SweepRunner(
    symbol='BTCUSDC',
    timeframe='1h',
    max_workers=30,
    batch_size=2000
)

results = runner.run_grid(
    param_grid={
        'bb_window': [15, 20, 25],
        'bb_num_std': [1.5, 2.0, 2.5],
        'atr_window': [10, 14, 20]
    },
    top_n=10
)

# 3. Meilleure combo
best_combo = results.iloc[0]['params']
print(f"Meilleur combo: {best_combo}")
print(f"Sharpe Ratio: {results.iloc[0]['sharpe_ratio']:.2f}")

# 4. Génération graphique
best_results = runner.get_backtest_results(best_combo)

chart_path = generate_backtest_chart(
    results_df=best_results,
    ohlcv_data=ohlcv,
    best_combo=best_combo,
    symbol='BTCUSDC',
    timeframe='1h',
    output_path='charts/best_backtest.html',
    show_browser=True
)

print(f"✅ Graphique: {chart_path}")
```

## 🔧 Customisation avancée

Pour modifier le graphique (couleurs, taille, layout):

```python
import plotly.graph_objects as go

# Après génération, recharger et modifier
import plotly.io as pio

fig = pio.read_json('charts/backtest.json')  # Si sauvegardé en JSON

# Modifier layout
fig.update_layout(
    template='plotly_white',  # Thème clair
    height=1200,              # Plus haut
    title_font_size=20
)

# Re-sauvegarder
fig.write_html('charts/backtest_custom.html')
```

## 📦 Dépendances

```txt
plotly>=5.14.0
kaleido>=0.2.1  # Pour export PNG/PDF
pandas>=1.5.0
```

## ⚠️ Notes importantes

1. **Mémoire**: Graphiques HTML peuvent être volumineux (5-20 MB) pour datasets longs
2. **Performance**: Génération prend 2-10s selon taille dataset
3. **Browser**: Chrome/Firefox recommandés pour interactivité optimale
4. **Timestamps**: Index DataFrame doit être pd.DatetimeIndex

## 🎯 Next Steps

- Implémenter filtres de plage temporelle (slider)
- Ajouter statistiques overlay (win rate, max DD, etc.)
- Export PDF/PNG automatique
- Annotations personnalisées pour événements clés
