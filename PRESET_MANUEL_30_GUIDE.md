# 🎯 Guide d'Utilisation - Preset Manuel 30 Workers

## 📋 Présentation

Le preset `manuel_30` permet d'utiliser **30 workers en parallèle** avec un batch size de **2000** pour maximiser l'utilisation des ressources (CPU 90%, RAM 80%, GPU 85%).

---

## ✅ Utilisation Directe (Recommandée)

### Méthode 1: Via paramètre `preset`

```python
from threadx.optimization.engine import SweepRunner

# Initialisation avec preset manuel_30
runner = SweepRunner(preset='manuel_30')

# Le runner utilise automatiquement:
# - max_workers = 30
# - batch_size = 2000
# - Optimisations GPU activées

# Exécution sweep
results = runner.run_grid(
    grid_spec={'params': {'bb_window': [10, 20, 30], 'bb_num_std': [1.5, 2.0, 2.5]}},
    real_data=ohlcv_data,
    symbol='BTCUSDC',
    timeframe='1h'
)
```

### Méthode 2: Charger preset manuellement

```python
from threadx.optimization.engine import SweepRunner
from threadx.optimization.presets.ranges import get_execution_preset

# Charger config preset
preset = get_execution_preset('manuel_30')
print(preset)
# {'max_workers': 30, 'batch_size': 2000, 'gpu_utilization_target': 0.85, ...}

# Passer valeurs manuellement
runner = SweepRunner(
    max_workers=preset['max_workers'],
    batch_size=preset['batch_size']
)

results = runner.run_grid(...)
```

---

## 🎛️ Presets Disponibles

### Liste des presets

```python
from threadx.optimization.presets.ranges import load_execution_presets

all_presets = load_execution_presets()
for category, presets in all_presets.items():
    print(f"\n{category}:")
    for name, config in presets.items():
        print(f"  - {name}: {config}")
```

### Presets principaux

| Preset | Workers | Batch Size | Utilisation |
|--------|---------|------------|-------------|
| `auto` | Auto | Auto | Détection automatique |
| `conservative` | 4 | 500 | Économie ressources |
| `balanced` | 8 | 1000 | Équilibre perf/ressources |
| `aggressive` | 16 | 1500 | Haute performance |
| **`manuel_30`** | **30** | **2000** | **Performance maximale** |
| `extreme` | 64 | 5000 | Serveurs puissants |

---

## 📊 Exemples Complets

### Exemple 1: Sweep simple avec manuel_30

```python
from threadx.optimization.engine import SweepRunner
from threadx.data.binance_loader import BinanceLoader

# Chargement données
loader = BinanceLoader()
ohlcv = loader.load('BTCUSDC', '1h', days=90)

# Sweep avec preset manuel_30
runner = SweepRunner(preset='manuel_30')

results = runner.run_grid(
    grid_spec={
        'params': {
            'bb_window': [15, 20, 25, 30],
            'bb_num_std': [1.5, 2.0, 2.5],
            'atr_window': [10, 14, 20]
        }
    },
    real_data=ohlcv,
    symbol='BTCUSDC',
    timeframe='1h'
)

print(f"✅ Sweep terminé: {len(results)} résultats")
print(f"Meilleur Sharpe: {results.iloc[0]['sharpe_ratio']:.2f}")
```

### Exemple 2: Override partiel du preset

```python
# Utiliser manuel_30 mais changer batch_size
runner = SweepRunner(
    preset='manuel_30',      # 30 workers du preset
    batch_size=3000          # Override batch size
)

# Utiliser manuel_30 mais changer workers
runner = SweepRunner(
    preset='manuel_30',      # batch_size=2000 du preset
    max_workers=20           # Override workers
)
```

### Exemple 3: Monte Carlo avec manuel_30

```python
runner = SweepRunner(preset='manuel_30')

results = runner.run_monte_carlo(
    mc_spec={
        'params': {
            'bb_window': {'min': 10, 'max': 50},
            'bb_num_std': {'min': 1.0, 'max': 3.0},
            'atr_window': {'min': 5, 'max': 30}
        },
        'n_samples': 10000  # 10k combinaisons aléatoires
    },
    real_data=ohlcv,
    symbol='BTCUSDC',
    timeframe='1h'
)
```

---

## 🎨 Intégration UI Streamlit

```python
import streamlit as st
from threadx.optimization.engine import SweepRunner

# Sélection preset dans UI
preset_name = st.selectbox(
    "Preset d'exécution",
    ['auto', 'conservative', 'balanced', 'aggressive', 'manuel_30', 'extreme']
)

# Utilisation
if st.button("Lancer Sweep"):
    runner = SweepRunner(preset=preset_name)

    with st.spinner(f"Sweep en cours (preset: {preset_name})..."):
        results = runner.run_grid(...)

    st.success(f"✅ Sweep terminé avec preset {preset_name}")
    st.dataframe(results.head(10))
```

---

## 🔍 Monitoring Ressources

Le preset manuel_30 affiche automatiquement l'utilisation ressources:

```
📊 Progrès: 1500/10000 (15.0%) |
⏱️  Écoulé: 5m 30s |
⏳ Restant: 31m 15s |
⚡ Vitesse: 4.53 combos/s

💻 CPU: 87.3% (16 cores) | 🧠 RAM: 76.2% (24.3 / 32.0 GB) |
🎮 RTX 5090: 82.5% (13.2 / 16.0 GB) | 🎮 RTX 2060: 68.1% (5.4 / 8.0 GB)
```

---

## ⚙️ Configuration Avancée

### Créer preset personnalisé

Éditer `src/threadx/optimization/presets/execution_presets.toml`:

```toml
[workers.custom_40]
max_workers = 40
batch_size = 2500
description = "Preset ultra-haute performance"
gpu_utilization_target = 0.90
cpu_utilization_target = 0.95
ram_utilization_target = 0.85
```

Utilisation:
```python
runner = SweepRunner(preset='custom_40')
```

---

## 🎯 Quand Utiliser Manuel_30 ?

### ✅ Recommandé pour:
- Sweeps avec >1000 combinaisons
- Grilles denses (ex: 5×5×5 = 125 combos)
- Monte Carlo >5000 samples
- Hardware puissant (CPU 16+ cores, GPU 16GB+)
- Backtests longue durée (>30 jours)

### ❌ Éviter si:
- Sweeps <100 combinaisons (overhead inutile)
- RAM <16GB (risque saturation)
- CPU <8 cores (trop de context switching)
- Backtests courts (<7 jours, trop rapide)

---

## 📊 Performances Attendues

| Config | Combos/s | Speedup vs Auto |
|--------|----------|----------------|
| Auto (4-8 workers) | ~1.2 | 1x (référence) |
| Balanced (8 workers) | ~2.5 | 2x |
| Aggressive (16 workers) | ~5.0 | 4x |
| **Manuel_30 (30 workers)** | **~10.0** | **8-10x** |
| Extreme (64 workers) | ~12.0 | 10-12x |

---

## 🐛 Troubleshooting

### Erreur: "Impossible charger preset 'manuel_30'"

**Cause**: Fichier `execution_presets.toml` introuvable

**Solution**:
```python
# Vérifier chemin
from pathlib import Path
from threadx.optimization.presets.ranges import EXECUTION_PRESETS_FILE

print(f"Fichier preset: {EXECUTION_PRESETS_FILE}")
print(f"Existe: {EXECUTION_PRESETS_FILE.exists()}")
```

### Warning: "Chunk GPU trop petit"

**Normal avec manuel_30**: Si dataset <50k lignes, chunks GPU peuvent être petits.

**Solution**: Augmenter durée backtest ou utiliser preset moins agressif.

---

## 📝 Résumé

```python
# ✅ UTILISATION SIMPLE (RECOMMANDÉE)
runner = SweepRunner(preset='manuel_30')
results = runner.run_grid(...)

# 🎛️ OVERRIDE PARTIEL
runner = SweepRunner(preset='manuel_30', max_workers=20)

# 📊 MONITORING AUTO
# Les logs affichent automatiquement CPU/RAM/GPU usage

# 🚀 RÉSULTAT: 8-10x speedup vs auto
```

---

**Auteur**: ThreadX Framework
**Version**: Phase 3 Complete
**Fichier**: `PRESET_MANUEL_30_GUIDE.md`
