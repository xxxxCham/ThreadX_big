# 🚀 OPTIMISATIONS THREADX - PLAN D'IMPLÉMENTATION COMPLET

**Date**: 31 Octobre 2025
**Objectif**: Utiliser 100% puissance calcul (CPU 90%+, RAM 80%+, GPU1 80%+, GPU2 pleine utilisation)

---

## ✅ OPTIMISATIONS DÉJÀ IMPLÉMENTÉES

### 1. **Estimation Temps Réel** ⏱️
**Fichier**: `src/threadx/optimization/engine.py`

**Implémenté**:
- ✅ Méthode `_update_progress_estimation()` - Calcul vitesse traitement
- ✅ Historique glissant (10 derniers points)
- ✅ Estimation basée sur `scénarios/seconde`
- ✅ Format temps lisible (Xh Ym Zs)
- ✅ Affichage dans `_log_progress()`:
  ```
  📊 Progrès: 1500/10000 (15.0%) |
  ⏱️  Écoulé: 5m 30s |
  ⏳ Restant: 31m 15s |
  ⚡ Vitesse: 4.53 combos/s
  ```

**Variables ajoutées**:
```python
self.completed_scenarios = 0
self.last_progress_time = None
self.progress_history = []  # (timestamp, completed_count)
self.estimated_time_remaining = None
self.avg_scenario_time = None
```

---

### 2. **Preset Manuel 30 Workers** 🔧
**Fichier**: `src/threadx/optimization/presets/execution_presets.toml` (nouveau)

**Implémenté**:
- ✅ Fichier TOML avec presets workers
- ✅ Preset `workers.manuel_30`:
  ```toml
  [workers.manuel_30]
  max_workers = 30
  batch_size = 2000
  gpu_utilization_target = 0.85
  cpu_utilization_target = 0.90
  ram_utilization_target = 0.80
  ```
- ✅ Preset combiné `combined.manuel_30_full_power`:
  ```toml
  [combined.manuel_30_full_power]
  max_workers = 30
  batch_size = 2000
  gpu_target = 0.85
  cpu_target = 0.90
  ram_target = 0.80
  estimated_speedup = "5-10x vs défaut"
  ```

**Fichier**: `src/threadx/optimization/presets/ranges.py`

**Implémenté**:
- ✅ Fonction `load_execution_presets()` - Charge config TOML
- ✅ Fonction `get_execution_preset(preset_name)` - Récupère preset par nom
- ✅ Export dans `__all__`

**Usage**:
```python
from threadx.optimization.presets import get_execution_preset

preset = get_execution_preset("manuel_30")
runner = SweepRunner(max_workers=preset["max_workers"])
```

---

## 🔨 OPTIMISATIONS À IMPLÉMENTER (Phase 2)

### 3. **Graphique Résultats Backtests** 📊
**Fichier à créer**: `src/threadx/visualization/backtest_charts.py`

**Fonctionnalités requises**:
```python
def generate_backtest_chart(
    results_df: pd.DataFrame,
    ohlcv_data: pd.DataFrame,
    best_combo: Dict,
    symbol: str,
    timeframe: str,
    output_path: str = "backtest_results.html"
):
    """
    Génère graphique interactif des résultats backtest.

    Affiche:
    - Bougies japonaises (OHLCV) sur période complète
    - Signaux ENTRÉE (flèches vertes ▲)
    - Signaux SORTIE (flèches rouges ▼)
    - Équité curve (overlay ou subplot)
    - Indicateurs utilisés (Bollinger Bands, ATR, etc.)

    Librairie recommandée: Plotly (interactif HTML)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Subplot 1: Prix + Signaux
    # Subplot 2: Équité
    # Subplot 3: Indicateurs (optionnel)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Prix & Signaux', 'Équité', 'Indicateurs')
    )

    # Bougies
    fig.add_trace(go.Candlestick(
        x=ohlcv_data.index,
        open=ohlcv_data['open'],
        high=ohlcv_data['high'],
        low=ohlcv_data['low'],
        close=ohlcv_data['close'],
        name='Prix'
    ), row=1, col=1)

    # Entrées (récupérer depuis trades_history)
    entries = ...  # Extraire timestamps et prix d'entrée
    fig.add_trace(go.Scatter(
        x=entries.index,
        y=entries['price'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='green'),
        name='Entrées'
    ), row=1, col=1)

    # Sorties
    exits = ...  # Extraire timestamps et prix de sortie
    fig.add_trace(go.Scatter(
        x=exits.index,
        y=exits['price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        name='Sorties'
    ), row=1, col=1)

    # Équité curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['equity'],
        mode='lines',
        name='Équité',
        line=dict(color='blue', width=2)
    ), row=2, col=1)

    # Indicateurs (ex: Bollinger Bands)
    if 'bb_upper' in indicators:
        fig.add_trace(go.Scatter(
            x=indicators.index,
            y=indicators['bb_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ), row=1, col=1)

    # Sauvegarder
    fig.write_html(output_path)
    logger.info(f"Graphique sauvegardé: {output_path}")
```

**Intégration**:
- Appeler après `run_grid()` ou `run_monte_carlo()`
- Extraire meilleur combo depuis `results_df`
- Charger données OHLCV utilisées
- Récupérer `trades_history` depuis stratégie

---

### 4. **Optimisation Puissance Calcul** 🔥

#### A. **Augmenter Batch Size Dynamique**
**Fichier**: `src/threadx/optimization/engine.py`

**Changements**:
```python
# AVANT (ligne 478):
batch_size = 1000

# APRÈS:
if self.max_workers >= 30:
    batch_size = 2000  # Pour preset manuel_30
elif self.max_workers >= 16:
    batch_size = 1500
else:
    batch_size = 1000
```

**Impact**: Moins de overhead soumission futures, GPU mieux saturé

---

#### B. **Optimisation GPU Load Balancing**
**Fichier**: `src/threadx/gpu/multi_gpu.py`

**Problème actuel**:
- GPU1 (RTX 5090) utilise seulement 2.5GB / 16GB = 15%
- GPU2 (RTX 2060) peu utilisé
- Balance par défaut: 75% / 25%

**Solution**:

```python
# Dans MultiGPUManager.__init__()

# AVANT:
self.device_balance = {
    "5090": 0.75,
    "2060": 0.25
}

# APRÈS (plus agressif):
self.device_balance = {
    "5090": 0.80,  # Augmenter à 80%
    "2060": 0.20
}

# ET activer auto-balance dynamique:
if auto_optimize:
    optimal_ratios = self.profile_auto_balance(
        sample_size=100_000,
        warmup=3,
        runs=5
    )
    self.set_balance(optimal_ratios)
```

**Fichier**: `src/threadx/indicators/gpu_integration.py`

**Changements**:
```python
# Réduire seuil GPU (ligne ~65):
# AVANT:
self.min_samples_for_gpu = 1000

# APRÈS:
self.min_samples_for_gpu = 500  # Utiliser GPU plus tôt
```

---

#### C. **Augmenter Chunk Size Données**
**Fichier**: `src/threadx/gpu/multi_gpu.py`

**Dans `_split_workload()`**:
```python
# Augmenter taille minimale chunks pour mieux saturer VRAM

MIN_CHUNK_SIZE_GPU = 50_000  # Au lieu de 10_000

# Validation chunk size
for chunk in chunks:
    if chunk.expected_size < MIN_CHUNK_SIZE_GPU and device != 'cpu':
        logger.warning(
            f"Chunk trop petit ({chunk.expected_size}) "
            f"pour GPU {device}, risque sous-utilisation VRAM"
        )
```

---

#### D. **Parallélisme Indicateurs Bank**
**Fichier**: `src/threadx/indicators/bank.py`

**Augmenter workers batch**:
```python
# Ligne ~400 dans batch_ensure_indicators()

# AVANT:
max_workers = 4

# APRÈS:
import os
max_workers = os.cpu_count() or 8  # Utiliser tous les cores CPU
```

---

#### E. **Prefetch Données GPU**
**Fichier**: `src/threadx/gpu/multi_gpu.py`

**Dans `_compute_chunk()`**:
```python
# Ajouter pinned memory pour transferts asynchrones

import cupy as cp

# Allouer pinned memory pool
if not hasattr(cp.cuda, '_pinned_pool'):
    cp.cuda.set_pinned_memory_allocator(
        cp.cuda.PinnedMemoryPool().malloc
    )

# Dans transfert GPU:
with cp.cuda.Stream(non_blocking=True):
    device_data = cp.asarray(chunk_data, order='C')  # Contiguous
```

**Impact**: Overlap transferts CPU↔GPU avec compute

---

#### F. **Augmenter Workers Dynamiquement**
**Fichier**: `src/threadx/optimization/engine.py`

**Dans `__init__()`**:
```python
# Si preset manuel_30 détecté:
if max_workers >= 30:
    # Désactiver ajustement adaptatif workers
    self._adaptive_workers = False
    logger.info("Mode haute performance: 30 workers fixes")

    # Augmenter batch soumission
    self._submission_batch_size = 3000

    # Pre-warm GPU
    if self.use_multigpu and self.gpu_manager:
        logger.info("Pre-warming GPUs...")
        self.gpu_manager.profile_auto_balance(
            sample_size=50_000,
            warmup=5  # Plus de warmup pour stabilité
        )
```

---

#### G. **Monitoring Utilisation Ressources**
**Fichier nouveau**: `src/threadx/utils/resource_monitor.py`

```python
"""
Monitoring utilisation ressources en temps réel.
"""

import psutil
import time
from typing import Dict, Optional
import cupy as cp

def get_resource_usage() -> Dict[str, float]:
    """
    Récupère utilisation actuelle CPU/RAM/GPU.

    Returns:
        {
            'cpu_percent': 45.2,
            'ram_percent': 62.1,
            'ram_used_gb': 15.3,
            'gpu0_percent': 25.8,
            'gpu0_vram_used_gb': 2.5,
            'gpu1_percent': 8.3,
            'gpu1_vram_used_gb': 0.8
        }
    """
    stats = {}

    # CPU
    stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)

    # RAM
    mem = psutil.virtual_memory()
    stats['ram_percent'] = mem.percent
    stats['ram_used_gb'] = mem.used / (1024**3)

    # GPUs
    try:
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                mem_info = cp.cuda.runtime.memGetInfo()
                used = (mem_info[1] - mem_info[0]) / (1024**3)
                total = mem_info[1] / (1024**3)

                stats[f'gpu{i}_percent'] = (used / total) * 100
                stats[f'gpu{i}_vram_used_gb'] = used
                stats[f'gpu{i}_vram_total_gb'] = total
    except:
        pass

    return stats

def log_resource_usage(logger):
    """Log périodique des ressources."""
    stats = get_resource_usage()
    logger.info(
        f"💻 CPU: {stats.get('cpu_percent', 0):.1f}% | "
        f"🧠 RAM: {stats.get('ram_percent', 0):.1f}% "
        f"({stats.get('ram_used_gb', 0):.1f} GB) | "
        f"🎮 GPU0: {stats.get('gpu0_percent', 0):.1f}% "
        f"({stats.get('gpu0_vram_used_gb', 0):.1f} GB) | "
        f"🎮 GPU1: {stats.get('gpu1_percent', 0):.1f}% "
        f"({stats.get('gpu1_vram_used_gb', 0):.1f} GB)"
    )
```

**Intégration dans `engine.py`**:
```python
from threadx.utils.resource_monitor import log_resource_usage

# Dans boucle exécution (tous les 500 combos):
if completed_count[0] % 500 == 0:
    self._log_progress()
    log_resource_usage(self.logger)  # Monitoring ressources
```

---

## 📊 RÉSUMÉ DES GAINS ATTENDUS

| Optimisation | Impact CPU | Impact RAM | Impact GPU1 | Impact GPU2 | Speedup |
|-------------|-----------|-----------|------------|------------|---------|
| **30 workers** | +70% → 90% | +50% → 80% | +30% → 60% | +20% → 40% | 3-4x |
| **Batch 2000** | - | - | +20% → 80% | +20% → 60% | 1.5x |
| **Chunk size↑** | - | - | +15% | +15% | 1.2x |
| **Prefetch async** | - | - | +10% | +10% | 1.3x |
| **Total combiné** | **90%+** | **80%+** | **85%+** | **70%+** | **~8-10x** |

---

## 🚀 CHECKLIST D'IMPLÉMENTATION

### Phase 1 (Déjà fait) ✅
- [x] Estimation temps réel (engine.py)
- [x] Preset manuel_30 TOML (execution_presets.toml)
- [x] Fonctions load/get execution preset (ranges.py)
- [x] Batch sizing dynamique (engine.py: 30+→2000, 16+→1500)
- [x] GPU threshold réduit (gpu_integration.py: 1000→500)

### Phase 2 (Déjà fait) ✅
- [x] Créer `resource_monitor.py` avec get_resource_usage(), log_resource_usage(), get_utilization_score()
- [x] Intégrer monitoring dans _log_progress() (tous les 500 combos)
- [x] Warning si score d'utilisation < 50%
- [x] Créer `backtest_charts.py` avec generate_backtest_chart()
- [x] Implémenter graphiques Plotly (candlesticks + BB + entrées/sorties + équité)
- [x] Créer `visualization/__init__.py`

### Phase 3 (Implémenté maintenant) ✅ 🆕
- [x] **Workers IndicatorBank → cpu_count()** (bank.py: max_workers = None auto)
- [x] **MIN_CHUNK_SIZE_GPU = 50000** (multi_gpu.py: constante + validation)
- [x] **Auto-balance GPU startup** (engine.py: profile_auto_balance au __init__)
- [x] **ETA ajusté durée plage** (engine.py: facteur correction ×duration_days/30)
- [ ] Intégrer génération graphiques dans UI (après run_grid/monte_carlo)
- [ ] Implémenter pinned memory async transfers (CuPy allocator)
- [ ] Intégrer appel dans UI après run
- [ ] Ajouter bouton "Voir Graphique" Streamlit

---

## 🔧 COMMANDES D'APPLICATION RAPIDE

```bash
# 1. Appliquer preset manuel_30
# Dans votre code UI/CLI:
from threadx.optimization.presets import get_execution_preset
preset = get_execution_preset("manuel_30")
runner = SweepRunner(max_workers=preset["max_workers"])

# 2. Forcer auto-balance GPU
gpu_manager = get_default_manager()
optimal = gpu_manager.profile_auto_balance(sample_size=100_000, warmup=5, runs=5)
gpu_manager.set_balance(optimal)

# 3. Monitoring ressources
from threadx.utils.resource_monitor import log_resource_usage
log_resource_usage(logger)  # Appeler périodiquement
```

---

**Conclusion**: Optimisations 1-2 implémentées. Optimisations 3-4 nécessitent modifications additionnelles décrites ci-dessus.
