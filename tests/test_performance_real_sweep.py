"""
Test de performance réel avec sweep
Objectif: Mesurer CPU/RAM/GPU pendant exécution et vérifier les optimisations
"""

import time
import psutil
import os
from datetime import datetime
import pandas as pd
import numpy as np

print("=" * 80)
print("🚀 TEST PERFORMANCE RÉEL - SWEEP THREADX")
print("=" * 80)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💻 CPU cores: {os.cpu_count()}")
print(f"🧠 RAM total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print()

# ==================== Préparation données test ====================
print("📊 Préparation données de test...")

# Générer données synthétiques OHLCV
np.random.seed(42)
n_bars = 5000  # 5000 barres (environ 7 mois en 1h)

dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")
close_prices = 50000 + np.cumsum(
    np.random.randn(n_bars) * 100
)  # Random walk autour 50k
high_prices = close_prices + np.random.rand(n_bars) * 200
low_prices = close_prices - np.random.rand(n_bars) * 200
open_prices = close_prices + np.random.randn(n_bars) * 50
volume = np.random.randint(100, 10000, n_bars)

test_data = pd.DataFrame(
    {
        "timestamp": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    }
)
test_data.set_index("timestamp", inplace=True)

print(
    f"✅ Données créées: {len(test_data)} barres ({test_data.index[0]} → {test_data.index[-1]})"
)
print()

# ==================== Import ThreadX ====================
print("📦 Import ThreadX modules...")

try:
    from threadx.optimization.engine import SweepRunner
    from threadx.indicators.bank import IndicatorBank, IndicatorSettings
    from threadx.optimization.scenarios import ScenarioSpec

    print("✅ Imports réussis")
except Exception as e:
    print(f"❌ Erreur import: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print()

# ==================== Configuration Sweep ====================
print("⚙️  Configuration sweep...")

# Paramètres de recherche (réduits pour test rapide)
grid_spec = ScenarioSpec(
    type="grid",
    params={
        "bb_window": [10, 20, 30],  # 3 valeurs
        "bb_num_std": [1.5, 2.0, 2.5],  # 3 valeurs
        "atr_window": [10, 14, 20],  # 3 valeurs
        "atr_multiplier": [1.5, 2.0, 2.5],  # 3 valeurs
    },
    sampler="grid",
)

total_combos = 3 * 3 * 3 * 3  # = 81 combinaisons
print(f"   → Combinaisons à tester: {total_combos}")
print(f"   → BB windows: {grid_spec.params['bb_window']}")
print(f"   → BB std: {grid_spec.params['bb_num_std']}")
print(f"   → ATR windows: {grid_spec.params['atr_window']}")
print(f"   → ATR mult: {grid_spec.params['atr_multiplier']}")
print()

# ==================== Test AVANT optimisations (8 workers fixe) ====================
print("=" * 80)
print("🔴 TEST BASELINE: IndicatorBank avec max_workers=8 (fixe)")
print("=" * 80)

# Force 8 workers comme avant optimisation
settings_baseline = IndicatorSettings(max_workers=8, use_gpu=False)
bank_baseline = IndicatorBank(settings_baseline)

runner_baseline = SweepRunner(
    indicator_bank=bank_baseline, max_workers=8, use_multigpu=False
)

print(f"   → IndicatorBank workers: {bank_baseline.settings.max_workers}")
print(f"   → SweepRunner workers: {runner_baseline.max_workers}")
print()

# Monitoring ressources AVANT
cpu_before = psutil.cpu_percent(interval=1)
ram_before = psutil.virtual_memory().percent

print("🏁 Démarrage sweep baseline...")
start_time = time.time()

try:
    results_baseline = runner_baseline.run_grid(
        grid_spec=grid_spec, real_data=test_data, symbol="BTCUSDC_TEST", timeframe="1h"
    )

    baseline_time = time.time() - start_time

    # Monitoring ressources PENDANT (approximation finale)
    cpu_during = psutil.cpu_percent(interval=0.1)
    ram_during = psutil.virtual_memory().percent

    print(f"\n✅ Baseline terminé:")
    print(f"   ⏱️  Temps: {baseline_time:.2f}s")
    print(f"   💻 CPU: {cpu_during:.1f}%")
    print(f"   🧠 RAM: {ram_during:.1f}%")
    print(f"   📊 Résultats: {len(results_baseline)} valides")

except Exception as e:
    print(f"❌ Erreur baseline: {e}")
    import traceback

    traceback.print_exc()
    baseline_time = None

print()
time.sleep(2)  # Pause entre tests

# ==================== Test APRÈS optimisations (auto workers) ====================
print("=" * 80)
print("🟢 TEST OPTIMISÉ: IndicatorBank avec max_workers=None (auto)")
print("=" * 80)

# Auto workers (doit détecter 32 cores)
settings_optimized = IndicatorSettings(
    max_workers=None, use_gpu=False  # Auto = cpu_count()
)
bank_optimized = IndicatorBank(settings_optimized)

# Preset manuel_30 (30 workers)
runner_optimized = SweepRunner(
    indicator_bank=bank_optimized, preset="manuel_30", use_multigpu=False
)

print(f"   → IndicatorBank workers: {bank_optimized.settings.max_workers}")
print(f"   → SweepRunner workers: {runner_optimized.max_workers}")
print(f"   → Batch size: {runner_optimized.batch_size}")
print()

print("🏁 Démarrage sweep optimisé...")
start_time = time.time()

try:
    results_optimized = runner_optimized.run_grid(
        grid_spec=grid_spec, real_data=test_data, symbol="BTCUSDC_TEST", timeframe="1h"
    )

    optimized_time = time.time() - start_time

    # Monitoring ressources PENDANT
    cpu_during_opt = psutil.cpu_percent(interval=0.1)
    ram_during_opt = psutil.virtual_memory().percent

    print(f"\n✅ Optimisé terminé:")
    print(f"   ⏱️  Temps: {optimized_time:.2f}s")
    print(f"   💻 CPU: {cpu_during_opt:.1f}%")
    print(f"   🧠 RAM: {ram_during_opt:.1f}%")
    print(f"   📊 Résultats: {len(results_optimized)} valides")

except Exception as e:
    print(f"❌ Erreur optimisé: {e}")
    import traceback

    traceback.print_exc()
    optimized_time = None

print()

# ==================== Comparaison ====================
print("=" * 80)
print("📊 COMPARAISON BASELINE vs OPTIMISÉ")
print("=" * 80)

if baseline_time and optimized_time:
    speedup = baseline_time / optimized_time
    time_saved = baseline_time - optimized_time

    print(f"\n⏱️  TEMPS D'EXÉCUTION:")
    print(f"   Baseline (8 workers):    {baseline_time:.2f}s")
    print(
        f"   Optimisé ({bank_optimized.settings.max_workers} workers):    {optimized_time:.2f}s"
    )
    print(f"   💾 Temps gagné:          {time_saved:.2f}s")
    print(f"   🚀 Speedup:              {speedup:.2f}x")

    print(f"\n💻 UTILISATION RESSOURCES:")
    print(f"   Baseline CPU:  ~{cpu_during:.1f}%")
    print(f"   Optimisé CPU:  ~{cpu_during_opt:.1f}%")
    print(f"   Différence:    +{cpu_during_opt - cpu_during:.1f}%")

    print(f"\n🎯 OBJECTIFS ATTEINTS:")
    if speedup >= 2.0:
        print(f"   ✅ Speedup {speedup:.1f}x >= 2.0x attendu")
    else:
        print(f"   ⚠️  Speedup {speedup:.1f}x < 2.0x (attendu avec 32 cores)")

    if cpu_during_opt >= 70:
        print(f"   ✅ CPU {cpu_during_opt:.1f}% >= 70% (bien utilisé)")
    else:
        print(f"   ⚠️  CPU {cpu_during_opt:.1f}% < 70% (sous-utilisé)")

else:
    print("❌ Impossible de comparer (erreurs dans tests)")

print()
print("=" * 80)
print("✅ TEST TERMINÉ")
print("=" * 80)
