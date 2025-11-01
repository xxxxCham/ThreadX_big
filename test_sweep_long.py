"""
Test de performance sur un sweep plus long pour identifier la dégradation.
"""

import time
import pandas as pd
import numpy as np
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec

# Données de test
dates = pd.date_range(start="2024-01-01", periods=5000, freq="1h")
test_data = pd.DataFrame(
    {
        "open": 50000 + np.random.randn(5000) * 100,
        "high": 50100 + np.random.randn(5000) * 100,
        "low": 49900 + np.random.randn(5000) * 100,
        "close": 50000 + np.random.randn(5000) * 100,
        "volume": np.random.randint(1000, 10000, 5000),
    },
    index=dates,
)

print("=" * 80)
print("🧪 TEST SWEEP LONG - SURVEILLANCE DÉGRADATION")
print("=" * 80)
print()

# Grid spec PLUS LARGE pour voir la dégradation
grid_spec = ScenarioSpec(
    type="grid",
    params={
        "bb_window": [10, 15, 20, 25, 30, 35, 40],  # 7 valeurs
        "bb_num_std": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5 valeurs
        "atr_window": [10, 14, 20],  # 3 valeurs
        "atr_multiplier": [1.0, 1.5, 2.0],  # 3 valeurs
    },
    sampler="grid",
)

total_combos = 7 * 5 * 3 * 3  # = 315 combinaisons

# Runner
runner = SweepRunner(max_workers=8)

print(f"📊 Données: {len(test_data)} barres")
print(f"🔧 Combinaisons: {total_combos}")
print()

# Exécution AVEC monitoring
print("⏱️  Démarrage avec monitoring...")
start = time.perf_counter()


# Wrap pour monitorer
class MonitoredRunner:
    def __init__(self, runner):
        self.runner = runner
        self.combo_times = []
        self.last_log = time.perf_counter()
        self.count = 0

    def run(self, *args, **kwargs):
        # Patch _evaluate_single_combination pour monitorer
        original_eval = self.runner._evaluate_single_combination

        def monitored_eval(*eval_args, **eval_kwargs):
            t0 = time.perf_counter()
            result = original_eval(*eval_args, **eval_kwargs)
            elapsed = time.perf_counter() - t0

            self.combo_times.append(elapsed)
            self.count += 1

            # Log toutes les 50 combos
            if self.count % 50 == 0:
                now = time.perf_counter()
                rate = 50 / (now - self.last_log)
                avg_time = np.mean(self.combo_times[-50:]) * 1000
                print(
                    f"   [{self.count}/{total_combos}] Vitesse: {rate:.1f} tests/sec, Temps moyen: {avg_time:.1f}ms"
                )
                self.last_log = now

            return result

        self.runner._evaluate_single_combination = monitored_eval
        results = self.runner.run_grid(*args, **kwargs)
        self.runner._evaluate_single_combination = original_eval

        return results


monitored = MonitoredRunner(runner)

results = monitored.run(
    grid_spec=grid_spec, real_data=test_data, symbol="TEST", timeframe="1h"
)

elapsed = time.perf_counter() - start

print()
print("=" * 80)
print("✅ RÉSULTATS")
print("=" * 80)
print(f"⏱️  Temps total: {elapsed:.2f}s")
print(f"📊 Résultats: {len(results)}")
print(f"🚀 Vitesse moyenne: {len(results) / elapsed:.1f} tests/sec")
print()

# Analyse de la dégradation
if len(monitored.combo_times) > 100:
    first_50 = monitored.combo_times[:50]
    last_50 = monitored.combo_times[-50:]

    print("📈 Analyse dégradation:")
    print(
        f"   Premier batch (1-50): {np.mean(first_50)*1000:.1f}ms/test ({1/np.mean(first_50):.1f} tests/sec)"
    )
    print(
        f"   Dernier batch: {np.mean(last_50)*1000:.1f}ms/test ({1/np.mean(last_50):.1f} tests/sec)"
    )
    print(f"   Ratio: {np.mean(last_50)/np.mean(first_50):.2f}x plus lent")
    print()

if len(results) > 0:
    print("Top 5 meilleures combinaisons:")
    top5 = results.nlargest(5, "pnl")
    for idx, row in top5.iterrows():
        print(
            f"  - PnL: {row['pnl']:.2f}, bb={row.get('bb_window', '?')}, std={row.get('bb_num_std', '?')}"
        )
