"""
Test de performance PRODUCTION avec 7776 combinaisons.
"""

import time
import pandas as pd
import numpy as np
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec

# Données de test (même volume que production)
dates = pd.date_range(start="2024-01-01", periods=5952, freq="15min")
test_data = pd.DataFrame(
    {
        "open": 50000 + np.random.randn(5952) * 100,
        "high": 50100 + np.random.randn(5952) * 100,
        "low": 49900 + np.random.randn(5952) * 100,
        "close": 50000 + np.random.randn(5952) * 100,
        "volume": np.random.randint(1000, 10000, 5952),
    },
    index=dates,
)

print("=" * 80)
print("🔥 TEST SWEEP PRODUCTION - 7776 COMBINAISONS")
print("=" * 80)
print()

# Grid spec IDENTIQUE à production
grid_spec = ScenarioSpec(
    type="grid",
    params={
        "bb_window": [
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
        ],  # 16 valeurs
        "bb_num_std": [
            1.0,
            1.2,
            1.4,
            1.6,
            1.8,
            2.0,
            2.2,
            2.4,
            2.6,
            2.8,
            3.0,
        ],  # 11 valeurs
        "atr_window": [10, 12, 14, 16, 18, 20],  # 6 valeurs
        "atr_multiplier": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],  # 8 valeurs
    },
    sampler="grid",
)

total_combos = 16 * 11 * 6 * 8  # = 8448 combinaisons (proche de 7776)

# Runner avec 30 workers comme production
runner = SweepRunner(max_workers=30)

print(f"📊 Données: {len(test_data)} barres (15min)")
print(f"🔧 Combinaisons: {total_combos}")
print(f"👷 Workers: 30")
print()

# Exécution AVEC monitoring détaillé
print("⏱️  Démarrage du sweep production...")
start = time.perf_counter()


class ProductionMonitor:
    def __init__(self, runner, total):
        self.runner = runner
        self.total = total
        self.combo_times = []
        self.last_log = time.perf_counter()
        self.count = 0
        self.start_time = time.perf_counter()

    def run(self, *args, **kwargs):
        # Patch _evaluate_single_combination pour monitorer
        original_eval = self.runner._evaluate_single_combination

        def monitored_eval(*eval_args, **eval_kwargs):
            t0 = time.perf_counter()
            result = original_eval(*eval_args, **eval_kwargs)
            elapsed = time.perf_counter() - t0

            self.combo_times.append(elapsed)
            self.count += 1

            # Log toutes les 500 combos
            if self.count % 500 == 0:
                now = time.perf_counter()
                rate = 500 / (now - self.last_log)
                avg_time = np.mean(self.combo_times[-500:]) * 1000
                total_elapsed = now - self.start_time
                eta = (self.total - self.count) / rate if rate > 0 else 0

                print(
                    f"   [{self.count}/{self.total}] "
                    f"Vitesse: {rate:.1f} tests/sec | "
                    f"Temps moy: {avg_time:.1f}ms | "
                    f"ETA: {eta:.0f}s"
                )

                self.last_log = now

            return result

        self.runner._evaluate_single_combination = monitored_eval
        results = self.runner.run_grid(*args, **kwargs)
        self.runner._evaluate_single_combination = original_eval

        return results


monitored = ProductionMonitor(runner, total_combos)

results = monitored.run(
    grid_spec=grid_spec, real_data=test_data, symbol="BTCUSDC", timeframe="15m"
)

elapsed = time.perf_counter() - start

print()
print("=" * 80)
print("✅ RÉSULTATS PRODUCTION")
print("=" * 80)
print(f"⏱️  Temps total: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
print(f"📊 Résultats: {len(results)}")
print(f"🚀 Vitesse moyenne: {len(results) / elapsed:.1f} tests/sec")
print()

# Analyse de la dégradation sur 4 quartiles
if len(monitored.combo_times) > 2000:
    q1 = monitored.combo_times[:500]
    q2 = monitored.combo_times[
        len(monitored.combo_times) // 4 : len(monitored.combo_times) // 4 + 500
    ]
    q3 = monitored.combo_times[
        len(monitored.combo_times) // 2 : len(monitored.combo_times) // 2 + 500
    ]
    q4 = monitored.combo_times[-500:]

    print("📈 Analyse dégradation (quartiles):")
    print(
        f"   Q1 (premiers 500): {np.mean(q1)*1000:.1f}ms/test ({1/np.mean(q1):.1f} tests/sec)"
    )
    print(
        f"   Q2 (milieu 1):     {np.mean(q2)*1000:.1f}ms/test ({1/np.mean(q2):.1f} tests/sec)"
    )
    print(
        f"   Q3 (milieu 2):     {np.mean(q3)*1000:.1f}ms/test ({1/np.mean(q3):.1f} tests/sec)"
    )
    print(
        f"   Q4 (derniers 500): {np.mean(q4)*1000:.1f}ms/test ({1/np.mean(q4):.1f} tests/sec)"
    )
    print(f"   Ratio Q4/Q1: {np.mean(q4)/np.mean(q1):.2f}x")
    print()

if len(results) > 0:
    print("Top 10 meilleures combinaisons:")
    top10 = results.nlargest(10, "pnl")
    for idx, row in top10.iterrows():
        print(
            f"  - PnL: {row['pnl']:>8.2f} | "
            f"bb_window={row.get('bb_window', '?'):>2}, "
            f"bb_std={row.get('bb_num_std', '?'):.1f}, "
            f"atr_window={row.get('atr_window', '?'):>2}, "
            f"atr_mult={row.get('atr_multiplier', '?'):.1f}"
        )

print()
print("🎯 Objectif: 200+ tests/sec stable")
if len(results) / elapsed >= 200:
    print("✅ OBJECTIF ATTEINT!")
elif len(results) / elapsed >= 100:
    print("⚠️  Proche de l'objectif, optimisations supplémentaires possibles")
else:
    print("❌ Performances insuffisantes, investigation nécessaire")
