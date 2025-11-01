"""
Test simplifié du sweep réel avec 81 combinaisons.
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
print("🚀 TEST SWEEP RÉEL - 81 COMBINAISONS")
print("=" * 80)
print()

# Grid spec
grid_spec = ScenarioSpec(
    type="grid",
    params={
        "bb_window": [10, 20, 30],
        "bb_num_std": [1.5, 2.0, 2.5],
        "atr_window": [10, 14, 20],
        "atr_multiplier": [1.5, 2.0, 2.5],
    },
    sampler="grid",
)

# Runner
runner = SweepRunner(max_workers=8)

print(f"📊 Données: {len(test_data)} barres")
print(f"🔧 Combinaisons: {3*3*3*3} = 81")
print()

# Exécution
print("⏱️  Démarrage...")
start = time.perf_counter()

results = runner.run_grid(
    grid_spec=grid_spec, real_data=test_data, symbol="TEST", timeframe="1h"
)

elapsed = time.perf_counter() - start

print()
print("=" * 80)
print("✅ RÉSULTATS")
print("=" * 80)
print(f"⏱️  Temps total: {elapsed:.2f}s")
print(f"📊 Résultats: {len(results)}")
print(f"🚀 Vitesse: {len(results) / elapsed:.1f} tests/sec")
print()

if len(results) > 0:
    print("Top 5 meilleures combinaisons:")
    top5 = results.nlargest(5, "pnl")
    for idx, row in top5.iterrows():
        print(
            f"  - PnL: {row['pnl']:.2f}, bb={row.get('bb_window', '?')}, std={row.get('bb_num_std', '?')}"
        )
