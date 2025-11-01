"""
Test de performance détaillé avec chronomètres
"""

import sys

sys.path.insert(0, "src")

import time
import numpy as np
import pandas as pd
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec

# Données de test
np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=500, freq="15min", tz="UTC")
df = pd.DataFrame(
    {
        "close": np.linspace(95000, 98000, 500) + np.random.randn(500) * 2000,
        "high": None,
        "low": None,
        "open": None,
        "volume": 100,
    },
    index=dates,
)
df["high"] = df["close"] + 500
df["low"] = df["close"] - 500
df["open"] = df["close"]

# Scénario
scenario = ScenarioSpec(
    type="grid",
    params={
        "bb_period": {"values": [15, 20, 25, 30, 35]},
        "bb_std": {"values": [1.5, 2.0, 2.5, 3.0, 3.5]},
        "entry_z": {"values": [1.0, 1.5, 2.0]},
        "atr_period": {"value": 14},
    },
)

print("\n" + "=" * 60)
print("📊 TEST DE PERFORMANCE DÉTAILLÉ")
print("=" * 60)

# Test avec état actuel
print("\n⏱️  TEST: Configuration actuelle (8 workers)")
runner = SweepRunner(max_workers=8, use_multigpu=False)

start = time.perf_counter()
results = runner.run_grid(scenario, df, "BTCUSDC", "15m")
duration = time.perf_counter() - start

print(f"\n✅ RÉSULTAT FINAL:")
print(f"   - Scénarios: {len(results)}")
print(f"   - Durée: {duration:.2f}s")
print(f"   - Vitesse: {len(results)/duration:.1f} tests/sec")
print(f"   - Temps moyen/test: {duration/len(results)*1000:.1f}ms")
print("=" * 60 + "\n")
