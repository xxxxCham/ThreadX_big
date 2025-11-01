#!/usr/bin/env python3
"""
Test: GPU vs CPU - Impact sur strategy_evaluation
==================================================

Compare les performances avec/sans GPU pour identifier si les CUDA Streams
ralentissent le processus parallèle.
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
close = np.linspace(95000, 98000, 500) + np.random.randn(500) * 2000
df = pd.DataFrame(
    {
        "close": close,
        "high": close + 500,
        "low": close - 500,
        "open": close,
        "volume": 100,
    },
    index=dates,
)

# Scénario de test (27 combos)
scenario = ScenarioSpec(
    type="grid",
    params={
        "bb_period": {"values": [20, 25, 30]},
        "bb_std": {"values": [2.0, 2.5, 3.0]},
        "entry_z": {"values": [1.5, 2.0, 2.5]},
        "atr_period": {"value": 14},
    },
)

print("=" * 60)
print("🧪 TEST 1: GPU ACTIVÉ (use_multigpu=True)")
print("=" * 60)
runner1 = SweepRunner(max_workers=8, use_multigpu=True)
start1 = time.time()
results1 = runner1.run_grid(df, scenario, symbol="TEST", timeframe="15m")
dur1 = time.time() - start1
print(
    f"\n✅ {len(results1)} scénarios en {dur1:.2f}s = {len(results1)/dur1:.1f} tests/sec\n"
)

print("=" * 60)
print("🧪 TEST 2: GPU DÉSACTIVÉ (use_multigpu=False)")
print("=" * 60)
runner2 = SweepRunner(max_workers=8, use_multigpu=False)
start2 = time.time()
results2 = runner2.run_grid(df, scenario, symbol="TEST", timeframe="15m")
dur2 = time.time() - start2
print(
    f"\n✅ {len(results2)} scénarios en {dur2:.2f}s = {len(results2)/dur2:.1f} tests/sec\n"
)

# Comparaison
speedup = dur1 / dur2
print("=" * 60)
print("📊 COMPARAISON FINALE")
print("=" * 60)
print(f"GPU ON  (use_multigpu=True):  {dur1:.2f}s ({len(results1)/dur1:.1f} tests/sec)")
print(f"GPU OFF (use_multigpu=False): {dur2:.2f}s ({len(results2)/dur2:.1f} tests/sec)")
print(f"\nRatio GPU/CPU: {speedup:.2f}x")
if speedup < 1.0:
    print(f"✅ GPU plus rapide de {(1-speedup)*100:.1f}%")
elif speedup > 1.0:
    print(f"⚠️  CPU plus rapide de {(speedup-1)*100:.1f}% - GPU RALENTIT !")
else:
    print(f"⚖️  Performances identiques")
print("=" * 60)
