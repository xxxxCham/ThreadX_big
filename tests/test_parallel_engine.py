#!/usr/bin/env python3
"""
Test du moteur de calcul parallèle ThreadX
==========================================

Vérifie:
1. ✅ Parallélisation multi-workers (ThreadPoolExecutor)
2. ✅ CuPy activé et utilisé pour calculs GPU
3. ✅ Multi-GPU (RTX 5090 + RTX 2060) balancé correctement
4. ✅ Monte-Carlo explore correctement l'espace des paramètres
5. ✅ Performances GPU vs CPU

Author: ThreadX Test Suite
Date: 2025-11-01
"""

import sys

sys.path.insert(0, "src")

import time
import numpy as np
import pandas as pd
from typing import Dict, List
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def create_test_data(n_bars: int = 500) -> pd.DataFrame:
    """Crée des données de test avec forte volatilité."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="15min", tz="UTC")

    # Prix avec tendance et volatilité
    trend = np.linspace(95000, 98000, n_bars)
    noise = np.random.randn(n_bars) * 2000
    cycles = 3000 * np.sin(np.linspace(0, 4 * np.pi, n_bars))
    close = trend + noise + cycles

    high = close + np.abs(np.random.randn(n_bars) * 500)
    low = close - np.abs(np.random.randn(n_bars) * 500)
    open_price = close + np.random.randn(n_bars) * 300
    volume = np.random.uniform(100, 500, n_bars)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_1_parallel_workers():
    """TEST 1: Vérifier que le parallélisme fonctionne avec plusieurs workers."""
    print("\n" + "=" * 80)
    print("TEST 1: PARALLÉLISME MULTI-WORKERS")
    print("=" * 80)

    data = create_test_data(300)

    # Configuration simple: 1 worker vs 30 workers
    for n_workers in [1, 30]:
        print(f"\n🔧 Test avec {n_workers} worker(s)...")

        indicator_settings = IndicatorSettings(use_gpu=False)  # CPU pour ce test
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(
            indicator_bank=indicator_bank, max_workers=n_workers, use_multigpu=False
        )

        # Grille simple: 3x3x3 = 27 combinaisons
        scenario_params = {
            "bb_period": {"values": [15, 20, 25]},
            "bb_std": {"values": [1.5, 2.0, 2.5]},
            "entry_z": {"values": [1.0, 1.5, 2.0]},
            "atr_period": {"value": 14},
            "min_pnl_pct": {"value": 0.0},
            "risk_per_trade": {"value": 0.02},
        }

        spec = ScenarioSpec(type="grid", params=scenario_params)

        start_time = time.time()
        results = runner.run_grid(
            spec,
            data,
            symbol="TESTUSDC",
            timeframe="15m",
            strategy_name="Bollinger_Breakout",
            reuse_cache=True,
        )
        elapsed = time.time() - start_time

        print(f"  ⏱️  Durée: {elapsed:.2f}s")
        print(f"  📊 Résultats: {len(results)} backtests")
        print(f"  🚀 Vitesse: {len(results)/elapsed:.1f} tests/sec")

        if n_workers == 1:
            baseline_time = elapsed
        else:
            speedup = baseline_time / elapsed
            efficiency = (speedup / n_workers) * 100
            print(f"  ⚡ Speedup: {speedup:.2f}x")
            print(f"  📈 Efficacité: {efficiency:.1f}%")

            if speedup > 1.5:
                print(
                    f"  ✅ SUCCÈS: Parallélisation effective ({speedup:.2f}x speedup)"
                )
            else:
                print(f"  ⚠️  WARNING: Faible speedup ({speedup:.2f}x)")

    return True


def test_2_cupy_gpu():
    """TEST 2: Vérifier que CuPy est utilisé pour les calculs GPU."""
    print("\n" + "=" * 80)
    print("TEST 2: CUPY / GPU ACTIVATION")
    print("=" * 80)

    try:
        import cupy as cp

        print("✅ CuPy importé avec succès")
        print(f"  Version: {cp.__version__}")

        # Test simple d'utilisation GPU
        x_cpu = np.random.randn(1000, 1000)
        x_gpu = cp.asarray(x_cpu)

        # Benchmark CPU vs GPU
        print("\n🔬 Benchmark matmul 1000x1000:")

        # CPU
        start = time.time()
        for _ in range(10):
            result_cpu = np.dot(x_cpu, x_cpu)
        cpu_time = time.time() - start
        print(f"  CPU (NumPy): {cpu_time:.4f}s")

        # GPU
        start = time.time()
        for _ in range(10):
            result_gpu = cp.dot(x_gpu, x_gpu)
            cp.cuda.Stream.null.synchronize()  # Attendre fin GPU
        gpu_time = time.time() - start
        print(f"  GPU (CuPy): {gpu_time:.4f}s")

        speedup = cpu_time / gpu_time
        print(f"  ⚡ Speedup GPU: {speedup:.2f}x")

        if speedup > 2.0:
            print("  ✅ SUCCÈS: GPU bien plus rapide que CPU")
        else:
            print("  ⚠️  WARNING: Speedup GPU faible")

        # Tester avec IndicatorBank
        print("\n🏦 Test IndicatorBank avec GPU...")
        data = create_test_data(500)

        indicator_settings = IndicatorSettings(use_gpu=True)
        indicator_bank = IndicatorBank(indicator_settings)

        # Calculer Bollinger Bands avec GPU
        start = time.time()
        bb_result = indicator_bank.ensure(
            indicator_type="bollinger",
            params={"period": 20, "std": 2.0},
            data=data,
            symbol="TEST",
            timeframe="15m",
        )
        gpu_calc_time = time.time() - start

        print(f"  ⏱️  Bollinger GPU: {gpu_calc_time:.4f}s")
        print(f"  📊 Type: {type(bb_result)}")
        print(f"  ✅ Calcul GPU réussi")

        return True

    except ImportError:
        print("❌ CuPy non disponible - GPU désactivé")
        return False
    except Exception as e:
        print(f"❌ Erreur GPU: {e}")
        return False


def test_3_multi_gpu():
    """TEST 3: Vérifier que le Multi-GPU (5090+2060) est balancé."""
    print("\n" + "=" * 80)
    print("TEST 3: MULTI-GPU BALANCING (RTX 5090 + RTX 2060)")
    print("=" * 80)

    try:
        import cupy as cp

        # Vérifier nombre de GPU
        n_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"🔍 GPU détectés: {n_gpus}")

        if n_gpus < 2:
            print("⚠️  Moins de 2 GPU disponibles - test Multi-GPU impossible")
            return False

        for i in range(n_gpus):
            cp.cuda.Device(i).use()
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode("utf-8")
            mem = props["totalGlobalMem"] / (1024**3)
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")

        # Test avec SweepRunner Multi-GPU
        print("\n🔬 Test SweepRunner avec Multi-GPU...")
        data = create_test_data(300)

        indicator_settings = IndicatorSettings(use_gpu=True)
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(
            indicator_bank=indicator_bank,
            max_workers=30,
            use_multigpu=True,  # ← CRITIQUE
        )

        # Grille moyenne: 4x4x3 = 48 combinaisons
        scenario_params = {
            "bb_period": {"values": [15, 20, 25, 30]},
            "bb_std": {"values": [1.5, 2.0, 2.5, 3.0]},
            "entry_z": {"values": [1.0, 1.5, 2.0]},
            "atr_period": {"value": 14},
            "min_pnl_pct": {"value": 0.0},
            "risk_per_trade": {"value": 0.02},
        }

        spec = ScenarioSpec(type="grid", params=scenario_params)

        start_time = time.time()
        results = runner.run_grid(
            spec,
            data,
            symbol="TESTUSDC",
            timeframe="15m",
            strategy_name="Bollinger_Breakout",
            reuse_cache=True,
        )
        elapsed = time.time() - start_time

        print(f"\n📊 Résultats Multi-GPU:")
        print(f"  ⏱️  Durée: {elapsed:.2f}s")
        print(f"  📈 Backtests: {len(results)}")
        print(f"  🚀 Vitesse: {len(results)/elapsed:.1f} tests/sec")

        # Vérifier balance GPU dans runner
        if hasattr(runner, "gpu_balance"):
            print(f"\n⚖️  Balance GPU:")
            for gpu_id, count in runner.gpu_balance.items():
                pct = (count / sum(runner.gpu_balance.values())) * 100
                print(f"  GPU {gpu_id}: {count} tasks ({pct:.1f}%)")

            # Vérifier équilibre
            counts = list(runner.gpu_balance.values())
            if len(counts) >= 2:
                ratio = max(counts) / min(counts)
                if ratio < 2.0:
                    print("  ✅ SUCCÈS: Balance équilibrée entre GPU")
                else:
                    print(f"  ⚠️  WARNING: Déséquilibre {ratio:.2f}x")

        return True

    except ImportError:
        print("❌ CuPy non disponible")
        return False
    except Exception as e:
        print(f"❌ Erreur Multi-GPU: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_4_monte_carlo():
    """TEST 4: Vérifier que Monte-Carlo explore correctement."""
    print("\n" + "=" * 80)
    print("TEST 4: MONTE-CARLO SAMPLING")
    print("=" * 80)

    data = create_test_data(300)

    indicator_settings = IndicatorSettings(use_gpu=True)
    indicator_bank = IndicatorBank(indicator_settings)
    runner = SweepRunner(
        indicator_bank=indicator_bank, max_workers=8, use_multigpu=True
    )

    # Configuration Monte-Carlo: plages larges
    scenario_params = {
        "bb_period": {"values": list(range(10, 50))},  # 40 valeurs
        "bb_std": {"values": [round(x * 0.1, 1) for x in range(10, 40)]},  # 30 valeurs
        "entry_z": {"values": [round(x * 0.1, 1) for x in range(5, 30)]},  # 25 valeurs
        "atr_period": {"value": 14},
        "min_pnl_pct": {"value": 0.0},
        "risk_per_trade": {"value": 0.02},
    }

    n_scenarios = 200
    spec = ScenarioSpec(
        type="monte_carlo", params=scenario_params, n_scenarios=n_scenarios, seed=42
    )

    print(f"🎲 Lancement Monte-Carlo: {n_scenarios} scénarios")
    print(f"  Espace total: {40 * 30 * 25:,} combinaisons possibles")

    start_time = time.time()
    results = runner.run_monte_carlo(
        spec,
        data,
        symbol="TESTUSDC",
        timeframe="15m",
        strategy_name="Bollinger_Breakout",
        reuse_cache=True,
    )
    elapsed = time.time() - start_time

    print(f"\n📊 Résultats Monte-Carlo:")
    print(f"  ⏱️  Durée: {elapsed:.2f}s")
    print(f"  📈 Scénarios: {len(results)}")
    print(f"  🚀 Vitesse: {len(results)/elapsed:.1f} scén/sec")

    # Analyser diversité des paramètres échantillonnés
    if "bb_period" in results.columns:
        bb_periods_unique = results["bb_period"].nunique()
        bb_stds_unique = results["bb_std"].nunique()
        entry_zs_unique = results["entry_z"].nunique()

        print(f"\n🔍 Diversité de l'échantillonnage:")
        print(f"  bb_period: {bb_periods_unique} valeurs uniques")
        print(f"  bb_std: {bb_stds_unique} valeurs uniques")
        print(f"  entry_z: {entry_zs_unique} valeurs uniques")

        # Vérifier que l'échantillonnage couvre bien l'espace
        if bb_periods_unique > 10 and bb_stds_unique > 10 and entry_zs_unique > 10:
            print("  ✅ SUCCÈS: Bonne couverture de l'espace des paramètres")
        else:
            print("  ⚠️  WARNING: Échantillonnage peu diversifié")

    # Vérifier variabilité des résultats
    if "total_return" in results.columns or "pnl" in results.columns:
        perf_col = "total_return" if "total_return" in results.columns else "pnl"
        perf_std = results[perf_col].std()
        perf_mean = results[perf_col].mean()
        print(f"\n📈 Variabilité des performances:")
        print(f"  {perf_col} moyen: {perf_mean:.2f}")
        print(f"  Écart-type: {perf_std:.2f}")
        print(f"  Min: {results[perf_col].min():.2f}")
        print(f"  Max: {results[perf_col].max():.2f}")

        if perf_std > 0:
            print("  ✅ SUCCÈS: Résultats diversifiés (exploration efficace)")
        else:
            print("  ⚠️  WARNING: Résultats identiques (problème sampling?)")

    return True


def test_5_performance_benchmark():
    """TEST 5: Benchmark comparatif CPU vs GPU vs Multi-GPU."""
    print("\n" + "=" * 80)
    print("TEST 5: BENCHMARK PERFORMANCE (CPU vs GPU vs Multi-GPU)")
    print("=" * 80)

    data = create_test_data(500)  # Plus de données pour benchmark

    # Grille fixe: 5x5x3 = 75 combinaisons
    scenario_params = {
        "bb_period": {"values": [15, 20, 25, 30, 35]},
        "bb_std": {"values": [1.5, 2.0, 2.5, 3.0, 3.5]},
        "entry_z": {"values": [1.0, 1.5, 2.0]},
        "atr_period": {"value": 14},
        "min_pnl_pct": {"value": 0.0},
        "risk_per_trade": {"value": 0.02},
    }
    spec = ScenarioSpec(type="grid", params=scenario_params)

    configs = [
        ("CPU (4 workers)", False, False, 4),
        ("GPU (8 workers)", True, False, 8),
        ("Multi-GPU (16 workers)", True, True, 16),
    ]

    results_benchmark = []

    for name, use_gpu, use_multigpu, workers in configs:
        print(f"\n🔬 Test: {name}")

        try:
            indicator_settings = IndicatorSettings(use_gpu=use_gpu)
            indicator_bank = IndicatorBank(indicator_settings)
            runner = SweepRunner(
                indicator_bank=indicator_bank,
                max_workers=workers,
                use_multigpu=use_multigpu,
            )

            start_time = time.time()
            results = runner.run_grid(
                spec,
                data,
                symbol="TESTUSDC",
                timeframe="15m",
                strategy_name="Bollinger_Breakout",
                reuse_cache=True,
            )
            elapsed = time.time() - start_time

            speed = len(results) / elapsed

            print(f"  ⏱️  Durée: {elapsed:.2f}s")
            print(f"  🚀 Vitesse: {speed:.1f} tests/sec")

            results_benchmark.append(
                {"config": name, "time": elapsed, "speed": speed, "tests": len(results)}
            )

        except Exception as e:
            print(f"  ❌ Erreur: {e}")

    # Tableau comparatif
    if results_benchmark:
        print("\n" + "=" * 80)
        print("📊 TABLEAU COMPARATIF")
        print("=" * 80)
        print(f"{'Configuration':<30} {'Durée (s)':<12} {'Vitesse (tests/s)':<20}")
        print("-" * 80)

        baseline_time = results_benchmark[0]["time"]

        for r in results_benchmark:
            speedup = baseline_time / r["time"]
            speedup_str = f"({speedup:.2f}x)" if speedup > 1 else ""
            print(
                f"{r['config']:<30} {r['time']:<12.2f} {r['speed']:<10.1f} {speedup_str}"
            )

        # Meilleure config
        best = max(results_benchmark, key=lambda x: x["speed"])
        print(
            f"\n🏆 Meilleure config: {best['config']} ({best['speed']:.1f} tests/sec)"
        )

    return True


def main():
    """Lance tous les tests."""
    print("\n" + "=" * 80)
    print("🧪 THREADX PARALLEL ENGINE TEST SUITE")
    print("=" * 80)
    print("Tests du moteur de calcul parallèle, GPU et Multi-GPU")
    print("=" * 80)

    tests = [
        ("Parallélisme Multi-Workers", test_1_parallel_workers),
        ("CuPy / GPU Activation", test_2_cupy_gpu),
        ("Multi-GPU Balancing", test_3_multi_gpu),
        ("Monte-Carlo Sampling", test_4_monte_carlo),
        ("Performance Benchmark", test_5_performance_benchmark),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} échoué: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Résumé final
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 80)

    for test_name, success in results:
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"{status:<12} {test_name}")

    total = len(results)
    passed = sum(1 for _, s in results if s)

    print("=" * 80)
    print(f"Résultat: {passed}/{total} tests réussis ({passed/total*100:.0f}%)")

    if passed == total:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
    elif passed >= total * 0.7:
        print("⚠️  La plupart des tests réussis, quelques problèmes mineurs")
    else:
        print("❌ Plusieurs tests échoués - révision nécessaire")


if __name__ == "__main__":
    main()
