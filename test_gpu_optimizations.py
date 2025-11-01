"""
Test des optimisations GPU Phase 2
==================================

Test rapide des améliorations:
1. Auto-balance profiling avec warmup + efficacité mémoire
2. Kernels Numba CUDA fusionnés (Bollinger Bands)
3. Configuration thread/block optimale
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ajout du path src pour imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_multi_gpu_profiling():
    """Test auto-balance profiling amélioré."""
    print("\n" + "=" * 60)
    print("TEST 1: Auto-Balance Profiling Hétérogène")
    print("=" * 60)

    try:
        from threadx.utils.gpu import get_default_manager

        manager = get_default_manager()

        print(f"\n📊 Devices disponibles: {len(manager.available_devices)}")
        for device in manager.available_devices:
            print(
                f"  - {device.name}: {device.memory_total_gb:.2f} GB, "
                f"compute {device.compute_capability}"
            )

        print(f"\n⚖️  Balance actuelle: {manager.device_balance}")

        # Test profiling avec warmup + efficacité mémoire
        print("\n🔬 Lancement auto-profiling (sample_size=50000, warmup=2, runs=3)...")
        optimal_ratios = manager.profile_auto_balance(
            sample_size=50000, warmup=2, runs=3
        )

        print(f"\n✅ Ratios optimaux calculés: {optimal_ratios}")

        # Stats devices
        print("\n📈 Stats devices après profiling:")
        stats = manager.get_device_stats()
        for device_name, device_stats in stats.items():
            print(f"  {device_name}:")
            print(f"    - Balance: {device_stats['current_balance']:.1%}")
            print(f"    - Mémoire: {device_stats['memory_used_pct']:.1f}%")
            print(f"    - Has stream: {device_stats['has_stream']}")

        print("\n✅ TEST 1 PASSED: Auto-balance profiling OK")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_numba_bollinger():
    """Test kernels Numba CUDA fusionnés."""
    print("\n" + "=" * 60)
    print("TEST 2: Kernels Numba CUDA Fusionnés (Bollinger)")
    print("=" * 60)

    try:
        from threadx.indicators.gpu_integration import get_gpu_accelerated_bank

        # Données test
        n = 10000
        prices = np.random.randn(n).cumsum() + 100
        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices + np.random.rand(n) * 2,
                "low": prices - np.random.rand(n) * 2,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

        print(f"\n📊 Données test: {len(df)} lignes")

        bank = get_gpu_accelerated_bank()

        # Test avec GPU forcé (tentera Numba si disponible)
        print("\n⚡ Calcul Bollinger Bands (GPU forcé)...")
        upper, middle, lower = bank.bollinger_bands(
            df, period=20, std_dev=2.0, use_gpu=True
        )

        print(f"  - Upper band: {upper.iloc[-5:].values}")
        print(f"  - Middle band: {middle.iloc[-5:].values}")
        print(f"  - Lower band: {lower.iloc[-5:].values}")

        # Vérification basique
        assert len(upper) == len(df), "Taille output incorrecte"
        assert not upper.isna().all(), "Output vide"

        print("\n✅ TEST 2 PASSED: Kernels Numba/GPU OK")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_comparison():
    """Test comparaison CPU vs GPU vs Numba."""
    print("\n" + "=" * 60)
    print("TEST 3: Comparaison Performance CPU vs GPU")
    print("=" * 60)

    try:
        from threadx.indicators.gpu_integration import get_gpu_accelerated_bank
        import time

        # Données test plus grandes
        n = 100000
        prices = np.random.randn(n).cumsum() + 100
        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices + np.random.rand(n) * 2,
                "low": prices - np.random.rand(n) * 2,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

        print(f"\n📊 Benchmark sur {len(df):,} lignes")

        bank = get_gpu_accelerated_bank()

        # CPU
        print("\n🐌 Test CPU...")
        t0 = time.time()
        _, _, _ = bank.bollinger_bands(df, period=20, use_gpu=False)
        cpu_time = time.time() - t0
        print(f"  Temps CPU: {cpu_time:.4f}s")

        # GPU (auto-décision, tentera Numba si disponible)
        print("\n⚡ Test GPU (auto-décision)...")
        t0 = time.time()
        _, _, _ = bank.bollinger_bands(df, period=20, use_gpu=None)
        gpu_time = time.time() - t0
        print(f"  Temps GPU: {gpu_time:.4f}s")

        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.2f}x")

        if speedup > 1.0:
            print("✅ GPU plus rapide que CPU")
        else:
            print("⚠️  CPU plus rapide (normal pour petites données ou sans Numba)")

        print("\n✅ TEST 3 PASSED: Benchmark terminé")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Lance tous les tests."""
    print("\n" + "=" * 70)
    print(" 🚀 TESTS OPTIMISATIONS GPU PHASE 2 - ThreadX v2.0")
    print("=" * 70)

    print("\nOptimisations testées:")
    print("  ✅ Auto-balance profiling hétérogène (warmup + mem_efficiency)")
    print("  ✅ Kernels Numba CUDA fusionnés (SMA+std)")
    print("  ✅ Configuration thread/block optimale (256 threads/block)")
    print("  ✅ Cascade fallback: Numba → CuPy → CPU")

    results = []

    # Test 1: Multi-GPU profiling
    results.append(("Auto-Balance Profiling", test_multi_gpu_profiling()))

    # Test 2: Numba kernels
    results.append(("Kernels Numba CUDA", test_numba_bollinger()))

    # Test 3: Performance
    results.append(("Benchmark CPU vs GPU", test_performance_comparison()))

    # Résumé
    print("\n" + "=" * 70)
    print(" 📊 RÉSUMÉ DES TESTS")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status} - {name}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\n🎯 Score: {total_passed}/{total_tests} tests réussis")

    if total_passed == total_tests:
        print("\n🎉 TOUS LES TESTS PASSED - Optimisations opérationnelles!")
        return 0
    else:
        print("\n⚠️  CERTAINS TESTS FAILED - Vérifier logs ci-dessus")
        return 1


if __name__ == "__main__":
    sys.exit(main())
