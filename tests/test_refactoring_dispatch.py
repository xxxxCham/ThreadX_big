#!/usr/bin/env python3
"""
Script de validation du refactoring pattern dispatch GPU/CPU.

Teste que les 3 indicateurs refactorés (bollinger_bands, atr, rsi)
fonctionnent correctement après centralisation du pattern dispatch.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import direct sans passer par __init__ pour éviter dépendances config
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gpu_integration", src_path / "threadx" / "indicators" / "gpu_integration.py"
)
gpu_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_module)

GPUAcceleratedIndicatorBank = gpu_module.GPUAcceleratedIndicatorBank


def create_test_data(n_rows: int = 1000) -> pd.DataFrame:
    """Crée des données de test réalistes."""
    np.random.seed(42)

    # Générer un prix de base avec tendance
    base_price = 100.0
    trend = np.linspace(0, 10, n_rows)
    noise = np.random.randn(n_rows) * 2

    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n_rows))
    low = close - np.abs(np.random.randn(n_rows))

    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_bollinger_bands():
    """Test Bollinger Bands après refactoring."""
    print("🔍 Test 1/3: Bollinger Bands")

    df = create_test_data(1000)
    bank = GPUAcceleratedIndicatorBank()

    try:
        upper, middle, lower = bank.bollinger_bands(df, period=20, std_dev=2.0)

        # Validations
        assert isinstance(upper, pd.Series), "Upper n'est pas une Series"
        assert isinstance(middle, pd.Series), "Middle n'est pas une Series"
        assert isinstance(lower, pd.Series), "Lower n'est pas une Series"

        assert len(upper) == len(df), f"Taille incorrecte: {len(upper)} vs {len(df)}"
        assert len(middle) == len(df), f"Taille incorrecte: {len(middle)} vs {len(df)}"
        assert len(lower) == len(df), f"Taille incorrecte: {len(lower)} vs {len(df)}"

        # Vérifier que upper > middle > lower (après warmup)
        valid_idx = middle.notna()
        assert (
            upper[valid_idx] >= middle[valid_idx]
        ).all(), "Upper doit être >= Middle"
        assert (
            middle[valid_idx] >= lower[valid_idx]
        ).all(), "Middle doit être >= Lower"

        print("  ✅ Bollinger Bands: OK")
        print(f"     - Upper: {upper.iloc[-1]:.2f}")
        print(f"     - Middle: {middle.iloc[-1]:.2f}")
        print(f"     - Lower: {lower.iloc[-1]:.2f}")
        return True

    except Exception as e:
        print(f"  ❌ Bollinger Bands: ÉCHEC - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_atr():
    """Test ATR après refactoring."""
    print("\n🔍 Test 2/3: ATR")

    df = create_test_data(1000)
    bank = GPUAcceleratedIndicatorBank()

    try:
        atr = bank.atr(df, period=14)

        # Validations
        assert isinstance(atr, pd.Series), "ATR n'est pas une Series"
        assert len(atr) == len(df), f"Taille incorrecte: {len(atr)} vs {len(df)}"

        # ATR doit être positif (après warmup)
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all(), "ATR doit être positif"

        print("  ✅ ATR: OK")
        print(f"     - Valeur finale: {atr.iloc[-1]:.4f}")
        print(f"     - Moyenne: {valid_atr.mean():.4f}")
        print(f"     - Max: {valid_atr.max():.4f}")
        return True

    except Exception as e:
        print(f"  ❌ ATR: ÉCHEC - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_rsi():
    """Test RSI après refactoring."""
    print("\n🔍 Test 3/3: RSI")

    df = create_test_data(1000)
    bank = GPUAcceleratedIndicatorBank()

    try:
        rsi = bank.rsi(df, period=14)

        # Validations
        assert isinstance(rsi, pd.Series), "RSI n'est pas une Series"
        assert len(rsi) == len(df), f"Taille incorrecte: {len(rsi)} vs {len(df)}"

        # RSI doit être entre 0 et 100 (après warmup)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI doit être >= 0"
        assert (valid_rsi <= 100).all(), "RSI doit être <= 100"

        print("  ✅ RSI: OK")
        print(f"     - Valeur finale: {rsi.iloc[-1]:.2f}")
        print(f"     - Moyenne: {valid_rsi.mean():.2f}")
        print(f"     - Min: {valid_rsi.min():.2f}, Max: {valid_rsi.max():.2f}")
        return True

    except Exception as e:
        print(f"  ❌ RSI: ÉCHEC - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dispatch_with_force_gpu():
    """Test que force_gpu fonctionne correctement."""
    print("\n🔍 Test 4/4: Force GPU/CPU")

    df = create_test_data(100)  # Petit dataset
    bank = GPUAcceleratedIndicatorBank()

    try:
        # Test avec force_gpu=True (petites données)
        rsi_force_gpu = bank.rsi(df, period=14, force_gpu=True)

        # Test avec force_gpu=False (grandes données)
        df_large = create_test_data(10000)
        rsi_force_cpu = bank.rsi(df_large, period=14, force_gpu=False)

        assert isinstance(
            rsi_force_gpu, pd.Series
        ), "RSI force_gpu n'est pas une Series"
        assert isinstance(
            rsi_force_cpu, pd.Series
        ), "RSI force_cpu n'est pas une Series"

        print("  ✅ Force GPU/CPU: OK")
        print(f"     - Force GPU (100 rows): {rsi_force_gpu.iloc[-1]:.2f}")
        print(f"     - Force CPU (10k rows): {rsi_force_cpu.iloc[-1]:.2f}")
        return True

    except Exception as e:
        print(f"  ❌ Force GPU/CPU: ÉCHEC - {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Exécute tous les tests."""
    print("=" * 70)
    print("🧪 VALIDATION REFACTORING PATTERN DISPATCH")
    print("=" * 70)
    print()

    results = []

    # Test 1: Bollinger Bands
    results.append(("Bollinger Bands", test_bollinger_bands()))

    # Test 2: ATR
    results.append(("ATR", test_atr()))

    # Test 3: RSI
    results.append(("RSI", test_rsi()))

    # Test 4: Force GPU/CPU
    results.append(("Force GPU/CPU", test_dispatch_with_force_gpu()))

    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Total: {passed}/{total} tests passés ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 TOUS LES TESTS PASSENT - REFACTORING VALIDÉ !")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) échoué(s) - VÉRIFIER LE CODE")
        return 1


if __name__ == "__main__":
    sys.exit(main())
