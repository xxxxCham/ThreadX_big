#!/usr/bin/env python3
"""Test du pipeline TokenDiversityDataSource - Option B"""

import sys
from pathlib import Path

# Ajout du chemin source
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_pipeline():
    print("=== Test Pipeline TokenDiversityDataSource - Option B ===")

    try:
        from threadx.data.unified_diversity_pipeline import (
            run_unified_diversity,
        )

        print("✓ Import pipeline réussi")

        # Test basique
        print("\n1. Test symbole unique BTCUSDT")
        results = run_unified_diversity(
            symbols=["BTCUSDT"], timeframe="1h", lookback_days=1, save_artifacts=False
        )

        print(f"✓ OHLCV récupéré: {len(results['ohlcv_data'])} symboles")
        print(f"✓ Indicateurs calculés: {len(results['indicators_data'])} symboles")
        print(f"✓ Métriques diversité: {len(results['diversity_metrics'])} entrées")
        print(f"✓ Durée: {results['metadata']['duration_seconds']:.2f}s")

        # Vérification structure OHLCV
        btc_data = results["ohlcv_data"]["BTCUSDT"]
        print(
            f"✓ Données BTC: {len(btc_data)} rows, colonnes: {list(btc_data.columns)}"
        )

        print("\n2. Test groupe L1")
        results2 = run_unified_diversity(
            groups=["L1"], timeframe="4h", lookback_days=2, save_artifacts=False
        )

        print(f"✓ Groupe L1: {len(results2['ohlcv_data'])} symboles traités")
        print(f"✓ Métriques: {len(results2['diversity_metrics'])} entrées")

        print("\n=== Tests réussis! ===")
        return True

    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
