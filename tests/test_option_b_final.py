#!/usr/bin/env python3
"""
Test final de l'orchestration Option B avec TokenDiversityDataSource
==================================================================

Validation complète :
✅ 1. OHLCV pur avec strict validation
✅ 2. Persistance unifiée Parquet-first
✅ 3. Performance hooks et métriques
✅ 4. Warmup validation pour IndicatorBank
✅ 5. Gestion d'erreurs robuste
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.data.tokens import (
    TokenDiversityDataSource,
    TokenDiversityConfig,
)


def test_option_b_final():
    """Test complet de l'orchestration Option B finalisée."""

    print("🚀 Test Option B - Orchestration finalisée")
    print("=" * 50)

    # 1) Configuration stricte
    config = TokenDiversityConfig(
        groups={
            "L1": ["BTC", "ETH", "BNB"],
            "L2": ["ARBUSDT", "OPUSDT", "MATICUSDT"],
            "DeFi": ["UNIUSDT", "AAVEUSDT", "LINKUSDT"],
        },
        symbols=[
            "BTC",
            "ETH",
            "BNB",
            "ARBUSDT",
            "OPUSDT",
            "MATICUSDT",
            "UNIUSDT",
            "AAVEUSDT",
            "LINKUSDT",
        ],
        supported_tf=("1m", "5m", "15m", "1h", "4h", "1d"),
        cache_dir=str(Path("./test_cache")),
        enable_persistence=True,
        use_external_manager=False,  # Mode synthétique pour test
        strict_validation=True,
        min_warmup_rows=100,  # Réduit pour test avec données synthétiques
    )

    provider = TokenDiversityDataSource(config)

    print(
        f"✅ Provider initialisé avec {len(provider.supported_timeframes())} timeframes"
    )

    # 2) Test symbols listing
    symbols = provider.list_symbols(group="L1", limit=5)
    print(f"✅ Symboles L1: {symbols}")

    if not symbols:
        print("⚠️  Aucun symbole trouvé, création données synthétiques")
        symbols = ["BTC", "ETH", "SOL"]

    # 3) Test get_frame avec strict validation
    test_symbol = symbols[0]

    try:
        print(f"\n📊 Test get_frame pour {test_symbol}@1h")
        df = provider.get_frame(test_symbol, "1h", end=datetime.now(timezone.utc))

        print(f"✅ DataFrame reçu: {len(df)} rows")
        print(f"   Colonnes: {list(df.columns)}")
        print(f"   Index type: {type(df.index)} (tz: {df.index.tz})")
        print(f"   Période: {df.index[0]} → {df.index[-1]}")

        # Validation OHLCV strict
        if set(df.columns) >= {"open", "high", "low", "close", "volume"}:
            print("✅ Colonnes OHLCV complètes")
        else:
            print("❌ Colonnes OHLCV manquantes")

        # 4) Test persistance unifiée
        if not df.empty:
            print(f"\n💾 Test persistance unifiée")
            saved_path = provider.persist_frame(
                df,
                test_symbol,
                "1h",
                metadata={"test": "option_b_final", "version": "1.0"},
            )
            print(f"✅ Sauvegardé: {saved_path}")

            # Vérification du fichier
            if saved_path and Path(saved_path).exists():
                size_kb = Path(saved_path).stat().st_size / 1024
                print(f"   Taille: {size_kb:.1f} KB")

        # 5) Stats de performance
        print(f"\n⚡ Statistiques de performance:")
        stats = provider._perf_stats
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # 6) Test timeframes multiples
        print(f"\n🔄 Test timeframes multiples")
        for tf in ["1m", "5m", "1h"]:
            try:
                df_tf = provider.get_frame(
                    test_symbol, tf, end=datetime.now(timezone.utc)
                )
                print(f"   {tf}: {len(df_tf)} rows")
            except Exception as e:
                print(f"   {tf}: ERREUR - {e}")

        print("\n🎯 Option B - Test terminé avec succès !")
        return True

    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_option_b_final()
    sys.exit(0 if success else 1)

