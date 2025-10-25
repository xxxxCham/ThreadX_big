"""Test rapide de l'adaptation aux chemins USDC"""

from threadx.data.tokens import (
    create_default_config,
    TokenDiversityDataSource,
)


def main():
    print("=" * 60)
    print("TEST ADAPTATION ARBORESCENCE CRYPTO_DATA_PARQUET/")
    print("=" * 60)

    # 1. Créer configuration
    print("\n📋 1. Création configuration...")
    config = create_default_config()
    print(f"   ✅ Config créée")
    print(f"   Symboles totaux: {len(config.symbols)}")
    print(f"   Groupes: {list(config.groups.keys())}")

    # 2. Créer provider
    print("\n🔌 2. Création provider...")
    provider = TokenDiversityDataSource(config)
    print(f"   ✅ Provider créé")

    # 3. Test groupes
    print("\n📁 3. Test list_groups()...")
    groups = provider.list_groups()
    print(f"   Groupes disponibles: {groups}")

    # 4. Test symboles L1
    print("\n💎 4. Test list_symbols('L1')...")
    l1_symbols = provider.list_symbols("L1")
    print(f"   Symboles L1: {l1_symbols}")

    # 5. Test validation
    print("\n✔️ 5. Test validate_symbol()...")
    print(f"   BTCUSDC valide: {provider.validate_symbol('BTCUSDC')}")
    print(f"   INVALID valide: {provider.validate_symbol('INVALID')}")

    # 6. Test fetch (si données disponibles)
    print("\n📊 6. Test fetch_ohlcv('BTCUSDC', '1h')...")
    try:
        df = provider.fetch_ohlcv("BTCUSDC", "1h", limit=10)
        print(f"   ✅ Chargement réussi!")
        print(f"   Lignes: {len(df)}")
        print(f"   Colonnes: {list(df.columns)}")
        print(f"   Période: {df.index[0]} → {df.index[-1]}")
        print(f"\n   Aperçu:")
        print(df.head(3).to_string())
    except FileNotFoundError as e:
        print(f"   ⚠️ Fichier non trouvé: {e}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    print("\n" + "=" * 60)
    print("✅ TEST TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()

