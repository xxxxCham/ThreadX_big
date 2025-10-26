#!/usr/bin/env python3
"""
Script de test pour vérifier que le chargement des données fonctionne.
"""
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.data_access import discover_tokens_and_timeframes, load_ohlcv, DATA_DIR

print("=" * 60)
print("TEST DE CHARGEMENT DES DONNÉES CRYPTO")
print("=" * 60)

print(f"\n📁 Dossier de données: {DATA_DIR}")
print(f"   Existe: {DATA_DIR.exists()}")

print("\n🔍 Découverte des tokens et timeframes...")
tokens, timeframes = discover_tokens_and_timeframes()

print(f"\n✅ Tokens trouvés ({len(tokens)}):")
for token in tokens:
    print(f"   - {token}")

print(f"\n✅ Timeframes trouvés ({len(timeframes)}):")
for tf in timeframes:
    print(f"   - {tf}")

if tokens and timeframes:
    print("\n📊 Test de chargement d'un fichier...")
    symbol = tokens[0]
    timeframe = timeframes[0]

    try:
        df = load_ohlcv(symbol, timeframe)
        print(f"\n✅ Chargement réussi: {symbol}/{timeframe}")
        print(f"   - Lignes: {len(df)}")
        print(f"   - Colonnes: {list(df.columns)}")
        print(f"   - Période: {df.index.min()} à {df.index.max()}")
        print("\n   Aperçu des données:")
        print(df.head(3).to_string())

        print("\n✅ TOUT FONCTIONNE CORRECTEMENT !")

    except Exception as e:
        print(f"\n❌ Erreur lors du chargement: {e}")
        sys.exit(1)
else:
    print("\n❌ Aucune donnée trouvée !")
    sys.exit(1)

print("\n" + "=" * 60)
