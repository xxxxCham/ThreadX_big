#!/usr/bin/env python3
"""
Script pour nettoyer TOUS les caches de ThreadX.
À exécuter si l'interface Streamlit ne détecte pas les fichiers.
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("NETTOYAGE COMPLET DES CACHES THREADX")
print("=" * 80)

# 1. Invalider les caches LRU de data_access
print("\n1️⃣  Invalidation des caches LRU...")
try:
    import threadx.data_access
    threadx.data_access._iter_data_files.cache_clear()
    threadx.data_access.discover_tokens_and_timeframes.cache_clear()
    print("   ✅ Cache LRU invalidé")
except Exception as e:
    print(f"   ⚠️  Erreur: {e}")

# 2. Vérifier les fichiers détectés
print("\n2️⃣  Vérification des fichiers détectés...")
try:
    from threadx.data_access import discover_tokens_and_timeframes, DATA_DIR, _iter_data_files

    print(f"   📁 DATA_DIR: {DATA_DIR}")

    all_files = list(_iter_data_files())
    print(f"   📄 Fichiers trouvés: {len(all_files)}")

    if all_files:
        print(f"\n   Exemples de fichiers:")
        for f in all_files[:5]:
            print(f"      - {f.name}")

    tokens, timeframes = discover_tokens_and_timeframes()
    print(f"\n   🪙 Tokens: {tokens}")
    print(f"   ⏰ Timeframes: {timeframes}")

    if not tokens:
        print("\n   ❌ AUCUN TOKEN TROUVÉ !")
        print("   Vérifiez que les fichiers sont dans crypto_data_parquet/ ou crypto_data_json/")
    else:
        print(f"\n   ✅ {len(tokens)} tokens détectés")

except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

# 3. Nettoyer le cache Streamlit (si disponible)
print("\n3️⃣  Nettoyage du cache Streamlit...")
try:
    import shutil
    streamlit_cache = Path.home() / ".streamlit" / "cache"
    if streamlit_cache.exists():
        shutil.rmtree(streamlit_cache)
        print(f"   ✅ Cache Streamlit supprimé: {streamlit_cache}")
    else:
        print(f"   ℹ️  Pas de cache Streamlit trouvé")
except Exception as e:
    print(f"   ⚠️  Erreur: {e}")

print("\n" + "=" * 80)
print("✅ NETTOYAGE TERMINÉ")
print("=" * 80)
print("\nRedémarrez Streamlit pour appliquer les changements:")
print("  pkill -f streamlit && streamlit run src/threadx/streamlit_app.py --server.port 8502")
print()
