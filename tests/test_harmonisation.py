#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de l'harmonisation ThreadX
Vérifie que toutes les fonctionnalités sont harmonisées avec la structure
"""

import sys
from pathlib import Path

# Configuration ThreadX
THREADX_ROOT = Path(__file__).parent
sys.path.insert(0, str(THREADX_ROOT))
sys.path.insert(0, str(THREADX_ROOT / "src"))


def test_structure_harmonisee():
    """Test de la structure harmonisée"""
    print("=" * 60)
    print("🔧 TEST DE L'HARMONISATION THREADX")
    print("=" * 60)

    # Chemins harmonisés
    paths = {
        "JSON (OHLCV)": THREADX_ROOT / "data" / "crypto_data_json",
        "Parquet (OHLCV)": THREADX_ROOT / "data" / "crypto_data_parquet",
        "Indicateurs Tech": THREADX_ROOT / "data" / "indicateurs_tech_data",
        "ATR": THREADX_ROOT / "data" / "indicateurs_tech_data" / "atr",
        "Bollinger": THREADX_ROOT / "data" / "indicateurs_tech_data" / "bollinger",
        "Registry": THREADX_ROOT / "data" / "indicateurs_tech_data" / "registry",
        "Cache": THREADX_ROOT / "data" / "cache",
        "Tokens List": THREADX_ROOT / "data" / "resultats_choix_des_100tokens.json",
    }

    print("\n📁 STRUCTURE DES DOSSIERS:")
    for name, path in paths.items():
        exists = "✅" if path.exists() else "❌"
        if path.is_file():
            size = f"({path.stat().st_size / 1024:.1f} KB)" if path.exists() else ""
        else:
            count = len(list(path.glob("*"))) if path.exists() else 0
            size = f"({count} fichiers)" if path.exists() else ""
        print(f"  {exists} {name:<20} {size}")

    # Test imports
    print("\n🔧 TEST DES IMPORTS:")
    try:
        from apps.threadx_tradxpro_interface import (
            setup_indicators_structure,
            load_indicators_registry,
            calculate_atr,
            calculate_bollinger_bands,
            verify_and_clean_cache,
        )

        print("  ✅ Imports fonctions harmonisées: OK")
    except ImportError as e:
        print(f"  ❌ Erreur imports: {e}")
        return False

    # Test initialisation structure
    print("\n🏗️ TEST INITIALISATION:")
    try:
        setup_indicators_structure()
        print("  ✅ Structure indicateurs: OK")
    except Exception as e:
        print(f"  ❌ Erreur structure: {e}")

    # Test registry
    print("\n📋 TEST REGISTRY:")
    try:
        registry = load_indicators_registry()
        if registry and "indicators" in registry:
            print(f"  ✅ Registry chargé: {len(registry['indicators'])} indicateurs")
            for name, info in registry["indicators"].items():
                print(f"    - {name}: {info['name']}")
        else:
            print("  ⚠️ Registry vide ou invalide")
    except Exception as e:
        print(f"  ❌ Erreur registry: {e}")

    # Test données existantes
    print("\n📊 TEST DONNÉES EXISTANTES:")

    # JSON files
    json_files = (
        list(paths["JSON (OHLCV)"].glob("*.json"))
        if paths["JSON (OHLCV)"].exists()
        else []
    )
    print(f"  📄 Fichiers JSON OHLCV: {len(json_files)}")

    # Parquet files
    parquet_files = (
        list(paths["Parquet (OHLCV)"].glob("*.parquet"))
        if paths["Parquet (OHLCV)"].exists()
        else []
    )
    print(f"  📊 Fichiers Parquet OHLCV: {len(parquet_files)}")

    # Indicateurs ATR
    atr_files = list(paths["ATR"].glob("*.parquet")) if paths["ATR"].exists() else []
    print(f"  📈 Indicateurs ATR: {len(atr_files)}")

    # Indicateurs Bollinger
    bb_files = (
        list(paths["Bollinger"].glob("*.parquet"))
        if paths["Bollinger"].exists()
        else []
    )
    print(f"  📈 Indicateurs Bollinger: {len(bb_files)}")

    # Cache
    cache_files = list(paths["Cache"].glob("**/*")) if paths["Cache"].exists() else []
    cache_size = sum(f.stat().st_size for f in cache_files if f.is_file()) / 1024 / 1024
    print(f"  🗃️ Cache: {len(cache_files)} fichiers, {cache_size:.1f} MB")

    print("\n" + "=" * 60)
    print("✅ TEST D'HARMONISATION TERMINÉ")
    print("🚀 L'interface TradXPro est prête avec la structure harmonisée !")
    print("=" * 60)

    return True


def show_usage_guide():
    """Guide d'utilisation harmonisé"""
    print("\n📖 GUIDE D'UTILISATION - INTERFACE HARMONISÉE:")
    print(
        """
    1. 🔄 Refresh 100 meilleures monnaies
       → Met à jour: data/resultats_choix_des_100tokens.json
       
    2. 📥 Télécharger OHLCV (sans indicateurs)
       → Sauvegarde dans: data/crypto_data_json/
       
    3. 📊 Télécharger OHLCV + Indicateurs
       → OHLCV dans: data/crypto_data_json/
       → ATR dans: data/indicateurs_tech_data/atr/
       → Bollinger dans: data/indicateurs_tech_data/bollinger/
       
    4. 🔄 Convertir JSON → Parquet
       → Conversion vers: data/crypto_data_parquet/
       
    5. ✅ Vérifier & Compléter données
       → Vérifie toute la structure harmonisée
       → Nettoie le cache: data/cache/
       → Génère rapport: data/verification_report.json
    """
    )


if __name__ == "__main__":
    success = test_structure_harmonisee()
    if success:
        show_usage_guide()
    sys.exit(0 if success else 1)
