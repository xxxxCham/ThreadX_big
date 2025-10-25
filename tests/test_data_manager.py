"""
Test simple du système de découverte de données
"""

import sys
from pathlib import Path

# Ajouter le chemin source
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Imports ThreadX
from apps.data_manager.discovery.local_scanner import (
    LocalDataScanner,
    create_demo_catalog,
)
from apps.data_manager.models import DataQuality


def test_demo_catalog():
    """Test du catalogue de démonstration"""
    print("🧪 Test du catalogue de démonstration...")

    catalog = create_demo_catalog()

    print(f"✅ Catalogue créé:")
    print(f"   - Symboles: {len(catalog.unique_symbols)}")
    print(f"   - Timeframes: {len(catalog.unique_timeframes)}")
    print(f"   - Indicateurs: {len(catalog.unique_indicators)}")
    print(f"   - Fichiers totaux: {catalog.total_files}")
    print(f"   - Taille: {catalog.size_mb:.1f} MB")

    # Vérifier la structure
    for symbol, symbol_data in catalog.symbols.items():
        print(f"\n📈 Symbole: {symbol}")
        for tf, tf_data in symbol_data.timeframes.items():
            print(f"   ⏰ Timeframe: {tf} ({tf_data.file_count} fichiers)")
            for indicator, files in tf_data.indicators.items():
                print(f"      📊 {indicator}: {len(files)} fichiers")
                for file_info in files:
                    print(
                        f"         - {file_info.file_name} ({file_info.size_bytes} bytes)"
                    )
                    print(f"           Paramètres: {file_info.parameters}")

    return catalog


def test_scanner():
    """Test du scanner avec des chemins fictifs"""
    print("\n🔍 Test du scanner...")

    scanner = LocalDataScanner()

    # Test avec des chemins fictifs (ne devrait rien trouver)
    test_paths = ["./nonexistent_path", "g:\\indicators_db_test"]

    catalog = scanner.scan_indicators_db(test_paths)
    print(f"✅ Scanner testé (chemins fictifs):")
    print(f"   - Chemins scannés: {len(catalog.root_paths)}")
    print(f"   - Symboles trouvés: {len(catalog.unique_symbols)}")

    return catalog


def test_data_quality():
    """Test des énumérations de qualité"""
    print("\n📋 Test des énumérations...")

    print("✅ DataQuality values:")
    for quality in DataQuality:
        print(f"   - {quality.name}: {quality.value}")

    # Test d'utilisation
    test_quality = DataQuality.PENDING
    print(f"✅ Test quality: {test_quality.value}")


if __name__ == "__main__":
    print("🚀 ThreadX Data Manager - Tests de base\n")

    try:
        # Tests
        test_data_quality()
        catalog1 = test_demo_catalog()
        catalog2 = test_scanner()

        print("\n🎉 Tous les tests réussis!")
        print(f"📊 Catalogue démo: {catalog1.total_files} fichiers")
        print(f"🔍 Scanner: {catalog2.total_files} fichiers")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
