"""
Test du preset manuel_30 - Vérification fonctionnement
"""

from threadx.optimization.engine import SweepRunner
from threadx.optimization.presets.ranges import get_execution_preset, load_execution_presets

def test_preset_loading():
    """Test 1: Chargement des presets"""
    print("=" * 60)
    print("TEST 1: Chargement presets")
    print("=" * 60)
    
    try:
        # Charger tous les presets
        all_presets = load_execution_presets()
        print(f"✅ Presets chargés: {list(all_presets.keys())}")
        
        # Charger manuel_30
        preset = get_execution_preset('manuel_30')
        print(f"\n✅ Preset manuel_30:")
        for key, value in preset.items():
            print(f"   - {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_sweeprunner_with_preset():
    """Test 2: Initialisation SweepRunner avec preset"""
    print("\n" + "=" * 60)
    print("TEST 2: SweepRunner avec preset='manuel_30'")
    print("=" * 60)
    
    try:
        # Initialisation avec preset
        runner = SweepRunner(preset='manuel_30', use_multigpu=False)
        
        print(f"✅ SweepRunner initialisé")
        print(f"   - max_workers: {runner.max_workers}")
        print(f"   - batch_size: {runner.batch_size}")
        
        # Vérifications
        assert runner.max_workers == 30, f"Expected 30 workers, got {runner.max_workers}"
        assert runner.batch_size == 2000, f"Expected batch 2000, got {runner.batch_size}"
        
        print("\n✅ Valeurs correctes!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sweeprunner_override():
    """Test 3: Override partiel du preset"""
    print("\n" + "=" * 60)
    print("TEST 3: Override partiel preset")
    print("=" * 60)
    
    try:
        # Override max_workers
        runner = SweepRunner(preset='manuel_30', max_workers=20, use_multigpu=False)
        
        print(f"✅ SweepRunner avec override max_workers=20")
        print(f"   - max_workers: {runner.max_workers} (devrait être 20)")
        print(f"   - batch_size: {runner.batch_size} (devrait être 2000 du preset)")
        
        assert runner.max_workers == 20, f"Expected 20, got {runner.max_workers}"
        assert runner.batch_size == 2000, f"Expected 2000, got {runner.batch_size}"
        
        print("\n✅ Override fonctionne!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_presets():
    """Test 4: Tous les presets"""
    print("\n" + "=" * 60)
    print("TEST 4: Tous les presets disponibles")
    print("=" * 60)
    
    presets_to_test = ['auto', 'conservative', 'balanced', 'aggressive', 'manuel_30']
    
    for preset_name in presets_to_test:
        try:
            runner = SweepRunner(preset=preset_name, use_multigpu=False)
            print(f"✅ {preset_name:15s} → workers={runner.max_workers:3d}, batch={runner.batch_size}")
        except Exception as e:
            print(f"❌ {preset_name:15s} → Erreur: {e}")


if __name__ == "__main__":
    print("\n🧪 TESTS PRESET MANUEL_30\n")
    
    results = []
    
    # Test 1: Chargement
    results.append(("Chargement presets", test_preset_loading()))
    
    # Test 2: SweepRunner avec preset
    results.append(("SweepRunner preset", test_sweeprunner_with_preset()))
    
    # Test 3: Override
    results.append(("Override partiel", test_sweeprunner_override()))
    
    # Test 4: Tous presets
    print("\n" + "=" * 60)
    test_all_presets()
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_success = sum(1 for _, success in results if success)
    print(f"\nRésultat: {total_success}/{len(results)} tests réussis")
    
    if total_success == len(results):
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("\n📝 Utilisation:")
        print("   runner = SweepRunner(preset='manuel_30')")
        print("   results = runner.run_grid(...)")
    else:
        print("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
