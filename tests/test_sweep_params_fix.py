"""
Test de validation du correctif: Paramètres par défaut dans Grid Sweep.

Vérifie que min_pnl_pct et autres params par défaut sont bien inclus
dans les combinaisons du sweep, même s'ils ne sont pas dans param_ranges.
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from threadx.optimization.scenarios import generate_param_grid
from threadx.ui.strategy_registry import parameter_specs_for, base_params_for


def test_scenario_params_include_defaults():
    """
    Simule la construction de scenario_params comme dans l'UI.
    Vérifie que TOUS les paramètres par défaut sont présents.
    """
    print("\n" + "=" * 70)
    print("TEST: Paramètres par défaut dans Grid Sweep")
    print("=" * 70)

    strategy = "Bollinger_Breakout"

    # Simuler param_ranges (paramètres optimisés)
    param_ranges = {
        "bb_period": (10, 50),
        "bb_std": (1.5, 3.0),
        "entry_z": (0.8, 2.0),
    }

    # Simuler configured_params (peut être vide ou incomplet)
    configured_params = {}  # ← VIDE comme souvent dans la vraie UI

    # Construction scenario_params SANS le fix
    scenario_params_OLD = {}
    for key, (min_v, max_v) in param_ranges.items():
        scenario_params_OLD[key] = {"values": [min_v, max_v]}

    # Ajouter configured_params (ne fait rien si vide)
    for key, value in configured_params.items():
        if key not in scenario_params_OLD:
            scenario_params_OLD[key] = {"value": value}

    print("\n❌ AVANT FIX (comportement bugué):")
    print(f"scenario_params contient {len(scenario_params_OLD)} paramètres:")
    for key in sorted(scenario_params_OLD.keys()):
        print(f"  - {key}")

    # Construction scenario_params AVEC le fix
    scenario_params_NEW = {}
    for key, (min_v, max_v) in param_ranges.items():
        scenario_params_NEW[key] = {"values": [min_v, max_v]}

    # 🔥 FIX: Ajouter TOUS les paramètres par défaut manquants
    all_param_specs = parameter_specs_for(strategy)
    base_strategy_params = base_params_for(strategy)

    for key, spec in all_param_specs.items():
        if key not in scenario_params_NEW:
            value = configured_params.get(
                key,
                base_strategy_params.get(
                    key, spec.get("default") if isinstance(spec, dict) else spec
                ),
            )
            scenario_params_NEW[key] = {"value": value}

    print("\n✅ APRÈS FIX (comportement correct):")
    print(f"scenario_params contient {len(scenario_params_NEW)} paramètres:")
    for key in sorted(scenario_params_NEW.keys()):
        val = scenario_params_NEW[key]
        if "value" in val:
            print(f"  - {key} = {val['value']}")
        else:
            print(f"  - {key} : {val}")

    # Vérifier que min_pnl_pct est présent avec valeur 0.0
    print("\n🔍 Vérification min_pnl_pct:")
    if "min_pnl_pct" not in scenario_params_OLD:
        print("  ❌ ABSENT dans version AVANT FIX (BUG !)")
    else:
        print(f"  ✓ Présent: {scenario_params_OLD['min_pnl_pct']}")

    if "min_pnl_pct" not in scenario_params_NEW:
        print("  ❌ ENCORE ABSENT après FIX (problème !)")
    else:
        val = scenario_params_NEW["min_pnl_pct"]
        print(f"  ✅ Présent après FIX: {val}")
        if val.get("value") == 0.0:
            print("  ✅ Valeur correcte: 0.0 (désactivé)")
        else:
            print(f"  ⚠️ Valeur inattendue: {val.get('value')}")

    # Générer les combinaisons
    print("\n🔬 Génération des combinaisons:")
    combos_OLD = generate_param_grid(scenario_params_OLD)
    combos_NEW = generate_param_grid(scenario_params_NEW)

    print(f"  AVANT FIX: {len(combos_OLD)} combinaisons")
    if combos_OLD:
        print(f"    Exemple combo[0]: {combos_OLD[0]}")
        if "min_pnl_pct" in combos_OLD[0]:
            print(f"      ✓ min_pnl_pct = {combos_OLD[0]['min_pnl_pct']}")
        else:
            print("      ❌ min_pnl_pct ABSENT (utilise défaut 0.01 ← BUG !)")

    print(f"\n  APRÈS FIX: {len(combos_NEW)} combinaisons")
    if combos_NEW:
        print(f"    Exemple combo[0]: {combos_NEW[0]}")
        if "min_pnl_pct" in combos_NEW[0]:
            print(f"      ✅ min_pnl_pct = {combos_NEW[0]['min_pnl_pct']}")
        else:
            print("      ❌ min_pnl_pct ENCORE ABSENT !")

    print("\n" + "=" * 70)
    if "min_pnl_pct" in combos_NEW[0] and combos_NEW[0]["min_pnl_pct"] == 0.0:
        print("✅ TEST RÉUSSI: min_pnl_pct=0.0 présent dans combinaisons")
        return True
    else:
        print("❌ TEST ÉCHOUÉ: min_pnl_pct manquant ou incorrect")
        return False


def test_all_default_params_present():
    """Vérifie que TOUS les paramètres de la stratégie sont dans les combos."""
    print("\n" + "=" * 70)
    print("TEST: Tous les paramètres par défaut présents")
    print("=" * 70)

    strategy = "Bollinger_Breakout"

    # Simuler param_ranges (seulement 3 params optimisés)
    param_ranges = {
        "bb_period": (20, 30),
        "entry_z": (1.0, 2.0),
    }

    configured_params = {}

    # Construction avec fix
    scenario_params = {}
    for key, (min_v, max_v) in param_ranges.items():
        scenario_params[key] = {"values": [min_v, max_v]}

    all_param_specs = parameter_specs_for(strategy)
    base_strategy_params = base_params_for(strategy)

    for key, spec in all_param_specs.items():
        if key not in scenario_params:
            value = configured_params.get(
                key,
                base_strategy_params.get(
                    key, spec.get("default") if isinstance(spec, dict) else spec
                ),
            )
            scenario_params[key] = {"value": value}

    # Générer combos
    combos = generate_param_grid(scenario_params)

    print(f"\nParamètres optimisés: {list(param_ranges.keys())}")
    print(f"Total paramètres dans spec: {len(all_param_specs)}")
    print(f"Total paramètres dans scenario: {len(scenario_params)}")
    print(f"Combinaisons générées: {len(combos)}")

    if not combos:
        print("❌ Aucune combinaison générée !")
        return False

    # Vérifier que tous les params sont dans combo[0]
    combo = combos[0]
    print(f"\nExemple combo[0] contient {len(combo)} paramètres:")

    missing = []
    for key in all_param_specs.keys():
        if key in combo:
            print(f"  ✓ {key} = {combo[key]}")
        else:
            print(f"  ❌ {key} MANQUANT")
            missing.append(key)

    if missing:
        print(f"\n❌ TEST ÉCHOUÉ: {len(missing)} paramètres manquants: {missing}")
        return False
    else:
        print("\n✅ TEST RÉUSSI: Tous les paramètres présents dans les combinaisons")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VALIDATION CORRECTIF: Paramètres par défaut dans Grid Sweep")
    print("=" * 70)

    try:
        test1_ok = test_scenario_params_include_defaults()
        test2_ok = test_all_default_params_present()

        print("\n" + "=" * 70)
        print("RÉSUMÉ DES TESTS")
        print("=" * 70)

        if test1_ok and test2_ok:
            print("✅ TOUS LES TESTS RÉUSSIS")
            print("\nIMPACT:")
            print("  1. min_pnl_pct=0.0 sera maintenant dans TOUTES les combinaisons")
            print("  2. Les trades ne seront plus filtrés (0 trades → X trades)")
            print("  3. Le capital va enfin varier entre les tests")
            print("\nRECOMMANDATION:")
            print("  Relancer le Grid Sweep dans Streamlit et vérifier:")
            print("  - Logs: 'Backtest terminé: X trades' (X > 0)")
            print("  - Capital final != 10,000")
            sys.exit(0)
        else:
            print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
            print("\nProblème détecté:")
            if not test1_ok:
                print("  - min_pnl_pct n'est pas correctement ajouté")
            if not test2_ok:
                print("  - D'autres paramètres par défaut sont manquants")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERREUR LORS DES TESTS: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
