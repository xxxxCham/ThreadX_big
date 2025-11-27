"""
Test du workflow complet Evolution Loop avec données mock.

Vérifie:
- Chargement sweep results
- Exécution séquentielle Analyst → Strategist → CodeWriter → Critic
- Génération rapport JSON
- Tous les agents fonctionnent correctement
"""

import json
import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH
if str(Path(__file__).parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_evolution_loop import EvolutionLoop


def test_evolution_loop_with_mock_data():
    """Test avec données d'exemple."""
    print("\n" + "=" * 80)
    print("🧪 TEST EVOLUTION LOOP - Workflow Complet")
    print("=" * 80)

    # Charger sweep results d'exemple
    sweep_path = Path(__file__).parent / "example_sweep_results.json"

    if not sweep_path.exists():
        print(f"❌ Fichier exemple non trouvé: {sweep_path}")
        print("   Créez d'abord example_sweep_results.json")
        return False

    with open(sweep_path, encoding="utf-8") as f:
        sweep_results = json.load(f)

    print(f"✅ Sweep results chargés: {sweep_path.name}")
    print(f"   Strategy: {sweep_results.get('strategy_name')}")
    print(f"   Baseline Sharpe: {sweep_results['baseline']['metrics']['sharpe_ratio']}")
    print(f"   Best Sharpe: {sweep_results.get('best_sharpe')}")
    print(f"   Total runs: {sweep_results['metadata']['total_runs']}")

    # Initialiser Evolution Loop
    print("\n🚀 Initialisation Evolution Loop...")

    output_dir = Path(__file__).parent.parent / "results" / "ai_evolution_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    loop = EvolutionLoop(
        output_dir=output_dir,
        debug=True,
    )

    print(f"✅ Loop initialisé")
    print(f"   Output: {loop.output_dir}")

    # Exécuter génération
    print("\n🧬 Exécution génération complète...")

    try:
        result = loop.run_generation(
            base_strategy=sweep_results["strategy_name"],
            sweep_results=sweep_results,
            task="improve_sharpe",
            generation_num=999,  # Test
        )

        # Vérifier résultat
        print("\n📊 RÉSUMÉ RÉSULTATS")
        print("=" * 80)

        print(f"\n✅ Status: {result['status']}")
        print(f"   Base Strategy: {result['base_strategy']}")
        print(f"   Task: {result['task']}")
        print(f"   Generation: {result['generation_num']}")

        # Vérifier toutes les étapes
        steps = result.get("steps", {})

        print("\n📋 Étapes exécutées:")

        for step_name in ["analyst", "strategist", "codewriter", "critic"]:
            if step_name in steps:
                step = steps[step_name]
                status = step.get("status", "unknown")
                duration = step.get("duration_sec", 0.0)

                status_emoji = "✅" if status in ["success", "approved"] else "❌"
                print(f"   {status_emoji} {step_name.capitalize()}: {status} ({duration:.1f}s)")

                # Détails par étape
                if step_name == "analyst":
                    print(f"      Patterns: {step.get('patterns_found', 0)}")
                    print(f"      Recommendations: {step.get('recommendations_count', 0)}")

                elif step_name == "strategist":
                    print(f"      Proposals: {step.get('proposals_count', 0)}")

                elif step_name == "codewriter":
                    print(f"      Filename: {step.get('filename', 'N/A')}")
                    print(f"      Class: {step.get('class_name', 'N/A')}")

                elif step_name == "critic":
                    print(f"      Recommendation: {step.get('recommendation', 'N/A')}")
                    if "result" in step and "test_quantitative" in step["result"]:
                        quant = step["result"]["test_quantitative"]
                        print(f"      Best Sharpe: {quant.get('best_sharpe', 'N/A')}")
                        print(f"      Worst DD: {quant.get('worst_drawdown', 'N/A')}%")

            else:
                print(f"   ⚠️  {step_name.capitalize()}: Non exécuté")

        # Vérifier fichier généré
        if result["status"] in ["approved", "rejected"]:
            if "codewriter" in steps and "filepath" in steps["codewriter"]:
                generated_file = Path(steps["codewriter"]["filepath"])
                if generated_file.exists():
                    print(f"\n✅ Fichier stratégie généré: {generated_file.name}")
                    print(f"   Path: {generated_file}")
                    print(f"   Size: {generated_file.stat().st_size} bytes")
                else:
                    print(f"\n⚠️  Fichier stratégie non trouvé: {generated_file}")

        # Vérifier rapport JSON
        if result.get("generation_num"):
            # Le rapport devrait être dans output_dir
            json_files = list(output_dir.glob("*.json"))
            if json_files:
                latest = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"\n✅ Rapport JSON sauvegardé: {latest.name}")
                print(f"   Size: {latest.stat().st_size} bytes")
            else:
                print(f"\n⚠️  Aucun rapport JSON trouvé dans {output_dir}")

        print("\n" + "=" * 80)

        # Vérifier succès
        if result["status"] in ["approved", "rejected"]:
            print("✅ TEST RÉUSSI: Workflow complet exécuté")
            return True
        else:
            print(f"⚠️  TEST PARTIEL: Status={result['status']}")
            return False

    except Exception as e:
        print(f"\n❌ ERREUR durant exécution:")
        print(f"   {type(e).__name__}: {str(e)}")

        import traceback
        traceback.print_exc()

        return False


def main():
    """Run test."""
    success = test_evolution_loop_with_mock_data()

    if success:
        print("\n✅ TESTS PASSÉS")
        sys.exit(0)
    else:
        print("\n❌ TESTS ÉCHOUÉS")
        sys.exit(1)


if __name__ == "__main__":
    main()
