"""
Script de diagnostic pour identifier pourquoi le nombre de combinaisons
change selon la durée des données de backtest.

Problème signalé:
- Sweep 6 mois: 310,000 combinaisons
- Sweep 3 jours: 288,000 combinaisons (ou 2,880,000?)

Le nombre devrait être IDENTIQUE si les plages de paramètres sont les mêmes.
"""

import pandas as pd
import logging
from threadx.optimization.scenarios import generate_param_grid

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def test_combination_count():
    """Test le nombre de combinaisons pour différentes durées de données"""

    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC: NOMBRE DE COMBINAISONS vs DURÉE DES DONNÉES")
    print("=" * 80 + "\n")

    # Définir les mêmes plages de paramètres pour les 2 tests
    param_ranges = {
        "bb_length": [10, 15, 20, 25, 30, 35, 40],  # 7 valeurs
        "bb_mult": [1.5, 2.0, 2.5, 3.0],  # 4 valeurs
        "atr_length": [10, 14, 21, 28],  # 4 valeurs
        "atr_mult": [1.0, 1.5, 2.0, 2.5],  # 4 valeurs
        "sl_atr_mult": [1.5, 2.0, 2.5, 3.0],  # 4 valeurs
        "tp_atr_mult": [2.0, 3.0, 4.0, 5.0],  # 4 valeurs
    }

    # Calculer le nombre théorique
    theoretical_count = 1
    for param_name, values in param_ranges.items():
        theoretical_count *= len(values)
        print(f"  {param_name:15s}: {len(values):3d} valeurs")

    print(f"\n📊 Nombre théorique de combinaisons: {theoretical_count:,}")
    print(f"   (7 × 4 × 4 × 4 × 4 × 4 = {theoretical_count:,})")

    # Générer les combinaisons avec generate_param_grid
    print(f"\n🔄 Génération des combinaisons avec generate_param_grid()...")
    combinations = generate_param_grid(param_ranges)
    actual_count = len(combinations)

    print(f"✅ Nombre réel généré: {actual_count:,}")

    if actual_count == theoretical_count:
        print(f"✅ OK: Nombre correct (théorique = réel)")
    else:
        print(f"❌ ERREUR: Différence détectée!")
        print(f"   Théorique: {theoretical_count:,}")
        print(f"   Réel:      {actual_count:,}")
        print(f"   Delta:     {actual_count - theoretical_count:,}")

    # Test avec différentes durées de données
    print(f"\n" + "=" * 80)
    print("🧪 TEST: Impact théorique de la durée des données")
    print("=" * 80 + "\n")

    test_cases = [
        ("3 jours", 288, 3),
        ("1 mois", 2880, 30),
        ("3 mois", 8640, 90),
        ("6 mois", 17280, 180),
    ]

    for label, n_bars, duration_days in test_cases:
        print(f"\n📅 {label}: {n_bars} barres ({duration_days} jours en 15m)")

        # Re-générer les combinaisons (ne devrait pas changer)
        combos = generate_param_grid(param_ranges)
        count = len(combos)

        print(f"  🔢 Combinaisons générées: {count:,}")

        if count != theoretical_count:
            print(f"  ❌ ANOMALIE DÉTECTÉE!")
            print(f"     Attendu: {theoretical_count:,}")
            print(f"     Obtenu:  {count:,}")
            print(f"     Delta:   {count - theoretical_count:,}")
        else:
            print(f"  ✅ OK: Nombre correct")

    # Test avec validation des paramètres
    print(f"\n" + "=" * 80)
    print("🔍 TEST: Validation des paramètres selon données disponibles")
    print("=" * 80 + "\n")

    # Vérifier si bb_length=40 est valide avec seulement 288 barres
    print(f"📊 Avec 288 barres (3 jours en 15m):")
    print(f"  - bb_length=40 → Warmup de 40 barres → 248 barres utilisables")
    print(f"  - bb_length=100 → Warmup de 100 barres → 188 barres utilisables")
    print(f"  - bb_length=200 → Warmup de 200 barres → 88 barres utilisables")
    print(f"\n⚠️  Si bb_length > 288, aucune barre utilisable!")

    print(f"\n📊 Avec 17,280 barres (6 mois en 15m, ~180 jours):")
    print(f"  - bb_length=40 → Warmup de 40 barres → 17,240 barres utilisables")
    print(f"  - bb_length=200 → Warmup de 200 barres → 17,080 barres utilisables")

    print(f"\n💡 Hypothèse:")
    print(f"  Le système pourrait FILTRER les combinaisons où:")
    print(f"  - bb_length + atr_length > nombre de barres disponibles")
    print(f"  - Pas assez de données pour calculer les indicateurs")

    # Vérifier dans le code source
    print(f"\n" + "=" * 80)
    print("📝 PROCHAINES ÉTAPES")
    print("=" * 80 + "\n")

    print(f"1. Vérifier si les PLAGES DE PARAMÈTRES sont identiques")
    print(f"2. Chercher tout code qui FILTRE les combinaisons basé sur les données")
    print(f"3. Vérifier si le warmup period réduit les combinaisons valides")
    print(f"4. Tracer l'exécution pour voir où les combinaisons disparaissent")

    print(f"\n✅ Test terminé\n")


if __name__ == "__main__":
    test_combination_count()
