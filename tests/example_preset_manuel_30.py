"""
Exemple rapide d'utilisation du preset manuel_30
Sweep paramétrique optimisé avec 30 workers et 2000 batch size
"""

from threadx.optimization.engine import SweepRunner
import logging

# Configuration logging pour voir les infos
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    print("\n" + "=" * 60)
    print("🚀 EXEMPLE PRESET MANUEL_30 - SWEEP OPTIMISÉ")
    print("=" * 60 + "\n")

    # Initialisation avec preset manuel_30
    print("📋 Initialisation du SweepRunner avec preset='manuel_30'...")
    runner = SweepRunner(preset="manuel_30", use_multigpu=True)

    print(f"\n✅ Configuration chargée:")
    print(f"   - Workers: {runner.max_workers}")
    print(f"   - Batch size: {runner.batch_size}")
    print(f"   - Multi-GPU: {runner.use_multigpu}")

    # Paramètres du sweep (petit exemple pour test rapide)
    param_ranges = {
        "bb_length": [15, 20, 25],  # 3 valeurs
        "bb_mult": [2.0, 2.5, 3.0],  # 3 valeurs
        "atr_length": [14, 21],  # 2 valeurs
        "atr_mult": [1.5, 2.0],  # 2 valeurs
        "sl_atr_mult": [2.0, 2.5],  # 2 valeurs
        "tp_atr_mult": [3.0, 4.0],  # 2 valeurs
    }

    total_combos = 3 * 3 * 2 * 2 * 2 * 2  # = 144 combinaisons

    print(f"\n📊 Paramètres du sweep:")
    print(f"   - Token: BTCUSDC")
    print(f"   - Période: 2024-01-01 → 2024-03-31 (3 mois)")
    print(f"   - Combinaisons: {total_combos:,}")
    print(f"   - Capital: 10,000 USDC")
    print(f"   - Levier: 3x")

    print(f"\n⏱️ Temps estimé:")
    print(f"   - Mode auto (8 workers): ~5-7 min")
    print(f"   - Mode manuel_30: ~30-60 sec (8-10x plus rapide)")

    # Demander confirmation
    response = input("\n🎯 Lancer le sweep ? (y/N): ")

    if response.lower() != "y":
        print("\n❌ Sweep annulé")
        return

    print("\n" + "=" * 60)
    print("🚀 LANCEMENT DU SWEEP")
    print("=" * 60 + "\n")

    try:
        # Lancement du sweep
        results = runner.run_grid(
            token="BTCUSDC",
            param_ranges=param_ranges,
            start_date="2024-01-01",
            end_date="2024-03-31",
            initial_capital=10000,
            leverage=3,
            fees_rate=0.0004,  # 0.04% fees
        )

        print("\n" + "=" * 60)
        print("✅ SWEEP TERMINÉ")
        print("=" * 60 + "\n")

        if results:
            # Top 5 résultats
            print("🏆 TOP 5 COMBINAISONS:\n")

            for i, result in enumerate(results[:5], 1):
                print(f"{i}. Sharpe: {result['metrics']['sharpe_ratio']:.2f}")
                print(f"   Return: {result['metrics']['total_return']*100:.2f}%")
                print(f"   Max DD: {result['metrics']['max_drawdown']*100:.2f}%")
                print(f"   Win Rate: {result['metrics']['win_rate']*100:.1f}%")
                print(f"   Params: {result['params']}")
                print()

            # Meilleure combinaison
            best = results[0]
            print(f"✨ MEILLEURE COMBINAISON:")
            print(f"   Sharpe: {best['metrics']['sharpe_ratio']:.2f}")
            print(f"   Return: {best['metrics']['total_return']*100:.2f}%")
            print(f"   Params: {best['params']}")

            # Optionnel: Générer un graphique
            try:
                from threadx.visualization.backtest_charts import (
                    generate_backtest_chart,
                )

                print("\n📊 Génération du graphique...")
                chart = generate_backtest_chart(
                    token="BTCUSDC",
                    start_date="2024-01-01",
                    end_date="2024-03-31",
                    params=best["params"],
                    initial_capital=10000,
                    leverage=3,
                )

                # Sauvegarder
                chart.write_html("best_backtest_manuel_30.html")
                print("✅ Graphique sauvegardé: best_backtest_manuel_30.html")

            except Exception as e:
                print(f"⚠️ Erreur génération graphique: {e}")

        else:
            print("⚠️ Aucun résultat valide trouvé")

    except KeyboardInterrupt:
        print("\n\n⚠️ Sweep interrompu par l'utilisateur")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
