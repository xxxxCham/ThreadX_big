"""
ThreadX Profiling Baseline - Étape 1 : cProfile complet
========================================================

Mesure baseline complète du système actuel avec cProfile.
Identifie les fonctions les plus coûteuses en temps CPU.

Usage:
    python profiling_baseline_step1.py

Outputs:
    - profiling_baseline.prof (données brutes)
    - profiling_baseline_report.txt (rapport texte)
    - Visualisation avec snakeviz si disponible
"""

import cProfile
import pstats
import io
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

from src.threadx.optimization.engine import UnifiedOptimizationEngine
from src.threadx.optimization.scenarios import generate_param_grid
from src.threadx.data_access import load_ohlcv
import pandas as pd


def run_baseline_sweep():
    """Exécute un sweep de taille raisonnable pour profiling."""

    print("=" * 80)
    print("🔬 PROFILING BASELINE - ThreadX Optimization Engine")
    print("=" * 80)
    print()

    # Chargement données test
    print("📊 Chargement données OHLCV...")
    try:
        df = load_ohlcv("BTCUSDT", "1h")
        print(f"   ✅ {len(df)} barres chargées")
    except Exception as e:
        print(f"   ⚠️  Erreur chargement: {e}")
        print("   📝 Utilisation de données synthétiques...")
        import numpy as np

        dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
        df = pd.DataFrame(
            {
                "open": np.random.randn(1000).cumsum() + 100,
                "high": np.random.randn(1000).cumsum() + 102,
                "low": np.random.randn(1000).cumsum() + 98,
                "close": np.random.randn(1000).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 1000),
            },
            index=dates,
        )

    # Génération grille de paramètres (taille réduite pour profiling)
    print("\n⚙️  Génération grille de paramètres...")
    param_spec = {
        "bb_period": [20, 30, 40],  # 3 valeurs
        "bb_std": [1.5, 2.0, 2.5],  # 3 valeurs
        "atr_multiplier": [1.5, 2.0],  # 2 valeurs
        "entry_z": [1.0, 1.5],  # 2 valeurs
        "risk_per_trade": [0.01, 0.02],  # 2 valeurs
    }
    # Total: 3 * 3 * 2 * 2 * 2 = 72 combinaisons
    combinations = generate_param_grid(param_spec)
    print(f"   ✅ {len(combinations)} combinaisons générées")

    # Configuration engine
    print("\n🚀 Configuration OptimizationEngine...")
    engine = UnifiedOptimizationEngine(
        max_workers=8,  # Réduit pour profiling plus clair
        use_gpu=False,  # CPU seulement pour profiling initial
        device_override="cpu",
    )
    print(f"   ✅ Engine configuré (max_workers=8)")

    # Exécution du sweep
    print("\n⏱️  DÉBUT PROFILING SWEEP...")
    print("-" * 80)

    results_df = engine.run_sweep(
        params=param_spec,
        data=df,
        symbol="BTCUSDT",
        timeframe="1h",
        initial_capital=10000.0,
        reuse_cache=True,
    )

    print("-" * 80)
    print(f"✅ PROFILING TERMINÉ")
    print(f"   📊 {len(results_df)} résultats générés")
    print()

    return results_df


def main():
    """Point d'entrée principal avec profiling cProfile."""

    # Création du profiler
    profiler = cProfile.Profile()

    # Activation profiling
    print("🔬 Activation cProfile...\n")
    profiler.enable()

    try:
        # Exécution du sweep profilé
        results = run_baseline_sweep()

    finally:
        # Désactivation profiling
        profiler.disable()
        print("\n🔬 Profiling terminé, génération des rapports...\n")

    # Sauvegarde données brutes
    output_prof = Path("profiling_baseline.prof")
    profiler.dump_stats(str(output_prof))
    print(f"💾 Données brutes sauvées: {output_prof}")

    # Génération rapport texte
    output_txt = Path("profiling_baseline_report.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("THREADX PROFILING BASELINE REPORT\n")
        f.write("=" * 100 + "\n\n")

        # Stats triées par temps cumulatif
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)

        f.write("📊 TOP 50 FONCTIONS PAR TEMPS CUMULATIF\n")
        f.write("-" * 100 + "\n")
        ps.sort_stats("cumulative")
        ps.print_stats(50)
        f.write(s.getvalue())

        # Stats triées par temps interne
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)

        f.write("\n\n📊 TOP 50 FONCTIONS PAR TEMPS INTERNE (TOTTIME)\n")
        f.write("-" * 100 + "\n")
        ps.sort_stats("tottime")
        ps.print_stats(50)
        f.write(s.getvalue())

        # Callers des fonctions les plus coûteuses
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)

        f.write("\n\n📊 CALLERS DES TOP 20 FONCTIONS\n")
        f.write("-" * 100 + "\n")
        ps.sort_stats("cumulative")
        ps.print_callers(20)
        f.write(s.getvalue())

    print(f"📄 Rapport texte sauvé: {output_txt}")

    # Affichage résumé dans console
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ TOP 20 FONCTIONS (TEMPS CUMULATIF)")
    print("=" * 80)
    ps = pstats.Stats(profiler)
    ps.sort_stats("cumulative")
    ps.print_stats(20)

    # Instructions snakeviz
    print("\n" + "=" * 80)
    print("📈 VISUALISATION INTERACTIVE")
    print("=" * 80)
    print(f"Pour visualiser avec snakeviz (installer si nécessaire):")
    print(f"  pip install snakeviz")
    print(f"  snakeviz {output_prof}")
    print()


if __name__ == "__main__":
    main()
