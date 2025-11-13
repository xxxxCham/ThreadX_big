"""
ThreadX - Profilage Runtime d'un Sweep Réel
============================================

Profileur détaillé qui trace les appels de fonctions et leur temps d'exécution
pendant un sweep paramétrique réel (mais avec peu de combinaisons).

Usage:
    python tools/profile_runtime_sweep.py

Output:
    - Temps par fonction
    - Nombre d'appels par fonction
    - Call tree complet
    - Bottlenecks runtime identifiés
"""

import cProfile
import io
import pstats
import sys
import time
from datetime import date
from pathlib import Path
from pstats import SortKey

# Ajouter le package root au sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Imports threadx
from threadx.data_access import load_ohlcv
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec


def run_minimal_sweep():
    """Exécute un sweep minimal pour profilage (peu de combinaisons)."""

    print("\n" + "="*80)
    print("EXÉCUTION SWEEP MINIMAL POUR PROFILAGE")
    print("="*80)

    # Configuration
    symbol = "BTCUSDC"
    timeframe = "15m"
    start = date(2024, 12, 1)
    end = date(2024, 12, 15)  # Seulement 15 jours pour rapidité

    # Chargement données
    print(f"\n[1/5] Chargement données {symbol} {timeframe} ({start} → {end})...")
    load_start = time.perf_counter()
    data = load_ohlcv(symbol, timeframe, start, end)
    load_time = time.perf_counter() - load_start
    print(f"  ✅ {len(data)} barres chargées en {load_time*1000:.2f} ms")

    # Création IndicatorBank
    print("\n[2/5] Création IndicatorBank...")
    bank_start = time.perf_counter()
    settings = IndicatorSettings(
        use_gpu=True,
        cache_dir="indicators_cache",
        ttl_seconds=3600,
        max_workers=8
    )
    bank = IndicatorBank(settings=settings)
    bank_time = time.perf_counter() - bank_start
    print(f"  ✅ IndicatorBank créé en {bank_time*1000:.2f} ms")

    # Création SweepRunner
    print("\n[3/5] Création SweepRunner...")
    runner_start = time.perf_counter()
    runner = SweepRunner(
        indicator_bank=bank,
        max_workers=4,  # Réduire workers pour profilage
        use_multigpu=True
    )
    runner_time = time.perf_counter() - runner_start
    print(f"  ✅ SweepRunner créé en {runner_time*1000:.2f} ms")

    # Définition du sweep (MINIMAL: 2x2x2 = 8 combinaisons)
    print("\n[4/5] Définition ScenarioSpec (8 combinaisons)...")
    spec = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"min": 20, "max": 30, "step": 10},  # 2 valeurs: 20, 30
            "bb_std": {"min": 2.0, "max": 2.5, "step": 0.5},  # 2 valeurs: 2.0, 2.5
            "atr_period": {"min": 14, "max": 21, "step": 7},  # 2 valeurs: 14, 21
        }
    )

    # Exécution du sweep
    print("\n[5/5] Exécution du sweep...")
    sweep_start = time.perf_counter()

    results = runner.run_grid(
        grid_spec=spec,
        real_data=data,
        symbol=symbol,
        timeframe=timeframe,
        strategy_name="Bollinger_Breakout"
    )

    sweep_time = time.perf_counter() - sweep_start

    print(f"\n✅ Sweep terminé!")
    print(f"  - Temps total: {sweep_time*1000:.2f} ms ({sweep_time:.2f}s)")
    print(f"  - Résultats: {len(results)} combinaisons")
    if len(results) > 0:
        print(f"  - Vitesse: {len(results) / sweep_time:.2f} tests/sec")

    return {
        "load_time": load_time,
        "bank_time": bank_time,
        "runner_time": runner_time,
        "sweep_time": sweep_time,
        "results": results,
        "num_combinations": len(results),
    }


def profile_with_cprofile():
    """Profile le sweep avec cProfile."""

    print("\n" + "="*80)
    print("PROFILAGE CPROFILE")
    print("="*80)

    profiler = cProfile.Profile()

    # Exécution profilée
    profiler.enable()
    metrics = run_minimal_sweep()
    profiler.disable()

    # Analyse des résultats
    print("\n" + "="*80)
    print("STATISTIQUES CPROFILE")
    print("="*80)

    # Buffer pour capturer output
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    # Tri par temps cumulatif
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)

    print("\n--- TOP 30 FONCTIONS PAR TEMPS CUMULATIF ---\n")
    stats.print_stats(30)
    print(stream.getvalue())

    # Tri par temps propre (time)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats(SortKey.TIME)

    print("\n" + "="*80)
    print("--- TOP 30 FONCTIONS PAR TEMPS PROPRE ---")
    print("="*80 + "\n")
    stats.print_stats(30)
    print(stream.getvalue())

    # Analyse des appelants (callers)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()

    print("\n" + "="*80)
    print("--- APPELANTS DES FONCTIONS CRITIQUES ---")
    print("="*80 + "\n")

    # Focus sur les fonctions threadx critiques
    critical_funcs = [
        "_execute_combinations",
        "_evaluate_single_combination",
        "_compute_batch_indicators",
        "backtest",
        "compute_bollinger",
        "compute_atr",
        "batch_ensure",
    ]

    for func_name in critical_funcs:
        print(f"\n>>> Appelants de '{func_name}':")
        try:
            stats.print_callers(func_name)
            print(stream.getvalue())
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs()
        except Exception as e:
            print(f"  (Fonction non trouvée: {e})")

    return metrics


def analyze_bottlenecks(metrics: dict):
    """Analyse les bottlenecks à partir des métriques."""

    print("\n" + "="*80)
    print("ANALYSE DES BOTTLENECKS")
    print("="*80)

    total_time = (
        metrics["load_time"]
        + metrics["bank_time"]
        + metrics["runner_time"]
        + metrics["sweep_time"]
    )

    print(f"\n{'Phase':<30} {'Temps (ms)':<15} {'% Total':<10}")
    print("-" * 80)

    phases = [
        ("Chargement données", metrics["load_time"]),
        ("Création IndicatorBank", metrics["bank_time"]),
        ("Création SweepRunner", metrics["runner_time"]),
        ("Exécution sweep", metrics["sweep_time"]),
    ]

    for phase_name, duration in phases:
        pct = (duration / total_time * 100) if total_time > 0 else 0
        print(f"{phase_name:<30} {duration*1000:>10.2f} ms   {pct:>6.2f}%")

    print("-" * 80)
    print(f"{'TOTAL':<30} {total_time*1000:>10.2f} ms   100.00%")

    # Métriques de performance
    print("\n" + "="*80)
    print("MÉTRIQUES DE PERFORMANCE")
    print("="*80)

    if metrics["num_combinations"] > 0:
        speed = metrics["num_combinations"] / metrics["sweep_time"]
        time_per_combo = metrics["sweep_time"] / metrics["num_combinations"] * 1000

        print(f"\n  Combinaisons testées: {metrics['num_combinations']}")
        print(f"  Vitesse: {speed:.2f} tests/sec")
        print(f"  Temps moyen/combo: {time_per_combo:.2f} ms")

        # Extrapolation pour grand sweep
        print("\n" + "-"*80)
        print("EXTRAPOLATION POUR SWEEP RÉEL")
        print("-" * 80)

        # Supposons 2903040 combinaisons (votre sweep actuel)
        real_sweep_combos = 2903040
        estimated_time_sec = real_sweep_combos / speed
        estimated_time_min = estimated_time_sec / 60
        estimated_time_hours = estimated_time_min / 60

        print(f"\n  Pour {real_sweep_combos:,} combinaisons:")
        print(f"    - Temps estimé: {estimated_time_sec:,.0f} sec")
        print(f"    - Soit: {estimated_time_min:,.2f} minutes")
        print(f"    - Soit: {estimated_time_hours:,.2f} heures")

        print("\n  ⚠️ ATTENTION:")
        print("    - Cette extrapolation suppose vitesse constante")
        print("    - Contention GPU/RAM peut réduire vitesse sur gros sweep")
        print("    - Cache IndicatorBank peut accélérer (hit ratio)")


def main():
    """Point d'entrée principal."""

    print("\n" + "="*80)
    print("ThreadX - Profilage Runtime d'un Sweep Réel")
    print("="*80)

    # Profiler avec cProfile
    metrics = profile_with_cprofile()

    # Analyser bottlenecks
    analyze_bottlenecks(metrics)

    print("\n✅ Profilage runtime terminé!\n")


if __name__ == "__main__":
    main()
