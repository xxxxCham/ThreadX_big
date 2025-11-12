"""
ThreadX Profiling Baseline - Étape 2 : Breakdown par composant
===============================================================

Profiling détaillé avec timers manuels pour identifier précisément
où le temps est consommé dans chaque composant.

Mesure :
- Temps génération signaux
- Temps simulation trades
- Temps calcul métriques
- Temps ensure_indicators
- Overhead multiprocessing

Usage:
    python profiling_baseline_step2.py

Outputs:
    - profiling_breakdown_report.txt
    - Console output avec % de temps par composant
"""

import sys
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.threadx.optimization.engine import UnifiedOptimizationEngine
from src.threadx.optimization.scenarios import generate_param_grid
from src.threadx.data_access import load_ohlcv


class ComponentTimer:
    """Timer pour mesurer le temps par composant."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.current_section = None
        self.section_start = None

    def start(self, section_name: str):
        """Démarre le timer pour une section."""
        self.current_section = section_name
        self.section_start = time.perf_counter()

    def stop(self):
        """Arrête le timer et enregistre le temps."""
        if self.section_start is not None:
            elapsed = time.perf_counter() - self.section_start
            self.timings[self.current_section].append(elapsed)
            self.section_start = None
            self.current_section = None

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Retourne les statistiques par section."""
        stats = {}
        total_time = sum(sum(times) for times in self.timings.values())

        for section, times in self.timings.items():
            total = sum(times)
            stats[section] = {
                "total": total,
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "count": len(times),
                "pct": (total / total_time * 100) if total_time > 0 else 0,
            }

        return stats

    def print_report(self):
        """Affiche un rapport formaté."""
        stats = self.get_stats()

        print("\n" + "=" * 100)
        print("⏱️  PROFILING BREAKDOWN PAR COMPOSANT")
        print("=" * 100)

        # Tri par % de temps décroissant
        sorted_sections = sorted(stats.items(), key=lambda x: x[1]["pct"], reverse=True)

        print(
            f"\n{'Composant':<40} {'Total (s)':<12} {'Mean (ms)':<12} {'%':<8} {'Calls':<8}"
        )
        print("-" * 100)

        for section, data in sorted_sections:
            print(
                f"{section:<40} {data['total']:>11.3f}s {data['mean']*1000:>11.2f}ms {data['pct']:>6.1f}% {data['count']:>7d}"
            )

        print("-" * 100)
        total_all = sum(d["total"] for d in stats.values())
        print(f"{'TOTAL':<40} {total_all:>11.3f}s")
        print("=" * 100)


def patch_backtest_engine_for_profiling(timer: ComponentTimer):
    """Patch BacktestEngine pour ajouter des timers."""
    from src.threadx.backtest.engine import BacktestEngine

    # Sauvegarder les méthodes originales
    original_generate_signals = BacktestEngine.generate_signals
    original_backtest_full = BacktestEngine.backtest_full

    # Wrapper pour generate_signals
    def timed_generate_signals(self, *args, **kwargs):
        timer.start("generate_signals")
        result = original_generate_signals(self, *args, **kwargs)
        timer.stop()
        return result

    # Wrapper pour backtest_full
    def timed_backtest_full(self, *args, **kwargs):
        timer.start("backtest_full")
        result = original_backtest_full(self, *args, **kwargs)
        timer.stop()
        return result

    # Application des patches
    BacktestEngine.generate_signals = timed_generate_signals
    BacktestEngine.backtest_full = timed_backtest_full


def patch_indicator_bank_for_profiling(timer: ComponentTimer):
    """Patch IndicatorBank pour mesurer ensure_indicators."""
    from src.threadx.indicators.bank import IndicatorBank

    original_ensure = IndicatorBank.ensure_indicators

    def timed_ensure_indicators(self, *args, **kwargs):
        timer.start("ensure_indicators")
        result = original_ensure(self, *args, **kwargs)
        timer.stop()
        return result

    IndicatorBank.ensure_indicators = timed_ensure_indicators


def run_profiled_sweep(timer: ComponentTimer):
    """Exécute un sweep avec profiling des composants."""

    print("=" * 100)
    print("🔬 PROFILING BREAKDOWN - ThreadX Composants")
    print("=" * 100)

    # Chargement données
    print("\n📊 Chargement données...")
    timer.start("data_loading")
    try:
        df = load_ohlcv("BTCUSDT", "1h")
        print(f"   ✅ {len(df)} barres chargées")
    except Exception as e:
        print(f"   ⚠️  Utilisation données synthétiques")
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
    timer.stop()

    # Génération paramètres
    print("\n⚙️  Génération paramètres...")
    timer.start("param_generation")
    param_spec = {
        "bb_period": [20, 30, 40],
        "bb_std": [1.5, 2.0, 2.5],
        "atr_multiplier": [1.5, 2.0],
        "entry_z": [1.0, 1.5],
        "risk_per_trade": [0.01, 0.02],
    }
    combinations = generate_param_grid(param_spec)
    print(f"   ✅ {len(combinations)} combinaisons")
    timer.stop()

    # Configuration engine
    print("\n🚀 Configuration engine...")
    timer.start("engine_setup")
    engine = UnifiedOptimizationEngine(
        max_workers=8, use_gpu=False, device_override="cpu"
    )
    timer.stop()

    # Exécution sweep
    print("\n⏱️  DÉBUT SWEEP PROFILÉ...")
    print("-" * 100)

    timer.start("sweep_execution_total")
    results_df = engine.run_sweep(
        params=param_spec,
        data=df,
        symbol="BTCUSDT",
        timeframe="1h",
        initial_capital=10000.0,
        reuse_cache=True,
    )
    timer.stop()

    print("-" * 100)
    print(f"✅ SWEEP TERMINÉ - {len(results_df)} résultats\n")

    return results_df


def main():
    """Point d'entrée principal."""

    # Création du timer
    timer = ComponentTimer()

    # Application des patches de profiling
    print("🔧 Application des patches de profiling...")
    patch_backtest_engine_for_profiling(timer)
    patch_indicator_bank_for_profiling(timer)
    print("   ✅ Patches appliqués\n")

    # Exécution du sweep profilé
    results = run_profiled_sweep(timer)

    # Affichage du rapport
    timer.print_report()

    # Sauvegarde rapport
    output_file = Path("profiling_breakdown_report.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        stats = timer.get_stats()

        f.write("=" * 100 + "\n")
        f.write("THREADX PROFILING BREAKDOWN REPORT\n")
        f.write("=" * 100 + "\n\n")

        sorted_sections = sorted(stats.items(), key=lambda x: x[1]["pct"], reverse=True)

        f.write(
            f"{'Composant':<40} {'Total (s)':<12} {'Mean (ms)':<12} {'Median (ms)':<12} {'%':<8} {'Calls':<8}\n"
        )
        f.write("-" * 100 + "\n")

        for section, data in sorted_sections:
            f.write(
                f"{section:<40} {data['total']:>11.3f}s {data['mean']*1000:>11.2f}ms {data['median']*1000:>11.2f}ms {data['pct']:>6.1f}% {data['count']:>7d}\n"
            )

        f.write("-" * 100 + "\n")
        total = sum(d["total"] for d in stats.values())
        f.write(f"{'TOTAL':<40} {total:>11.3f}s\n")
        f.write("\n")

        # Détails par section
        f.write("\n" + "=" * 100 + "\n")
        f.write("DÉTAILS PAR COMPOSANT\n")
        f.write("=" * 100 + "\n\n")

        for section, data in sorted_sections:
            f.write(f"\n{section}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Total time:   {data['total']:.3f}s ({data['pct']:.1f}%)\n")
            f.write(f"  Mean time:    {data['mean']*1000:.2f}ms\n")
            f.write(f"  Median time:  {data['median']*1000:.2f}ms\n")
            f.write(f"  Std dev:      {data['std']*1000:.2f}ms\n")
            f.write(f"  Min time:     {data['min']*1000:.2f}ms\n")
            f.write(f"  Max time:     {data['max']*1000:.2f}ms\n")
            f.write(f"  Call count:   {data['count']}\n")

    print(f"\n💾 Rapport détaillé sauvé: {output_file}")

    # Calcul tests/sec
    stats = timer.get_stats()
    if "sweep_execution_total" in stats:
        total_time = stats["sweep_execution_total"]["total"]
        n_combos = len(results)
        tests_per_sec = n_combos / total_time

        print("\n" + "=" * 100)
        print("📊 PERFORMANCE BASELINE")
        print("=" * 100)
        print(f"  Combinaisons testées: {n_combos}")
        print(f"  Temps total:          {total_time:.2f}s")
        print(f"  ⚡ VITESSE:            {tests_per_sec:.1f} tests/sec")
        print(f"  Temps/combo:          {(total_time/n_combos)*1000:.2f}ms")
        print("=" * 100)


if __name__ == "__main__":
    main()
