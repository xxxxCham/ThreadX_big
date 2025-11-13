"""
ThreadX - Profilage des Imports et Communications Inter-Modules
================================================================

Script de profilage approfondi pour identifier les temps de chargement
et les dépendances critiques lors des sweeps.

Usage:
    python tools/profile_imports.py

Output:
    - Temps de chargement de chaque module
    - Graphe de dépendances
    - Bottlenecks identifiés
"""

import importlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ajouter le package root au sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ===================================
# 1. PROFILAGE DES IMPORTS
# ===================================

class ImportProfiler:
    """Profileur d'imports avec mesure temps réel."""

    def __init__(self):
        self.timings: dict[str, float] = {}
        self.dependencies: dict[str, list[str]] = defaultdict(list)
        self.original_import = __builtins__.__import__

    def __enter__(self):
        """Active le profilage."""
        __builtins__.__import__ = self._profiling_import
        return self

    def __exit__(self, *args):
        """Désactive le profilage."""
        __builtins__.__import__ = self.original_import

    def _profiling_import(self, name, *args, **kwargs):
        """Import hook avec mesure de temps."""
        start = time.perf_counter()
        module = self.original_import(name, *args, **kwargs)
        elapsed = time.perf_counter() - start

        # Enregistrer uniquement les modules threadx
        if name.startswith('threadx'):
            self.timings[name] = self.timings.get(name, 0) + elapsed

            # Tracker dépendances (caller → callee)
            frame = sys._getframe(1)
            caller = frame.f_globals.get('__name__', '<unknown>')
            if caller.startswith('threadx'):
                self.dependencies[caller].append(name)

        return module

    def print_report(self):
        """Affiche le rapport de profilage."""
        print("\n" + "="*80)
        print("RAPPORT DE PROFILAGE DES IMPORTS")
        print("="*80)

        # Tri par temps décroissant
        sorted_timings = sorted(
            self.timings.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\n{'Module':<50} {'Temps (ms)':<15} {'% Total':<10}")
        print("-" * 80)

        total_time = sum(self.timings.values())
        for module, duration in sorted_timings:
            pct = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{module:<50} {duration*1000:>10.2f} ms   {pct:>6.2f}%")

        print("-" * 80)
        print(f"{'TOTAL':<50} {total_time*1000:>10.2f} ms   100.00%")

        # Top 10 bottlenecks
        print("\n" + "="*80)
        print("TOP 10 BOTTLENECKS")
        print("="*80)
        for module, duration in sorted_timings[:10]:
            print(f"  {module}: {duration*1000:.2f} ms")

        # Graphe de dépendances (top modules)
        print("\n" + "="*80)
        print("GRAPHE DE DÉPENDANCES (TOP 15)")
        print("="*80)

        top_modules = [mod for mod, _ in sorted_timings[:15]]
        for module in top_modules:
            deps = [d for d in self.dependencies.get(module, []) if d in top_modules]
            if deps:
                print(f"\n{module} →")
                for dep in deps:
                    print(f"  └─ {dep}")


# ===================================
# 2. TEST DES IMPORTS CRITIQUES
# ===================================

def profile_critical_imports():
    """Profile les imports critiques du système de sweep."""

    print("\n" + "="*80)
    print("PROFILAGE DES IMPORTS CRITIQUES")
    print("="*80)

    critical_modules = [
        # Interface UI
        "threadx.streamlit_app",
        "threadx.ui.page_backtest_optimization",
        "threadx.ui.strategy_registry",

        # Moteur optimisation
        "threadx.optimization.engine",
        "threadx.optimization.scenarios",

        # Moteur backtest
        "threadx.backtest.engine",

        # Stratégies
        "threadx.strategy.bb_atr",
        "threadx.strategy.bollinger_dual",

        # Indicateurs
        "threadx.indicators.bollinger",
        "threadx.indicators.xatr",
        "threadx.indicators.bank",

        # GPU
        "threadx.gpu.device_manager",
        "threadx.gpu.multi_gpu",

        # Données
        "threadx.data_access",
    ]

    results = {}

    for module_name in critical_modules:
        print(f"\nImport de {module_name}...", end=" ")

        try:
            start = time.perf_counter()
            importlib.import_module(module_name)
            elapsed = time.perf_counter() - start

            results[module_name] = elapsed
            print(f"✅ {elapsed*1000:.2f} ms")

        except Exception as e:
            print(f"❌ ERREUR: {e}")
            results[module_name] = -1

    # Rapport final
    print("\n" + "="*80)
    print("RÉSUMÉ IMPORTS CRITIQUES")
    print("="*80)

    sorted_results = sorted(
        [(mod, t) for mod, t in results.items() if t >= 0],
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{'Module':<50} {'Temps (ms)':<15}")
    print("-" * 80)

    for module, duration in sorted_results:
        print(f"{module:<50} {duration*1000:>10.2f} ms")

    total = sum(t for t in results.values() if t >= 0)
    print("-" * 80)
    print(f"{'TOTAL':<50} {total*1000:>10.2f} ms")

    return results


# ===================================
# 3. ANALYSE CYCLE DE VIE SWEEP
# ===================================

def analyze_sweep_lifecycle():
    """Analyse le cycle de vie complet d'un sweep (sans exécution réelle)."""

    print("\n" + "="*80)
    print("ANALYSE CYCLE DE VIE SWEEP")
    print("="*80)

    steps = []

    # Step 1: Import des modules
    step_start = time.perf_counter()
    from threadx.optimization.engine import SweepRunner
    from threadx.indicators.bank import IndicatorBank, IndicatorSettings
    from threadx.optimization.scenarios import ScenarioSpec
    steps.append(("Import modules optimisation", time.perf_counter() - step_start))

    # Step 2: Création IndicatorBank
    step_start = time.perf_counter()
    settings = IndicatorSettings(
        use_gpu=True,
        cache_dir="indicators_cache",
        ttl_seconds=3600
    )
    bank = IndicatorBank(settings=settings)
    steps.append(("Création IndicatorBank", time.perf_counter() - step_start))

    # Step 3: Création SweepRunner
    step_start = time.perf_counter()
    runner = SweepRunner(
        indicator_bank=bank,
        max_workers=30,
        use_multigpu=True
    )
    steps.append(("Création SweepRunner", time.perf_counter() - step_start))

    # Step 4: Génération ScenarioSpec
    step_start = time.perf_counter()
    spec = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"min": 10, "max": 50, "step": 5},
            "bb_std": {"min": 1.5, "max": 3.0, "step": 0.5},
            "atr_period": {"min": 7, "max": 21, "step": 2},
            "atr_multiplier": {"min": 1.0, "max": 3.0, "step": 0.5},
        }
    )
    steps.append(("Génération ScenarioSpec", time.perf_counter() - step_start))

    # Step 5: Import stratégie
    step_start = time.perf_counter()
    from threadx.strategy.bb_atr import BBAtrStrategy
    steps.append(("Import stratégie BB+ATR", time.perf_counter() - step_start))

    # Step 6: Création instance stratégie
    step_start = time.perf_counter()
    strategy = BBAtrStrategy(symbol="BTCUSDC", timeframe="15m")
    steps.append(("Création instance stratégie", time.perf_counter() - step_start))

    # Rapport
    print("\n" + "="*80)
    print("TEMPS PAR ÉTAPE")
    print("="*80)

    print(f"\n{'Étape':<50} {'Temps (ms)':<15}")
    print("-" * 80)

    for step_name, duration in steps:
        print(f"{step_name:<50} {duration*1000:>10.2f} ms")

    total = sum(duration for _, duration in steps)
    print("-" * 80)
    print(f"{'TOTAL OVERHEAD INITIALISATION':<50} {total*1000:>10.2f} ms")

    return steps


# ===================================
# 4. MAIN
# ===================================

def main():
    """Point d'entrée principal."""

    print("\n" + "="*80)
    print("ThreadX - Profilage des Imports et Communications")
    print("="*80)

    # 1. Profilage complet avec hook
    print("\n[1/3] Profilage complet des imports...")
    with ImportProfiler() as profiler:
        profile_critical_imports()

    profiler.print_report()

    # 2. Analyse cycle de vie
    print("\n[2/3] Analyse cycle de vie sweep...")
    analyze_sweep_lifecycle()

    # 3. Recommandations
    print("\n" + "="*80)
    print("RECOMMANDATIONS")
    print("="*80)

    print("""
    1. Imports Lourds:
       - Identifier modules > 100ms
       - Considérer lazy loading si possible
       - Vérifier imports circulaires

    2. Optimisations GPU:
       - Vérifier si GPU Manager est initialisé trop tôt
       - Considérer singleton GPU Manager global

    3. IndicatorBank:
       - Vérifier taille cache TTL
       - Analyser hit/miss ratio

    4. Stratégies:
       - Confirmer que caching fonctionne
       - Vérifier overhead Numba JIT compilation

    5. Communication Inter-Modules:
       - Tracer appels répétés inutiles
       - Optimiser serialization/deserialization
    """)

    print("\n✅ Profilage terminé!\n")


if __name__ == "__main__":
    main()
