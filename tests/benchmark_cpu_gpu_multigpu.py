"""
Benchmark intensif: Comparaison CPU vs GPU vs Multi-GPU
========================================================

Teste la performance du moteur ThreadX sur une grille paramétrique large:
- 320 scénarios (16 bb_period × 5 bb_std × 4 atr_mult)
- 2000 barres de données synthétiques
- 30 workers pour tous les modes

Génère des graphiques de comparaison et sauvegarde les résultats JSON.

Author: ThreadX Framework
Version: 1.0
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from threadx.optimization.engine import SweepRunner
from threadx.config import get_settings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_test_data(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Génère des données OHLCV synthétiques pour le benchmark.

    Args:
        n_bars: Nombre de barres
        seed: Seed pour reproductibilité

    Returns:
        DataFrame avec index DatetimeIndex et colonnes OHLCV
    """
    np.random.seed(seed)

    # Index temporel (barres 1h sur ~83 jours)
    start_date = pd.Timestamp("2024-01-01", tz="UTC")
    dates = pd.date_range(start=start_date, periods=n_bars, freq="1h")

    # Génération prix (random walk avec tendance)
    base_price = 50000.0
    returns = np.random.normal(0.0001, 0.02, n_bars)  # Drift + volatilité
    close = base_price * np.exp(np.cumsum(returns))

    # OHLC avec réalisme
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
    open_price = close * np.random.uniform(0.995, 1.005, n_bars)

    # Volume aléatoire
    volume = np.random.lognormal(15, 1, n_bars)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    logger.info(
        f"✅ Données générées: {len(df)} barres, {df.index[0]} → {df.index[-1]}"
    )
    return df


def run_benchmark(
    mode: str,
    n_scenarios: int = 320,
    n_bars: int = 2000,
    n_workers: int = 30,
) -> Dict[str, Any]:
    """
    Exécute un benchmark pour un mode donné (cpu, gpu, multi-gpu).

    Args:
        mode: "cpu", "gpu", ou "multi-gpu"
        n_scenarios: Nombre de scénarios à tester
        n_bars: Nombre de barres de données
        n_workers: Nombre de workers parallèles

    Returns:
        Dict avec résultats (mode, durée, vitesse, etc.)
    """
    # Configuration mode
    use_gpu = mode in ["gpu", "multi-gpu"]
    use_multigpu = mode == "multi-gpu"

    # Génération données
    df = generate_test_data(n_bars=n_bars)

    # Configuration runner
    runner = SweepRunner(
        max_workers=n_workers,
        use_multigpu=use_multigpu,
    )

    # Spec de la grille
    # 16 × 5 × 4 = 320 combinaisons
    grid_spec = {
        "name": f"{mode.upper()} Benchmark",
        "description": f"Test intensif {mode}",
        "type": "grid",
        "params": {
            "bb_period": list(range(10, 42, 2)),  # 16 valeurs: 10,12,...,40
            "bb_std": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5 valeurs
            "atr_mult": [1.0, 1.5, 2.0, 2.5],  # 4 valeurs
        },
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 BENCHMARK: {mode.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"  Scénarios: {n_scenarios}")
    logger.info(f"  Barres: {n_bars}")
    logger.info(f"  Workers: {n_workers}")
    logger.info(f"  GPU: {use_gpu}, Multi-GPU: {use_multigpu}")
    logger.info("")

    # Exécution du sweep
    start = time.time()
    try:
        results = runner.run_grid(
            grid_spec=grid_spec,
            real_data=df,
            symbol="BTCUSDC",
            timeframe="1h",
            strategy_name="Bollinger_Breakout",
            reuse_cache=False,
        )
        duration = time.time() - start

        # Métriques
        n_executed = len(results)
        speed = n_executed / duration if duration > 0 else 0

        logger.info(f"\n⏱️  Durée totale: {duration:.2f}s")
        logger.info(f"🚀 Vitesse: {speed:.2f} scénarios/sec")
        logger.info(f"✅ Scénarios exécutés: {n_executed}")

        return {
            "mode": mode,
            "n_scenarios": n_scenarios,
            "n_bars": n_bars,
            "workers": n_workers,
            "duration": duration,
            "speed": speed,
            "n_executed": n_executed,
            "success": True,
        }

    except Exception as e:
        logger.error(f"❌ Erreur benchmark {mode}: {e}")
        return {
            "mode": mode,
            "n_scenarios": n_scenarios,
            "n_bars": n_bars,
            "workers": n_workers,
            "duration": 0.0,
            "speed": 0.0,
            "n_executed": 0,
            "success": False,
            "error": str(e),
        }


def plot_results(results: list[Dict[str, Any]]) -> None:
    """
    Génère des graphiques de comparaison des benchmarks.

    Args:
        results: Liste des résultats de benchmark
    """
    # Filtrer les succès uniquement
    valid_results = [r for r in results if r.get("success", False)]

    if not valid_results:
        logger.warning("⚠️ Aucun résultat valide à tracer")
        return

    # DataFrame pour seaborn
    df = pd.DataFrame(valid_results)

    # Style seaborn
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Durée
    sns.barplot(data=df, x="mode", y="duration", ax=axes[0], palette="viridis")
    axes[0].set_title("⏱️ Durée d'exécution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Mode", fontsize=12)
    axes[0].set_ylabel("Durée (secondes)", fontsize=12)
    axes[0].grid(axis="y", alpha=0.3)

    # 2) Vitesse
    sns.barplot(data=df, x="mode", y="speed", ax=axes[1], palette="coolwarm")
    axes[1].set_title("🚀 Vitesse de traitement", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Mode", fontsize=12)
    axes[1].set_ylabel("Scénarios/sec", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3)

    # 3) Speedup (ratio vs CPU)
    if "cpu" in df["mode"].values:
        cpu_duration = df.loc[df["mode"] == "cpu", "duration"].iloc[0]
        df["speedup"] = cpu_duration / df["duration"]

        sns.barplot(data=df, x="mode", y="speedup", ax=axes[2], palette="rocket")
        axes[2].set_title("📈 Speedup (vs CPU)", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("Mode", fontsize=12)
        axes[2].set_ylabel("Facteur d'accélération", fontsize=12)
        axes[2].axhline(
            y=1.0, color="red", linestyle="--", linewidth=1.5, label="Baseline CPU"
        )
        axes[2].legend()
        axes[2].grid(axis="y", alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "CPU benchmark manquant",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[2].transAxes,
        )

    plt.tight_layout()

    # Sauvegarde
    output_path = Path("benchmark_cpu_gpu_multigpu.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"📊 Graphiques sauvegardés: {output_path.absolute()}")

    plt.show()


def main():
    """Point d'entrée principal du benchmark."""
    logger.info("=" * 80)
    logger.info("🏁 BENCHMARK INTENSIF : CPU vs GPU vs Multi-GPU")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Scénarios: 320 (16×5×4 grille)")
    logger.info("  - Barres: 2000")
    logger.info("  - Workers: 30 (tous modes)")
    logger.info("")

    # Exécution des 3 benchmarks
    results = []

    # 1) CPU
    result_cpu = run_benchmark("cpu", n_scenarios=320, n_bars=2000)
    results.append(result_cpu)

    # 2) GPU
    result_gpu = run_benchmark("gpu", n_scenarios=320, n_bars=2000)
    results.append(result_gpu)

    # 3) Multi-GPU
    result_multigpu = run_benchmark("multi-gpu", n_scenarios=320, n_bars=2000)
    results.append(result_multigpu)

    # Sauvegarde JSON
    output_json = Path("benchmark_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Résultats JSON sauvegardés: {output_json.absolute()}")

    # Génération graphiques
    plot_results(results)

    # Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 80)
    for r in results:
        if r.get("success"):
            logger.info(
                f"{r['mode'].upper():12s} | "
                f"Durée: {r['duration']:7.2f}s | "
                f"Vitesse: {r['speed']:6.2f} scénarios/sec | "
                f"Scénarios: {r['n_executed']}"
            )
        else:
            logger.info(
                f"{r['mode'].upper():12s} | ❌ Erreur: {r.get('error', 'Unknown')}"
            )

    logger.info("=" * 80)
    logger.info("✅ Benchmark terminé avec succès!")


if __name__ == "__main__":
    main()
