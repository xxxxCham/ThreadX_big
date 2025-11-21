#!/usr/bin/env python
"""
POC Orchestrateur Multi-Agents - Test Workflow Autonome
========================================================

Script de démonstration du système multi-agents autonome ThreadX v2.0.

Ce POC teste:
1. Orchestrateur coordination 3 agents
2. Boucle optimisation 7 étapes
3. Convergence automatique
4. Mémoire évitant repropositions
5. Export résultats

Usage:
    python tools/poc_orchestrator.py

Prérequis:
    - Ollama running avec models: deepseek-r1:70b, gpt-oss:20b
    - GPU disponible pour backtests
    - Données OHLCV (demo: génération synthétique)

Durée estimée: 10-15 minutes (5 itérations)

Author: ThreadX Framework
Version: 1.0 - Multi-Agent POC
"""

import sys
from pathlib import Path

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

import numpy as np
import pandas as pd

from threadx.llm.orchestrator import OptimizationConfig, OptimizationOrchestrator
from threadx.utils.log import get_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = get_logger(__name__)


def generate_synthetic_ohlcv(
    n_bars: int = 10_000, timeframe: str = "15m", seed: int = 42
) -> pd.DataFrame:
    """
    Génère données OHLCV synthétiques pour POC.

    Args:
        n_bars: Nombre de barres
        timeframe: Timeframe (ex: '15m', '1h')
        seed: Seed random pour reproductibilité

    Returns:
        DataFrame avec colonnes [timestamp, open, high, low, close, volume]
    """
    logger.info(f"Generating {n_bars} synthetic OHLCV bars ({timeframe})...")

    np.random.seed(seed)

    # Générer prix avec marche aléatoire + tendance
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_bars)  # Drift + volatilité
    prices = base_price * np.exp(np.cumsum(returns))

    # OHLC autour du prix de clôture
    close = prices
    open_prices = close * (1 + np.random.normal(0, 0.005, n_bars))
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))

    # Volume aléatoire
    volume = np.random.lognormal(10, 1, n_bars)

    # Timestamps UTC datetime64 (requis par BacktestEngine)
    start_time = pd.Timestamp("2023-01-01", tz="UTC")
    freq_map = {"15m": "15min", "1h": "1H", "1d": "1D"}
    timestamps = pd.date_range(start=start_time, periods=n_bars, freq=freq_map.get(timeframe, "15min"), tz="UTC")

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=timestamps,  # Index datetime64 UTC
    )

    logger.info(f"✅ Generated {len(df)} bars (price range: {df['close'].min():.2f} - {df['close'].max():.2f})")

    return df


def main():
    """Fonction principale POC."""
    logger.info("=" * 80)
    logger.info("🤖 POC ORCHESTRATEUR MULTI-AGENTS - ThreadX v2.0")
    logger.info("=" * 80)

    # ==========================================================================
    # 1. GÉNÉRATION DONNÉES SYNTHÉTIQUES
    # ==========================================================================
    data = generate_synthetic_ohlcv(n_bars=5000, timeframe="15m", seed=42)

    # ==========================================================================
    # 2. CONFIGURATION OPTIMISATION
    # ==========================================================================
    config = OptimizationConfig(
        strategy_name="ma_crossover",  # Stratégie MA crossover
        initial_params={
            "fast_period": 10,
            "slow_period": 20,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 2.0,
        },
        target_sharpe=1.8,  # Objectif Sharpe 1.8
        max_iterations=5,  # 5 itérations max pour POC
        convergence_threshold=2,  # Arrêt si 2 cycles stagnation
        proposals_per_iteration=3,  # 3 propositions par cycle
        memory_size=10,
        export_dir=Path("./output/poc_orchestrator"),  # Export résultats
    )

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  - Stratégie: {config.strategy_name}")
    logger.info(f"  - Params initiaux: {config.initial_params}")
    logger.info(f"  - Target Sharpe: {config.target_sharpe}")
    logger.info(f"  - Max iterations: {config.max_iterations}")
    logger.info(f"  - Export dir: {config.export_dir}")

    # ==========================================================================
    # 3. INITIALISATION ORCHESTRATEUR
    # ==========================================================================
    logger.info("\n🔧 Initializing Orchestrator...")

    try:
        orchestrator = OptimizationOrchestrator(
            config=config,
            data=data,
            analyst_model="deepseek-r1:70b",
            strategist_model="gpt-oss:20b",
            critic_model="deepseek-r1:70b",
            gpu_id=0,  # Premier GPU
            debug=True,  # Logs détaillés
        )

        logger.info(f"✅ Orchestrator initialized: {orchestrator}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        logger.error("Vérifiez:")
        logger.error("  1. Ollama running: ollama list")
        logger.error("  2. Models installed: deepseek-r1:70b, gpt-oss:20b")
        logger.error("  3. GPU disponible: nvidia-smi")
        sys.exit(1)

    # ==========================================================================
    # 4. LANCEMENT BOUCLE AUTONOME
    # ==========================================================================
    logger.info("\n🚀 Starting autonomous optimization loop...\n")

    try:
        result = orchestrator.run()

        # ==========================================================================
        # 5. AFFICHAGE RÉSULTATS
        # ==========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("🏆 RÉSULTATS FINAUX")
        logger.info("=" * 80)

        logger.info(f"\n📊 Performance:")
        logger.info(f"  - Best Sharpe: {result['best_score']:.3f}")
        logger.info(f"  - Converged: {result['converged']}")
        logger.info(f"  - Reason: {result['reason']}")
        logger.info(f"  - Total backtests: {result['total_backtests']}")
        logger.info(f"  - Execution time: {result['execution_time']:.1f}s")

        logger.info(f"\n🎯 Best Parameters:")
        for param, value in result["best_params"].items():
            logger.info(f"  - {param}: {value}")

        logger.info(f"\n📈 Iterations History:")
        for it in result["iterations"]:
            logger.info(
                f"  - Iteration {it['iteration']}: "
                f"Sharpe={it['score']:.3f}, "
                f"time={it['execution_time']:.1f}s"
            )

        # ==========================================================================
        # 6. GRAPHIQUE CONVERGENCE (optionnel)
        # ==========================================================================
        try:
            import matplotlib.pyplot as plt

            plot_data = orchestrator.get_convergence_plot_data()

            plt.figure(figsize=(10, 6))
            plt.plot(plot_data["iterations"], plot_data["scores"], marker="o")
            plt.axhline(
                y=config.target_sharpe,
                color="r",
                linestyle="--",
                label=f"Target ({config.target_sharpe})",
            )
            plt.xlabel("Iteration")
            plt.ylabel("Sharpe Ratio")
            plt.title("Convergence Optimisation Multi-Agents")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = config.export_dir / "convergence_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            logger.info(f"\n📊 Convergence plot saved: {plot_path}")

        except ImportError:
            logger.warning("Matplotlib not available, skipping convergence plot")

        logger.info("\n✅ POC COMPLETED SUCCESSFULLY")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Optimization interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n❌ Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
