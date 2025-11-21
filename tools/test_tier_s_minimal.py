#!/usr/bin/env python3
"""Test minimal intégration Tier S."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.backtest.performance import summarize
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def main():
    """Test que summarize() calcule Tier S automatiquement."""
    logger.info("=" * 60)
    logger.info("🧪 TEST TIER S INTEGRATION MINIMALE")
    logger.info("=" * 60)

    # Créer données synthétiques réalistes
    np.random.seed(42)
    n_returns = 1000

    # Returns avec Sharpe ~1.5 (target Tier S = 1.8)
    returns = pd.Series(np.random.normal(0.001, 0.015, n_returns))

    # Trades DataFrame (requis pour Tier S)
    trades_data = []
    for i in range(150):  # 150 trades
        entry_time = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i * 2)
        exit_time = entry_time + pd.Timedelta(hours=12)

        pnl = np.random.normal(0.02, 0.05)  # +2% moyenne avec volatilité
        side = "LONG" if np.random.random() > 0.5 else "SHORT"

        trades_data.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "pnl": pnl,  # Colonne requise par performance.profit_factor()
                "side": side,
                "qty": 1.0,
                "entry_price": 100.0,
                "exit_price": 100.0 * (1 + pnl),
            }
        )

    trades = pd.DataFrame(trades_data)

    logger.info(f"✅ Generated {len(returns)} returns, {len(trades)} trades")

    # ==========================================================================
    # TEST: summarize() calcule Tier S automatiquement
    # ==========================================================================
    logger.info("\n🚀 Calling summarize()...")

    # Equity curve (optionnelle mais recommandée)
    initial_capital = 10000.0
    equity = pd.Series(initial_capital * (1 + returns).cumprod())

    summary = summarize(
        trades=trades,
        returns=returns,
        initial_capital=initial_capital,
        risk_free=0.0,
        periods_per_year=365,
    )

    logger.info(f"✅ Summary calculated: {len(summary)} metrics")

    # ==========================================================================
    # VÉRIFICATION TIER S
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 TIER S METRICS CHECK")
    logger.info("=" * 60)

    tier_s_metrics = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "profit_factor_tier_s",
        "recovery_factor",
        "expectancy_pct",
        "sqn",
        "outlier_adjusted_sharpe",
    ]

    present_count = 0
    for metric in tier_s_metrics:
        if metric in summary:
            present_count += 1
            logger.info(f"  ✅ {metric}: {summary[metric]:.3f}")
        else:
            logger.warning(f"  ❌ {metric}: MISSING")

    logger.info(f"\n📊 Tier S Metrics Present: {present_count}/{len(tier_s_metrics)}")

    # Vérifier validation
    if "tier_s_validation" in summary:
        val = summary["tier_s_validation"]
        logger.info("\n🎯 TIER S VALIDATION:")
        logger.info(f"  Passed: {val.get('passed', False)}")
        logger.info(f"  Score: {val.get('score', 0):.1f}/100")
        logger.info(f"  Tier S Passed: {val.get('tier_s_passed', 0)}/10")
        logger.info(f"  AI-Gold: {val.get('ai_evolved_gold', False)}")

        if val.get("failed_metrics"):
            logger.info(f"  Failed: {', '.join(val['failed_metrics'])}")
    else:
        logger.warning("\n⚠️  tier_s_validation NOT FOUND (check HAS_TIER_S flag)")

    # Vérifier thresholds
    if "tier_s_thresholds" in summary:
        logger.info(
            f"\n✅ tier_s_thresholds present ({len(summary['tier_s_thresholds'])} thresholds)"
        )
    else:
        logger.warning("\n⚠️  tier_s_thresholds NOT FOUND")

    # ==========================================================================
    # RÉSULTAT
    # ==========================================================================
    logger.info("\n" + "=" * 60)

    if present_count >= 7:
        logger.info("✅ TEST PASSED: Tier S intégration fonctionne!")
        logger.info("   performance.summarize() calcule Tier S automatiquement")
        logger.info("   Architecture correcte: Engine → summarize() → Tier S")
        return 0
    else:
        logger.error("❌ TEST FAILED: Tier S metrics manquantes")
        logger.error(f"   Présentes: {present_count}/{len(tier_s_metrics)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
