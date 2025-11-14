"""
Script de test pour la stratégie MA Crossover
==============================================

Objectif: Valider le moteur de calcul avec une stratégie simple et connue.

Ce script:
1. Charge les données BTC historiques
2. Exécute la stratégie MA Crossover
3. Affiche les résultats détaillés
4. Compare avec les calculs attendus

Usage:
    python test_ma_crossover.py
"""

import sys
from pathlib import Path

import pandas as pd

# Ajouter le src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.data_access import load_ohlcv
from threadx.strategy.ma_crossover import MACrossoverParams, MACrossoverStrategy
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def test_ma_crossover_basic():
    """Test basique de la stratégie MA Crossover"""
    print("\n" + "=" * 60)
    print("TEST MA CROSSOVER - VALIDATION MOTEUR DE CALCUL")
    print("=" * 60 + "\n")

    # Chargement données
    print("📥 Chargement des données BTC/USDC 15m...")
    df = load_ohlcv(
        symbol="BTCUSDC",
        timeframe="15m",
        start="2024-12-01",
        end="2025-01-31",
    )
    print(f"✅ {len(df)} barres chargées ({df.index[0]} → {df.index[-1]})\n")

    # Configuration stratégie SIMPLE
    params = {
        "fast_period": 10,
        "slow_period": 30,
        "stop_loss_pct": 2.0,  # Stop loss fixe 2%
        "take_profit_pct": 4.0,  # TP fixe 4%
        "risk_per_trade": 0.01,  # 1% du capital risqué
        "leverage": 1.0,  # PAS de levier
        "max_hold_bars": 100,
        "fee_bps": 4.5,
        "slippage_bps": 0.0,
    }

    print("📋 Paramètres:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # Exécution backtest
    print("🚀 Exécution backtest...\n")
    strategy = MACrossoverStrategy()
    initial_capital = 10000.0

    equity_curve, stats = strategy.backtest(df, params, initial_capital)

    # Affichage résultats
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS BACKTEST")
    print("=" * 60 + "\n")

    print(f"💰 Capital:")
    print(f"  Initial:       {stats.initial_capital:>12,.2f} USDC")
    print(f"  Final:         {stats.final_equity:>12,.2f} USDC")
    print(f"  PnL:           {stats.total_pnl:>12,.2f} USDC ({stats.total_pnl_pct:+.2f}%)")
    print()

    print(f"📈 Performance:")
    print(f"  Total trades:  {stats.total_trades:>12}")
    print(f"  Win trades:    {stats.win_trades:>12} ({stats.win_rate_pct:.1f}%)")
    print(f"  Loss trades:   {stats.loss_trades:>12}")
    print(f"  Avg win:       {stats.avg_win or 0:>12,.2f} USDC")
    print(f"  Avg loss:      {stats.avg_loss or 0:>12,.2f} USDC")
    print(f"  Profit factor: {stats.profit_factor or 0:>12,.2f}")
    print()

    print(f"⚠️  Risque:")
    print(f"  Max DD:        {stats.max_drawdown:>12,.2f} USDC ({stats.max_drawdown_pct:.2f}%)")
    print(f"  DD duration:   {stats.max_drawdown_duration_bars:>12} bars")
    print(f"  Sharpe ratio:  {stats.sharpe_ratio or 0:>12,.2f}")
    print(f"  Sortino ratio: {stats.sortino_ratio or 0:>12,.2f}")
    print()

    print(f"💸 Frais:")
    print(f"  Total fees:    {stats.total_fees_paid:>12,.2f} USDC")
    print()

    # Analyse détaillée
    print("\n" + "=" * 60)
    print("🔍 ANALYSE DÉTAILLÉE")
    print("=" * 60 + "\n")

    # Vérification cohérence DD
    if abs(stats.max_drawdown_pct) > 50:
        print("⚠️  ALERTE: Drawdown > 50% détecté!")
        print(f"   → DD = {stats.max_drawdown_pct:.2f}%")
        print("   → Possible bug dans le moteur de calcul\n")
    else:
        print(f"✅ Drawdown acceptable: {stats.max_drawdown_pct:.2f}%\n")

    # Vérification cohérence PnL vs capital
    expected_capital = initial_capital + stats.total_pnl
    capital_diff = abs(expected_capital - stats.final_equity)

    print(f"🧮 Vérification cohérence capital:")
    print(f"   Capital initial:  {initial_capital:,.2f} USDC")
    print(f"   + Total PnL:      {stats.total_pnl:+,.2f} USDC")
    print(f"   = Attendu:        {expected_capital:,.2f} USDC")
    print(f"   Capital final:    {stats.final_equity:,.2f} USDC")
    print(f"   Différence:       {capital_diff:,.2f} USDC")

    if capital_diff > 1.0:
        print("   ⚠️  INCOHÉRENCE détectée (> 1 USDC)\n")
    else:
        print("   ✅ Cohérent (< 1 USDC différence)\n")

    # Vérification stops
    if stats.total_trades > 0:
        print(f"🛡️  Vérification stops:")
        print(f"   Params stop loss:  {params['stop_loss_pct']}%")
        print(f"   Params TP:         {params['take_profit_pct']}%")

        # Analyse distribution pertes
        if stats.avg_loss:
            max_expected_loss_pct = params["stop_loss_pct"] + 0.5  # Marge slippage
            print(f"   Perte max théorique: -{max_expected_loss_pct}%")

            # TODO: Ajouter analyse détaillée des trades individuels
            print("   → Besoin d'analyser les trades.csv pour valider\n")
    else:
        print("⚠️  Aucun trade généré!\n")

    # Sauvegarde résultats
    print("\n" + "=" * 60)
    print("💾 SAUVEGARDE")
    print("=" * 60 + "\n")

    output_path = Path("CSV/test_ma_crossover_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export stats
    stats_df = pd.DataFrame(
        [
            {
                "strategy": "MA_Crossover",
                "capital_initial": stats.initial_capital,
                "capital_final": stats.final_equity,
                "pnl": stats.total_pnl,
                "pnl_pct": stats.total_pnl_pct,
                "total_trades": stats.total_trades,
                "win_rate": stats.win_rate_pct,
                "max_dd": stats.max_drawdown,
                "max_dd_pct": stats.max_drawdown_pct,
                "sharpe": stats.sharpe_ratio,
                "sortino": stats.sortino_ratio,
                **params,
            }
        ]
    )

    stats_df.to_csv(output_path, index=False)
    print(f"✅ Résultats sauvés: {output_path}")

    # Equity curve
    equity_path = Path("CSV/test_ma_crossover_equity.csv")
    equity_curve.to_csv(equity_path, header=["equity"])
    print(f"✅ Equity curve sauvée: {equity_path}\n")

    print("=" * 60)
    print("✨ TEST TERMINÉ")
    print("=" * 60 + "\n")

    # Verdict
    print("📋 VERDICT:\n")

    checks_passed = 0
    checks_total = 3

    if stats.total_trades > 0:
        print("✅ Des trades ont été générés")
        checks_passed += 1
    else:
        print("❌ Aucun trade généré")

    if abs(stats.max_drawdown_pct) < 50:
        print("✅ Drawdown raisonnable (< 50%)")
        checks_passed += 1
    else:
        print("❌ Drawdown excessif (> 50%)")

    if capital_diff < 1.0:
        print("✅ Cohérence capital validée")
        checks_passed += 1
    else:
        print("❌ Incohérence capital détectée")

    print(f"\n🎯 Score: {checks_passed}/{checks_total} checks passés\n")

    if checks_passed == checks_total:
        print("✅ Le moteur de calcul semble FONCTIONNEL")
        print("   → Vous pouvez utiliser cette stratégie comme référence\n")
    else:
        print("⚠️  Des problèmes ont été détectés")
        print("   → Analysez les détails ci-dessus pour identifier les bugs\n")


if __name__ == "__main__":
    try:
        test_ma_crossover_basic()
    except Exception as e:
        logger.error(f"Erreur durant le test: {e}", exc_info=True)
        print(f"\n❌ TEST ÉCHOUÉ: {e}\n")
        sys.exit(1)
