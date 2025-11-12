"""
Test du filtrage des dates pour vérifier que le backtest utilise bien la période correcte
"""

import pandas as pd
from threadx.data_access import load_ohlcv
from threadx.strategy.bb_atr import BBAtrStrategy


def test_date_filtering():
    """Vérifie que le backtest utilise bien les données filtrées par date."""

    print("=" * 60)
    print("TEST DU FILTRAGE PAR DATES")
    print("=" * 60)

    symbol = "BTCUSDC"
    timeframe = "15m"

    # Test 1: Charger 6 mois de données
    print("\n📊 Test 1: Chargement 6 mois de données")
    df_6months = load_ohlcv(symbol, timeframe, start="2024-05-01", end="2024-10-31")
    print(f"   Données chargées: {len(df_6months)} barres")
    print(f"   Période: {df_6months.index[0]} → {df_6months.index[-1]}")
    print(f"   Durée: {(df_6months.index[-1] - df_6months.index[0]).days} jours")

    # Test 2: Charger 3 jours de données
    print("\n📊 Test 2: Chargement 3 jours de données")
    df_3days = load_ohlcv(symbol, timeframe, start="2024-10-29", end="2024-10-31")
    print(f"   Données chargées: {len(df_3days)} barres")
    print(f"   Période: {df_3days.index[0]} → {df_3days.index[-1]}")
    print(f"   Durée: {(df_3days.index[-1] - df_3days.index[0]).days} jours")

    # Vérification
    print("\n✅ Vérification du filtrage:")
    if len(df_6months) > len(df_3days) * 10:
        print(
            f"   OK: 6 mois ({len(df_6months)} barres) >> 3 jours ({len(df_3days)} barres)"
        )
    else:
        print(f"   ❌ PROBLÈME: Les données ne sont pas proportionnelles!")
        print(f"      6 mois: {len(df_6months)} barres")
        print(f"      3 jours: {len(df_3days)} barres")
        print(f"      Ratio: {len(df_6months) / len(df_3days):.1f}x")

    # Test 3: Backtest avec les 2 DataFrames
    print("\n🔧 Test 3: Backtest sur les deux périodes")

    strategy = BBAtrStrategy(symbol=symbol, timeframe=timeframe)
    params = {
        "bb_length": 20,
        "bb_mult": 2.0,
        "atr_length": 14,
        "atr_mult": 1.5,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
    }

    import time

    # Backtest 6 mois
    print("\n   ⏱️ Backtest 6 mois...")
    start_time = time.time()
    equity_6m, stats_6m = strategy.backtest(
        df=df_6months,
        params=params,
        initial_capital=10000.0,
        fee_bps=4.5,
        slippage_bps=0.0,
    )
    time_6m = time.time() - start_time
    print(f"      Durée: {time_6m:.2f}s")
    print(f"      Trades: {stats_6m.total_trades}")
    print(f"      PnL: {stats_6m.total_pnl_pct:.2f}%")

    # Backtest 3 jours
    print("\n   ⏱️ Backtest 3 jours...")
    start_time = time.time()
    equity_3d, stats_3d = strategy.backtest(
        df=df_3days,
        params=params,
        initial_capital=10000.0,
        fee_bps=4.5,
        slippage_bps=0.0,
    )
    time_3d = time.time() - start_time
    print(f"      Durée: {time_3d:.2f}s")
    print(f"      Trades: {stats_3d.total_trades}")
    print(f"      PnL: {stats_3d.total_pnl_pct:.2f}%")

    # Vérification finale
    print("\n" + "=" * 60)
    print("📊 RÉSULTAT FINAL")
    print("=" * 60)

    speedup = time_6m / time_3d
    data_ratio = len(df_6months) / len(df_3days)

    print(f"Ratio de données: {data_ratio:.1f}x (6 mois vs 3 jours)")
    print(f"Ratio de temps: {speedup:.1f}x (6 mois vs 3 jours)")

    # Le temps devrait être proportionnel aux données
    # Avec 60x plus de données, le temps devrait être ~60x plus long
    # Tolérance: ±50%
    expected_speedup = data_ratio
    tolerance = 0.5

    if abs(speedup - expected_speedup) / expected_speedup < tolerance:
        print(f"\n✅ TEST RÉUSSI !")
        print(
            f"   Le temps est proportionnel aux données ({speedup:.1f}x vs {expected_speedup:.1f}x attendu)"
        )
    else:
        print(f"\n❌ TEST ÉCHOUÉ !")
        print(f"   Le temps n'est PAS proportionnel aux données!")
        print(f"   Obtenu: {speedup:.1f}x")
        print(f"   Attendu: ~{expected_speedup:.1f}x (±{tolerance*100:.0f}%)")
        print(f"   Différence: {abs(speedup - expected_speedup):.1f}x")

        if speedup < 2:
            print(
                f"\n⚠️ DIAGNOSTIC: Le backtest utilise probablement les MÊMES données dans les deux cas!"
            )
            print(
                f"   Vérifiez que strategy.backtest() utilise bien le DataFrame filtré."
            )


if __name__ == "__main__":
    test_date_filtering()
