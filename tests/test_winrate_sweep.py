"""Sweep rapide pour valider le fix du win_rate."""

import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec

print("🚀 SWEEP DE VALIDATION WIN_RATE")
print("=" * 70)

# Charger les données BTCUSDC 15m
data_path = (
    Path(__file__).parent
    / "src"
    / "threadx"
    / "data"
    / "crypto_data_parquet"
    / "BTCUSDC_15m.parquet"
)
print(f"\n📁 Chargement des données: {data_path.name}")
df = pd.read_parquet(data_path)

# Convertir timestamp en index datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)

# Prendre les 60 derniers jours
df = df.tail(60 * 24 * 4)  # 60 jours * 24h * 4 périodes 15m
print(f"📊 Données: {len(df)} lignes ({df.index[0]} → {df.index[-1]})")

# Configuration du sweep (réduit pour test rapide)
spec = ScenarioSpec(
    type="grid",
    params={
        "bb_period": {"values": [20, 30]},  # 2 valeurs
        "bb_std": {"values": [1.5, 2.0]},  # 2 valeurs (plus larges)
        "entry_z": {"value": 0.5},  # Plus permissif
        "atr_period": {"values": [7]},  # 1 valeur
        "atr_multiplier": {"value": 2.0},  # Plus serré
        "risk_per_trade": {"value": 0.015},
        "min_pnl_pct": {"value": 0.0},
        "max_hold_bars": {"value": 50},
        "spacing_bars": {"value": 3},  # Plus court
        "trend_period": {"value": 14},
        "entry_logic": {"value": "AND"},
        "trailing_stop": {"value": True},
        "leverage": {"value": 3.5},
    },
    seed=42,
)

print(f"\n📊 Configuration:")
print(f"   Token: BTCUSDC")
print(f"   Timeframe: 15m")
print(f"   Période: 60 jours")
print(f"   Scénarios: 3×2×2 = 12 combinaisons")
print(f"   Workers: 8")

# Créer le runner
runner = SweepRunner(max_workers=8, use_multigpu=True)

# Exécuter le sweep
print(f"\n⏳ Exécution du sweep...")
results_df = runner.run_grid(
    grid_spec=spec,
    real_data=df,
    symbol="BTCUSDC",
    timeframe="15m",
    strategy_name="Bollinger_Breakout",
)

print(f"\n✅ Sweep terminé: {len(results_df)} résultats")

# Filtrer les scénarios avec trades
with_trades = results_df[results_df["total_trades"] > 0].copy()

if len(with_trades) == 0:
    print("\n⚠️  Aucun scénario n'a généré de trades")
    print("   Essayez d'élargir les paramètres (bb_std, entry_z, etc.)")
else:
    print(f"\n📈 {len(with_trades)} scénarios avec trades")

    # Trier par PnL
    with_trades = with_trades.sort_values("pnl_pct", ascending=False)

    print(f"\n🏆 TOP 5 SCÉNARIOS:")
    print("=" * 70)

    for i, row in with_trades.head(5).iterrows():
        print(
            f"\n{i+1}. BB({row['bb_period']:.0f}, {row['bb_std']:.1f}) + ATR({row['atr_period']:.0f}, {row['atr_multiplier']:.1f})"
        )
        print(f"   PnL:        {row['pnl_pct']:>8.2f}%")
        print(f"   Sharpe:     {row['sharpe']:>8.2f}")
        print(f"   Trades:     {row['total_trades']:>8.0f}")

        # ⭐ VÉRIFICATION WIN_RATE ⭐
        win_rate = row.get("win_rate", 0.0)
        if win_rate > 0:
            print(f"   Win Rate:   {win_rate*100:>8.2f}% ✅ FONCTIONNE !")
        else:
            print(f"   Win Rate:   {win_rate*100:>8.2f}% ⚠️  (suspect si PnL > 0)")

        print(f"   Max DD:     {row['max_drawdown']:>8.2f}")

    # Statistiques win_rate
    print(f"\n📊 STATISTIQUES WIN_RATE:")
    print("=" * 70)

    win_rates = with_trades["win_rate"].values
    non_zero = sum(1 for wr in win_rates if wr > 0)

    print(f"   Scénarios avec win_rate > 0: {non_zero}/{len(with_trades)}")

    if non_zero > 0:
        avg_wr = with_trades[with_trades["win_rate"] > 0]["win_rate"].mean()
        min_wr = with_trades[with_trades["win_rate"] > 0]["win_rate"].min()
        max_wr = with_trades[with_trades["win_rate"] > 0]["win_rate"].max()

        print(f"   Win rate moyen:  {avg_wr*100:.2f}%")
        print(f"   Win rate min:    {min_wr*100:.2f}%")
        print(f"   Win rate max:    {max_wr*100:.2f}%")

        print(f"\n✅ LE FIX FONCTIONNE ! Win rates affichés correctement.")
    else:
        print(f"\n⚠️  Tous les win_rate sont à 0% - problème possible")

    # Sauvegarder les résultats
    output_file = "sweep_winrate_validation.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Résultats sauvegardés: {output_file}")
