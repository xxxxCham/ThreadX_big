"""Test rapide pour vérifier le fix du win_rate."""

import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
from threadx.strategy.bb_atr import BBAtrStrategy

# Charger les données BTCUSDC 15m
data_file = Path("src/threadx/data/crypto_data_parquet/BTCUSDC_15m.parquet")
if not data_file.exists():
    print(f"❌ Fichier de données introuvable: {data_file}")
    sys.exit(1)

df = pd.read_parquet(data_file)
# Assurer que l'index est DatetimeIndex
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
elif not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

print(f"✅ Données chargées: {len(df)} barres BTCUSDC/15m")

# Tester avec les meilleurs paramètres du sweep
best_params = {
    "bb_period": 44,
    "bb_std": 3.0,
    "entry_z": 0.8,
    "atr_period": 7,
    "atr_multiplier": 3.0,
    "risk_per_trade": 0.015,
    "min_pnl_pct": 0.0,
    "max_hold_bars": 30,
    "spacing_bars": 3,
    "trend_period": 14,
    "entry_logic": "AND",
    "trailing_stop": True,
    "leverage": 3.5,
}

print(
    f"\n🔧 Test avec paramètres optimaux: BB({best_params['bb_period']},{best_params['bb_std']}) + ATR({best_params['atr_period']},{best_params['atr_multiplier']})"
)

# Instancier la stratégie
strategy = BBAtrStrategy(symbol="BTCUSDC", timeframe="15m")

# Backtest
equity_curve, run_stats = strategy.backtest(
    df=df.tail(5000),  # Dernières 5000 barres pour test rapide
    params=best_params,
    initial_capital=10000.0,
    fee_bps=4.5,
    slippage_bps=0.0,
)

print(f"\n📊 RÉSULTATS DU BACKTEST:")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Total PnL:       ${run_stats.total_pnl:.2f} ({run_stats.total_pnl_pct:.2f}%)")
print(
    f"Sharpe Ratio:    {run_stats.sharpe_ratio:.2f}"
    if run_stats.sharpe_ratio
    else "Sharpe Ratio:    N/A"
)
print(
    f"Max Drawdown:    ${run_stats.max_drawdown:.2f} ({run_stats.max_drawdown_pct:.2f}%)"
)
print(f"Total Trades:    {run_stats.total_trades}")
print(f"Win Trades:      {run_stats.win_trades}")
print(f"Loss Trades:     {run_stats.loss_trades}")
print(f"Win Rate:        {run_stats.win_rate_pct:.2f}% ✅ NOUVEAU CALCUL")
print(
    f"Profit Factor:   {run_stats.profit_factor:.2f}"
    if run_stats.profit_factor
    else "Profit Factor:   N/A"
)
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Vérification
if run_stats.total_trades > 0:
    expected_wr = (run_stats.win_trades / run_stats.total_trades) * 100.0
    if abs(run_stats.win_rate_pct - expected_wr) < 0.01:
        print(f"\n✅ WIN_RATE CORRECTEMENT CALCULÉ!")
        print(
            f"   {run_stats.win_trades} trades gagnants / {run_stats.total_trades} total = {run_stats.win_rate_pct:.2f}%"
        )
    else:
        print(f"\n❌ ERREUR WIN_RATE:")
        print(f"   Attendu: {expected_wr:.2f}%")
        print(f"   Obtenu:  {run_stats.win_rate_pct:.2f}%")
else:
    print("\n⚠️  Aucun trade généré avec ces paramètres")
