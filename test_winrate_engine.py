"""Test rapide pour vérifier le fix du win_rate via l'engine."""

import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from threadx.optimization.engine import SweepRunner

print("🧪 TEST DU FIX WIN_RATE")
print("=" * 60)

# Créer un runner
runner = SweepRunner(data_dir="src/threadx/data/crypto_data_parquet")

# Paramètres du meilleur scénario du sweep
best_combo = {
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
    f"\n📊 Test combo: BB({best_combo['bb_period']},{best_combo['bb_std']}) + ATR({best_combo['atr_period']},{best_combo['atr_multiplier']})"
)

# Évaluer le combo
result = runner._evaluate_combo(
    combo=best_combo,
    symbol="BTCUSDC",
    timeframe="15m",
    strategy_name="Bollinger_Breakout",
    days=30,  # 30 jours pour test rapide
)

print(f"\n📈 RÉSULTATS:")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"PnL:             ${result.get('pnl', 0):.2f} ({result.get('pnl_pct', 0):.2f}%)")
print(f"Sharpe:          {result.get('sharpe', 0):.2f}")
print(f"Max Drawdown:    {result.get('max_drawdown', 0):.2f}")
print(f"Total Trades:    {result.get('total_trades', 0)}")
print(
    f"Win Rate:        {result.get('win_rate', 0):.4f} ({result.get('win_rate', 0)*100:.2f}%) ✅"
)
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Vérification
if result.get("win_rate", 0) > 0:
    print(f"\n✅ WIN_RATE N'EST PLUS À 0% !")
    print(f"   Le fix fonctionne correctement.")
    print(f"   Win rate = {result.get('win_rate', 0)*100:.2f}%")
else:
    if result.get("total_trades", 0) > 0:
        print(
            f"\n⚠️  WIN_RATE TOUJOURS À 0% malgré {result.get('total_trades', 0)} trades"
        )
        print(f"   Le fix ne fonctionne pas encore.")
    else:
        print(f"\n⚠️  Aucun trade généré, impossible de vérifier le win_rate")
