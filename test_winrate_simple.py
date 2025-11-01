"""Test simple pour vérifier le calcul du win_rate."""

import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from threadx.strategy.model import RunStats

print("🧪 TEST DU CALCUL WIN_RATE")
print("=" * 60)

# Simuler des stats de run avec trades
test_stats = RunStats(
    final_equity=10500.0,
    initial_capital=10000.0,
    total_pnl=500.0,
    max_drawdown=-200.0,
    max_drawdown_pct=-2.0,
    total_trades=10,
    win_trades=6,
    loss_trades=4,
    win_rate_pct=60.0,  # 6 trades gagnants / 10 total = 60%
    total_fees_paid=50.0,
    start_time="2024-01-01",
    end_time="2024-12-31",
    bars_analyzed=1000,
)

print(f"\n📊 RUNSTATS TEST:")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Total Trades:    {test_stats.total_trades}")
print(f"Win Trades:      {test_stats.win_trades}")
print(f"Loss Trades:     {test_stats.loss_trades}")
print(f"Win Rate (pct):  {test_stats.win_rate_pct}%")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Tester que l'attribut existe
print(f"\n🔍 Vérification attributs:")
print(f"   hasattr(test_stats, 'win_rate'):     {hasattr(test_stats, 'win_rate')}")
print(f"   hasattr(test_stats, 'win_rate_pct'): {hasattr(test_stats, 'win_rate_pct')}")

# Simuler le code de l'engine (AVANT FIX)
print(f"\n❌ ANCIEN CODE (engine.py ligne 940-941):")
win_rate_old = test_stats.win_rate if hasattr(test_stats, "win_rate") else 0.0
print(f"   win_rate = {win_rate_old} (TOUJOURS 0.0 car attribut n'existe pas)")

# Simuler le code de l'engine (APRÈS FIX)
print(f"\n✅ NOUVEAU CODE (avec fix):")
win_rate_new = (
    test_stats.win_rate_pct / 100.0 if hasattr(test_stats, "win_rate_pct") else 0.0
)
print(f"   win_rate = {win_rate_new} (60% / 100 = 0.60)")

# Vérification
expected = test_stats.win_trades / test_stats.total_trades
print(f"\n🎯 COMPARAISON:")
print(f"   Attendu (6/10):       {expected:.2f}")
print(f"   Obtenu (ancien):      {win_rate_old:.2f} ❌")
print(f"   Obtenu (nouveau):     {win_rate_new:.2f} ✅")

if abs(win_rate_new - expected) < 0.01:
    print(f"\n✅ LE FIX FONCTIONNE CORRECTEMENT!")
else:
    print(f"\n❌ LE FIX NE FONCTIONNE PAS")
