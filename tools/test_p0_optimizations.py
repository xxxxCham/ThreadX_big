"""
ThreadX - Test Optimisations P0.1 + P0.5
=========================================

Teste les corrections:
- P0.1: Workers 8 → 30+
- P0.5: Balance GPU 100% RTX 2060 → 66% RTX 5080 / 34% RTX 2060

Usage:
    python tools/test_p0_optimizations.py
"""

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from threadx.data_access import load_ohlcv
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.optimization.engine import SweepRunner
from threadx.gpu.multi_gpu import get_default_manager


def test_workers_and_gpu_balance():
    """Teste workers auto + balance GPU."""

    print("\n" + "="*80)
    print("TEST P0.1 + P0.5: Workers Auto + Balance Multi-GPU")
    print("="*80)

    # 1. Vérifier balance GPU
    print("\n[1/4] Vérification balance Multi-GPU...")
    manager = get_default_manager()

    print(f"  GPUs détectés: {len(manager._gpu_devices)}")
    for device in manager._gpu_devices:
        print(f"    - {device.name}: {device.memory_total_gb:.1f} GB VRAM")

    print(f"\n  Balance configurée:")
    for device_name, ratio in manager.device_balance.items():
        print(f"    - {device_name}: {ratio*100:.1f}%")

    # Vérification attendue
    expected_5080 = 0.66
    expected_2060 = 0.34

    actual_5080 = manager.device_balance.get("5080", 0)
    actual_2060 = manager.device_balance.get("2060", 0)

    if abs(actual_5080 - expected_5080) < 0.01 and abs(actual_2060 - expected_2060) < 0.01:
        print(f"\n  ✅ Balance Multi-GPU correcte (5080: {actual_5080*100:.0f}%, 2060: {actual_2060*100:.0f}%)")
    else:
        print(f"\n  ⚠️ Balance incorrecte!")
        print(f"     Attendu: 5080: 66%, 2060: 34%")
        print(f"     Actuel: 5080: {actual_5080*100:.0f}%, 2060: {actual_2060*100:.0f}%")

    # 2. Vérifier workers auto
    print("\n[2/4] Vérification workers auto...")

    settings = IndicatorSettings(use_gpu=True)
    bank = IndicatorBank(settings)

    runner = SweepRunner(
        indicator_bank=bank,
        max_workers=None,  # Auto !
        use_multigpu=True
    )

    print(f"  Workers calculés automatiquement: {runner.max_workers}")

    if runner.max_workers >= 30:
        print(f"  ✅ Workers suffisants ({runner.max_workers} ≥ 30)")
    else:
        print(f"  ⚠️ Workers insuffisants ({runner.max_workers} < 30)")

    # 3. Benchmark mini sweep
    print("\n[3/4] Benchmark mini sweep (20 combinaisons)...")

    data = load_ohlcv("BTCUSDC", "15m", date(2024, 12, 1), date(2024, 12, 10))
    print(f"  Données: {len(data)} barres")

    # Définir 20 combinaisons
    from threadx.optimization.scenarios import ScenarioSpec

    spec = ScenarioSpec(
        type="grid",
        params={
            "bb_period": {"values": [15, 20, 25, 30]},      # 4 valeurs
            "bb_std": {"values": [2.0, 2.5]},               # 2 valeurs
            "atr_period": {"values": [10, 14, 21]},         # 3 valeurs (total: 4×2 = 8)
        }
    )

    # Mesurer temps
    t_start = time.perf_counter()

    try:
        results = runner.run_grid(
            grid_spec=spec,
            real_data=data,
            symbol="BTCUSDC",
            timeframe="15m",
            strategy_name="Bollinger_Breakout"
        )

        t_elapsed = time.perf_counter() - t_start

        print(f"\n  ✅ Sweep terminé!")
        print(f"     Temps: {t_elapsed:.2f} sec")
        print(f"     Résultats: {len(results)} combinaisons")
        print(f"     Vitesse: {len(results) / t_elapsed:.2f} tests/sec")

        # Extrapolation pour 2.9M combinaisons
        if len(results) > 0:
            speed_per_sec = len(results) / t_elapsed
            eta_hours = 2903040 / speed_per_sec / 3600

            print(f"\n  📊 Extrapolation pour 2,903,040 combinaisons:")
            print(f"     ETA: {eta_hours:.2f} heures")

            if eta_hours < 20:
                print(f"     ✅ Performance EXCELLENTE (<20h) !")
            elif eta_hours < 40:
                print(f"     ✅ Performance BONNE (20-40h)")
            elif eta_hours < 60:
                print(f"     ⚠️ Performance MOYENNE (40-60h)")
            else:
                print(f"     ❌ Performance FAIBLE (>60h)")

    except Exception as e:
        print(f"\n  ❌ Erreur pendant sweep: {e}")
        import traceback
        traceback.print_exc()

    # 4. Vérifier monitoring GPU
    print("\n[4/4] Vérification utilisation GPU...")
    print("  💡 Lancez 'nvidia-smi dmon -s u' dans un autre terminal pour monitorer en temps réel")
    print("     Objectif: RTX 5080 à 60-80%, RTX 2060 à 60-80%")

    print("\n" + "="*80)
    print("TEST TERMINÉ")
    print("="*80)


if __name__ == "__main__":
    test_workers_and_gpu_balance()
