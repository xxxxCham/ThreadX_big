"""
Test du monitoring GPU temps réel avec pynvml.

Ce script démarre le SystemMonitor et affiche les métriques GPU
toutes les 2 secondes pendant 10 secondes.

Usage:
    python test_gpu_monitoring.py
"""

import time
from threadx.ui.system_monitor import SystemMonitor


def main():
    """Test du monitoring GPU."""
    print("🔍 Démarrage du monitoring GPU...")
    print("=" * 80)

    # Créer le moniteur (1 snapshot toutes les 0.5s, historique de 120 = 1 minute)
    monitor = SystemMonitor(interval=0.5, max_history=120)

    # Démarrer la collecte
    monitor.start()

    # Laisser le temps de collecter des données
    print("\n📊 Collecte de métriques pendant 10 secondes...\n")

    for i in range(5):
        time.sleep(2)

        # Récupérer le dernier snapshot
        snapshot = monitor.get_latest_snapshot()

        if snapshot:
            print(f"⏱️  Temps: {i * 2 + 2}s")
            print(f"  CPU: {snapshot.cpu_percent:.1f}%")
            print(f"  RAM: {snapshot.memory_percent:.1f}%")
            print(
                f"  GPU 1 (5080): {snapshot.gpu1_percent:.1f}% (VRAM: {snapshot.gpu1_memory_percent:.1f}%, Temp: {snapshot.gpu1_temperature:.0f}°C, Power: {snapshot.gpu1_power_usage:.1f}W)"
            )
            print(
                f"  GPU 2 (2060): {snapshot.gpu2_percent:.1f}% (VRAM: {snapshot.gpu2_memory_percent:.1f}%, Temp: {snapshot.gpu2_temperature:.0f}°C, Power: {snapshot.gpu2_power_usage:.1f}W)"
            )
            print()

    # Arrêter le monitoring
    monitor.stop()

    # Récupérer l'historique
    history = monitor.get_history_dataframe()

    if not history.empty:
        print("=" * 80)
        print("📈 RÉSUMÉ DES MÉTRIQUES (dernières 10s)")
        print("=" * 80)

        print("\n🖥️  CPU:")
        print(f"  Moyenne: {history['cpu'].mean():.1f}%")
        print(f"  Max: {history['cpu'].max():.1f}%")

        print("\n💾 RAM:")
        print(f"  Moyenne: {history['memory'].mean():.1f}%")
        print(f"  Max: {history['memory'].max():.1f}%")

        print("\n🎮 GPU 1 (5080):")
        print(f"  Utilisation moyenne: {history['gpu1'].mean():.1f}%")
        print(f"  VRAM moyenne: {history['gpu1_mem'].mean():.1f}%")
        print(
            f"  Température moyenne: {history['gpu1_temp'].mean():.0f}°C (max: {history['gpu1_temp'].max():.0f}°C)"
        )
        print(
            f"  Consommation moyenne: {history['gpu1_power'].mean():.1f}W (max: {history['gpu1_power'].max():.1f}W)"
        )

        print("\n🎮 GPU 2 (2060):")
        print(f"  Utilisation moyenne: {history['gpu2'].mean():.1f}%")
        print(f"  VRAM moyenne: {history['gpu2_mem'].mean():.1f}%")
        print(
            f"  Température moyenne: {history['gpu2_temp'].mean():.0f}°C (max: {history['gpu2_temp'].max():.0f}°C)"
        )
        print(
            f"  Consommation moyenne: {history['gpu2_power'].mean():.1f}W (max: {history['gpu2_power'].max():.1f}W)"
        )

        print("\n" + "=" * 80)
        print("✅ Monitoring GPU fonctionnel!")
        print("=" * 80)

    else:
        print("\n⚠️  Aucune donnée collectée")


if __name__ == "__main__":
    main()
