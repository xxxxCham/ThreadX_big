#!/usr/bin/env python3
"""
Monitoring Temps Réel ThreadX - Type Gestionnaire de Tâches
============================================================

Affiche en temps réel :
- CPU : % utilisation par core + global
- RAM : % utilisation + Go utilisés/totaux
- GPU 1 (RTX 5080) : % utilisation, VRAM, température, puissance
- GPU 2 (RTX 2060) : % utilisation, VRAM, température, puissance
- SSD : % utilisation, read/write

Rafraîchissement automatique toutes les 0.5s

Usage:
    python tools/realtime_monitor.py

Raccourcis:
    Ctrl+C : Quitter

Author: ThreadX Framework
"""

import os
import sys
import time
import platform
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import psutil
except ImportError:
    print("❌ psutil non installé. Installation: pip install psutil")
    sys.exit(1)

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    print("⚠️  pynvml non installé (monitoring GPU limité)")
    NVML_AVAILABLE = False


def clear_screen():
    """Efface l'écran (compatible Windows/Linux)."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_cpu_info() -> Dict[str, Any]:
    """Récupère infos CPU."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()

    return {
        'percent': cpu_percent,
        'count': cpu_count,
        'freq_current': cpu_freq.current if cpu_freq else 0,
        'freq_max': cpu_freq.max if cpu_freq else 0,
        'per_core': psutil.cpu_percent(interval=0, percpu=True),
    }


def get_ram_info() -> Dict[str, Any]:
    """Récupère infos RAM."""
    mem = psutil.virtual_memory()

    return {
        'percent': mem.percent,
        'used_gb': mem.used / (1024**3),
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
    }


def get_gpu_info() -> Dict[int, Dict[str, Any]]:
    """Récupère infos GPUs via pynvml."""
    if not NVML_AVAILABLE:
        return {}

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        gpus = {}
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Nom
            name_bytes = pynvml.nvmlDeviceGetName(handle)
            name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else str(name_bytes)

            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory
            except:
                gpu_util = 0
                mem_util = 0

            # Mémoire
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem_info.used / (1024**3)
                mem_total_gb = mem_info.total / (1024**3)
                mem_percent = (mem_info.used / mem_info.total) * 100
            except:
                mem_used_gb = 0
                mem_total_gb = 0
                mem_percent = 0

            # Température
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0

            # Puissance
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0
            except:
                power_w = 0

            # Fréquence
            try:
                clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except:
                clock_mhz = 0

            gpus[i] = {
                'name': name,
                'gpu_percent': gpu_util,
                'mem_percent': mem_percent,
                'mem_used_gb': mem_used_gb,
                'mem_total_gb': mem_total_gb,
                'temperature': temp,
                'power_w': power_w,
                'clock_mhz': clock_mhz,
            }

        pynvml.nvmlShutdown()
        return gpus

    except Exception as e:
        return {}


def get_disk_info() -> Dict[str, Any]:
    """Récupère infos disques."""
    try:
        # Partition principale (C: sur Windows)
        partitions = psutil.disk_partitions()
        main_partition = None

        # Chercher C: ou /
        for part in partitions:
            if 'C:' in part.mountpoint or part.mountpoint == '/':
                main_partition = part.mountpoint
                break

        if not main_partition and partitions:
            main_partition = partitions[0].mountpoint

        if main_partition:
            usage = psutil.disk_usage(main_partition)
            io_counters = psutil.disk_io_counters()

            return {
                'mountpoint': main_partition,
                'percent': usage.percent,
                'used_gb': usage.used / (1024**3),
                'total_gb': usage.total / (1024**3),
                'read_mb': io_counters.read_bytes / (1024**2) if io_counters else 0,
                'write_mb': io_counters.write_bytes / (1024**2) if io_counters else 0,
            }
    except:
        pass

    return {
        'mountpoint': 'N/A',
        'percent': 0,
        'used_gb': 0,
        'total_gb': 0,
        'read_mb': 0,
        'write_mb': 0,
    }


def create_bar(percent: float, width: int = 40) -> str:
    """Crée une barre de progression."""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)

    # Couleur selon le pourcentage
    if percent < 30:
        color = '\033[92m'  # Vert
    elif percent < 70:
        color = '\033[93m'  # Jaune
    else:
        color = '\033[91m'  # Rouge

    reset = '\033[0m'

    return f"{color}{bar}{reset} {percent:5.1f}%"


def display_monitor():
    """Affiche le monitoring en temps réel."""
    iteration = 0

    while True:
        try:
            iteration += 1
            clear_screen()

            # Header
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("=" * 100)
            print(f"  🖥️  MONITORING TEMPS RÉEL ThreadX  |  {now}  |  Refresh: {iteration}")
            print("=" * 100)

            # CPU
            cpu = get_cpu_info()
            print(f"\n📊 CPU ({cpu['count']} cores @ {cpu['freq_current']:.0f} MHz)")
            print("─" * 100)
            print(f"  Global : {create_bar(cpu['percent'], 50)}")

            # Cores (afficher par groupe de 4)
            cores = cpu['per_core']
            print(f"\n  Cores :")
            for i in range(0, len(cores), 4):
                core_group = cores[i:i+4]
                line = "    "
                for j, core_pct in enumerate(core_group):
                    core_num = i + j
                    line += f"Core {core_num:2d}: {create_bar(core_pct, 20)}  "
                print(line)

            # RAM
            ram = get_ram_info()
            print(f"\n💾 RAM")
            print("─" * 100)
            print(f"  Utilisée : {create_bar(ram['percent'], 50)}  ({ram['used_gb']:.1f} / {ram['total_gb']:.1f} GB)")

            # GPUs
            gpus = get_gpu_info()

            if gpus:
                for gpu_id, gpu_info in gpus.items():
                    print(f"\n🎮 GPU {gpu_id} - {gpu_info['name']}")
                    print("─" * 100)
                    print(f"  Utilisation GPU  : {create_bar(gpu_info['gpu_percent'], 50)}")
                    print(f"  VRAM             : {create_bar(gpu_info['mem_percent'], 50)}  ({gpu_info['mem_used_gb']:.2f} / {gpu_info['mem_total_gb']:.2f} GB)")
                    print(f"  Température      : {gpu_info['temperature']:3d}°C")
                    print(f"  Puissance        : {gpu_info['power_w']:6.1f} W")
                    print(f"  Clock            : {gpu_info['clock_mhz']:5d} MHz")
            else:
                print("\n🎮 GPUs")
                print("─" * 100)
                print("  ⚠️  Aucun GPU détecté ou pynvml non disponible")

            # Disk
            disk = get_disk_info()
            print(f"\n💿 Disque ({disk['mountpoint']})")
            print("─" * 100)
            print(f"  Utilisation : {create_bar(disk['percent'], 50)}  ({disk['used_gb']:.0f} / {disk['total_gb']:.0f} GB)")
            print(f"  I/O         : Read: {disk['read_mb']:.0f} MB  |  Write: {disk['write_mb']:.0f} MB")

            # Footer
            print("\n" + "=" * 100)
            print("  Appuyez sur Ctrl+C pour quitter")
            print("=" * 100)

            # Refresh interval
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n✅ Monitoring arrêté")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            time.sleep(1)


def main():
    """Point d'entrée principal."""
    print("\n🚀 Lancement du monitoring temps réel...\n")
    time.sleep(1)

    display_monitor()


if __name__ == "__main__":
    main()
