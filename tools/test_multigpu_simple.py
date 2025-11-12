#!/usr/bin/env python3
"""
Test Multi-GPU Simple - ThreadX
================================

Teste la répartition de charge sur 2 GPUs avec calcul réel.

Author: ThreadX Framework
"""

import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import cupy as cp
    import numpy as np
    from threadx.gpu.device_manager import list_devices
    from threadx.gpu.multi_gpu import MultiGPUManager

    CUPY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def print_section(title: str):
    """Print section separator."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_gpu_detection():
    """Test 1: Détection des GPUs."""
    print_section("TEST 1: Détection GPUs")

    devices = list_devices()
    gpu_devices = [d for d in devices if d.device_id >= 0]

    logger.info(f"Nombre total de devices: {len(devices)}")
    logger.info(f"Nombre de GPUs: {len(gpu_devices)}")

    for dev in gpu_devices:
        logger.info(f"  GPU {dev.device_id} ({dev.name}): {dev.memory_total_gb:.2f} GB, CC {dev.compute_capability[0]}.{dev.compute_capability[1]}")

    return len(gpu_devices) >= 2


def test_single_gpu_compute():
    """Test 2: Calcul sur chaque GPU individuellement."""
    print_section("TEST 2: Calcul sur Chaque GPU")

    n = 50_000_000  # 50M éléments
    logger.info(f"Taille des données: {n:,} éléments")

    # CPU baseline
    logger.info("\n📊 CPU (NumPy):")
    x_cpu = np.random.rand(n).astype(np.float32)
    y_cpu = np.random.rand(n).astype(np.float32)

    start = time.time()
    z_cpu = x_cpu * y_cpu + np.sin(x_cpu) + np.sqrt(y_cpu)
    cpu_time = time.time() - start
    logger.info(f"   Temps: {cpu_time*1000:.2f} ms")

    # Chaque GPU
    device_count = cp.cuda.runtime.getDeviceCount()
    gpu_times = {}

    for device_id in range(device_count):
        with cp.cuda.Device(device_id):
            logger.info(f"\n📊 GPU {device_id}:")

            x_gpu = cp.random.rand(n, dtype=cp.float32)
            y_gpu = cp.random.rand(n, dtype=cp.float32)

            # Warmup
            for _ in range(3):
                _ = x_gpu * y_gpu + cp.sin(x_gpu) + cp.sqrt(y_gpu)
            cp.cuda.Stream.null.synchronize()

            # Benchmark
            start = time.time()
            z_gpu = x_gpu * y_gpu + cp.sin(x_gpu) + cp.sqrt(y_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            gpu_times[device_id] = gpu_time

            logger.info(f"   Temps: {gpu_time*1000:.2f} ms")
            logger.info(f"   Speedup: {speedup:.2f}x vs CPU")

    return gpu_times


def test_multi_gpu_parallel():
    """Test 3: Calcul multi-GPU en parallèle."""
    print_section("TEST 3: Multi-GPU Parallèle")

    n_total = 100_000_000  # 100M éléments total
    device_count = cp.cuda.runtime.getDeviceCount()

    if device_count < 2:
        logger.warning("⚠️ Moins de 2 GPUs disponibles, test multi-GPU ignoré")
        return

    logger.info(f"Total éléments: {n_total:,}")
    logger.info(f"Répartition: {device_count} GPUs")

    # Split manuel des données
    n_per_gpu = n_total // device_count
    logger.info(f"Par GPU: {n_per_gpu:,} éléments")

    # Préparation données CPU
    x_cpu = np.random.rand(n_total).astype(np.float32)
    y_cpu = np.random.rand(n_total).astype(np.float32)

    # CPU baseline
    logger.info("\n📊 CPU séquentiel (baseline):")
    start = time.time()
    z_cpu = x_cpu * y_cpu + np.sin(x_cpu) + np.sqrt(y_cpu)
    cpu_time = time.time() - start
    logger.info(f"   Temps: {cpu_time*1000:.2f} ms")

    # Multi-GPU parallèle
    logger.info("\n📊 Multi-GPU parallèle:")

    import threading

    results = {}
    threads = []

    def compute_on_gpu(device_id, start_idx, end_idx):
        """Compute sur un GPU spécifique."""
        with cp.cuda.Device(device_id):
            x_gpu = cp.array(x_cpu[start_idx:end_idx])
            y_gpu = cp.array(y_cpu[start_idx:end_idx])

            # Warmup
            _ = x_gpu * y_gpu
            cp.cuda.Stream.null.synchronize()

            # Compute
            start_time = time.time()
            z_gpu = x_gpu * y_gpu + cp.sin(x_gpu) + cp.sqrt(y_gpu)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start_time

            results[device_id] = {
                'time': elapsed,
                'result': z_gpu.get(),
            }

    # Launch threads
    start = time.time()
    for device_id in range(device_count):
        start_idx = device_id * n_per_gpu
        end_idx = start_idx + n_per_gpu if device_id < device_count - 1 else n_total

        thread = threading.Thread(
            target=compute_on_gpu,
            args=(device_id, start_idx, end_idx)
        )
        threads.append(thread)
        thread.start()

    # Wait
    for thread in threads:
        thread.join()

    total_time = time.time() - start

    logger.info(f"   Temps total: {total_time*1000:.2f} ms")
    logger.info(f"   Speedup: {cpu_time / total_time:.2f}x vs CPU")

    # Par GPU
    for device_id in sorted(results.keys()):
        logger.info(f"   GPU {device_id}: {results[device_id]['time']*1000:.2f} ms")

    return total_time


def test_multigpu_manager():
    """Test 4: ThreadX MultiGPUManager."""
    print_section("TEST 4: ThreadX MultiGPUManager")

    try:
        manager = MultiGPUManager()
        logger.info("✅ MultiGPUManager créé")

        # Auto-profiling
        logger.info("\n📊 Auto-profiling balance...")
        ratios = manager.profile_auto_balance(sample_size=100_000, warmup=2, runs=5)

        logger.info("\n📊 Balance optimale:")
        for device_name, ratio in ratios.items():
            logger.info(f"   {device_name}: {ratio*100:.1f}%")

        return True

    except Exception as e:
        logger.error(f"❌ Erreur MultiGPUManager: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("  🔬 TEST MULTI-GPU SIMPLE - ThreadX")
    print("="*80)

    results = {}

    # Tests
    results["detection"] = test_gpu_detection()

    if not results["detection"]:
        logger.error("❌ Pas assez de GPUs détectés (besoin de 2+)")
        return

    results["single_gpu"] = test_single_gpu_compute()
    results["multi_gpu_parallel"] = test_multi_gpu_parallel()
    results["multigpu_manager"] = test_multigpu_manager()

    # Résumé
    print_section("RÉSUMÉ")

    logger.info(f"Detection GPUs: {'✅' if results['detection'] else '❌'}")
    logger.info(f"Calcul single GPU: {'✅' if results['single_gpu'] else '❌'}")
    logger.info(f"Multi-GPU parallèle: {'✅' if results['multi_gpu_parallel'] else '❌'}")
    logger.info(f"MultiGPUManager: {'✅' if results['multigpu_manager'] else '❌'}")

    print("\n" + "="*80)
    print("  ✅ Tests terminés")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
