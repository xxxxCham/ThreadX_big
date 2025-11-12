#!/usr/bin/env python3
"""
Diagnostic Complet GPU - ThreadX
=================================

Détecte et analyse tous les devices disponibles :
- GPUs NVIDIA (dGPU)
- GPU intégré CPU (iGPU)
- Configuration CUDA
- Utilisation mémoire
- Performance benchmark

Author: ThreadX Framework
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def print_section(title: str):
    """Print section separator."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_cupy_detection():
    """Test 1: CuPy GPU detection."""
    print_section("TEST 1: CuPy GPU Detection")

    try:
        import cupy as cp
        print("✅ CuPy importé avec succès")

        # Nombre de GPUs
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"📊 Nombre de GPUs détectés par CuPy: {device_count}")

        if device_count == 0:
            print("⚠️  AUCUN GPU détecté par CuPy !")
            return False

        # Détails de chaque GPU
        for device_id in range(device_count):
            with cp.cuda.Device(device_id):
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                mem_info = cp.cuda.runtime.memGetInfo()

                full_name = props["name"].decode("utf-8")
                compute_cap = (props["major"], props["minor"])
                memory_total_gb = mem_info[1] / (1024**3)
                memory_free_gb = mem_info[0] / (1024**3)
                memory_used_gb = memory_total_gb - memory_free_gb

                print(f"\n🎮 GPU {device_id}:")
                print(f"   Nom: {full_name}")
                print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                print(f"   VRAM Total: {memory_total_gb:.2f} GB")
                print(f"   VRAM Libre: {memory_free_gb:.2f} GB")
                print(f"   VRAM Utilisée: {memory_used_gb:.2f} GB ({memory_used_gb/memory_total_gb*100:.1f}%)")

        return True

    except ImportError:
        print("❌ CuPy non installé")
        print("   Installation: pip install cupy-cuda12x")
        return False
    except Exception as e:
        print(f"❌ Erreur CuPy: {e}")
        return False


def test_pynvml_detection():
    """Test 2: pynvml (nvidia-smi) GPU detection."""
    print_section("TEST 2: PyNVML (nvidia-smi) GPU Detection")

    try:
        import pynvml
        print("✅ pynvml importé avec succès")

        # Init NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"📊 Nombre de GPUs détectés par NVML: {device_count}")

        if device_count == 0:
            print("⚠️  AUCUN GPU détecté par NVML !")
            pynvml.nvmlShutdown()
            return False

        # Détails de chaque GPU
        for device_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # GPU utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory
            except:
                gpu_util = 0
                mem_util = 0

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

            print(f"\n🎮 GPU {device_id}:")
            print(f"   Nom: {name}")
            print(f"   VRAM Total: {mem_info.total / (1024**3):.2f} GB")
            print(f"   VRAM Utilisée: {mem_info.used / (1024**3):.2f} GB ({mem_info.used / mem_info.total * 100:.1f}%)")
            print(f"   GPU Utilization: {gpu_util}%")
            print(f"   Memory Utilization: {mem_util}%")
            print(f"   Température: {temp}°C")
            print(f"   Consommation: {power_w:.1f} W")

        pynvml.nvmlShutdown()
        return True

    except ImportError:
        print("❌ pynvml non installé")
        print("   Installation: pip install nvidia-ml-py3")
        return False
    except Exception as e:
        print(f"❌ Erreur NVML: {e}")
        return False


def test_threadx_device_manager():
    """Test 3: ThreadX device_manager."""
    print_section("TEST 3: ThreadX Device Manager")

    try:
        from threadx.gpu.device_manager import list_devices, is_available, check_nccl_support

        print("✅ device_manager importé avec succès")

        # GPU disponibles
        gpu_available = is_available()
        print(f"📊 GPUs disponibles: {gpu_available}")

        # Liste des devices
        devices = list_devices()
        print(f"📊 Nombre de devices détectés: {len(devices)}")

        for dev in devices:
            print(f"\n🎮 Device: {dev.name}")
            print(f"   ID: {dev.device_id}")
            print(f"   Nom complet: {dev.full_name}")
            if dev.device_id != -1:  # Pas CPU
                print(f"   VRAM Total: {dev.memory_total_gb:.2f} GB")
                print(f"   VRAM Libre: {dev.memory_free_gb:.2f} GB")
                print(f"   Compute Capability: {dev.compute_capability[0]}.{dev.compute_capability[1]}")

        # NCCL support
        nccl_available = check_nccl_support()
        print(f"\n📊 NCCL Support: {nccl_available}")

        return len(devices) > 1  # Au moins 1 GPU + CPU

    except ImportError as e:
        print(f"❌ Erreur import device_manager: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur device_manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_igpu_detection():
    """Test 4: Détection iGPU (GPU intégré CPU)."""
    print_section("TEST 4: Détection iGPU (GPU Intégré CPU)")

    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        igpu_found = False

        for device_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name_bytes = pynvml.nvmlDeviceGetName(handle)

            # Decode bytes to string
            if isinstance(name_bytes, bytes):
                name = name_bytes.decode('utf-8')
            else:
                name = str(name_bytes)

            # Patterns pour iGPU
            igpu_patterns = [
                "Intel",
                "UHD",
                "Iris",
                "Integrated",
                "AMD Radeon Vega",
                "AMD Ryzen",
            ]

            is_igpu = any(pattern.lower() in name.lower() for pattern in igpu_patterns)

            if is_igpu:
                igpu_found = True
                print(f"✅ iGPU détecté: {name} (GPU {device_id})")

                # Infos
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except:
                    gpu_util = 0

                print(f"   VRAM: {mem_info.total / (1024**3):.2f} GB")
                print(f"   Utilisation: {gpu_util}%")

        pynvml.nvmlShutdown()

        if not igpu_found:
            print("ℹ️  Aucun iGPU détecté (ou non accessible via CUDA)")
            print("   Note: Les iGPU Intel/AMD ne sont généralement pas accessibles via CUDA")

        return igpu_found

    except ImportError:
        print("❌ pynvml non installé")
        return False
    except Exception as e:
        print(f"❌ Erreur détection iGPU: {e}")
        return False


def test_simple_compute():
    """Test 5: Calcul GPU simple."""
    print_section("TEST 5: Calcul GPU Simple")

    try:
        import cupy as cp
        import numpy as np
        import time

        # Test data
        n = 10_000_000
        print(f"Test avec {n:,} éléments...")

        # CPU
        print("\n📊 CPU (NumPy):")
        x_cpu = np.random.rand(n).astype(np.float32)
        y_cpu = np.random.rand(n).astype(np.float32)

        start = time.time()
        z_cpu = x_cpu * y_cpu + np.sin(x_cpu)
        cpu_time = time.time() - start
        print(f"   Temps: {cpu_time*1000:.2f} ms")

        # GPU 0
        device_count = cp.cuda.runtime.getDeviceCount()
        for device_id in range(device_count):
            with cp.cuda.Device(device_id):
                print(f"\n📊 GPU {device_id}:")
                x_gpu = cp.random.rand(n, dtype=cp.float32)
                y_gpu = cp.random.rand(n, dtype=cp.float32)

                # Warmup
                _ = x_gpu * y_gpu
                cp.cuda.Stream.null.synchronize()

                # Benchmark
                start = time.time()
                z_gpu = x_gpu * y_gpu + cp.sin(x_gpu)
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.time() - start

                speedup = cpu_time / gpu_time if gpu_time > 0 else 0

                print(f"   Temps: {gpu_time*1000:.2f} ms")
                print(f"   Speedup: {speedup:.2f}x vs CPU")

        return True

    except ImportError:
        print("❌ CuPy non installé")
        return False
    except Exception as e:
        print(f"❌ Erreur calcul GPU: {e}")
        return False


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("  🔬 DIAGNOSTIC GPU COMPLET - ThreadX")
    print("="*80)

    results = {}

    # Tests
    results["cupy_detection"] = test_cupy_detection()
    results["pynvml_detection"] = test_pynvml_detection()
    results["threadx_device_manager"] = test_threadx_device_manager()
    results["igpu_detection"] = test_igpu_detection()
    results["simple_compute"] = test_simple_compute()

    # Résumé
    print_section("RÉSUMÉ")

    for test_name, success in results.items():
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"{status:12s} {test_name}")

    # Recommandations
    print("\n" + "="*80)
    print("  💡 RECOMMANDATIONS")
    print("="*80 + "\n")

    if not results["cupy_detection"]:
        print("⚠️  CuPy n'est pas installé ou ne détecte aucun GPU")
        print("   → Installer: pip install cupy-cuda12x")
        print("   → Vérifier driver NVIDIA: nvidia-smi")

    if not results["pynvml_detection"]:
        print("⚠️  pynvml n'est pas installé ou ne détecte aucun GPU")
        print("   → Installer: pip install nvidia-ml-py3")

    if results["cupy_detection"] and results["pynvml_detection"]:
        import cupy as cp
        cupy_count = cp.cuda.runtime.getDeviceCount()

        import pynvml
        pynvml.nvmlInit()
        nvml_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()

        if cupy_count != nvml_count:
            print(f"⚠️  Incohérence détection GPU:")
            print(f"   CuPy détecte: {cupy_count} GPU(s)")
            print(f"   NVML détecte: {nvml_count} GPU(s)")
            print("   → Possible problème de configuration CUDA")

    if results["igpu_detection"]:
        print("ℹ️  iGPU détecté et actif")
        print("   → Peut consommer des ressources")
        print("   → Vérifier si utilisé par ThreadX ou autre application")

    print("\n" + "="*80)
    print("  ✅ Diagnostic terminé")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
