"""
ThreadX GPU Startup Diagnostics
================================

Module de diagnostic GPU au démarrage pour afficher l'état multi-GPU.
Appelé automatiquement par threadx_gpu_init.py pour informer l'utilisateur.
"""

import sys
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def check_and_report_gpus() -> dict:
    """
    Vérifie et affiche l'état des GPUs au démarrage.

    Returns:
        Dict avec infos GPU : {"gpu_count", "devices", "multi_gpu_enabled", "primary_gpu"}
    """
    result = {
        "gpu_count": 0,
        "devices": [],
        "multi_gpu_enabled": False,
        "primary_gpu": None,
        "cupy_available": False,
    }

    # Import CuPy avec gestion gracieuse
    try:
        import cupy as cp
        result["cupy_available"] = True
    except ImportError:
        logger.warning("⚠️  CuPy non disponible - Mode CPU uniquement")
        return result

    # Détection GPUs brute
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        result["gpu_count"] = gpu_count

        if gpu_count == 0:
            logger.info("ℹ️  Aucun GPU détecté - Mode CPU uniquement")
            return result

        # Collecte infos devices
        devices_info = []
        for i in range(gpu_count):
            try:
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode('utf-8')
                compute_cap = f"{props['major']}.{props['minor']}"

                with cp.cuda.Device(i):
                    mem_info = cp.cuda.runtime.memGetInfo()
                    mem_total_gb = mem_info[1] / (1024**3)

                devices_info.append({
                    "id": i,
                    "name": name,
                    "compute_capability": compute_cap,
                    "memory_gb": mem_total_gb,
                })
            except Exception as e:
                logger.warning(f"Erreur lecture GPU {i}: {e}")

        result["devices"] = devices_info

        # Log résumé
        if gpu_count == 1:
            gpu = devices_info[0]
            result["primary_gpu"] = gpu["name"]
            logger.info("=" * 60)
            logger.info(f"🎯 GPU PRINCIPAL DÉTECTÉ")
            logger.info(f"   {gpu['name']}")
            logger.info(f"   Mémoire: {gpu['memory_gb']:.1f} GB")
            logger.info(f"   Compute Capability: {gpu['compute_capability']}")
            logger.info("=" * 60)

        elif gpu_count >= 2:
            result["multi_gpu_enabled"] = True
            result["primary_gpu"] = devices_info[0]["name"]

            logger.info("=" * 60)
            logger.info(f"💎 MULTI-GPU DÉTECTÉ : {gpu_count} GPUs")
            logger.info("=" * 60)
            for gpu in devices_info:
                logger.info(f"   GPU {gpu['id']}: {gpu['name']}")
                logger.info(f"      └─ {gpu['memory_gb']:.1f} GB VRAM | CC {gpu['compute_capability']}")
            logger.info("=" * 60)
            logger.info("✅ Le système Multi-GPU est prêt à être utilisé")
            logger.info("   (Activable dans Sidebar > Multi-GPU)")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Erreur détection GPU: {e}")

    return result


def suppress_pytorch_warnings():
    """
    Supprime les warnings PyTorch pour compute capability non supportée.

    ThreadX utilise CuPy (pas PyTorch) pour les calculs GPU, donc ces warnings
    sont non pertinents et créent de la confusion.
    """
    import warnings

    # Filtrer warnings CUDA de PyTorch
    warnings.filterwarnings(
        "ignore",
        message=".*CUDA capability.*not compatible.*PyTorch.*",
        category=UserWarning,
    )

    # Filtrer warnings généraux PyTorch CUDA
    warnings.filterwarnings(
        "ignore",
        message=".*PyTorch install.*CUDA capabilities.*",
        category=UserWarning,
    )

    logger.debug("🔇 Warnings PyTorch CUDA supprimés (ThreadX utilise CuPy)")


def initialize_multi_gpu_manager():
    """
    Initialise le MultiGPUManager dès le démarrage si 2+ GPUs détectés.

    Permet d'afficher les infos de balance immédiatement au lieu d'attendre
    la première optimisation (lazy loading).
    """
    try:
        from threadx.gpu.multi_gpu import get_default_manager
        from threadx.gpu.device_manager import list_devices

        devices = list_devices(use_cache=False)
        gpu_devices = [d for d in devices if d.device_id >= 0]

        if len(gpu_devices) >= 2:
            # Force création du manager (affichera ses propres logs de config)
            manager = get_default_manager()
            # Note: Balance et NCCL sont loggés par le manager lui-même
            return True
        else:
            logger.debug("Multi-GPU Manager non initialisé (< 2 GPUs)")
            return False

    except Exception as e:
        logger.warning(f"⚠️  Impossible d'initialiser Multi-GPU Manager: {e}")
        return False
