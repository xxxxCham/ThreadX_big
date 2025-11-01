"""
ThreadX - Resource Monitoring Utilities
========================================

Monitoring utilisation CPU/RAM/GPU en temps réel pour optimisation.

Usage:
    >>> from threadx.utils.resource_monitor import get_resource_usage, log_resource_usage
    >>> stats = get_resource_usage()
    >>> print(f"CPU: {stats['cpu_percent']:.1f}%")
    >>> log_resource_usage(logger)  # Log formaté
"""

import time
from typing import Dict, Optional
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Imports optionnels
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil non disponible, monitoring CPU/RAM désactivé")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy non disponible, monitoring GPU désactivé")


def get_resource_usage() -> Dict[str, float]:
    """
    Récupère utilisation actuelle CPU/RAM/GPU.

    Returns:
        Dict avec stats ressources:
        {
            'cpu_percent': 45.2,
            'ram_percent': 62.1,
            'ram_used_gb': 15.3,
            'ram_total_gb': 32.0,
            'gpu0_percent': 25.8,
            'gpu0_vram_used_gb': 2.5,
            'gpu0_vram_total_gb': 16.0,
            'gpu1_percent': 8.3,
            'gpu1_vram_used_gb': 0.8,
            'gpu1_vram_total_gb': 8.0
        }

    Example:
        >>> stats = get_resource_usage()
        >>> if stats['gpu0_percent'] < 30:
        >>>     logger.warning("GPU0 sous-utilisé!")
    """
    stats = {}

    # CPU
    if PSUTIL_AVAILABLE:
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        stats["cpu_count"] = psutil.cpu_count(logical=False)
    else:
        stats["cpu_percent"] = 0.0
        stats["cpu_count"] = 0

    # RAM
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        stats["ram_percent"] = mem.percent
        stats["ram_used_gb"] = mem.used / (1024**3)
        stats["ram_total_gb"] = mem.total / (1024**3)
        stats["ram_available_gb"] = mem.available / (1024**3)
    else:
        stats["ram_percent"] = 0.0
        stats["ram_used_gb"] = 0.0
        stats["ram_total_gb"] = 0.0

    # GPUs
    if CUPY_AVAILABLE:
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            stats["gpu_count"] = gpu_count

            for i in range(gpu_count):
                with cp.cuda.Device(i):
                    try:
                        # Mémoire GPU
                        mem_info = cp.cuda.runtime.memGetInfo()
                        free_bytes = mem_info[0]
                        total_bytes = mem_info[1]
                        used_bytes = total_bytes - free_bytes

                        used_gb = used_bytes / (1024**3)
                        total_gb = total_bytes / (1024**3)
                        percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0

                        stats[f"gpu{i}_percent"] = percent
                        stats[f"gpu{i}_vram_used_gb"] = used_gb
                        stats[f"gpu{i}_vram_total_gb"] = total_gb
                        stats[f"gpu{i}_vram_free_gb"] = free_bytes / (1024**3)

                        # Propriétés device
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        stats[f"gpu{i}_name"] = props["name"].decode("utf-8")

                    except Exception as e:
                        logger.debug(f"Erreur stats GPU{i}: {e}")
                        stats[f"gpu{i}_percent"] = 0.0
                        stats[f"gpu{i}_vram_used_gb"] = 0.0
        except Exception as e:
            logger.debug(f"Erreur accès GPUs: {e}")
            stats["gpu_count"] = 0
    else:
        stats["gpu_count"] = 0

    return stats


def log_resource_usage(custom_logger=None):
    """
    Log formaté de l'utilisation ressources.

    Args:
        custom_logger: Logger optionnel (utilise logger module si None)

    Example:
        >>> from threadx.utils.log import get_logger
        >>> logger = get_logger(__name__)
        >>> log_resource_usage(logger)
        # 💻 CPU: 45.2% (8 cores) | 🧠 RAM: 62.1% (15.3 / 32.0 GB) |
        # 🎮 GPU0: 25.8% (2.5 / 16.0 GB) | 🎮 GPU1: 8.3% (0.8 / 8.0 GB)
    """
    log = custom_logger or logger
    stats = get_resource_usage()

    # Format CPU
    cpu_str = f"💻 CPU: {stats.get('cpu_percent', 0):.1f}%"
    if stats.get("cpu_count", 0) > 0:
        cpu_str += f" ({stats['cpu_count']} cores)"

    # Format RAM
    ram_str = (
        f"🧠 RAM: {stats.get('ram_percent', 0):.1f}% "
        f"({stats.get('ram_used_gb', 0):.1f} / {stats.get('ram_total_gb', 0):.1f} GB)"
    )

    # Format GPUs
    gpu_parts = []
    gpu_count = stats.get("gpu_count", 0)
    for i in range(gpu_count):
        gpu_pct = stats.get(f"gpu{i}_percent", 0)
        gpu_used = stats.get(f"gpu{i}_vram_used_gb", 0)
        gpu_total = stats.get(f"gpu{i}_vram_total_gb", 0)
        gpu_name = stats.get(f"gpu{i}_name", f"GPU{i}")

        gpu_parts.append(
            f"🎮 {gpu_name}: {gpu_pct:.1f}% ({gpu_used:.1f} / {gpu_total:.1f} GB)"
        )

    # Log complet
    if gpu_parts:
        log.info(f"{cpu_str} | {ram_str} | {' | '.join(gpu_parts)}")
    else:
        log.info(f"{cpu_str} | {ram_str}")


def check_resource_saturation(
    cpu_threshold: float = 85.0,
    ram_threshold: float = 85.0,
    gpu_threshold: float = 85.0,
) -> Dict[str, bool]:
    """
    Vérifie si ressources sont saturées (proche de la limite).

    Args:
        cpu_threshold: Seuil CPU (%)
        ram_threshold: Seuil RAM (%)
        gpu_threshold: Seuil GPU (%)

    Returns:
        Dict avec flags saturation:
        {
            'cpu_saturated': False,
            'ram_saturated': True,
            'gpu0_saturated': False,
            'gpu1_saturated': False,
            'any_saturated': True
        }
    """
    stats = get_resource_usage()
    result = {}

    # CPU
    result["cpu_saturated"] = stats.get("cpu_percent", 0) >= cpu_threshold

    # RAM
    result["ram_saturated"] = stats.get("ram_percent", 0) >= ram_threshold

    # GPUs
    gpu_count = stats.get("gpu_count", 0)
    for i in range(gpu_count):
        gpu_pct = stats.get(f"gpu{i}_percent", 0)
        result[f"gpu{i}_saturated"] = gpu_pct >= gpu_threshold

    # Any
    result["any_saturated"] = any(v for k, v in result.items() if k != "any_saturated")

    return result


def get_utilization_score() -> float:
    """
    Calcule score d'utilisation global (0-100%).

    Plus le score est élevé, mieux les ressources sont utilisées.
    Objectif: >80% pour optimisation maximale.

    Returns:
        Score global (moyenne pondérée CPU/RAM/GPU)

    Example:
        >>> score = get_utilization_score()
        >>> if score < 50:
        >>>     logger.warning(f"Sous-utilisation ressources: {score:.1f}%")
    """
    stats = get_resource_usage()

    weights = []
    values = []

    # CPU (poids 0.3)
    if stats.get("cpu_percent", 0) > 0:
        weights.append(0.3)
        values.append(stats["cpu_percent"])

    # RAM (poids 0.2)
    if stats.get("ram_percent", 0) > 0:
        weights.append(0.2)
        values.append(stats["ram_percent"])

    # GPUs (poids 0.5 total, réparti entre GPUs)
    gpu_count = stats.get("gpu_count", 0)
    if gpu_count > 0:
        gpu_weight = 0.5 / gpu_count
        for i in range(gpu_count):
            gpu_pct = stats.get(f"gpu{i}_percent", 0)
            if gpu_pct > 0:
                weights.append(gpu_weight)
                values.append(gpu_pct)

    # Calcul score pondéré
    if not weights:
        return 0.0

    total_weight = sum(weights)
    weighted_sum = sum(w * v for w, v in zip(weights, values))
    score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return score


__all__ = [
    "get_resource_usage",
    "log_resource_usage",
    "check_resource_saturation",
    "get_utilization_score",
]
