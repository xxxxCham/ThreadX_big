"""
ThreadX Device Manager - GPU Detection & Management
===================================================

Gestion unified des devices GPU/CPU avec détection automatique et mapping
des noms conviviaux ("5090", "2060") vers les IDs CuPy.

Architecture:
- Détection multi-GPU via CuPy avec fallback NumPy
- Mapping nom ↔ ID pour RTX 5090, RTX 2060, etc.
- Vérification NCCL pour synchronisation multi-GPU
- Interface unifiée xp() pour code device-agnostic
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Detection CuPy/GPU support
try:
    import cupy as cp

    CUPY_AVAILABLE = True
    logger.info("CuPy détecté - Support GPU activé")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logger.info("CuPy indisponible - Fallback CPU NumPy")

import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    """
    Informations sur un device GPU.

    Attributes:
        device_id: ID CuPy du device (0, 1, etc.)
        name: Nom convivial ("5090", "2060", "cpu")
        full_name: Nom complet du GPU
        memory_total: Mémoire totale en bytes
        memory_free: Mémoire libre en bytes
        compute_capability: Version compute CUDA (ex. (8, 6))
        is_available: True si device utilisable
    """

    device_id: int
    name: str
    full_name: str
    memory_total: int
    memory_free: int
    compute_capability: tuple[int, int]
    is_available: bool

    @property
    def memory_total_gb(self) -> float:
        """Mémoire totale en Go."""
        return self.memory_total / (1024**3)

    @property
    def memory_free_gb(self) -> float:
        """Mémoire libre en Go."""
        return self.memory_free / (1024**3)

    @property
    def memory_used_pct(self) -> float:
        """Pourcentage mémoire utilisée."""
        if self.memory_total == 0:
            return 0.0
        return ((self.memory_total - self.memory_free) / self.memory_total) * 100


def _parse_gpu_name(gpu_name: str) -> str:
    """
    Extrait un nom convivial depuis le nom complet GPU.

    Args:
        gpu_name: Nom complet GPU (ex. "NVIDIA GeForce RTX 5090")

    Returns:
        Nom convivial (ex. "5090")
    """
    # Patterns courants pour RTX series
    gpu_name_upper = gpu_name.upper()

    if "RTX 5080" in gpu_name_upper:
        return "5080"
    elif "RTX 4090" in gpu_name_upper:
        return "4090"
    elif "RTX 4080" in gpu_name_upper:
        return "4080"
    elif "RTX 3090" in gpu_name_upper:
        return "3090"
    elif "RTX 3080" in gpu_name_upper:
        return "3080"
    elif "RTX 2080" in gpu_name_upper:
        return "2080"
    elif "RTX 2070" in gpu_name_upper:
        return "2070"
    elif "RTX 2060" in gpu_name_upper:
        return "2060"
    elif "GTX 1080" in gpu_name_upper:
        return "1080"
    elif "GTX 1070" in gpu_name_upper:
        return "1070"
    elif "GTX 1060" in gpu_name_upper:
        return "1060"

    # Fallback: prendre les derniers chiffres
    import re

    numbers = re.findall(r"\d+", gpu_name)
    if numbers:
        return numbers[-1]  # Dernier nombre trouvé

    # Fallback ultime
    return gpu_name.split()[-1] if gpu_name else "unknown"


def is_available() -> bool:
    """
    Vérifie si au moins un GPU est disponible.

    Returns:
        True si GPU(s) détecté(s), False sinon
    """
    if not CUPY_AVAILABLE:
        return False

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        return device_count > 0
    except Exception as e:
        logger.debug(f"Erreur détection GPU: {e}")
        return False


def list_devices() -> List[DeviceInfo]:
    """
    Liste tous les devices disponibles (GPU + CPU fallback).

    Returns:
        Liste des DeviceInfo triés par performance décroissante

    Example:
        >>> devices = list_devices()
        >>> for dev in devices:
        ...     print(f"{dev.name}: {dev.memory_total_gb:.1f}GB")
        5090: 32.0GB
        2060: 6.0GB
        cpu: 0.0GB
    """
    devices = []

    if CUPY_AVAILABLE:
        try:
            device_count = cp.cuda.runtime.getDeviceCount() if cp else 0
            logger.info(f"Détection de {device_count} GPU(s)")

            for device_id in range(device_count):
                try:
                    if cp:
                        with cp.cuda.Device(device_id):
                            # Propriétés du device
                            props = cp.cuda.runtime.getDeviceProperties(device_id)

                            # Mémoire
                            mem_info = cp.cuda.runtime.memGetInfo()
                        memory_free = mem_info[0]
                        memory_total = mem_info[1]

                        # Nom convivial
                        full_name = props["name"].decode("utf-8")
                        friendly_name = _parse_gpu_name(full_name)

                        # Compute capability
                        compute_cap = (props["major"], props["minor"])

                        device_info = DeviceInfo(
                            device_id=device_id,
                            name=friendly_name,
                            full_name=full_name,
                            memory_total=memory_total,
                            memory_free=memory_free,
                            compute_capability=compute_cap,
                            is_available=True,
                        )

                        devices.append(device_info)
                        logger.info(
                            f"GPU {device_id} ({friendly_name}): "
                            f"{device_info.memory_total_gb:.1f}GB, "
                            f"CC {compute_cap[0]}.{compute_cap[1]}"
                        )

                except Exception as e:
                    logger.warning(f"Erreur lecture GPU {device_id}: {e}")

        except Exception as e:
            logger.error(f"Erreur énumération GPU: {e}")

    # Ajout CPU comme fallback
    cpu_device = DeviceInfo(
        device_id=-1,
        name="cpu",
        full_name="CPU NumPy Fallback",
        memory_total=0,
        memory_free=0,
        compute_capability=(0, 0),
        is_available=True,
    )
    devices.append(cpu_device)

    # Tri par performance (mémoire totale décroissante, CPU en dernier)
    devices.sort(key=lambda d: (d.device_id == -1, -d.memory_total))

    return devices


def get_device_by_name(name: str) -> Optional[DeviceInfo]:
    """
    Récupère un device par son nom convivial.

    Args:
        name: Nom du device ("5090", "2060", "cpu")

    Returns:
        DeviceInfo si trouvé, None sinon

    Example:
        >>> gpu = get_device_by_name("5090")
        >>> if gpu:
        ...     print(f"RTX 5090: {gpu.memory_total_gb:.1f}GB")
    """
    devices = list_devices()
    for device in devices:
        if device.name.lower() == name.lower():
            return device
    return None


def get_device_by_id(device_id: int) -> Optional[DeviceInfo]:
    """
    Récupère un device par son ID CuPy.

    Args:
        device_id: ID CuPy du device (0, 1, etc.) ou -1 pour CPU

    Returns:
        DeviceInfo si trouvé, None sinon
    """
    devices = list_devices()
    for device in devices:
        if device.device_id == device_id:
            return device
    return None


def check_nccl_support() -> bool:
    """
    Vérifie si NCCL est disponible pour la synchronisation multi-GPU.

    Returns:
        True si NCCL utilisable, False sinon
    """
    if not CUPY_AVAILABLE:
        return False

    try:
        # Test import NCCL
        import cupy.cuda.nccl  # noqa: F401

        # Test basique
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 2:
            logger.debug("NCCL disponible mais <2 GPU détectés")
            return False

        logger.info("Support NCCL activé pour synchronisation multi-GPU")
        return True

    except (ImportError, AttributeError) as e:
        logger.debug(f"NCCL indisponible: {e}")
        return False


def xp(device_name: Optional[str] = None) -> Any:
    """
    Retourne le module array approprié (CuPy ou NumPy).

    Args:
        device_name: Nom du device optionnel ("5090", "cpu", etc.)
                    Si None, utilise GPU par défaut ou NumPy

    Returns:
        Module cupy ou numpy selon disponibilité

    Example:
        >>> # Code device-agnostic
        >>> xp_module = xp("5090")  # CuPy si 5090 disponible
        >>> arr = xp_module.array([1, 2, 3])
        >>> result = xp_module.sum(arr)
    """
    if device_name == "cpu":
        return np

    if not CUPY_AVAILABLE:
        return np

    if device_name:
        device = get_device_by_name(device_name)
        if not device or device.device_id == -1:
            return np

        # Définir device actuel
        try:
            if cp:
                cp.cuda.Device(device.device_id).use()
            return cp
        except Exception as e:
            logger.warning(f"Erreur activation device {device_name}: {e}")
            return np

    # GPU par défaut
    try:
        if cp and cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except Exception:
        pass

    return np


def get_memory_info(device_name: str) -> Dict[str, float]:
    """
    Récupère les infos mémoire pour un device.

    Args:
        device_name: Nom du device

    Returns:
        Dict avec 'total_gb', 'free_gb', 'used_pct'

    Raises:
        ValueError: Si device introuvable
    """
    device = get_device_by_name(device_name)
    if not device:
        raise ValueError(f"Device '{device_name}' non trouvé")

    if device.device_id == -1:  # CPU
        return {"total_gb": 0.0, "free_gb": 0.0, "used_pct": 0.0}

    if not CUPY_AVAILABLE:
        raise ValueError("CuPy requis pour infos mémoire GPU")

    try:
        if cp:
            with cp.cuda.Device(device.device_id):
                mem_info = cp.cuda.runtime.memGetInfo()
                return {
                    "free": mem_info[0] / (1024**3),  # GB
                    "total": mem_info[1] / (1024**3),  # GB
                    "used": (mem_info[1] - mem_info[0]) / (1024**3),  # GB
                }
        else:
            return {"free": 0.0, "total": 0.0, "used": 0.0}
            free_bytes = mem_info[0]
            total_bytes = mem_info[1]

            return {
                "total_gb": total_bytes / (1024**3),
                "free_gb": free_bytes / (1024**3),
                "used_pct": ((total_bytes - free_bytes) / total_bytes) * 100,
            }
    except Exception as e:
        logger.error(f"Erreur lecture mémoire {device_name}: {e}")
        raise


# Export des exceptions CuPy si disponibles
if CUPY_AVAILABLE and cp:
    CudaMemoryError = cp.cuda.memory.OutOfMemoryError
    CudaRuntimeError = cp.cuda.runtime.CUDARuntimeError
else:
    # Fallback exceptions
    class CudaMemoryError(RuntimeError):
        """Exception pour erreurs mémoire GPU (fallback)"""

        pass

    class CudaRuntimeError(RuntimeError):
        """Exception pour erreurs runtime GPU (fallback)"""

        pass
