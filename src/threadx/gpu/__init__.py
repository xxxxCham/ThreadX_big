"""
ThreadX GPU Utilities - Phase 5 Multi-GPU Support
==================================================

Gestionnaire multi-GPU avec distribution de charge automatique.

Modules:
    device_manager: Détection et gestion des devices GPU/CPU
    multi_gpu: Orchestration multi-GPU avec auto-balancing
"""

from .device_manager import (
    is_available,
    list_devices,
    get_device_by_name,
    get_device_by_id,
    check_nccl_support,
    xp,
    DeviceInfo,
)

from .multi_gpu import (
    MultiGPUManager,
    DeviceUnavailableError,
    GPUMemoryError,
    ShapeMismatchError,
    NonVectorizableFunctionError,
    get_default_manager,
)

__all__ = [
    # Device utilities
    "is_available",
    "list_devices",
    "get_device_by_name",
    "get_device_by_id",
    "check_nccl_support",
    "xp",
    "DeviceInfo",
    # Multi-GPU manager
    "MultiGPUManager",
    "get_default_manager",
    # Exceptions
    "DeviceUnavailableError",
    "GPUMemoryError",
    "ShapeMismatchError",
    "NonVectorizableFunctionError",
]

__version__ = "5.0.0"
