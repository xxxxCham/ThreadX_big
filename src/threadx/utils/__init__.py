#!/usr/bin/env python3
"""
ThreadX Utils Package - Enhanced with Phase 9
==============================================

Utilities and helper modules for ThreadX framework.

Available modules:
- log: Centralized logging utilities
- gpu: GPU utilities and device management (Phase 5)
- timing: Performance measurement and throughput tracking (Phase 9)
- cache: High-performance LRU/TTL caching infrastructure (Phase 9)
- xp: Device-agnostic computing (NumPy/CuPy abstraction) (Phase 9)

Usage:
    # Legacy utilities
    from threadx.utils.log import get_logger
    from threadx.utils.gpu import is_available, MultiGPUManager

    # Phase 9 utilities
    from threadx.utils.timing import measure_throughput, Timer
    from threadx.utils.cache import cached, LRUCache, TTLCache
    from threadx.utils import xp as xpmod
    xp = xpmod.xp
    gpu_available = xpmod.gpu_available
    to_device = xpmod.to_device
    to_host = xpmod.to_host
"""

from threadx.utils.log import get_logger

# Import common imports module (Phase 2 DRY refactoring)
from . import common_imports

# Import Phase 9 utilities with graceful fallback
try:
    # Core timing utilities
    from .timing import (
        Timer,
        PerformanceMetrics,
        measure_throughput,
        track_memory,
        combined_measurement,
        performance_context,
    )

    # Caching infrastructure
    from .cache import (
        LRUCache,
        TTLCache,
        CacheStats,
        CacheEvent,
        cached,
        lru_cache,
        ttl_cache,
        indicators_cache,
        generate_stable_key,
    )

    # Device-agnostic computing
    from .xp import (
        xp,
        gpu_available,
        get_gpu_devices,
        to_device,
        to_host,
        device_synchronize,
        get_array_info,
        ensure_array_type,
        memory_pool_info,
        clear_memory_pool,
        asnumpy,
        ascupy,
        benchmark_operation,
    )

    PHASE_9_AVAILABLE = True

except ImportError as e:
    import logging

    logging.getLogger(__name__).warning(f"Phase 9 utilities not fully available: {e}")
    PHASE_9_AVAILABLE = False

__all__ = [
    "get_logger",
    "common_imports",  # Phase 2 DRY refactoring module
    # Phase 9 exports (if available)
    "Timer",
    "PerformanceMetrics",
    "measure_throughput",
    "track_memory",
    "combined_measurement",
    "performance_context",
    "LRUCache",
    "TTLCache",
    "CacheStats",
    "CacheEvent",
    "cached",
    "lru_cache",
    "ttl_cache",
    "indicators_cache",
    "generate_stable_key",
    "xp",
    "gpu_available",
    "get_gpu_devices",
    "to_device",
    "to_host",
    "device_synchronize",
    "get_array_info",
    "ensure_array_type",
    "memory_pool_info",
    "clear_memory_pool",
    "asnumpy",
    "ascupy",
    "benchmark_operation",
    "PHASE_9_AVAILABLE",
]

__version__ = "1.0.0"
__author__ = "ThreadX Team"
