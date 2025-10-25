"""Backend-agnostic array helpers with lazy CuPy support."""

from __future__ import annotations

import importlib
import logging
import time
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# Public placeholders updated at runtime.
cp: Optional[Any] = None
CUPY_AVAILABLE: bool = False

# Internal caches.
_CUPY_IMPORT_ERROR: Optional[BaseException] = None
_CUPY_MODULE: Optional[Any] = None
_XP_BACKEND: Any = np
_BACKEND_OVERRIDE: Optional[str] = None  # "numpy", "cupy" or None (auto)


def _get_device_manager() -> Optional[Any]:
    try:
        from threadx.utils.gpu import device_manager  # type: ignore

        return device_manager
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("device_manager unavailable: %s", exc)
        return None


def _get_cupy() -> Optional[Any]:
    global _CUPY_MODULE, _CUPY_IMPORT_ERROR, CUPY_AVAILABLE, cp

    if _CUPY_MODULE is not None:
        return _CUPY_MODULE
    if _CUPY_IMPORT_ERROR is not None:
        return None

    try:
        module = importlib.import_module("cupy")
        _CUPY_MODULE = module
        cp = module
        CUPY_AVAILABLE = True
        logger.debug("CuPy successfully imported")
        return module
    except Exception as exc:  # pragma: no cover - import failure path is CI common
        _CUPY_IMPORT_ERROR = exc
        CUPY_AVAILABLE = False
        cp = None
        logger.debug("CuPy import failed: %s", exc)
        return None


def refresh_cupy_cache() -> None:
    """Reset cached CuPy module information."""
    global _CUPY_MODULE, _CUPY_IMPORT_ERROR, CUPY_AVAILABLE, cp

    _CUPY_MODULE = None
    _CUPY_IMPORT_ERROR = None
    CUPY_AVAILABLE = False
    cp = None


def _sync_cupy_state() -> Optional[Any]:
    module = _get_cupy()
    return module


def _set_backend(backend: Any) -> Any:
    global _XP_BACKEND, xp

    if backend is not _XP_BACKEND:
        _XP_BACKEND = backend
        logger.debug("Backend switched to %s", "cupy" if backend is cp else "numpy")
    xp = _XP_BACKEND
    return _XP_BACKEND


def gpu_available() -> bool:
    """Return True if GPU execution is possible with the current environment."""
    module = _sync_cupy_state()
    if module is None:
        return False

    manager = _get_device_manager()
    if manager is not None and hasattr(manager, "is_available"):
        try:
            available = bool(manager.is_available())
            logger.debug("device_manager reports GPU available: %s", available)
            return available
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("device_manager.is_available raised: %s", exc)

    try:
        count = module.cuda.runtime.getDeviceCount()
        logger.debug("Detected %s CUDA device(s)", count)
        return count > 0
    except Exception as exc:  # pragma: no cover - CUDA probing failure
        logger.debug("CuPy runtime detection failed: %s", exc)
        return False


def _select_backend(prefer_gpu: bool) -> Any:
    if _BACKEND_OVERRIDE == "numpy":
        return np
    if _BACKEND_OVERRIDE == "cupy":
        module = _sync_cupy_state()
        if module is not None and gpu_available():
            return module
        logger.debug("Requested CuPy backend but unavailable; falling back to NumPy")
        return np

    if prefer_gpu and gpu_available():
        module = _sync_cupy_state()
        if module is not None:
            return module

    return np


def get_xp(prefer_gpu: bool = True) -> Any:
    """Return the numerical backend module (NumPy or CuPy)."""
    backend = _select_backend(prefer_gpu)
    return _set_backend(backend)


# Public shorthand mirroring NumPy/CuPy API.
xp: Any = get_xp(prefer_gpu=False)


def is_gpu_backend() -> bool:
    return _XP_BACKEND is not np


def get_backend_name() -> str:
    backend = "cupy" if is_gpu_backend() and _sync_cupy_state() is not None else "numpy"
    return backend


def configure_backend(preferred: str) -> None:
    """Force a backend ("numpy", "cupy", or "auto")."""
    valid = {"numpy", "cupy", "auto"}
    if preferred not in valid:
        raise ValueError(
            f"Invalid backend '{preferred}'. Expected one of {sorted(valid)}"
        )

    global _BACKEND_OVERRIDE
    _BACKEND_OVERRIDE = None if preferred == "auto" else preferred
    get_xp(prefer_gpu=True)


def to_device(array: Any, dtype: Optional[Any] = None) -> Any:
    backend = get_xp()
    if backend is np:
        return np.array(array, dtype=dtype, copy=False)

    module = _sync_cupy_state()
    if module is None:
        raise RuntimeError("CuPy backend requested but unavailable")
    if hasattr(module, "asarray"):
        return module.asarray(array, dtype=dtype)
    return module.array(array, dtype=dtype)


def to_host(array: Any) -> np.ndarray:
    module = _sync_cupy_state()
    if module is None or isinstance(array, np.ndarray):
        return np.asarray(array)
    return module.asnumpy(array)


def asnumpy(array: Any) -> np.ndarray:
    return to_host(array)


def ascupy(array: Any, dtype: Optional[Any] = None) -> Any:
    module = _sync_cupy_state()
    if module is None:
        raise RuntimeError("CuPy is not available on this system")
    if isinstance(array, module.ndarray):
        return array if dtype is None else array.astype(dtype)
    return module.asarray(array, dtype=dtype)


def ensure_array_type(
    array: Any, dtype: Optional[Any] = None, *, xp_module: Any = None
) -> Any:
    backend = xp_module or get_xp()
    if backend is np:
        return np.asarray(array, dtype=dtype)

    module = _sync_cupy_state()
    if module is None:
        return np.asarray(array, dtype=dtype)
    return module.asarray(array, dtype=dtype)


def memory_pool_info() -> Mapping[str, Union[int, float, str]]:
    module = _sync_cupy_state()
    if module is None:
        return {"backend": "numpy", "used_bytes": 0, "total_bytes": 0}

    pool = module.get_default_memory_pool()
    pinned = module.cuda.get_default_pinned_memory_pool()
    info: MutableMapping[str, Union[int, float, str]] = {
        "backend": "cupy",
        "used_bytes": pool.used_bytes(),
        "total_bytes": pool.total_bytes(),
        "pinned_used": pinned.used_bytes(),
        "pinned_total": pinned.total_bytes(),
    }
    return info


def clear_memory_pool() -> None:
    module = _sync_cupy_state()
    if module is not None:
        module.get_default_memory_pool().free_all_blocks()
        module.cuda.get_default_pinned_memory_pool().free_all_blocks()


@contextmanager
def _gpu_timer(module: Any) -> Iterable[Tuple[Callable[[], float], Any]]:
    start = module.cuda.Event()
    end = module.cuda.Event()
    stream = module.cuda.get_current_stream()
    start.record(stream)
    yield (lambda: _elapsed_gpu_time(module, start, end, stream), stream)


def _elapsed_gpu_time(module: Any, start: Any, end: Any, stream: Any) -> float:
    end.record(stream)
    end.synchronize()
    return module.cuda.get_elapsed_time(start, end) / 1000.0


def benchmark_operation(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Tuple[Any, float]:
    """Execute *func* and return (result, elapsed_seconds)."""
    backend = get_xp()
    if backend is np:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    module = _sync_cupy_state()
    if module is None:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    with _gpu_timer(module) as (timer, stream):
        result = func(*args, **kwargs)
        elapsed = timer()
    stream.synchronize()
    return result, elapsed


def device_synchronize() -> None:
    module = _sync_cupy_state()
    if module is not None:
        module.cuda.get_current_stream().synchronize()


def get_array_info(array: Any) -> Mapping[str, Any]:
    module = _sync_cupy_state()
    backend = "numpy"
    device: Union[str, int] = "cpu"

    if module is not None and isinstance(array, getattr(module, "ndarray", ())):
        backend = "cupy"
        try:
            device = module.cuda.get_current_device().id
        except Exception:  # pragma: no cover - device query failure
            device = "gpu"
        shape = getattr(array, "shape", ())
        dtype = getattr(array, "dtype", "unknown")
        size = int(getattr(array, "size", module.asnumpy(array).size))
        memory_mb = float(getattr(array, "nbytes", module.asnumpy(array).nbytes)) / (
            1024**2
        )
        return {
            "backend": backend,
            "device": device,
            "shape": shape,
            "dtype": dtype,
            "size": size,
            "memory_mb": memory_mb,
        }

    arr = np.asarray(array)
    return {
        "backend": backend,
        "device": device,
        "shape": arr.shape,
        "dtype": arr.dtype,
        "size": int(arr.size),
        "memory_mb": float(arr.nbytes) / (1024**2),
    }


__all__ = [
    "np",
    "cp",
    "CUPY_AVAILABLE",
    "gpu_available",
    "get_xp",
    "xp",
    "is_gpu_backend",
    "get_backend_name",
    "configure_backend",
    "to_device",
    "to_host",
    "asnumpy",
    "ascupy",
    "ensure_array_type",
    "memory_pool_info",
    "clear_memory_pool",
    "benchmark_operation",
    "device_synchronize",
    "get_array_info",
    "refresh_cupy_cache",
]
