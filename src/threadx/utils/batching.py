"""
ThreadX Utils Module - Phase 9
Batching Utilities for Large Dataset Processing.

Provides efficient batch processing for large time series and datasets:
- Memory-efficient batch generation
- Configurable batch sizes with adaptive sizing
- Progress tracking and performance monitoring
- Integration with ThreadX caching and timing utilities

Designed for processing large indicator calculations, backtesting sweeps,
and Monte Carlo simulations without memory overflow.
"""

import logging
from typing import Iterator, Any, Optional, Union, Callable, Tuple, List
from dataclasses import dataclass
import numpy as np

# Import ThreadX logger - fallback to standard logging if not available
try:
    from threadx.utils.log import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


# Import ThreadX Settings - fallback if not available
try:
    from threadx.config import load_settings

    SETTINGS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency during tests
    SETTINGS_AVAILABLE = False

# Import ThreadX utils if available
try:
    from threadx.utils.timing import Timer, performance_context

    TIMING_AVAILABLE = True
except ImportError:
    TIMING_AVAILABLE = False

try:
    from threadx.utils import xp as xpmod

    xp = xpmod.xp
    get_array_info = xpmod.get_array_info
    XP_AVAILABLE = True
except ImportError:
    XP_AVAILABLE = False


logger = get_logger(__name__)


@dataclass
class BatchInfo:
    """Information about a data batch."""

    batch_id: int
    start_idx: int
    end_idx: int
    size: int
    total_batches: int
    progress_pct: float
    memory_mb: Optional[float] = None


def _get_default_batch_size() -> int:
    """Get default batch size from ThreadX settings."""
    if SETTINGS_AVAILABLE:
        try:
            settings = load_settings()
            return settings.VECTORIZATION_BATCH_SIZE
        except Exception:
            pass
    return 10000  # Fallback default


def batch_generator(
    data: Any,
    batch_size: Optional[int] = None,
    overlap: int = 0,
    *,
    track_memory: bool = True,
    progress_callback: Optional[Callable[[BatchInfo], None]] = None,
) -> Iterator[Tuple[Any, BatchInfo]]:
    """
    Generate batches from input data with optional overlap.

    Memory-efficient batch generation for large datasets. Supports
    overlap for windowed operations (e.g., moving averages).

    Parameters
    ----------
    data : array-like
        Input data to batch. Must support len() and slicing.
    batch_size : int, optional
        Size of each batch. If None, uses ThreadX settings default.
    overlap : int, default 0
        Number of elements to overlap between batches.
    track_memory : bool, default True
        Whether to track memory usage of batches.
    progress_callback : callable, optional
        Callback function called for each batch with BatchInfo.

    Yields
    ------
    tuple[Any, BatchInfo]
        Batch data and batch information.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100000)
    >>>
    >>> for batch, info in batch_generator(data, batch_size=5000):
    ...     result = np.mean(batch)  # Process batch
    ...     print(f"Batch {info.batch_id}: {info.progress_pct:.1f}% complete")

    >>> # With overlap for moving window operations
    >>> window_size = 20
    >>> for batch, info in batch_generator(data, batch_size=1000, overlap=window_size-1):
    ...     moving_avg = np.convolve(batch, np.ones(window_size)/window_size, mode='valid')

    Notes
    -----
    - Overlap allows for consistent windowed operations across batch boundaries
    - Memory tracking helps identify optimal batch sizes
    - Progress callback enables UI updates during long operations
    """
    if batch_size is None:
        batch_size = _get_default_batch_size()

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if overlap < 0:
        raise ValueError("Overlap must be non-negative")

    if overlap >= batch_size:
        raise ValueError("Overlap must be less than batch size")

    try:
        data_length = len(data)
    except TypeError:
        raise TypeError("Data must support len() operation")

    if data_length == 0:
        logger.warning("Empty data provided to batch_generator")
        return

    # Calculate number of batches
    effective_step = batch_size - overlap
    if effective_step <= 0:
        raise ValueError("Effective step size (batch_size - overlap) must be positive")

    total_batches = max(
        1, (data_length - overlap + effective_step - 1) // effective_step
    )

    logger.info(
        f"Starting batch processing: {data_length} items, "
        f"batch_size={batch_size}, overlap={overlap}, "
        f"total_batches={total_batches}"
    )

    batch_id = 0
    start_idx = 0

    while start_idx < data_length:
        # Calculate batch boundaries
        end_idx = min(start_idx + batch_size, data_length)
        actual_size = end_idx - start_idx

        # Extract batch
        try:
            batch_data = data[start_idx:end_idx]
        except Exception as e:
            logger.error(f"Failed to extract batch {batch_id}: {e}")
            break

        # Calculate progress
        progress_pct = (batch_id + 1) / total_batches * 100.0

        # Track memory if requested
        memory_mb = None
        if track_memory and XP_AVAILABLE:
            try:
                array_info = get_array_info(batch_data)
                memory_mb = array_info.get("memory_mb", 0.0)
            except Exception as e:
                logger.debug(f"Memory tracking failed for batch {batch_id}: {e}")

        # Create batch info
        info = BatchInfo(
            batch_id=batch_id,
            start_idx=start_idx,
            end_idx=end_idx,
            size=actual_size,
            total_batches=total_batches,
            progress_pct=progress_pct,
            memory_mb=memory_mb,
        )

        # Call progress callback
        if progress_callback:
            try:
                progress_callback(info)
            except Exception as e:
                logger.warning(f"Progress callback failed for batch {batch_id}: {e}")

        yield batch_data, info

        # Move to next batch
        batch_id += 1
        start_idx += effective_step

        # Break if this was the last possible batch
        if end_idx >= data_length:
            break

    logger.info(f"Batch processing completed: {batch_id} batches processed")


def adaptive_batch_size(
    data_size: int,
    target_memory_mb: float = 256.0,
    element_size_bytes: int = 8,
    min_batch_size: int = 100,
    max_batch_size: Optional[int] = None,
) -> int:
    """
    Calculate adaptive batch size based on memory constraints.

    Automatically determines optimal batch size to stay within memory limits
    while maximizing processing efficiency.

    Parameters
    ----------
    data_size : int
        Total number of elements in dataset.
    target_memory_mb : float, default 256.0
        Target memory usage per batch in MB.
    element_size_bytes : int, default 8
        Approximate size per element in bytes (e.g., 8 for float64).
    min_batch_size : int, default 100
        Minimum allowed batch size.
    max_batch_size : int, optional
        Maximum allowed batch size. If None, uses ThreadX settings.

    Returns
    -------
    int
        Recommended batch size.

    Examples
    --------
    >>> # For 1M float64 elements, targeting 256MB per batch
    >>> batch_size = adaptive_batch_size(1_000_000, element_size_bytes=8)
    >>> print(f"Recommended batch size: {batch_size}")

    >>> # For smaller memory constraint
    >>> batch_size = adaptive_batch_size(1_000_000, target_memory_mb=64.0)
    """
    if max_batch_size is None:
        max_batch_size = _get_default_batch_size()

    # Calculate batch size based on memory target
    target_bytes = target_memory_mb * 1024 * 1024
    calculated_size = int(target_bytes / element_size_bytes)

    # Apply constraints
    batch_size = max(min_batch_size, calculated_size)
    batch_size = min(batch_size, max_batch_size)
    batch_size = min(batch_size, data_size)  # Can't be larger than data

    logger.info(
        f"Adaptive batch size: {batch_size} "
        f"(target: {target_memory_mb}MB, element_size: {element_size_bytes}B)"
    )

    return batch_size


def batch_process(
    data: Any,
    processor_func: Callable[[Any], Any],
    batch_size: Optional[int] = None,
    *,
    combine_func: Optional[Callable[[List[Any]], Any]] = None,
    track_performance: bool = True,
    progress_callback: Optional[Callable[[BatchInfo], None]] = None,
) -> Any:
    """
    Process data in batches and optionally combine results.

    High-level batch processing with automatic result combination.
    Includes performance tracking and progress reporting.

    Parameters
    ----------
    data : array-like
        Input data to process.
    processor_func : callable
        Function to apply to each batch. Should accept batch data and return result.
    batch_size : int, optional
        Batch size. If None, uses adaptive sizing.
    combine_func : callable, optional
        Function to combine batch results. If None, returns list of results.
    track_performance : bool, default True
        Whether to measure and log performance.
    progress_callback : callable, optional
        Progress callback for UI updates.

    Returns
    -------
    Any
        Combined results or list of batch results.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100000)
    >>>
    >>> # Simple batch processing
    >>> results = batch_process(
    ...     data,
    ...     lambda batch: np.mean(batch),
    ...     combine_func=lambda results: np.array(results)
    ... )

    >>> # With custom batch size and progress tracking
    >>> def progress_update(info):
    ...     print(f"Processing: {info.progress_pct:.1f}% complete")
    >>>
    >>> results = batch_process(
    ...     data,
    ...     lambda batch: expensive_calculation(batch),
    ...     batch_size=5000,
    ...     progress_callback=progress_update
    ... )
    """
    if batch_size is None:
        # Use adaptive batch sizing
        try:
            element_size = 8  # Default float64
            if hasattr(data, "dtype"):
                element_size = data.dtype.itemsize
            batch_size = adaptive_batch_size(len(data), element_size_bytes=element_size)
        except Exception:
            batch_size = _get_default_batch_size()

    results = []
    total_items = 0

    # Setup performance tracking
    perf_context = None
    if track_performance and TIMING_AVAILABLE:
        try:
            perf_context = performance_context(
                "batch_process", task_count=len(data), unit_of_work="item"
            )
            perf_context.__enter__()
        except Exception as e:
            logger.debug(f"Performance tracking setup failed: {e}")

    try:
        # Process batches
        for batch_data, batch_info in batch_generator(
            data, batch_size=batch_size, progress_callback=progress_callback
        ):
            try:
                batch_result = processor_func(batch_data)
                results.append(batch_result)
                total_items += batch_info.size

            except Exception as e:
                logger.error(f"Processing failed for batch {batch_info.batch_id}: {e}")
                # Continue with other batches
                continue

        # Combine results
        if combine_func and results:
            try:
                final_result = combine_func(results)
                logger.info(
                    f"Batch processing completed: {total_items} items, combined results"
                )
                return final_result
            except Exception as e:
                logger.error(f"Result combination failed: {e}")
                # Fall back to returning list

        logger.info(
            f"Batch processing completed: {total_items} items, {len(results)} batches"
        )
        return results

    finally:
        # Cleanup performance tracking
        if perf_context:
            try:
                perf_context.__exit__(None, None, None)
            except Exception as e:
                logger.debug(f"Performance tracking cleanup failed: {e}")


# Convenience functions for common patterns
def batch_apply(
    data: Any, func: Callable[[Any], Any], batch_size: Optional[int] = None, **kwargs
) -> List[Any]:
    """Apply function to data in batches, returning list of results."""
    return batch_process(data, func, batch_size, **kwargs)


def batch_reduce(
    data: Any,
    func: Callable[[Any], Any],
    reduce_func: Callable[[List[Any]], Any],
    batch_size: Optional[int] = None,
    **kwargs,
) -> Any:
    """Apply function to data in batches and reduce results."""
    return batch_process(data, func, batch_size, combine_func=reduce_func, **kwargs)


def chunked(iterable: Any, chunk_size: int) -> Iterator[List[Any]]:
    """
    Simple chunking utility for iterables.

    Alternative to batch_generator for simple use cases without overlap
    or advanced features.

    Parameters
    ----------
    iterable : iterable
        Input iterable to chunk.
    chunk_size : int
        Size of each chunk.

    Yields
    ------
    list
        Chunks of the iterable.

    Examples
    --------
    >>> data = range(100)
    >>> for chunk in chunked(data, 10):
    ...     print(f"Processing {len(chunk)} items")
    """
    iterator = iter(iterable)
    while True:
        chunk = list(iterator.__next__() for _ in range(chunk_size))
        if not chunk:
            break
        yield chunk
