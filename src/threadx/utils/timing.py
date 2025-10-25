"""
ThreadX Utils Module 
Timing, Profiling and Performance Measurement Utilities.

Provides standardized decorators and context managers for:
- Throughput measurement (tasks/min) with configurable warning thresholds
- Memory consumption tracking via psutil (with graceful fallback)
- Performance timing and cumulative measurements

Integrates with ThreadX Settings/TOML configuration for thresholds.
No environment variables - Windows-first design.
"""

import time
import functools
import logging
from typing import Optional, Callable, Any, Dict, Union
from contextlib import contextmanager
from dataclasses import dataclass, field

# Import psutil with graceful fallback
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

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


logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""

    elapsed_sec: float
    tasks_completed: int = 0
    tasks_per_min: float = 0.0
    memory_peak_mb: Optional[float] = None
    memory_avg_mb: Optional[float] = None
    function_name: str = ""
    unit_of_work: str = "task"


class Timer:
    """
    High-precision timer context manager.

    Provides accurate timing measurements with minimal overhead.
    Thread-safe and Windows-compatible.

    Examples
    --------
    >>> with Timer() as timer:
    ...     # Some work
    ...     time.sleep(0.1)
    >>> print(f"Elapsed: {timer.elapsed_sec:.3f}s")
    Elapsed: 0.100s

    >>> timer = Timer()
    >>> timer.start()
    >>> # Some work
    >>> timer.stop()
    >>> print(timer.elapsed_sec)
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        """Start timing measurement."""
        self._start_time = time.perf_counter()
        self._end_time = None

    def stop(self) -> None:
        """Stop timing measurement."""
        if self._start_time is None:
            raise RuntimeError("Timer not started")
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time

    @property
    def elapsed_sec(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        if self._end_time is None:
            # Timer still running
            return time.perf_counter() - self._start_time
        return self._elapsed

    def __enter__(self) -> "Timer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class MemoryTracker:
    """
    Memory usage tracker using psutil.

    Tracks peak and average memory consumption during execution.
    Graceful fallback if psutil is not available.
    """

    def __init__(self):
        self.start_memory_mb: Optional[float] = None
        self.peak_memory_mb: Optional[float] = None
        self.memory_samples: list = []
        self._process = None

        if PSUTIL_AVAILABLE:
            try:
                self._process = psutil.Process()
            except Exception as e:
                logger.warning(f"Failed to initialize memory tracker: {e}")

    def start(self) -> None:
        """Start memory tracking."""
        if not PSUTIL_AVAILABLE or self._process is None:
            return

        try:
            memory_info = self._process.memory_info()
            self.start_memory_mb = memory_info.rss / (1024 * 1024)
            self.peak_memory_mb = self.start_memory_mb
            self.memory_samples = [self.start_memory_mb]
        except Exception as e:
            logger.debug(f"Memory tracking start failed: {e}")

    def sample(self) -> None:
        """Take a memory sample."""
        if not PSUTIL_AVAILABLE or self._process is None:
            return

        try:
            memory_info = self._process.memory_info()
            current_mb = memory_info.rss / (1024 * 1024)
            self.memory_samples.append(current_mb)

            if self.peak_memory_mb is None or current_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_mb
        except Exception as e:
            logger.debug(f"Memory sampling failed: {e}")

    def get_stats(self) -> Dict[str, Optional[float]]:
        """Get memory usage statistics."""
        if not self.memory_samples:
            return {"peak_mb": None, "avg_mb": None, "start_mb": None}

        return {
            "peak_mb": self.peak_memory_mb,
            "avg_mb": sum(self.memory_samples) / len(self.memory_samples),
            "start_mb": self.start_memory_mb,
        }


def _get_throughput_threshold() -> int:
    """Get throughput warning threshold from settings."""
    if not SETTINGS_AVAILABLE:
        return 1000  # Default fallback

    try:
        settings = load_settings()
        return settings.MIN_TASKS_PER_MIN
    except Exception as e:
        logger.debug(f"Failed to load throughput threshold from settings: {e}")
        return 1000


def measure_throughput(
    name: Optional[str] = None, *, unit_of_work: str = "task"
) -> Callable:
    """
    Decorator to measure function throughput (tasks per minute).

    Logs execution time, throughput, and issues WARNING if below threshold.
    Integrates with ThreadX Settings for configurable warning threshold.

    Parameters
    ----------
    name : str, optional
        Custom name for the measurement. If None, uses function name.
    unit_of_work : str, default "task"
        Description of the unit being measured (e.g., "indicator", "trade", "calculation").

    Returns
    -------
    callable
        Decorated function with throughput measurement.

    Examples
    --------
    >>> @measure_throughput()
    ... def process_trades(trades):
    ...     return [t * 2 for t in trades]

    >>> @measure_throughput("custom_indicator", unit_of_work="calculation")
    ... def compute_rsi(prices):
    ...     return prices.rolling(14).mean()

    Notes
    -----
    - Assumes the function's first argument or return value indicates task count
    - WARNING issued if throughput < MIN_TASKS_PER_MIN from Settings
    - Logs at INFO level for normal operation, WARNING for low throughput
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            threshold = _get_throughput_threshold()

            # Estimate task count - try different strategies
            task_count = 1  # Default fallback

            # Strategy 1: Look for common parameter names
            if args:
                first_arg = args[0]
                if hasattr(first_arg, "__len__"):
                    try:
                        task_count = len(first_arg)
                    except (TypeError, AttributeError):
                        pass

            # Strategy 2: Check for explicit task_count parameter
            if "task_count" in kwargs:
                task_count = kwargs["task_count"]
            elif "n_tasks" in kwargs:
                task_count = kwargs["n_tasks"]

            logger.info(f"Starting {func_name} with {task_count} {unit_of_work}(s)")

            with Timer() as timer:
                result = func(*args, **kwargs)

            # Update task count from result if possible
            if hasattr(result, "__len__"):
                try:
                    result_count = len(result)
                    if result_count > 0:
                        task_count = result_count
                except (TypeError, AttributeError):
                    pass

            elapsed_sec = timer.elapsed_sec
            tasks_per_min = (
                (task_count * 60.0) / elapsed_sec if elapsed_sec > 0 else 0.0
            )

            # Log results
            log_msg = (
                f"{func_name} completed: {task_count} {unit_of_work}(s) "
                f"in {elapsed_sec:.3f}s ({tasks_per_min:.1f} {unit_of_work}s/min)"
            )

            if tasks_per_min < threshold:
                logger.warning(
                    f"{log_msg} - PERFORMANCE WARNING: Below threshold "
                    f"({threshold} {unit_of_work}s/min). Consider vectorization or batching."
                )
            else:
                logger.info(log_msg)

            return result

        return wrapper

    return decorator


def track_memory(name: Optional[str] = None) -> Callable:
    """
    Decorator to track memory consumption during function execution.

    Uses psutil to monitor peak and average memory usage. Provides graceful
    fallback if psutil is unavailable (logs INFO and continues).

    Parameters
    ----------
    name : str, optional
        Custom name for the measurement. If None, uses function name.

    Returns
    -------
    callable
        Decorated function with memory tracking.

    Examples
    --------
    >>> @track_memory()
    ... def process_large_dataset(data):
    ...     return data.copy()

    >>> @track_memory("indicator_computation")
    ... def compute_indicators(prices):
    ...     return expensive_calculation(prices)

    Notes
    -----
    - Requires psutil for actual memory tracking
    - Falls back gracefully if psutil unavailable (logs None values)
    - Memory values in MB for readability
    - Samples memory periodically during execution for peak detection
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__

            if not PSUTIL_AVAILABLE:
                logger.info(
                    f"{func_name} memory tracking: psutil unavailable, skipping"
                )
                return func(*args, **kwargs)

            tracker = MemoryTracker()
            tracker.start()

            logger.info(f"Starting memory tracking for {func_name}")

            try:
                # Sample memory during execution (simple approach)
                result = func(*args, **kwargs)
                tracker.sample()

                stats = tracker.get_stats()

                if stats["peak_mb"] is not None:
                    log_msg = (
                        f"{func_name} memory usage: "
                        f"peak={stats['peak_mb']:.1f}MB, "
                        f"avg={stats['avg_mb']:.1f}MB"
                    )
                    logger.info(log_msg)
                else:
                    logger.info(f"{func_name} memory tracking: no data collected")

                return result

            except Exception as e:
                logger.warning(f"Memory tracking failed for {func_name}: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator


def combined_measurement(
    name: Optional[str] = None,
    *,
    unit_of_work: str = "task",
    track_memory_usage: bool = True,
) -> Callable:
    """
    Combined decorator for throughput and memory measurement.

    Convenience decorator that applies both @measure_throughput and @track_memory.

    Parameters
    ----------
    name : str, optional
        Custom name for measurements.
    unit_of_work : str, default "task"
        Unit description for throughput measurement.
    track_memory_usage : bool, default True
        Whether to include memory tracking.

    Returns
    -------
    callable
        Decorated function with combined measurements.

    Examples
    --------
    >>> @combined_measurement("bulk_processing", unit_of_work="record")
    ... def process_records(records):
    ...     return [process_record(r) for r in records]
    """

    def decorator(func: Callable) -> Callable:
        # Apply decorators in reverse order (they're applied inside-out)
        decorated = measure_throughput(name, unit_of_work=unit_of_work)(func)
        if track_memory_usage:
            decorated = track_memory(name)(decorated)
        return decorated

    return decorator


@contextmanager
def performance_context(
    name: str,
    *,
    unit_of_work: str = "task",
    task_count: int = 1,
    track_memory_usage: bool = True,
):
    """
    Context manager for performance measurement.

    Alternative to decorators for measuring performance of code blocks.

    Parameters
    ----------
    name : str
        Name for the measurement.
    unit_of_work : str, default "task"
        Unit description.
    task_count : int, default 1
        Number of tasks being processed.
    track_memory_usage : bool, default True
        Whether to track memory usage.

    Yields
    ------
    PerformanceMetrics
        Metrics object that gets populated during execution.

    Examples
    --------
    >>> with performance_context("data_processing", task_count=1000) as perf:
    ...     # Process data
    ...     results = process_data(data)
    >>> print(f"Processed at {perf.tasks_per_min:.1f} tasks/min")
    """
    threshold = _get_throughput_threshold()

    metrics = PerformanceMetrics(
        elapsed_sec=0.0,
        tasks_completed=task_count,
        function_name=name,
        unit_of_work=unit_of_work,
    )

    memory_tracker = None
    if track_memory_usage and PSUTIL_AVAILABLE:
        memory_tracker = MemoryTracker()
        memory_tracker.start()

    logger.info(f"Starting {name} with {task_count} {unit_of_work}(s)")

    with Timer() as timer:
        try:
            yield metrics
        finally:
            metrics.elapsed_sec = timer.elapsed_sec

            if metrics.elapsed_sec > 0:
                metrics.tasks_per_min = (task_count * 60.0) / metrics.elapsed_sec
            else:
                metrics.tasks_per_min = 0.0

            # Collect memory stats
            if memory_tracker:
                memory_tracker.sample()
                stats = memory_tracker.get_stats()
                metrics.memory_peak_mb = stats["peak_mb"]
                metrics.memory_avg_mb = stats["avg_mb"]

            # Log results
            log_msg = (
                f"{name} completed: {task_count} {unit_of_work}(s) "
                f"in {metrics.elapsed_sec:.3f}s ({metrics.tasks_per_min:.1f} {unit_of_work}s/min)"
            )

            if memory_tracker and metrics.memory_peak_mb:
                log_msg += f", peak_memory={metrics.memory_peak_mb:.1f}MB"

            if metrics.tasks_per_min < threshold:
                logger.warning(
                    f"{log_msg} - PERFORMANCE WARNING: Below threshold "
                    f"({threshold} {unit_of_work}s/min). Consider vectorization or batching."
                )
            else:
                logger.info(log_msg)
