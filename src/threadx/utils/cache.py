"""
ThreadX Utils Module - Phase 9
Caching Infrastructure with LRU and TTL Support.

Provides high-performance, thread-safe caching with:
- LRU (Least Recently Used) eviction policy
- TTL (Time To Live) expiration
- Combined LRU+TTL caching via decorator
- Comprehensive statistics and observability
- Stable key generation for reproducible caching
- Thread-safe operations with fine-grained locking

Designed for integration with ThreadX Indicators Bank without API changes.
Windows-first, no environment variables, TOML/Settings configuration.
"""

import time
import threading
import hashlib
import pickle
from typing import (
    Any,
    Optional,
    Callable,
    Dict,
    List,
    Tuple,
    Union,
    Generic,
    TypeVar,
    Hashable,
    NamedTuple,
)
from dataclasses import dataclass, field
from functools import wraps
from collections import OrderedDict
import logging

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

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class CacheEvent(NamedTuple):
    """Cache event for observability callbacks."""

    event_type: str  # 'hit', 'miss', 'eviction', 'expiration', 'clear'
    key: str
    namespace: Optional[str] = None
    value_size: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """Cache statistics container."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    current_size: int = 0
    capacity: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hits + self.misses

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU (Least Recently Used) cache implementation.

    Features:
    - O(1) get/set operations using OrderedDict
    - Thread-safe with fine-grained locking
    - Comprehensive statistics tracking
    - Configurable capacity with automatic eviction
    - Optional observability callbacks

    Parameters
    ----------
    capacity : int
        Maximum number of items to store.
    on_cache_event : callable, optional
        Callback function for cache events.

    Examples
    --------
    >>> cache = LRUCache[str, int](capacity=100)
    >>> cache.set("key1", 42)
    >>> value = cache.get("key1")  # Returns 42
    >>> cache.contains("key1")    # Returns True
    >>> stats = cache.stats()
    >>> print(f"Hit rate: {stats.hit_rate:.1f}%")
    """

    def __init__(
        self,
        capacity: int,
        on_cache_event: Optional[Callable[[CacheEvent], None]] = None,
    ):
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")

        self._capacity = capacity
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._stats = CacheStats(capacity=capacity)
        self._lock = threading.RLock()
        self._on_cache_event = on_cache_event

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value by key, updating LRU order.

        Parameters
        ----------
        key : Hashable
            Cache key.
        default : Any, optional
            Default value if key not found.

        Returns
        -------
        Any
            Cached value or default.
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache[key]
                self._cache.move_to_end(key)
                self._stats.hits += 1

                if self._on_cache_event:
                    self._on_cache_event(CacheEvent("hit", str(key)))

                return value
            else:
                self._stats.misses += 1

                if self._on_cache_event:
                    self._on_cache_event(CacheEvent("miss", str(key)))

                return default

    def set(self, key: K, value: V) -> None:
        """
        Set key-value pair, evicting LRU item if needed.

        Parameters
        ----------
        key : Hashable
            Cache key.
        value : Any
            Value to cache.
        """
        with self._lock:
            if key in self._cache:
                # Update existing key
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new key
                self._cache[key] = value
                self._stats.current_size = len(self._cache)

                # Evict LRU item if over capacity
                if len(self._cache) > self._capacity:
                    evicted_key, evicted_value = self._cache.popitem(last=False)
                    self._stats.evictions += 1
                    self._stats.current_size = len(self._cache)

                    if self._on_cache_event:
                        self._on_cache_event(CacheEvent("eviction", str(evicted_key)))

    def contains(self, key: K) -> bool:
        """Check if key exists in cache without updating LRU order."""
        with self._lock:
            return key in self._cache

    def remove(self, key: K) -> bool:
        """Remove key from cache. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.current_size = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._stats.current_size = 0

            if self._on_cache_event:
                self._on_cache_event(CacheEvent("clear", "all"))

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            # Return a copy to avoid concurrent modification
            stats_copy = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                current_size=len(self._cache),
                capacity=self._capacity,
            )
            return stats_copy

    @property
    def size(self) -> int:
        """Current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def capacity(self) -> int:
        """Maximum cache capacity."""
        return self._capacity


class TTLCache(Generic[K, V]):
    """
    Thread-safe TTL (Time To Live) cache implementation.

    Features:
    - Automatic expiration based on TTL
    - Lazy expiration on access + explicit purge methods
    - Thread-safe operations
    - Comprehensive statistics
    - Optional observability callbacks

    Parameters
    ----------
    ttl_seconds : float
        Time to live in seconds.
    on_cache_event : callable, optional
        Callback function for cache events.

    Examples
    --------
    >>> cache = TTLCache[str, int](ttl_seconds=300)  # 5 minute TTL
    >>> cache.set("key1", 42)
    >>> # ... after 6 minutes ...
    >>> cache.get("key1")  # Returns None (expired)
    >>> cache.purge_expired()  # Explicit cleanup
    """

    def __init__(
        self,
        ttl_seconds: float,
        on_cache_event: Optional[Callable[[CacheEvent], None]] = None,
    ):
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")

        self._ttl_seconds = ttl_seconds
        self._cache: Dict[K, Tuple[V, float]] = {}  # value, expiry_time
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._on_cache_event = on_cache_event

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value by key, checking expiration.

        Parameters
        ----------
        key : Hashable
            Cache key.
        default : Any, optional
            Default value if key not found or expired.

        Returns
        -------
        Any
            Cached value or default.
        """
        current_time = time.time()

        with self._lock:
            if key in self._cache:
                value, expiry_time = self._cache[key]

                if current_time <= expiry_time:
                    # Valid, not expired
                    self._stats.hits += 1

                    if self._on_cache_event:
                        self._on_cache_event(CacheEvent("hit", str(key)))

                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
                    self._stats.expirations += 1
                    self._stats.misses += 1

                    if self._on_cache_event:
                        self._on_cache_event(CacheEvent("expiration", str(key)))

                    return default
            else:
                self._stats.misses += 1

                if self._on_cache_event:
                    self._on_cache_event(CacheEvent("miss", str(key)))

                return default

    def set(self, key: K, value: V) -> None:
        """
        Set key-value pair with TTL expiration.

        Parameters
        ----------
        key : Hashable
            Cache key.
        value : Any
            Value to cache.
        """
        expiry_time = time.time() + self._ttl_seconds

        with self._lock:
            self._cache[key] = (value, expiry_time)

    def contains(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        current_time = time.time()

        with self._lock:
            if key in self._cache:
                _, expiry_time = self._cache[key]
                if current_time <= expiry_time:
                    return True
                else:
                    # Expired, clean up
                    del self._cache[key]
                    self._stats.expirations += 1
                    return False
            return False

    def remove(self, key: K) -> bool:
        """Remove key from cache. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def purge_expired(self) -> int:
        """
        Remove all expired entries.

        Returns
        -------
        int
            Number of entries removed.
        """
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, (value, expiry_time) in self._cache.items():
                if current_time > expiry_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1

                if self._on_cache_event:
                    self._on_cache_event(CacheEvent("expiration", str(key)))

        return len(expired_keys)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()

            if self._on_cache_event:
                self._on_cache_event(CacheEvent("clear", "all"))

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            stats_copy = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                current_size=len(self._cache),
                capacity=0,  # TTL cache has no fixed capacity
            )
            return stats_copy

    @property
    def size(self) -> int:
        """Current cache size (including potentially expired items)."""
        with self._lock:
            return len(self._cache)

    @property
    def ttl_seconds(self) -> float:
        """Time to live in seconds."""
        return self._ttl_seconds


def generate_stable_key(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    namespace: Optional[str] = None,
) -> str:
    """
    Generate stable, deterministic cache key.

    Creates a consistent hash from function signature and arguments,
    suitable for cross-session caching with stable keys.

    Parameters
    ----------
    func : callable
        Function being cached.
    args : tuple
        Function positional arguments.
    kwargs : dict
        Function keyword arguments.
    namespace : str, optional
        Optional namespace prefix.

    Returns
    -------
    str
        Stable cache key.

    Notes
    -----
    - Handles nested data structures
    - Warns about float arguments (potential precision issues)
    - Uses pickle + SHA256 for deterministic hashing
    - Includes function name and module for uniqueness
    """
    try:
        # Start with function signature
        func_sig = f"{func.__module__}.{func.__qualname__}"

        # Check for potentially unstable float arguments
        def check_floats(obj, path=""):
            if isinstance(obj, float):
                if abs(obj) > 1e-10:  # Not close to zero
                    logger.debug(
                        f"Float in cache key at {path}: {obj} - ensure precision consistency"
                    )
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    check_floats(item, f"{path}[{i}]")
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    check_floats(value, f"{path}.{key}")

        check_floats(args, "args")
        check_floats(kwargs, "kwargs")

        # Create stable representation
        key_data = {
            "func": func_sig,
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else None,
            "namespace": namespace,
        }

        # Serialize and hash
        serialized = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        key_hash = hashlib.sha256(serialized).hexdigest()

        # Create readable key with hash
        readable_part = f"{func.__name__}"
        if namespace:
            readable_part = f"{namespace}.{readable_part}"

        return f"{readable_part}_{key_hash[:16]}"

    except Exception as e:
        logger.warning(f"Failed to generate stable cache key for {func.__name__}: {e}")
        # Fallback to simpler key
        return f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items())) if kwargs else None))}"


def cached(
    ttl: Optional[int] = None,
    lru: Optional[int] = None,
    key_fn: Optional[Callable] = None,
    namespace: Optional[str] = None,
    stats_logging: bool = True,
) -> Callable:
    """
    Comprehensive caching decorator with LRU and/or TTL support.

    Provides flexible caching with configurable eviction policies.
    Can use LRU-only, TTL-only, or combined LRU+TTL caching.

    Parameters
    ----------
    ttl : int, optional
        Time to live in seconds. If None, no TTL expiration.
    lru : int, optional
        LRU cache capacity. If None, no LRU eviction.
    key_fn : callable, optional
        Custom key generation function. If None, uses generate_stable_key.
    namespace : str, optional
        Cache namespace for key prefixing.
    stats_logging : bool, default True
        Whether to log cache statistics periodically.

    Returns
    -------
    callable
        Decorated function with caching.

    Examples
    --------
    Basic TTL caching:
    >>> @cached(ttl=300)  # 5 minutes
    ... def compute_indicator(symbol, period):
    ...     return expensive_calculation(symbol, period)

    Combined LRU + TTL:
    >>> @cached(ttl=600, lru=1000)  # 10min TTL, 1000 item LRU
    ... def fetch_market_data(symbol, timeframe):
    ...     return api_call(symbol, timeframe)

    Custom key function:
    >>> def custom_key(func, args, kwargs, namespace):
    ...     return f"custom_{args[0]}_{kwargs.get('period', 'default')}"
    >>> @cached(ttl=300, key_fn=custom_key)
    ... def custom_computation(data, period=10):
    ...     return process(data, period)

    Integration with ThreadX Indicators Bank:
    >>> # Wrap existing indicator functions without changing signatures
    >>> @cached(ttl=3600, lru=500, namespace="indicators")
    ... def compute_bollinger_bands(prices, period, std_dev):
    ...     return calculate_bands(prices, period, std_dev)

    Notes
    -----
    - Thread-safe caching with comprehensive statistics
    - Stable key generation prevents cache invalidation issues
    - Integrates with ThreadX logging system
    - Performance optimized for hot-path indicator calculations
    - No changes needed to existing function signatures
    """

    # Cache instance storage (per decorated function)
    _cache_registry: Dict[str, Union[LRUCache, TTLCache, Tuple[LRUCache, TTLCache]]] = (
        {}
    )
    _stats_last_logged: Dict[str, float] = {}

    def _get_stats_interval() -> float:
        """Get stats logging interval from settings."""
        if SETTINGS_AVAILABLE:
            try:
                settings = load_settings()
                return settings.CACHE_STATS_LOG_INTERVAL_SEC
            except Exception:
                pass
        return 300.0  # 5 minute default

    def _log_cache_stats(func_name: str, cache_obj: Any) -> None:
        """Log cache statistics if interval has passed."""
        if not stats_logging:
            return

        current_time = time.time()
        last_logged = _stats_last_logged.get(func_name, 0)
        interval = _get_stats_interval()

        if current_time - last_logged >= interval:
            try:
                if hasattr(cache_obj, "stats"):
                    stats = cache_obj.stats()
                    logger.info(
                        f"Cache stats for {func_name}: "
                        f"hit_rate={stats.hit_rate:.1f}%, "
                        f"hits={stats.hits}, misses={stats.misses}, "
                        f"size={stats.current_size}"
                    )
                elif isinstance(cache_obj, tuple):
                    # Combined LRU+TTL
                    lru_cache, ttl_cache = cache_obj
                    lru_stats = lru_cache.stats()
                    ttl_stats = ttl_cache.stats()
                    logger.info(
                        f"Combined cache stats for {func_name}: "
                        f"LRU hit_rate={lru_stats.hit_rate:.1f}%, "
                        f"TTL hit_rate={ttl_stats.hit_rate:.1f}%, "
                        f"LRU size={lru_stats.current_size}, TTL size={ttl_stats.current_size}"
                    )

                _stats_last_logged[func_name] = current_time

            except Exception as e:
                logger.debug(f"Failed to log cache stats for {func_name}: {e}")

    def decorator(func: Callable) -> Callable:
        func_key = f"{func.__module__}.{func.__qualname__}"

        # Initialize cache for this function
        if func_key not in _cache_registry:
            if lru and ttl:
                # Combined LRU + TTL
                lru_cache = LRUCache(capacity=lru)
                ttl_cache = TTLCache(ttl_seconds=ttl)
                _cache_registry[func_key] = (lru_cache, ttl_cache)
                logger.info(
                    f"Initialized combined LRU+TTL cache for {func.__name__} (lru={lru}, ttl={ttl}s)"
                )
            elif lru:
                # LRU only
                _cache_registry[func_key] = LRUCache(capacity=lru)
                logger.info(
                    f"Initialized LRU cache for {func.__name__} (capacity={lru})"
                )
            elif ttl:
                # TTL only
                _cache_registry[func_key] = TTLCache(ttl_seconds=ttl)
                logger.info(f"Initialized TTL cache for {func.__name__} (ttl={ttl}s)")
            else:
                raise ValueError("Must specify either 'lru' capacity or 'ttl' seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                cache_key = key_fn(func, args, kwargs, namespace)
            else:
                cache_key = generate_stable_key(func, args, kwargs, namespace)

            cache_obj = _cache_registry[func_key]

            # Try to get from cache
            cached_result = None

            if isinstance(cache_obj, tuple):
                # Combined LRU + TTL - check both
                lru_cache, ttl_cache = cache_obj

                # Try LRU first (faster access pattern)
                cached_result = lru_cache.get(cache_key)
                if cached_result is None:
                    # Try TTL cache
                    cached_result = ttl_cache.get(cache_key)
                    if cached_result is not None:
                        # Found in TTL, promote to LRU
                        lru_cache.set(cache_key, cached_result)

            else:
                # Single cache (LRU or TTL)
                cached_result = cache_obj.get(cache_key)

            if cached_result is not None:
                # Cache hit - log stats and return
                _log_cache_stats(func.__name__, cache_obj)
                return cached_result

            # Cache miss - compute result
            result = func(*args, **kwargs)

            # Store in cache
            if isinstance(cache_obj, tuple):
                # Store in both caches
                lru_cache, ttl_cache = cache_obj
                lru_cache.set(cache_key, result)
                ttl_cache.set(cache_key, result)
            else:
                cache_obj.set(cache_key, result)

            # Log stats periodically
            _log_cache_stats(func.__name__, cache_obj)

            return result

        # Add cache management methods to wrapper
        def cache_stats():
            cache_obj = _cache_registry[func_key]
            if isinstance(cache_obj, tuple):
                lru_cache, ttl_cache = cache_obj
                return {"lru": lru_cache.stats(), "ttl": ttl_cache.stats()}
            else:
                return cache_obj.stats()

        def cache_clear():
            cache_obj = _cache_registry[func_key]
            if isinstance(cache_obj, tuple):
                lru_cache, ttl_cache = cache_obj
                lru_cache.clear()
                ttl_cache.clear()
            else:
                cache_obj.clear()

        def cache_info():
            cache_obj = _cache_registry[func_key]
            if isinstance(cache_obj, tuple):
                lru_cache, ttl_cache = cache_obj
                return {
                    "type": "combined_lru_ttl",
                    "lru_capacity": lru_cache.capacity,
                    "lru_size": lru_cache.size,
                    "ttl_seconds": ttl_cache.ttl_seconds,
                    "ttl_size": ttl_cache.size,
                }
            elif hasattr(cache_obj, "capacity"):
                return {
                    "type": "lru",
                    "capacity": cache_obj.capacity,
                    "size": cache_obj.size,
                }
            else:
                return {
                    "type": "ttl",
                    "ttl_seconds": cache_obj.ttl_seconds,
                    "size": cache_obj.size,
                }

        # Attach management methods
        wrapper.cache_stats = cache_stats
        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info

        return wrapper

    return decorator


# Convenience functions for common caching patterns
def lru_cache(capacity: int, namespace: Optional[str] = None) -> Callable:
    """LRU-only caching decorator."""
    return cached(lru=capacity, namespace=namespace)


def ttl_cache(ttl_seconds: int, namespace: Optional[str] = None) -> Callable:
    """TTL-only caching decorator."""
    return cached(ttl=ttl_seconds, namespace=namespace)


def indicators_cache(ttl_seconds: int = 3600, lru_capacity: int = 1000) -> Callable:
    """
    Specialized caching decorator for ThreadX indicators.

    Optimized for indicator computation patterns with sensible defaults.
    """
    return cached(ttl=ttl_seconds, lru=lru_capacity, namespace="indicators")
