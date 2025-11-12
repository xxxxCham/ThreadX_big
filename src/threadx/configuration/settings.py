"""Settings dataclass for ThreadX configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Centralised application configuration."""

    # Paths
    DATA_ROOT: str = "./data"
    RAW_JSON: str = "{data_root}/raw/json"
    PROCESSED: str = "{data_root}/processed"
    INDICATORS: str = "{data_root}/indicators"
    RUNS: str = "{data_root}/runs"
    LOGS: str = "./logs"
    CACHE: str = "./cache"
    CONFIG: str = "./config"

    # GPU
    GPU_DEVICES: list[str] = field(default_factory=lambda: ["5090", "2060"])
    LOAD_BALANCE: dict[str, float] = field(
        default_factory=lambda: {"5090": 0.75, "2060": 0.25}
    )
    MEMORY_THRESHOLD: float = 0.8
    AUTO_FALLBACK: bool = True
    ENABLE_GPU: bool = True

    # Performance
    TARGET_TASKS_PER_MIN: int = 2500
    VECTORIZATION_BATCH_SIZE: int = 10000
    CACHE_TTL_SEC: int = 3600
    MAX_WORKERS: int = 4
    MEMORY_LIMIT_MB: int = 8192

    # Trading
    SUPPORTED_TF: tuple[str, ...] = (
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    )
    DEFAULT_TIMEFRAME: str = "1h"
    BASE_CURRENCY: str = "USDC"
    FEE_RATE: float = 0.001
    SLIPPAGE_RATE: float = 0.0005

    # Backtesting
    INITIAL_CAPITAL: float = 10000.0
    MAX_POSITIONS: int = 10
    POSITION_SIZE: float = 0.1
    STOP_LOSS: float = 0.02
    TAKE_PROFIT: float = 0.04

    # Logging
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 100
    MAX_FILES: int = 10
    LOG_ROTATE: bool = True
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Security
    READ_ONLY_DATA: bool = True
    VALIDATE_PATHS: bool = True
    ALLOW_ABSOLUTE_PATHS: bool = False
    SECURITY_MAX_FILE_SIZE_MB: int = 1000

    # Monte Carlo
    DEFAULT_SIMULATIONS: int = 10000
    MAX_SIMULATIONS: int = 1000000
    DEFAULT_STEPS: int = 252
    MC_SEED: int = 42
    CONFIDENCE_LEVELS: list[float] = field(default_factory=lambda: [0.95, 0.99])

    # Cache
    CACHE_ENABLE: bool = True
    CACHE_MAX_SIZE_MB: int = 2048
    CACHE_TTL_SECONDS: int = 3600
    CACHE_COMPRESSION: bool = True
    CACHE_STRATEGY: str = "LRU"


DEFAULT_SETTINGS = Settings()

__all__ = ["Settings", "DEFAULT_SETTINGS"]

# Export global settings instance
# ----- Lazy singleton accessor (autocache) -----


def get_settings(_cache={"value": None}):
    """
    Retourne une instance Settings unique (mise en cache au niveau du process).
    Évite de ré-instancier Settings à chaque import.
    """
    if _cache["value"] is None:
        _cache["value"] = Settings()
    return _cache["value"]


S = get_settings()




