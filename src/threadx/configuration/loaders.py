"""TOML configuration loader for ThreadX."""
# type: ignore  # Trop d'erreurs de type, analyse désactivée

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - fallback for Python <3.11
    import tomllib
except ImportError:  # pragma: no cover - fallback path
    import tomli as tomllib

from .errors import ConfigurationError, PathValidationError
from .settings import DEFAULT_SETTINGS, Settings

logger = logging.getLogger(__name__)


def load_config_dict(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigurationError(
            str(config_path), "Configuration file not found"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            str(config_path), "Invalid TOML syntax", details=str(exc)
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigurationError(
            str(config_path), "Unexpected error while loading config", details=str(exc)
        ) from exc


class TOMLConfigLoader:
    """Load and validate ThreadX configuration files."""

    DEFAULT_CONFIG_NAME = "paths.toml"

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = self._resolve_config_path(config_path)
        self.config_data = load_config_dict(self.config_path)
        self._validated_paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------
    def _resolve_config_path(self, provided: Optional[Union[str, Path]]) -> Path:
        if provided:
            candidate = Path(provided)
            if candidate.exists():
                return candidate
            raise ConfigurationError(str(candidate), "Configuration file not found")

        search_paths = [
            Path.cwd() / self.DEFAULT_CONFIG_NAME,
            Path.cwd().parent / self.DEFAULT_CONFIG_NAME,
            Path(__file__).resolve().parents[2] / self.DEFAULT_CONFIG_NAME,
        ]

        for candidate in search_paths:
            if candidate.exists():
                return candidate

        searched = "\n".join(str(path) for path in search_paths)
        raise ConfigurationError(None, "Configuration file not found", details=searched)

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def get_section(self, name: str) -> Dict[str, Any]:
        return dict(self.config_data.get(name, {}))

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        return self.get_section(section).get(key, default)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_config(self) -> List[str]:
        errors: List[str] = []
        self._validated_paths.clear()
        required_sections = ["paths", "gpu", "performance", "trading"]
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Missing required configuration section: {section}")

        errors.extend(self._validate_paths(check_only=True))
        errors.extend(self._validate_gpu_config(check_only=True))
        errors.extend(self._validate_performance_config(check_only=True))
        return errors

    def _validate_paths(self, check_only: bool = False) -> List[str]:
        errors: List[str] = []
        paths_section = self.get_section("paths")
        security = self.get_section("security")
        allow_abs = bool(security.get("allow_absolute_paths", False))

        data_root = paths_section.get("data_root", "./data")
        if not isinstance(data_root, str):
            errors.append("paths.data_root must be a string")
            data_root = "./data"

        resolved_paths: Dict[str, str] = {"data_root": data_root}

        for key, value in paths_section.items():
            if not isinstance(value, str):
                continue
            if not allow_abs and Path(value).is_absolute():
                errors.append(f"Absolute path not allowed for {key}: {value}")
                continue
            resolved_paths[key] = value.format(data_root=data_root)

        if not check_only:
            self._validated_paths.update(resolved_paths)

        return errors

    def _validate_gpu_config(self, check_only: bool = False) -> List[str]:
        errors: List[str] = []
        gpu_section = self.get_section("gpu")

        load_balance = gpu_section.get("load_balance", {})
        if isinstance(load_balance, dict) and load_balance:
            total = sum(load_balance.values())
            if not (0.99 <= total <= 1.01):
                errors.append("GPU load balance ratios must sum to 1.0")
        elif load_balance not in ({}, None):
            errors.append("gpu.load_balance must be a mapping of ratios")

        threshold = gpu_section.get("memory_threshold", 0.8)
        if not isinstance(threshold, (int, float)) or not (
            0.1 <= float(threshold) <= 1.0
        ):
            errors.append("gpu.memory_threshold must be between 0.1 and 1.0")

        return errors

    def _validate_performance_config(self, check_only: bool = False) -> List[str]:
        errors: List[str] = []
        perf_section = self.get_section("performance")
        for key in (
            "target_tasks_per_min",
            "vectorization_batch_size",
            "cache_ttl_sec",
            "max_workers",
        ):
            value = perf_section.get(key)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"performance.{key} must be a positive number")
        return errors

    # ------------------------------------------------------------------
    # Settings construction
    # ------------------------------------------------------------------
    def create_settings(self, **overrides: Any) -> Settings:
        errors = self.validate_config()
        if errors:
            raise ConfigurationError(
                str(self.config_path),
                "Invalid configuration",
                details="\n".join(errors),
            )

        self._validated_paths.clear()
        path_errors = self._validate_paths(check_only=False)
        if path_errors:
            raise PathValidationError("; ".join(path_errors))

        paths = self._validated_paths
        gpu = self.get_section("gpu")
        performance = self.get_section("performance")
        trading = self.get_section("trading")
        backtesting = self.get_section("backtesting")
        logging_section = self.get_section("logging")
        security = self.get_section("security")
        monte_carlo = self.get_section("monte_carlo")
        cache = self.get_section("cache")

        defaults = DEFAULT_SETTINGS
        return Settings(
            DATA_ROOT=overrides.get(
                "data_root", paths.get("data_root", defaults.DATA_ROOT)
            ),
            RAW_JSON=overrides.get(
                "raw_json", paths.get("raw_json", defaults.RAW_JSON)
            ),
            PROCESSED=overrides.get(
                "processed", paths.get("processed", defaults.PROCESSED)
            ),
            INDICATORS=overrides.get(
                "indicators", paths.get("indicators", defaults.INDICATORS)
            ),
            RUNS=overrides.get("runs", paths.get("runs", defaults.RUNS)),
            LOGS=overrides.get("logs", paths.get("logs", defaults.LOGS)),
            CACHE=overrides.get("cache", paths.get("cache", defaults.CACHE)),
            CONFIG=overrides.get("config", paths.get("config", defaults.CONFIG)),
            GPU_DEVICES=gpu.get("devices", defaults.GPU_DEVICES),
            LOAD_BALANCE=gpu.get("load_balance", defaults.LOAD_BALANCE),
            MEMORY_THRESHOLD=gpu.get("memory_threshold", defaults.MEMORY_THRESHOLD),
            AUTO_FALLBACK=gpu.get("auto_fallback", defaults.AUTO_FALLBACK),
            ENABLE_GPU=overrides.get(
                "enable_gpu", gpu.get("enable_gpu", defaults.ENABLE_GPU)
            ),
            TARGET_TASKS_PER_MIN=performance.get(
                "target_tasks_per_min", defaults.TARGET_TASKS_PER_MIN
            ),
            VECTORIZATION_BATCH_SIZE=performance.get(
                "vectorization_batch_size", defaults.VECTORIZATION_BATCH_SIZE
            ),
            CACHE_TTL_SEC=performance.get("cache_ttl_sec", defaults.CACHE_TTL_SEC),
            MAX_WORKERS=performance.get("max_workers", defaults.MAX_WORKERS),
            MEMORY_LIMIT_MB=performance.get(
                "memory_limit_mb", defaults.MEMORY_LIMIT_MB
            ),
            SUPPORTED_TF=tuple(
                trading.get("supported_timeframes", defaults.SUPPORTED_TF)
            ),
            DEFAULT_TIMEFRAME=trading.get(
                "default_timeframe", defaults.DEFAULT_TIMEFRAME
            ),
            BASE_CURRENCY=trading.get("base_currency", defaults.BASE_CURRENCY),
            FEE_RATE=trading.get("fee_rate", defaults.FEE_RATE),
            SLIPPAGE_RATE=trading.get("slippage_rate", defaults.SLIPPAGE_RATE),
            INITIAL_CAPITAL=backtesting.get(
                "initial_capital", defaults.INITIAL_CAPITAL
            ),
            MAX_POSITIONS=backtesting.get("max_positions", defaults.MAX_POSITIONS),
            POSITION_SIZE=backtesting.get("position_size", defaults.POSITION_SIZE),
            STOP_LOSS=backtesting.get("stop_loss", defaults.STOP_LOSS),
            TAKE_PROFIT=backtesting.get("take_profit", defaults.TAKE_PROFIT),
            LOG_LEVEL=overrides.get(
                "log_level", logging_section.get("level", defaults.LOG_LEVEL)
            ),
            MAX_FILE_SIZE_MB=logging_section.get(
                "max_file_size_mb", defaults.MAX_FILE_SIZE_MB
            ),
            MAX_FILES=logging_section.get("max_files", defaults.MAX_FILES),
            LOG_ROTATE=logging_section.get("log_rotate", defaults.LOG_ROTATE),
            LOG_FORMAT=logging_section.get("format", defaults.LOG_FORMAT),
            READ_ONLY_DATA=security.get("read_only_data", defaults.READ_ONLY_DATA),
            VALIDATE_PATHS=security.get("validate_paths", defaults.VALIDATE_PATHS),
            ALLOW_ABSOLUTE_PATHS=security.get(
                "allow_absolute_paths", defaults.ALLOW_ABSOLUTE_PATHS
            ),
            SECURITY_MAX_FILE_SIZE_MB=security.get(
                "max_file_size_mb", defaults.SECURITY_MAX_FILE_SIZE_MB
            ),
            DEFAULT_SIMULATIONS=monte_carlo.get(
                "default_simulations", defaults.DEFAULT_SIMULATIONS
            ),
            MAX_SIMULATIONS=monte_carlo.get(
                "max_simulations", defaults.MAX_SIMULATIONS
            ),
            DEFAULT_STEPS=monte_carlo.get("default_steps", defaults.DEFAULT_STEPS),
            MC_SEED=monte_carlo.get("seed", defaults.MC_SEED),
            CONFIDENCE_LEVELS=list(
                monte_carlo.get("confidence_levels", defaults.CONFIDENCE_LEVELS)
            ),
            CACHE_ENABLE=cache.get("enable", defaults.CACHE_ENABLE),
            CACHE_MAX_SIZE_MB=cache.get("max_size_mb", defaults.CACHE_MAX_SIZE_MB),
            CACHE_TTL_SECONDS=cache.get("ttl_seconds", defaults.CACHE_TTL_SECONDS),
            CACHE_COMPRESSION=cache.get("compression", defaults.CACHE_COMPRESSION),
            CACHE_STRATEGY=cache.get("strategy", defaults.CACHE_STRATEGY),
        )

    def load_config(self, cli_overrides: Optional[Dict[str, Any]] = None) -> Settings:
        overrides = cli_overrides or {}
        return self.create_settings(**overrides)

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------
    @staticmethod
    def create_cli_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="ThreadX configuration loader")
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--data-root", dest="data_root", type=str)
        parser.add_argument("--log-level", dest="log_level", type=str)
        parser.add_argument("--enable-gpu", dest="enable_gpu", action="store_true")
        parser.add_argument("--disable-gpu", dest="disable_gpu", action="store_true")
        parser.add_argument("--print-config", action="store_true")
        return parser


_settings_cache: Optional[Settings] = None


def load_settings(
    config_path: Union[str, Path] = "paths.toml",
    cli_args: Optional[Sequence[str]] = None,
) -> Settings:
    parser = TOMLConfigLoader.create_cli_parser()
    args = (
        parser.parse_args(cli_args) if cli_args is not None else parser.parse_args([])
    )

    overrides: Dict[str, Any] = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.log_level:
        overrides["log_level"] = args.log_level
    if args.enable_gpu:
        overrides["enable_gpu"] = True
    if args.disable_gpu:
        overrides["enable_gpu"] = False

    resolved_config = args.config or config_path
    loader = TOMLConfigLoader(resolved_config)
    settings = loader.create_settings(**overrides)

    if args.print_config:
        print_config(settings)

    return settings


def get_settings(force_reload: bool = False) -> Settings:
    global _settings_cache
    if _settings_cache is None or force_reload:
        _settings_cache = load_settings()
    return _settings_cache


def print_config(settings: Optional[Settings] = None) -> None:
    cfg = settings or get_settings()
    print("ThreadX Configuration")
    print("=" * 50)
    print("\n[PATHS]")
    print(f"Data Root: {cfg.DATA_ROOT}")
    print(f"Indicators: {cfg.INDICATORS}")
    print(f"Runs: {cfg.RUNS}")
    print(f"Logs: {cfg.LOGS}")

    print("\n[GPU]")
    print(f"Devices: {cfg.GPU_DEVICES}")
    print(f"Load Balance: {cfg.LOAD_BALANCE}")
    print(f"GPU Enabled: {cfg.ENABLE_GPU}")

    print("\n[PERFORMANCE]")
    print(f"Target Tasks/Min: {cfg.TARGET_TASKS_PER_MIN}")
    print(f"Max Workers: {cfg.MAX_WORKERS}")

    print("\n[TRADING]")
    print(f"Supported TF: {cfg.SUPPORTED_TF}")
    print(f"Default TF: {cfg.DEFAULT_TIMEFRAME}")

    print("\n[SECURITY]")
    print(f"Read Only: {cfg.READ_ONLY_DATA}")
    print(f"Validate Paths: {cfg.VALIDATE_PATHS}")


__all__ = [
    "ConfigurationError",
    "PathValidationError",
    "TOMLConfigLoader",
    "load_config_dict",
    "load_settings",
    "get_settings",
    "print_config",
]
