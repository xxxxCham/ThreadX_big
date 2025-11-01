"""
ThreadX OHLCV Data Normalization Module
========================================

Handles normalization of OHLCV (Open, High, Low, Close, Volume) data from various formats.
Ensures consistent structure, timezone handling, and column naming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for OHLCV normalization."""

    timezone: str = "UTC"
    required_columns: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    datetime_column: Optional[str] = None  # Auto-detect if None
    lowercase_columns: bool = True
    ensure_utc_index: bool = True
    sort_by_time: bool = True


@dataclass
class NormalizationReport:
    """Report of normalization results."""

    success: bool
    transformations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Default normalization configuration
DEFAULT_NORMALIZATION_CONFIG = NormalizationConfig(
    timezone="UTC",
    required_columns=["open", "high", "low", "close", "volume"],
    datetime_column=None,
    lowercase_columns=True,
    ensure_utc_index=True,
    sort_by_time=True,
)


def _detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the datetime column in the dataframe.

    Looks for common naming patterns: 'time', 'timestamp', 'datetime', 'date'.

    Args:
        df: The dataframe to search

    Returns:
        The name of the datetime column or None if not found
    """
    candidates = ["time", "timestamp", "datetime", "date"]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def _normalize_datetime_index(
    df: pd.DataFrame, datetime_col: Optional[str] = None, timezone: str = "UTC"
) -> tuple[pd.DataFrame, List[str]]:
    """
    Normalize the datetime index of the dataframe.

    Args:
        df: The dataframe to normalize
        datetime_col: The datetime column name (auto-detect if None)
        timezone: Target timezone (default: 'UTC')

    Returns:
        Tuple of (normalized_dataframe, transformation_list)
    """
    transformations = []

    # Detect datetime column if not provided
    if datetime_col is None:
        datetime_col = _detect_datetime_column(df)

    # Handle datetime column
    if datetime_col and datetime_col in df.columns:
        try:
            # Detect if timestamps are in milliseconds (> 1e10) or seconds
            sample_val = df[datetime_col].iloc[0] if len(df) > 0 else 0
            try:
                sample_num = float(sample_val)
                is_milliseconds = abs(sample_num) > 1e10
                unit = 'ms' if is_milliseconds else 's'
            except (ValueError, TypeError):
                unit = None  # Let pandas auto-detect

            df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, unit=unit, errors="coerce")
            df = df.set_index(datetime_col)
            unit_str = f" (unit={unit})" if unit else ""
            transformations.append(f"Set '{datetime_col}' as index with UTC timezone{unit_str}")
        except Exception as e:
            logger.warning(f"Failed to set datetime column '{datetime_col}': {e}")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert index to datetime
        try:
            # Detect if index values are in milliseconds
            sample_val = df.index[0] if len(df) > 0 else 0
            try:
                sample_num = float(sample_val)
                is_milliseconds = abs(sample_num) > 1e10
                unit = 'ms' if is_milliseconds else 's'
            except (ValueError, TypeError):
                unit = None

            df.index = pd.to_datetime(df.index, utc=True, unit=unit, errors="coerce")
            unit_str = f" (unit={unit})" if unit else ""
            transformations.append(f"Converted index to datetime with UTC timezone{unit_str}")
        except Exception as e:
            logger.warning(f"Failed to convert index to datetime: {e}")

    # Ensure UTC timezone
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        transformations.append("Localized datetime index to UTC")
    elif isinstance(df.index, pd.DatetimeIndex) and df.index.tz != timezone:
        df.index = df.index.tz_convert(timezone)
        transformations.append(f"Converted datetime index timezone from {df.index.tz} to {timezone}")

    return df, transformations


def _normalize_columns(df: pd.DataFrame, lowercase: bool = True) -> tuple[pd.DataFrame, List[str]]:
    """
    Normalize column names.

    Args:
        df: The dataframe to normalize
        lowercase: Convert column names to lowercase

    Returns:
        Tuple of (normalized_dataframe, transformation_list)
    """
    transformations = []

    if lowercase:
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.lower()
        if df.columns.tolist() != original_columns:
            transformations.append("Converted column names to lowercase")

    return df, transformations


def _validate_required_columns(
    df: pd.DataFrame, required_columns: List[str]
) -> tuple[bool, List[str]]:
    """
    Validate that required columns are present.

    Args:
        df: The dataframe to validate
        required_columns: List of required column names

    Returns:
        Tuple of (all_present, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def normalize_ohlcv(
    df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
) -> tuple[pd.DataFrame, NormalizationReport]:
    """
    Normalize OHLCV data to a standard format.

    Args:
        df: Input dataframe (OHLCV data)
        config: Normalization configuration (uses DEFAULT if None)

    Returns:
        Tuple of (normalized_dataframe, NormalizationReport)
    """
    if config is None:
        config = DEFAULT_NORMALIZATION_CONFIG

    report = NormalizationReport(success=True)
    df = df.copy()

    # Step 1: Normalize datetime index
    try:
        df, dt_transforms = _normalize_datetime_index(df, config.datetime_column, config.timezone)
        report.transformations.extend(dt_transforms)
    except Exception as e:
        report.errors.append(f"Datetime normalization failed: {e}")
        report.success = False

    # Step 2: Normalize column names
    try:
        df, col_transforms = _normalize_columns(df, config.lowercase_columns)
        report.transformations.extend(col_transforms)
    except Exception as e:
        report.errors.append(f"Column normalization failed: {e}")
        report.success = False

    # Step 3: Validate required columns
    all_present, missing = _validate_required_columns(df, config.required_columns)
    if not all_present:
        msg = f"Missing required columns: {', '.join(missing)}"
        report.errors.append(msg)
        report.success = False
    else:
        report.transformations.append(f"Validated presence of required columns: {', '.join(config.required_columns)}")

    # Step 4: Sort by time (optional)
    if config.sort_by_time and isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.sort_index()
            report.transformations.append("Sorted data by datetime index")
        except Exception as e:
            report.warnings.append(f"Failed to sort by datetime: {e}")

    return df, report
