"""
ThreadX Utils Module - Phase 9
Vector and Array Validation Utilities.

Provides comprehensive validation for array shapes, dtypes, and data quality:
- Shape and dimension validation with detailed error messages
- Data type checking and conversion recommendations
- NaN/infinity detection and handling suggestions
- Performance-oriented validation (non-blocking warnings)
- Integration with ThreadX logging for actionable feedback

Designed for hot-path validation in indicator calculations and backtesting.
"""

import logging
import warnings
from typing import Any, Optional, Tuple, List, Dict, Union, Callable
from dataclasses import dataclass
import numpy as np

# Import ThreadX logger - fallback to standard logging if not available
try:
    from threadx.utils.log import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


# Import ThreadX xp utils if available
try:
    from threadx.utils.xp import xp, get_array_info, asnumpy

    XP_AVAILABLE = True
except ImportError:
    XP_AVAILABLE = False

# Import ThreadX Settings - fallback if not available
try:
    from threadx.config import load_settings

    SETTINGS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency during tests
    SETTINGS_AVAILABLE = False


logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of array validation."""

    is_valid: bool
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]
    array_info: Dict[str, Any]


class ArrayValidator:
    """
    Comprehensive array validator for ThreadX operations.

    Provides fast, non-blocking validation with actionable feedback.
    Designed for use in hot paths without significant performance impact.

    Examples
    --------
    >>> validator = ArrayValidator()
    >>>
    >>> # Basic validation
    >>> result = validator.validate(data, expected_shape=(1000,), expected_dtype=np.float64)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(f"Error: {error}")

    >>> # Custom validation with callback
    >>> def handle_issues(result):
    ...     if result.warnings:
    ...         log_performance_warning(result.warnings)
    >>>
    >>> validator.validate(data, callback=handle_issues)
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Parameters
        ----------
        strict_mode : bool, default False
            If True, treats warnings as errors.
        """
        self.strict_mode = strict_mode
        self._validation_cache = {}

    def validate(
        self,
        array: Any,
        *,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[np.dtype] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        check_finite: bool = True,
        check_contiguous: bool = False,
        callback: Optional[Callable[[ValidationResult], None]] = None,
        name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate array properties and data quality.

        Performs comprehensive validation with performance-oriented design.
        Issues warnings for non-critical issues, errors for blocking problems.

        Parameters
        ----------
        array : array-like
            Array to validate.
        expected_shape : tuple, optional
            Expected array shape. None means any shape is acceptable.
        expected_dtype : numpy.dtype, optional
            Expected data type.
        min_size : int, optional
            Minimum required array size.
        max_size : int, optional
            Maximum allowed array size.
        allow_nan : bool, default False
            Whether NaN values are acceptable.
        allow_inf : bool, default False
            Whether infinite values are acceptable.
        check_finite : bool, default True
            Whether to check for finite values.
        check_contiguous : bool, default False
            Whether to check for contiguous memory layout.
        callback : callable, optional
            Callback function to handle validation results.
        name : str, optional
            Name for logging/debugging purposes.

        Returns
        -------
        ValidationResult
            Comprehensive validation results.
        """
        warnings_list = []
        errors_list = []
        suggestions_list = []

        array_name = name or "array"

        # Basic array conversion and info
        try:
            if XP_AVAILABLE:
                array_info = get_array_info(array)
                # Convert to numpy for validation if needed
                if hasattr(array, "get"):  # CuPy array
                    np_array = asnumpy(array)
                else:
                    np_array = np.asarray(array)
            else:
                np_array = np.asarray(array)
                array_info = {
                    "device": "cpu",
                    "shape": np_array.shape,
                    "dtype": np_array.dtype,
                    "memory_mb": np_array.nbytes / (1024 * 1024),
                    "is_contiguous": np_array.flags.c_contiguous,
                }
        except Exception as e:
            errors_list.append(f"Failed to convert {array_name} to array: {e}")
            return ValidationResult(
                is_valid=False,
                warnings=warnings_list,
                errors=errors_list,
                suggestions=["Ensure input is array-like (list, numpy array, etc.)"],
                array_info={},
            )

        # Shape validation
        if expected_shape is not None:
            if np_array.shape != expected_shape:
                error_msg = f"{array_name} shape mismatch: expected {expected_shape}, got {np_array.shape}"
                if self.strict_mode:
                    errors_list.append(error_msg)
                else:
                    warnings_list.append(error_msg)

                suggestions_list.append(
                    f"Reshape or subset {array_name} to match expected dimensions"
                )

        # Size validation
        array_size = np_array.size

        if min_size is not None and array_size < min_size:
            errors_list.append(f"{array_name} too small: {array_size} < {min_size}")
            suggestions_list.append(
                f"Provide at least {min_size} data points for reliable calculation"
            )

        if max_size is not None and array_size > max_size:
            warnings_list.append(f"{array_name} very large: {array_size} > {max_size}")
            suggestions_list.append("Consider batch processing for large datasets")

        # Data type validation
        if expected_dtype is not None:
            if np_array.dtype != expected_dtype:
                warning_msg = f"{array_name} dtype mismatch: expected {expected_dtype}, got {np_array.dtype}"
                warnings_list.append(warning_msg)

                # Suggest conversion if reasonable
                if np.can_cast(np_array.dtype, expected_dtype, casting="safe"):
                    suggestions_list.append(
                        f"Convert {array_name} to {expected_dtype} using astype()"
                    )
                else:
                    suggestions_list.append(
                        f"Check data precision requirements for {array_name}"
                    )

        # Data quality checks (only for numeric arrays)
        if np.issubdtype(np_array.dtype, np.number):
            try:
                # Check for NaN values
                nan_count = np.isnan(np_array).sum()
                if nan_count > 0:
                    if allow_nan:
                        warnings_list.append(
                            f"{array_name} contains {nan_count} NaN values"
                        )
                    else:
                        errors_list.append(
                            f"{array_name} contains {nan_count} NaN values (not allowed)"
                        )
                        suggestions_list.append(
                            "Remove or fill NaN values before processing"
                        )

                # Check for infinite values
                if check_finite:
                    inf_count = np.isinf(np_array).sum()
                    if inf_count > 0:
                        if allow_inf:
                            warnings_list.append(
                                f"{array_name} contains {inf_count} infinite values"
                            )
                        else:
                            errors_list.append(
                                f"{array_name} contains {inf_count} infinite values (not allowed)"
                            )
                            suggestions_list.append("Clip or remove infinite values")

                # Check data range for potential issues
                if array_size > 0 and not (nan_count == array_size):
                    try:
                        data_min = np.nanmin(np_array)
                        data_max = np.nanmax(np_array)

                        # Warn about extreme values that might cause numerical issues
                        if data_max > 1e10:
                            warnings_list.append(
                                f"{array_name} contains very large values (max: {data_max:.2e})"
                            )
                            suggestions_list.append(
                                "Consider scaling data to prevent numerical overflow"
                            )

                        if data_min < -1e10:
                            warnings_list.append(
                                f"{array_name} contains very small values (min: {data_min:.2e})"
                            )
                            suggestions_list.append(
                                "Consider scaling data to prevent numerical underflow"
                            )

                        # Check for constant arrays
                        if data_min == data_max and array_size > 1:
                            warnings_list.append(
                                f"{array_name} is constant (all values = {data_min})"
                            )
                            suggestions_list.append(
                                "Constant arrays may cause division by zero in calculations"
                            )

                    except Exception as e:
                        logger.debug(f"Data range check failed for {array_name}: {e}")

            except Exception as e:
                warnings_list.append(f"Data quality check failed for {array_name}: {e}")

        # Memory layout checks
        if check_contiguous and not array_info.get("is_contiguous", True):
            warnings_list.append(f"{array_name} is not contiguous in memory")
            suggestions_list.append("Use np.ascontiguousarray() for better performance")

        # Performance warnings
        memory_mb = array_info.get("memory_mb", 0)
        if memory_mb > 1000:  # > 1GB
            warnings_list.append(f"{array_name} is very large ({memory_mb:.1f}MB)")
            suggestions_list.append(
                "Consider batch processing or memory-efficient algorithms"
            )

        # Create result
        is_valid = len(errors_list) == 0

        result = ValidationResult(
            is_valid=is_valid,
            warnings=warnings_list,
            errors=errors_list,
            suggestions=suggestions_list,
            array_info=array_info,
        )

        # Log results
        self._log_validation_result(result, array_name)

        # Call callback if provided
        if callback:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Validation callback failed: {e}")

        return result

    def _log_validation_result(self, result: ValidationResult, array_name: str) -> None:
        """Log validation results at appropriate levels."""
        if result.errors:
            for error in result.errors:
                logger.error(f"Validation error for {array_name}: {error}")

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"Validation warning for {array_name}: {warning}")

        if result.suggestions and (result.errors or result.warnings):
            logger.info(
                f"Suggestions for {array_name}: {'; '.join(result.suggestions)}"
            )

    def validate_multiple(
        self, arrays: Dict[str, Any], **validation_kwargs
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple arrays with same criteria.

        Parameters
        ----------
        arrays : dict
            Dictionary of name -> array to validate.
        **validation_kwargs
            Validation parameters passed to validate().

        Returns
        -------
        dict
            Dictionary of name -> ValidationResult.
        """
        results = {}

        for name, array in arrays.items():
            results[name] = self.validate(array, name=name, **validation_kwargs)

        return results


# Convenience functions for common validation patterns
def validate_price_data(
    prices: Any, min_length: int = 2, name: str = "prices"
) -> ValidationResult:
    """
    Validate price/OHLCV data arrays.

    Specialized validation for financial time series data.

    Parameters
    ----------
    prices : array-like
        Price data array.
    min_length : int, default 2
        Minimum required length for calculations.
    name : str, default "prices"
        Name for logging.

    Returns
    -------
    ValidationResult
        Validation results.
    """
    validator = ArrayValidator()

    return validator.validate(
        prices,
        expected_dtype=np.float64,
        min_size=min_length,
        allow_nan=False,
        allow_inf=False,
        name=name,
    )


def validate_indicator_params(
    period: int,
    multiplier: Optional[float] = None,
    *,
    min_period: int = 1,
    max_period: int = 1000,
) -> ValidationResult:
    """
    Validate common indicator parameters.

    Parameters
    ----------
    period : int
        Period parameter (e.g., moving average window).
    multiplier : float, optional
        Multiplier parameter (e.g., standard deviation multiplier).
    min_period : int, default 1
        Minimum allowed period.
    max_period : int, default 1000
        Maximum reasonable period.

    Returns
    -------
    ValidationResult
        Validation results.
    """
    warnings_list = []
    errors_list = []
    suggestions_list = []

    # Validate period
    if not isinstance(period, int):
        errors_list.append(f"Period must be integer, got {type(period).__name__}")
    elif period < min_period:
        errors_list.append(f"Period too small: {period} < {min_period}")
    elif period > max_period:
        warnings_list.append(f"Period very large: {period} > {max_period}")
        suggestions_list.append(
            "Large periods may require more data and cause edge effects"
        )

    # Validate multiplier if provided
    if multiplier is not None:
        if not isinstance(multiplier, (int, float)):
            errors_list.append(
                f"Multiplier must be numeric, got {type(multiplier).__name__}"
            )
        elif multiplier <= 0:
            errors_list.append(f"Multiplier must be positive, got {multiplier}")
        elif multiplier > 10:
            warnings_list.append(f"Multiplier very large: {multiplier}")
            suggestions_list.append("Large multipliers may produce extreme values")

    return ValidationResult(
        is_valid=len(errors_list) == 0,
        warnings=warnings_list,
        errors=errors_list,
        suggestions=suggestions_list,
        array_info={"period": period, "multiplier": multiplier},
    )


def check_array_compatibility(
    *arrays: Any, operation: str = "operation"
) -> ValidationResult:
    """
    Check if arrays are compatible for operations.

    Validates shape compatibility, device compatibility, and dtype compatibility.

    Parameters
    ----------
    *arrays : array-like
        Arrays to check for compatibility.
    operation : str, default "operation"
        Description of the operation for error messages.

    Returns
    -------
    ValidationResult
        Compatibility validation results.
    """
    if len(arrays) < 2:
        return ValidationResult(
            is_valid=True, warnings=[], errors=[], suggestions=[], array_info={}
        )

    warnings_list = []
    errors_list = []
    suggestions_list = []

    # Convert all arrays and get info
    array_infos = []
    np_arrays = []

    for i, array in enumerate(arrays):
        try:
            if XP_AVAILABLE:
                info = get_array_info(array)
                np_array = (
                    asnumpy(array) if hasattr(array, "get") else np.asarray(array)
                )
            else:
                np_array = np.asarray(array)
                info = {
                    "device": "cpu",
                    "shape": np_array.shape,
                    "dtype": np_array.dtype,
                }

            array_infos.append(info)
            np_arrays.append(np_array)

        except Exception as e:
            errors_list.append(f"Failed to process array {i}: {e}")
            return ValidationResult(
                is_valid=False,
                warnings=warnings_list,
                errors=errors_list,
                suggestions=["Ensure all inputs are array-like"],
                array_info={},
            )

    # Check shape compatibility
    shapes = [info["shape"] for info in array_infos]
    if not all(np.broadcast_shapes(shapes[0], shape) for shape in shapes[1:]):
        errors_list.append(f"Arrays not broadcast-compatible for {operation}: {shapes}")
        suggestions_list.append("Reshape arrays or use compatible dimensions")

    # Check device compatibility
    devices = [info["device"] for info in array_infos]
    if len(set(devices)) > 1:
        warnings_list.append(f"Arrays on different devices for {operation}: {devices}")
        suggestions_list.append("Move arrays to same device for optimal performance")

    # Check dtype compatibility
    dtypes = [info["dtype"] for info in array_infos]
    if len(set(str(dtype) for dtype in dtypes)) > 1:
        warnings_list.append(f"Mixed dtypes in {operation}: {dtypes}")
        suggestions_list.append(
            "Convert arrays to common dtype to avoid precision loss"
        )

    return ValidationResult(
        is_valid=len(errors_list) == 0,
        warnings=warnings_list,
        errors=errors_list,
        suggestions=suggestions_list,
        array_info={"shapes": shapes, "devices": devices, "dtypes": dtypes},
    )


# Global validator instance for convenience
default_validator = ArrayValidator()

# Convenience aliases
validate = default_validator.validate
validate_arrays = default_validator.validate_multiple
