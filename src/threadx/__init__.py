"""
ThreadX Main Package-
Main package initialization for ThreadX framework.
"""

__version__ = "1.0.0"
__author__ = "ThreadX Team"
__description__ = "High-performance backtesting framework with GPU acceleration"

# Import core configuration
from .config import (
    ConfigurationError,
    PathValidationError,
    Settings,
    get_settings,
    load_settings,
)

__all__ = [
    "ConfigurationError",
    "PathValidationError",
    "Settings",
    "get_settings",
    "load_settings",
]
