"""
ThreadX Main Package - 
Main package initialization for ThreadX framework.
"""

__version__ = "1.0.0"
__author__ = "ThreadX Team"
__description__ = "High-performance backtesting framework with GPU acceleration"

# Import core configuration
from .config import (
    Settings,
    get_settings,
    load_settings,
    ConfigurationError,
    PathValidationError,
)

__all__ = [
    "Settings",
    "get_settings",
    "load_settings",
    "ConfigurationError",
    "PathValidationError",
]
