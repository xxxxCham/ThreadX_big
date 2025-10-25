"""
ThreadX CLI Module
==================

Command-line interface for ThreadX backtesting framework.
Provides async commands for data validation, indicator building,
backtesting, and parameter optimization via ThreadXBridge.

Usage:
    python -m threadx.cli --help
    python -m threadx.cli data validate <path>
    python -m threadx.cli backtest run --strategy <name>
    python -m threadx.cli optimize sweep --param <name>

Author: ThreadX Framework
Version: Prompt 9 - CLI Bridge Interface
"""

from .main import app

__all__ = ["app"]
