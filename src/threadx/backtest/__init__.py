#!/usr/bin/env python3
"""
ThreadX Backtest Module
======================

Production-ready backtesting framework with performance analytics.

Phase 6 - Performance Metrics:
- Comprehensive financial metrics calculation
- GPU-accelerated computations with CPU fallback
- Risk-adjusted returns (Sharpe, Sortino)
- Trade analysis (profit factor, win rate, expectancy)
- Drawdown visualization and analysis
- Robust error handling and edge case management

Features:
✅ Vectorized CPU/GPU-aware implementations
✅ Standard financial metrics with proper annualization
✅ Matplotlib-based visualization (headless compatible)
✅ Type-safe API with comprehensive logging
✅ Deterministic testing with seed=42
✅ Windows 11 compatible with relative paths

Integration:
- Compatible with ThreadX Engine (Phase 5) outputs
- TOML configuration support (when available)
- Structured logging for performance monitoring
- No environment variables dependency

Usage:
    from threadx.backtest.performance import summarize, plot_drawdown

    # Calculate comprehensive metrics
    metrics = summarize(trades_df, returns_series, initial_capital=10000)

    # Generate drawdown visualization
    plot_path = plot_drawdown(equity_series, save_path=Path("./reports/dd.png"))
"""

from threadx.backtest.performance import (
    # Core equity and drawdown functions
    equity_curve,
    max_drawdown,
    drawdown_series,
    # Risk-adjusted metrics
    sharpe_ratio,
    sortino_ratio,
    # Trade-based metrics
    profit_factor,
    win_rate,
    expectancy,
    # Comprehensive analysis
    summarize,
    # Visualization
    plot_drawdown,
    # GPU capability detection
    HAS_CUPY,
    xp,
)

__all__ = [
    # Core functions
    "equity_curve",
    "max_drawdown",
    "drawdown_series",
    # Risk metrics
    "sharpe_ratio",
    "sortino_ratio",
    # Trade metrics
    "profit_factor",
    "win_rate",
    "expectancy",
    # Integration
    "summarize",
    "plot_drawdown",
    # Utilities
    "HAS_CUPY",
    "xp",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "ThreadX Team"
__description__ = "ThreadX Phase 6 - Performance Metrics Module"
