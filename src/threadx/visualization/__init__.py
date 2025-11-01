"""
ThreadX Visualization Module
=============================

Génération de graphiques interactifs pour visualiser résultats backtests.
"""

from .backtest_charts import (
    generate_backtest_chart,
    generate_multi_timeframe_chart,
)

__all__ = [
    "generate_backtest_chart",
    "generate_multi_timeframe_chart",
]
