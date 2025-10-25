"""
Lightweight backtest bridge used by the Streamlit UI.

This module purposely avoids importing the heavy engine to keep the demo UI
responsive.  It provides a small `run_backtest` helper that computes simple
metrics from an OHLCV DataFrame and returns them in a convenient container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import logging

import numpy as np
import pandas as pd

from .strategy_registry import list_strategies

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Simplified backtest result structure consumed by the UI pages."""

    equity: pd.Series
    metrics: Dict[str, Any] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _compute_equity(close: pd.Series) -> pd.Series:
    """Compute a naive equity curve from close prices."""
    returns = close.pct_change().fillna(0.0)
    equity = (1.0 + returns).cumprod()
    equity.name = "equity"
    return equity


def _build_placeholder_trades(close: pd.Series) -> List[Dict[str, Any]]:
    """Build a minimal trades list to visualise basic activity."""
    if len(close) < 3:
        return []

    rolling = close.rolling(window=3, min_periods=3).mean()
    signal = close > rolling
    trades: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for timestamp, is_long in signal.items():
        price = float(close.loc[timestamp])
        if is_long and current is None:
            current = {"entry_time": timestamp, "entry_price": price}
        elif not is_long and current is not None:
            current["exit_time"] = timestamp
            current["exit_price"] = price
            current["pnl"] = price - current["entry_price"]
            trades.append(current)
            current = None

    if current is not None:
        current["exit_time"] = close.index[-1]
        current["exit_price"] = float(close.iloc[-1])
        current["pnl"] = current["exit_price"] - current["entry_price"]
        trades.append(current)

    return trades


def run_backtest(df: pd.DataFrame, strategy: str, params: Dict[str, Any]) -> BacktestResult:
    """Execute a lightweight backtest on the provided OHLCV DataFrame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Le DataFrame d'entree est vide ou invalide.")

    available_strategies = set(list_strategies())
    if strategy not in available_strategies:
        raise ValueError(f"Strategie '{strategy}' non disponible.")

    if "close" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'close'.")

    close = pd.to_numeric(df["close"], errors="coerce")
    if close.isna().all():
        raise ValueError("La colonne 'close' ne contient pas de valeurs numeriques exploitables.")

    try:
        equity = _compute_equity(close)

        returns = equity.pct_change().fillna(0.0)
        metrics: Dict[str, Any] = {
            "total_return": float(equity.iloc[-1] - 1.0),
            "annualized_volatility": float(np.std(returns) * np.sqrt(252)),
            "sharpe_ratio": float((returns.mean() / returns.std()) * np.sqrt(252))
            if returns.std() > 0
            else 0.0,
        }

        trades = _build_placeholder_trades(close)
        metadata = {"strategy": strategy, "params": params}

        return BacktestResult(
            equity=equity,
            metrics=metrics,
            trades=trades,
            metadata=metadata,
        )

    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.error("Erreur lors du backtest: %s", exc, exc_info=True)
        return BacktestResult(
            equity=pd.Series(dtype=float),
            metrics={},
            trades=[],
            metadata={"error": str(exc)},
        )



