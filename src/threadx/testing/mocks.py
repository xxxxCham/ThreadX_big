"""
Mocks centralisés pour ThreadX - utilisés uniquement en cas de fallback.

Ces mocks permettent aux applications de fonctionner même si certains modules
ThreadX ne sont pas disponibles ou chargés correctement.
"""
# type: ignore  # Trop d'erreurs de type, analyse désactivée

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union, List


class MockSettings:
    """Mock pour threadx.config.settings.Settings."""

    @staticmethod
    def get(key: str, default=None):
        """Retourne une valeur de configuration mock."""
        mock_values = {
            "gpu.enable_gpu": False,
            "gpu.devices": ["cpu"],
            "performance.max_workers": 1,
            "performance.cache_ttl_sec": 300,
            "paths.data_root": "./data",
            "paths.logs": "./logs",
            "paths.cache": "./cache",
        }
        return mock_values.get(key, default)


def get_mock_logger(name: str) -> logging.Logger:
    """Mock pour threadx.utils.log.get_logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def setup_mock_logging_once():
    """Mock pour threadx.utils.log.setup_logging_once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class MockBank:
    """Mock pour threadx.indicators.bank.Bank."""

    def ensure(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        symbol: str = "",
        timeframe: str = "",
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Simule l'assurance d'un indicateur."""
        time.sleep(0.1)  # Simule le calcul

        if isinstance(data, pd.DataFrame) and "close" in data.columns:
            close_series = data["close"]
        elif isinstance(data, pd.Series):
            close_series = data
        else:
            # Fallback si pas de données correctes
            close_series = pd.Series(np.arange(100) + 100.0)

        if indicator_type == "bollinger":
            # Convertir en arrays NumPy pour compatibilité de type
            upper = (close_series + 2.0).values
            middle = close_series.values
            lower = (close_series - 2.0).values
            return (upper, middle, lower)
        elif indicator_type == "atr":
            # Retourner un array NumPy
            return np.ones(len(close_series)) * 0.5
        else:
            # Retourner un array NumPy
            return np.random.randn(len(close_series)) * 0.1 + 0.5

    def batch_ensure(
        self,
        indicator_type: str,
        params_list: List[Dict[str, Any]],
        data: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
    ) -> Dict[str, Any]:
        """Simule le calcul batch d'indicateurs."""
        time.sleep(len(params_list) * 0.01)

        results = {}
        for i, params in enumerate(params_list):
            # Utiliser le nom du paramètre comme clé
            param_key = f"{indicator_type}_{i}"
            results[param_key] = self.ensure(
                indicator_type, params, data, symbol, timeframe
            )

        return results


class MockRunResult:
    """Mock pour threadx.backtest.engine.RunResult."""

    def __init__(self, returns, trades, meta=None):
        self.returns = returns
        self.trades = trades
        self.equity = (1 + returns).cumprod() * meta.get("initial_capital", 10000)
        self.meta = meta or {
            "status": "completed",
            "execution_time": 1.0,
            "device": "cpu",
        }


class MockBacktestEngine:
    """Mock pour threadx.backtest.engine.BacktestEngine."""

    def __init__(self, controller=None):
        self.controller = controller

    def run(
        self,
        df_1m: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
        symbol: str = "BTCUSDC",
        timeframe: str = "1m",
        seed: int = 42,
        use_gpu: bool = False,
    ) -> MockRunResult:
        """Simule l'exécution d'un backtest."""
        # Définir la graine pour déterminisme
        np.random.seed(seed)

        # Simuler le calcul avec interruptions potentielles
        total_steps = len(df_1m)
        step_interval = max(1, total_steps // 10)  # 10 checkpoints

        for i in range(0, total_steps, step_interval):
            # Checkpoints pour interruption
            if self.controller:
                try:
                    self.controller.check_interruption(f"step {i}/{total_steps}")
                except KeyboardInterrupt:
                    # Créer un résultat partiel avec status="interrupted"
                    returns = pd.Series(
                        np.random.randn(i) * 0.01, index=df_1m.index[:i]
                    )
                    trades = pd.DataFrame(
                        {"entry_time": [], "exit_time": [], "pnl": [], "side": []}
                    )
                    meta = {
                        "status": "interrupted",
                        "execution_time": 0.5,
                        "device": "cpu" if not use_gpu else "gpu",
                        "initial_capital": params.get("initial_capital", 10000),
                    }
                    return MockRunResult(returns, trades, meta)

            # Simuler le travail
            time.sleep(0.01)

        # Simuler un résultat normal
        n_trades = min(20, len(df_1m) // 50)
        trade_indices = np.random.choice(
            range(len(df_1m) - 10), size=n_trades, replace=False
        )

        # Générer des returns aléatoires avec une tendance
        returns = pd.Series(np.random.randn(len(df_1m)) * 0.01, index=df_1m.index)

        # Créer des trades synthétiques
        trades = pd.DataFrame(
            {
                "entry_time": df_1m.index[trade_indices],
                "exit_time": [
                    t + pd.Timedelta(minutes=np.random.randint(10, 100))
                    for t in df_1m.index[trade_indices]
                ],
                "pnl": np.random.randn(n_trades) * 100,
                "side": np.random.choice(["LONG", "SHORT"], size=n_trades),
                "entry_price": df_1m["close"].iloc[trade_indices].values,
                "exit_price": df_1m["close"].iloc[trade_indices].values
                * (1 + np.random.randn(n_trades) * 0.01),
            }
        )

        meta = {
            "status": "completed",
            "execution_time": np.random.rand() * 2 + 0.5,  # Entre 0.5 et 2.5 secondes
            "device": "cpu" if not use_gpu else "gpu",
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "initial_capital": params.get("initial_capital", 10000),
            "seed": seed,
        }

        return MockRunResult(returns, trades, meta)


class MockPerformanceCalculator:
    """Mock pour threadx.performance.metrics.PerformanceCalculator."""

    @staticmethod
    def summarize(returns: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        """Calcule des métriques de performance mock."""
        # Simulation de métriques réalistes
        total_return = returns.sum()
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        max_dd = (returns.cumsum() - returns.cumsum().expanding().max()).min()

        # Si trades a des données PnL, utiliser pour win rate
        if "pnl" in trades.columns and len(trades) > 0:
            win_rate = (trades["pnl"] > 0).mean()
            profit_factor = (
                abs(trades[trades["pnl"] > 0]["pnl"].sum())
                / abs(trades[trades["pnl"] < 0]["pnl"].sum())
                if abs(trades[trades["pnl"] < 0]["pnl"].sum()) > 0
                else float("inf")
            )
        else:
            win_rate = 0.5
            profit_factor = 1.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": len(trades),
        }


# Mocks pour l'UI
def mock_plot_equity(equity: pd.Series, save_path: Union[str, None] = None) -> str:
    """Mock pour threadx.ui.charts.plot_equity."""
    print(f"[MOCK] Generating equity chart with {len(equity)} points")
    return save_path or "equity_chart.png"


def mock_plot_drawdown(equity: pd.Series, save_path: Union[str, None] = None) -> str:
    """Mock pour threadx.ui.charts.plot_drawdown."""
    print(f"[MOCK] Generating drawdown chart from {len(equity)} points")
    return save_path or "drawdown_chart.png"


def mock_render_trades_table(trades: pd.DataFrame) -> Dict[str, Any]:
    """Mock pour threadx.ui.tables.render_trades_table."""
    return {
        "data": trades,
        "summary": {
            "total_trades": len(trades),
            "profitable_trades": (
                len(trades[trades["pnl"] > 0]) if "pnl" in trades.columns else 0
            ),
        },
    }


def mock_render_metrics_table(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Mock pour threadx.ui.tables.render_metrics_table."""
    return {"data": pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])}


def mock_export_table(df: pd.DataFrame, path: str) -> str:
    """Mock pour threadx.ui.tables.export_table."""
    print(f"[MOCK] Exporting table with {len(df)} rows to {path}")
    return path


# Classe de contrôleur mock pour les tests
class MockBacktestController:
    """Mock pour threadx.backtest.engine.BacktestController."""

    def __init__(self):
        self.is_stopped = False
        self.is_paused = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_stopped = True

    def reset(self):
        self.is_stopped = False
        self.is_paused = False

    def check_interruption(self, step_info=None):
        if self.is_stopped:
            raise KeyboardInterrupt("Backtest stopped")
        while self.is_paused and not self.is_stopped:
            time.sleep(0.1)
        if self.is_stopped:
            raise KeyboardInterrupt("Backtest stopped")


# Export des mocks principaux
__all__ = [
    "MockSettings",
    "get_mock_logger",
    "setup_mock_logging_once",
    "MockBank",
    "MockBacktestEngine",
    "MockRunResult",
    "MockPerformanceCalculator",
    "MockBacktestController",
    "mock_plot_equity",
    "mock_plot_drawdown",
    "mock_render_trades_table",
    "mock_render_metrics_table",
    "mock_export_table",
]
