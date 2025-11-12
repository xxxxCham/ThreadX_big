"""
Lightweight backtest bridge used by the Streamlit UI.

This module provides TWO backtest implementations:
1. run_backtest() - Lightweight CPU-only for quick demos
2. run_backtest_gpu() - Full GPU-accelerated with BacktestEngine

The GPU version connects to the production BacktestEngine with:
- Multi-GPU support (RTX 5090 75% + RTX 2060 25%)
- IndicatorBank for cached GPU computations
- Real-time system monitoring
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .strategy_registry import list_strategies

logger = logging.getLogger(__name__)

# Lazy imports pour éviter overhead si GPU non utilisé
_ENGINE_IMPORTS_DONE = False
_BacktestEngine = None
_IndicatorBank = None
_get_global_monitor = None


def _ensure_gpu_imports():
    """Lazy import des modules GPU lourds."""
    global _ENGINE_IMPORTS_DONE, _BacktestEngine, _IndicatorBank, _get_global_monitor

    if _ENGINE_IMPORTS_DONE:
        return

    try:
        from threadx.backtest.engine import BacktestEngine
        from threadx.indicators.bank import IndicatorBank

        from .system_monitor import get_global_monitor

        _BacktestEngine = BacktestEngine
        _IndicatorBank = IndicatorBank
        _get_global_monitor = get_global_monitor
        _ENGINE_IMPORTS_DONE = True

        logger.info("✅ GPU modules importés avec succès")
    except ImportError as e:
        logger.warning(f"⚠️ Impossible d'importer modules GPU: {e}")
        _ENGINE_IMPORTS_DONE = False


@dataclass
class BacktestResult:
    """Simplified backtest result structure consumed by the UI pages."""

    equity: pd.Series
    metrics: dict[str, Any] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_window(params: dict[str, Any], default: int = 20) -> int:
    """Extract an integer window/period from params with sensible bounds."""
    for key in ("window", "lookback", "period"):
        if key in params:
            try:
                value = int(params[key])
                return max(value, 2)
            except (TypeError, ValueError):
                continue
    return max(int(default), 2)


def _generate_position(close: pd.Series, params: dict[str, Any]) -> pd.Series:
    """Generate a long-only position series based on simplified Bollinger logic."""
    window = _resolve_window(params)
    if len(close) < window:
        return pd.Series(0.0, index=close.index)

    std_mult = float(params.get("std", 2.0) or 2.0)
    signal_window = max(1, int(params.get("signal_window", 3) or 3))
    gap_input = float(params.get("price_gap_pct", 0.5) or 0.5)
    gap_ratio = abs(gap_input) / 100.0 if abs(gap_input) > 1 else abs(gap_input) / 100.0
    bandwidth_threshold = float(params.get("bandwidth_threshold", 0.0) or 0.0)
    use_bandwidth_filter = bool(params.get("use_bandwidth_filter", False))
    confirm_breakout = bool(params.get("confirm_breakout", False))

    rolling_mean = close.rolling(window, min_periods=window).mean()
    rolling_std = close.rolling(window, min_periods=window).std(ddof=0)
    rolling_std = rolling_std.replace(0, np.nan)

    z_score = (close - rolling_mean) / rolling_std
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    bandwidth = (2 * std_mult * rolling_std) / rolling_mean.replace(0, np.nan).abs()
    if use_bandwidth_filter:
        bandwidth_ok = (bandwidth.fillna(0.0) >= bandwidth_threshold)
    else:
        bandwidth_ok = pd.Series(True, index=close.index)

    long_trigger = z_score <= (-std_mult * (1.0 + gap_ratio))
    exit_trigger = z_score >= (std_mult * 0.5 if confirm_breakout else 0.0)

    if signal_window > 1:
        long_trigger = (
            long_trigger.rolling(signal_window, min_periods=1).mean() >= 0.6
        )
        exit_trigger = (
            exit_trigger.rolling(signal_window, min_periods=1).mean() >= 0.5
        )

    enter_condition = long_trigger & bandwidth_ok
    exit_condition = exit_trigger

    position = pd.Series(0.0, index=close.index)
    in_trade = False

    for idx, timestamp in enumerate(close.index):
        can_enter = bool(enter_condition.iloc[idx])
        can_exit = bool(exit_condition.iloc[idx])

        if not in_trade and can_enter:
            in_trade = True
        elif in_trade and can_exit:
            in_trade = False

        position.iloc[idx] = 1.0 if in_trade else 0.0

    return position


def _compute_equity(close: pd.Series, params: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Compute equity curve and position series for the lightweight backtest."""
    position = _generate_position(close, params)
    returns = close.pct_change().fillna(0.0)
    strategy_returns = returns * position

    equity = (1.0 + strategy_returns).cumprod()
    if not equity.empty:
        equity.iloc[0] = 1.0
    equity.name = "equity"
    return equity, position


def _build_placeholder_trades(close: pd.Series, position: pd.Series) -> list[dict[str, Any]]:
    """Convert the position series into placeholder trades for the UI."""
    if close.empty or position.empty:
        return []

    trades: list[dict[str, Any]] = []
    in_trade = False
    current: dict[str, Any] | None = None

    for timestamp, pos in position.items():
        price = float(close.loc[timestamp])
        if pos >= 0.5 and not in_trade:
            in_trade = True
            current = {
                "entry_time": timestamp,
                "entry_price": price,
                "side": "LONG",
            }
        elif pos < 0.5 and in_trade and current is not None:
            current["exit_time"] = timestamp
            current["exit_price"] = price
            current["pnl"] = current["exit_price"] - current["entry_price"]
            trades.append(current)
            in_trade = False
            current = None

    if in_trade and current is not None:
        current["exit_time"] = close.index[-1]
        current["exit_price"] = float(close.iloc[-1])
        current["pnl"] = current["exit_price"] - current["entry_price"]
        trades.append(current)

    return trades


def run_backtest(df: pd.DataFrame, strategy: str, params: dict[str, Any]) -> BacktestResult:
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
        equity, position = _compute_equity(close, params)

        returns = equity.pct_change().fillna(0.0)
        metrics: dict[str, Any] = {
            "total_return": float(equity.iloc[-1] - 1.0),
            "annualized_volatility": float(np.std(returns) * np.sqrt(252)),
            "sharpe_ratio": float((returns.mean() / returns.std()) * np.sqrt(252))
            if returns.std() > 0
            else 0.0,
        }

        trades = _build_placeholder_trades(close, position)
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


def run_backtest_gpu(
    df: pd.DataFrame,
    strategy: str,
    params: dict[str, Any],
    *,
    symbol: str = "BTCUSDC",
    timeframe: str = "1m",
    use_gpu: bool = True,
    enable_monitoring: bool = True,
) -> BacktestResult:
    """
    Exécute un backtest GPU-accelerated avec le moteur de production.

    Cette version utilise le vrai BacktestEngine avec :
    - Multi-GPU (RTX 5090 75% + RTX 2060 25%)
    - IndicatorBank pour cache GPU
    - Monitoring système temps réel (CPU, GPU1, GPU2)
    - Calculs d'indicateurs optimisés

    Args:
        df: DataFrame OHLCV avec colonnes [open, high, low, close, volume]
        strategy: Nom de la stratégie (doit être dans registry)
        params: Paramètres de stratégie (entry_z, k_sl, leverage, etc.)
        symbol: Symbole tradé (pour cache IndicatorBank)
        timeframe: Timeframe des données
        use_gpu: Active/désactive GPU (True par défaut)
        enable_monitoring: Active monitoring système temps réel

    Returns:
        BacktestResult avec equity, metrics, trades, metadata

    Raises:
        ValueError: Si données invalides
        RuntimeError: Si erreur GPU non récupérable

    Example:
        >>> result = run_backtest_gpu(
        ...     df_ohlcv,
        ...     strategy="bollinger_reversion",
        ...     params={"entry_z": 2.0, "k_sl": 1.5, "leverage": 3},
        ...     use_gpu=True,
        ...     enable_monitoring=True
        ... )
        >>> print(f"GPU utilisé: {result.metadata.get('gpu_enabled')}")
        >>> print(f"Multi-GPU: {result.metadata.get('multi_gpu_enabled')}")
    """
    # Import lazy des modules GPU
    _ensure_gpu_imports()

    if not _ENGINE_IMPORTS_DONE:
        logger.warning("⚠️ Modules GPU non disponibles, fallback vers run_backtest() CPU")
        return run_backtest(df, strategy, params)

    # Validation entrées
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Le DataFrame d'entrée est vide ou invalide.")

    available_strategies = set(list_strategies())
    if strategy not in available_strategies:
        raise ValueError(f"Stratégie '{strategy}' non disponible.")

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Le DataFrame doit contenir les colonnes OHLCV: {missing_cols}")

    start_time = time.time()
    logger.info(f"🚀 Démarrage backtest GPU: {symbol} {timeframe}, GPU={use_gpu}, monitoring={enable_monitoring}")

    # Démarrage monitoring système
    monitor = None
    if enable_monitoring and _get_global_monitor:
        try:
            monitor = _get_global_monitor()
            if not monitor.is_running():
                monitor.start()
                logger.info("📊 Monitoring système démarré")
        except Exception as e:
            logger.warning(f"⚠️ Erreur démarrage monitoring: {e}")
            monitor = None

    try:
        # 1. Initialisation du moteur GPU avec multi-GPU
        engine = _BacktestEngine(use_multi_gpu=True)
        logger.info(f"✅ BacktestEngine initialisé: GPU={engine.gpu_available}, Multi-GPU={engine.use_multi_gpu}")

        # 2. Calcul des indicateurs via IndicatorBank (avec cache GPU)
        bank = _IndicatorBank()

        # Paramètres Bollinger Bands
        bb_period = params.get("bb_period", params.get("window", params.get("period", 20)))
        bb_std = params.get("bb_std", params.get("std", 2.0))

        # Paramètres ATR
        atr_period = params.get("atr_period", 14)

        logger.info(f"📊 Calcul indicateurs: BB(period={bb_period}, std={bb_std}), ATR(period={atr_period})")

        indicators = {}

        # Bollinger Bands
        try:
            bollinger_result = bank.ensure(
                "bollinger",
                {"period": int(bb_period), "std": float(bb_std)},
                df,
                symbol=symbol,
                timeframe=timeframe,
            )
            indicators["bollinger"] = bollinger_result
            logger.info("✅ Bollinger Bands calculés")
        except Exception as e:
            logger.warning(f"⚠️ Erreur Bollinger Bands: {e}")
            indicators["bollinger"] = None

        # ATR
        try:
            atr_result = bank.ensure(
                "atr",
                {"period": int(atr_period)},
                df,
                symbol=symbol,
                timeframe=timeframe,
            )
            indicators["atr"] = atr_result
            logger.info("✅ ATR calculé")
        except Exception as e:
            logger.warning(f"⚠️ Erreur ATR: {e}")
            indicators["atr"] = None

        # 3. Paramètres pour BacktestEngine
        engine_params = {
            "entry_z": params.get("entry_z", 2.0),
            "k_sl": params.get("k_sl", 1.5),
            "leverage": params.get("leverage", 1.0),
            "initial_capital": params.get("initial_capital", 10000.0),
            "fees_bps": params.get("fees_bps", 10.0),
            "slip_bps": params.get("slip_bps", 5.0),
        }

        # 4. Exécution du backtest avec GPU
        logger.info(f"⚡ Exécution backtest avec use_gpu={use_gpu}")
        result = engine.run(
            df_1m=df,
            indicators=indicators,
            params=engine_params,
            symbol=symbol,
            timeframe=timeframe,
            use_gpu=use_gpu,
            seed=42,
        )

        # 5. Conversion RunResult → BacktestResult
        # Calcul métriques simplifiées pour UI
        returns = result.returns
        total_return = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1.0)
        volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0

        metrics = {
            "total_return": total_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
        }

        # Conversion trades DataFrame → List[Dict]
        trades_list = []
        if not result.trades.empty:
            for _, trade_row in result.trades.iterrows():
                trades_list.append({
                    "entry_time": trade_row["entry_ts"],
                    "exit_time": trade_row["exit_ts"],
                    "entry_price": trade_row["price_entry"],
                    "exit_price": trade_row["price_exit"],
                    "pnl": trade_row["pnl"],
                    "side": trade_row.get("side", "LONG"),
                })

        # Métadonnées enrichies
        metadata = {
            "strategy": strategy,
            "params": params,
            "gpu_enabled": result.meta.get("mode") in ["single_gpu", "multi_gpu"],
            "multi_gpu_enabled": result.meta.get("mode") == "multi_gpu",
            "devices_used": result.meta.get("devices", []),
            "gpu_balance": result.meta.get("gpu_balance", {}),
            "execution_time_sec": time.time() - start_time,
            "engine_meta": result.meta,
        }

        # Ajout stats monitoring si disponible
        if monitor:
            try:
                stats = monitor.get_stats_summary()
                metadata["monitoring_stats"] = stats
                logger.info(f"📊 Monitoring stats: CPU_mean={stats.get('cpu_mean', 0):.1f}%, "
                           f"GPU1_mean={stats.get('gpu1_mean', 0):.1f}%, "
                           f"GPU2_mean={stats.get('gpu2_mean', 0):.1f}%")
            except Exception as e:
                logger.warning(f"⚠️ Erreur récupération stats monitoring: {e}")

        duration = time.time() - start_time
        logger.info(f"✅ Backtest GPU terminé en {duration:.2f}s, "
                   f"{len(trades_list)} trades, "
                   f"Sharpe={sharpe:.2f}")

        return BacktestResult(
            equity=result.equity,
            metrics=metrics,
            trades=trades_list,
            metadata=metadata,
        )

    except Exception as exc:
        duration = time.time() - start_time
        logger.error(f"❌ Erreur backtest GPU après {duration:.2f}s: {exc}", exc_info=True)

        # Fallback vers version CPU simple
        logger.warning("⚠️ Fallback vers backtest CPU simple")
        return run_backtest(df, strategy, params)



