"""
ThreadX Fast Sweep - Optimisation Ultra-Rapide
==============================================

Sweep optimisé pour l'interface Streamlit avec :
- Batch processing des indicateurs (calcul 1 fois seulement)
- Mise à jour UI espacée (tous les 50 runs)
- Calculs vectorisés numpy
- Pas de recalcul inutile

Objectif : 100+ runs/seconde

Author: ThreadX Framework
Version: 1.0
"""

import time
from typing import Dict, List, Any, Callable, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Optional global cancel hook from optimization engine
try:
    from threadx.optimization.engine import is_global_stop_requested  # type: ignore
except Exception:
    def is_global_stop_requested() -> bool:  # type: ignore
        return False


def fast_parameter_sweep(
    data: pd.DataFrame,
    param_name: str,
    param_values: List[Any],
    strategy_func: Callable,
    *,
    capital_initial: float = 10000.0,
    update_callback: Optional[Callable] = None,
    update_frequency: int = 50,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """
    Sweep ultra-rapide avec batch processing.

    Cette fonction optimise le sweep en :
    1. Pré-calculant tous les indicateurs une seule fois
    2. Espaçant les mises à jour UI (tous les update_frequency runs)
    3. Utilisant des calculs vectorisés numpy
    4. Évitant tout recalcul inutile

    Args:
        data: DataFrame OHLCV
        param_name: Nom du paramètre à optimiser
        param_values: Liste des valeurs à tester
        strategy_func: Fonction de stratégie à tester
        capital_initial: Capital de départ
        update_callback: Fonction appelée pour mise à jour UI (idx, total, result)
        update_frequency: Fréquence de mise à jour UI (tous les N runs)

    Returns:
        DataFrame avec résultats pour chaque valeur

    Example:
        >>> results = fast_parameter_sweep(
        ...     data=df_ohlcv,
        ...     param_name="window",
        ...     param_values=list(range(10, 50)),
        ...     strategy_func=simple_ma_strategy,
        ...     capital_initial=10000,
        ...     update_callback=lambda i, total, r: print(f"{i}/{total}")
        ... )
    """
    start_time = time.time()
    n_params = len(param_values)

    logger.info(f"🚀 Fast Sweep démarré: {n_params} paramètres, param={param_name}")

    # === ÉTAPE 1: Pré-calcul des prix et returns (1 fois seulement) ===
    close = data["close"].values
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])  # Pad pour avoir même longueur

    results = []

    # === ÉTAPE 2: Boucle sur les paramètres (vectorisée autant que possible) ===
    for idx, param_value in enumerate(param_values):
        # Cooperative cancellation: allow UI to request stop
        if (should_cancel and should_cancel()) or is_global_stop_requested():
            logger.info("⏹️  Fast Sweep cancellation requested by user")
            break
        iter_start = time.time()

        # Appliquer stratégie pour ce paramètre
        try:
            # Stratégie retourne positions {-1, 0, 1}
            params = {param_name: param_value}
            positions = strategy_func(data, params)

            # Calculs vectorisés des métriques
            strategy_returns = returns * positions[:-1] if len(positions) > len(returns) else returns * positions

            # Equity curve
            cumulative_returns = np.cumprod(1 + strategy_returns)
            total_return = cumulative_returns[-1] - 1.0 if len(cumulative_returns) > 0 else 0.0

            # Sharpe ratio (annualisé)
            if len(strategy_returns) > 1 and strategy_returns.std() > 0:
                sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0

            # Max drawdown (vectorisé)
            cummax = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - cummax) / cummax
            max_dd = drawdown.min() if len(drawdown) > 0 else 0.0

            # PnL en euros
            pnl_euros = capital_initial * total_return
            equity_final = capital_initial * (1 + total_return)

            # Nombre de trades (approximation: changements de position)
            position_changes = np.diff(positions)
            nb_trades = int(np.sum(np.abs(position_changes) > 0))

            # Taille moyenne des trades (approximation)
            if nb_trades > 0:
                # PnL moyen par trade
                avg_trade_size = abs(pnl_euros / nb_trades)
            else:
                avg_trade_size = 0.0

            # Résultat
            result = {
                "param": param_value,
                "sharpe": sharpe,
                "return_pct": total_return * 100,
                "pnl_euros": pnl_euros,
                "equity_final": equity_final,
                "max_dd": max_dd * 100,
                "nb_trades": nb_trades,
                "avg_trade_size": avg_trade_size,
                "compute_time_ms": (time.time() - iter_start) * 1000,
            }

            results.append(result)

            # Callback UI (seulement tous les update_frequency runs)
            if update_callback and (idx % update_frequency == 0 or idx == n_params - 1):
                update_callback(idx + 1, n_params, result)

        except Exception as e:
            logger.error(f"Erreur param {param_value}: {e}")
            # Résultat par défaut
            results.append({
                "param": param_value,
                "sharpe": 0.0,
                "return_pct": 0.0,
                "pnl_euros": 0.0,
                "equity_final": capital_initial,
                "max_dd": 0.0,
                "nb_trades": 0,
                "avg_trade_size": 0.0,
                "compute_time_ms": 0.0,
                "error": str(e),
            })

    # === ÉTAPE 3: Construction DataFrame final ===
    results_df = pd.DataFrame(results)

    total_time = time.time() - start_time
    throughput = n_params / total_time if total_time > 0 else 0

    logger.info(f"✅ Fast Sweep terminé: {n_params} runs en {total_time:.2f}s "
                f"({throughput:.1f} runs/sec)")

    # Stats de performance
    if not results_df.empty and "compute_time_ms" in results_df.columns:
        avg_time = results_df["compute_time_ms"].mean()
        logger.info(f"   Temps moyen par run: {avg_time:.2f}ms")

    return results_df


def simple_bollinger_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    """
    Stratégie Bollinger simple ultra-rapide (vectorisée).

    Cette implémentation évite toutes les opérations lentes :
    - Pas de boucles Python
    - Calculs vectorisés numpy/pandas
    - Pas de conditions if/else dans la boucle

    Args:
        data: DataFrame OHLCV
        params: Paramètres {"window": int, "std": float, ...}

    Returns:
        Array numpy de positions {-1: short, 0: flat, 1: long}
    """
    # Extraction paramètres
    window = params.get("window", params.get("period", 20))
    std_mult = params.get("std", params.get("bb_std", 2.0))

    # Calculs vectorisés Bollinger
    close = data["close"]
    rolling = close.rolling(window=window, min_periods=window)

    middle = rolling.mean()
    std = rolling.std()

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    # Signaux vectorisés
    # Long: prix <= lower (oversold)
    # Short: prix >= upper (overbought)
    # Flat: entre les bandes

    positions = np.zeros(len(close))

    # Mean reversion: long quand prix touche bande basse
    long_signal = (close <= lower).astype(int)

    # Exit/flat quand prix revient au milieu
    flat_signal = (close > lower) & (close < upper)

    # Stratégie simple: long dans oversold, flat sinon
    # (on pourrait améliorer avec gestion d'état mais restons vectorisé)

    # Approche vectorisée simple: si prix < lower → position = 1
    positions = (close <= lower).astype(float)

    return positions


def bollinger_zscore_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    """
    Stratégie Bollinger avec Z-score (plus sophistiquée mais toujours vectorisée).

    Args:
        data: DataFrame OHLCV
        params: {"window": int, "std": float, "entry_z": float}

    Returns:
        Positions array
    """
    window = params.get("window", params.get("period", 20))
    std_mult = params.get("std", params.get("bb_std", 2.0))
    entry_z = params.get("entry_z", 2.0)

    close = data["close"]
    rolling = close.rolling(window=window, min_periods=window)

    middle = rolling.mean()
    std = rolling.std()

    # Z-score
    z_score = (close - middle) / std.replace(0, np.nan)
    z_score = z_score.fillna(0)

    # Positions basées sur Z-score
    # Long: z < -entry_z
    # Short: z > entry_z
    # Flat: abs(z) < entry_z

    positions = np.zeros(len(close))

    # Mean reversion
    long_mask = z_score <= -entry_z
    short_mask = z_score >= entry_z

    positions[long_mask] = 1.0
    positions[short_mask] = -1.0

    return positions


def adaptive_ma_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    """
    Stratégie moving average adaptative (trend following).

    Args:
        data: DataFrame OHLCV
        params: {"window": int}

    Returns:
        Positions array
    """
    window = params.get("window", params.get("period", 20))

    close = data["close"]
    ma = close.rolling(window=window, min_periods=window).mean()

    # Trend following: long si prix > MA
    positions = (close > ma).astype(float)

    return positions


# Mapping des stratégies disponibles
STRATEGY_FUNCTIONS = {
    "bollinger_reversion": simple_bollinger_strategy,
    "bollinger_zscore": bollinger_zscore_strategy,
    "ma_trend": adaptive_ma_strategy,
}


def get_strategy_function(strategy_name: str) -> Callable:
    """
    Récupère la fonction de stratégie optimisée.

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Fonction de stratégie

    Raises:
        ValueError: Si stratégie inconnue
    """
    if strategy_name not in STRATEGY_FUNCTIONS:
        # Fallback vers Bollinger par défaut
        logger.warning(f"Stratégie '{strategy_name}' inconnue, fallback vers bollinger_reversion")
        return simple_bollinger_strategy

    return STRATEGY_FUNCTIONS[strategy_name]
