"""
ThreadX - Stochastic Oscillator Indicator
=========================================

Calcul GPU-optimisé du Stochastic Oscillator (%K et %D).

Le Stochastic Oscillator compare le prix de clôture actuel à sa fourchette 
sur une période donnée. Il génère deux lignes:
- %K (ligne rapide): Position relative dans la fourchette
- %D (ligne lente): Moyenne mobile de %K

Utilisé pour identifier les zones de surachat (>80) et survente (<20).

Formules:
    %K = 100 * (Close - LowestLow(n)) / (HighestHigh(n) - LowestLow(n))
    %D = SMA(%K, k)

Exemple:
    >>> from threadx.indicators.stochastic import stochastic_rsi, StochasticSettings
    >>> settings = StochasticSettings(k_period=14, d_period=3, smooth_k=3)
    >>> k_values, d_values = stochastic_rsi(high, low, close, settings)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from threadx.utils.log import get_logger

logger = get_logger(__name__)


# ==========================================
# SETTINGS
# ==========================================


@dataclass
class StochasticSettings:
    """
    Configuration pour le Stochastic Oscillator.

    Attributes:
        k_period: Période pour %K (défaut: 14)
        d_period: Période pour %D - SMA de %K (défaut: 3)
        smooth_k: Période de lissage pour %K (défaut: 3)
        use_gpu: Utiliser GPU si disponible (défaut: False - CPU suffit)
        overbought: Seuil de surachat (défaut: 80)
        oversold: Seuil de survente (défaut: 20)
    """

    k_period: int = 14
    d_period: int = 3
    smooth_k: int = 3
    use_gpu: bool = False
    overbought: float = 80.0
    oversold: float = 20.0

    def __post_init__(self):
        if self.k_period < 1:
            raise ValueError(f"k_period must be >= 1, got {self.k_period}")
        if self.d_period < 1:
            raise ValueError(f"d_period must be >= 1, got {self.d_period}")
        if self.smooth_k < 1:
            raise ValueError(f"smooth_k must be >= 1, got {self.smooth_k}")


# ==========================================
# NUMBA OPTIMIZED CORE
# ==========================================


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _rolling_min_max_numba(
    high: np.ndarray, low: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les min/max glissants sur une période.

    Args:
        high: Prix high (n,)
        low: Prix low (n,)
        period: Période de calcul

    Returns:
        Tuple (highest_high, lowest_low)
    """
    n = len(high)
    highest = np.full(n, np.nan, dtype=np.float64)
    lowest = np.full(n, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        start = i - period + 1
        highest[i] = np.max(high[start : i + 1])
        lowest[i] = np.min(low[start : i + 1])

    return highest, lowest


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _simple_moving_average_numba(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calcule une SMA simple.

    Args:
        values: Valeurs d'entrée
        period: Période de la moyenne

    Returns:
        SMA
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        result[i] = np.mean(values[i - period + 1 : i + 1])

    return result


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _stochastic_core_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int,
    d_period: int,
    smooth_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcul Numba du Stochastic Oscillator.

    Args:
        high: Prix high (n,)
        low: Prix low (n,)
        close: Prix close (n,)
        k_period: Période %K
        d_period: Période %D
        smooth_k: Période de lissage %K

    Returns:
        Tuple (%K, %D)
    """
    n = len(close)

    # 1. Calcul highest high et lowest low
    highest, lowest = _rolling_min_max_numba(high, low, k_period)

    # 2. Calcul %K brut
    k_raw = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        hl_range = highest[i] - lowest[i]
        if hl_range > 1e-10:  # Éviter division par zéro
            k_raw[i] = 100.0 * (close[i] - lowest[i]) / hl_range
        else:
            k_raw[i] = 50.0  # Valeur neutre si range = 0

    # 3. Lissage %K si demandé
    if smooth_k > 1:
        k_smooth = _simple_moving_average_numba(k_raw, smooth_k)
    else:
        k_smooth = k_raw

    # 4. Calcul %D (SMA de %K)
    d_values = _simple_moving_average_numba(k_smooth, d_period)

    return k_smooth, d_values


# ==========================================
# PUBLIC API
# ==========================================


def stochastic_oscillator(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    settings: StochasticSettings | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule le Stochastic Oscillator (%K et %D).

    Args:
        high: Prix high
        low: Prix low
        close: Prix close
        settings: Configuration (défaut: StochasticSettings())

    Returns:
        Tuple (%K, %D) - arrays numpy de même taille que l'entrée

    Example:
        >>> k, d = stochastic_oscillator(df['high'], df['low'], df['close'])
        >>> # Détection survente
        >>> oversold = (k < 20) & (d < 20)
    """
    if settings is None:
        settings = StochasticSettings()

    # Conversion en numpy
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # Validation
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, close must have same length")

    # Calcul
    k_values, d_values = _stochastic_core_numba(
        high=high,
        low=low,
        close=close,
        k_period=settings.k_period,
        d_period=settings.d_period,
        smooth_k=settings.smooth_k,
    )

    logger.debug(
        f"Stochastic calculé: k_period={settings.k_period}, d_period={settings.d_period}, "
        f"smooth_k={settings.smooth_k}, n_valid=%K={(~np.isnan(k_values)).sum()}"
    )

    return k_values, d_values


def stochastic_signals(
    k_values: np.ndarray,
    d_values: np.ndarray,
    overbought: float = 80.0,
    oversold: float = 20.0,
) -> dict[str, np.ndarray]:
    """
    Génère des signaux de trading basés sur le Stochastic.

    Args:
        k_values: Valeurs %K
        d_values: Valeurs %D
        overbought: Seuil de surachat (défaut: 80)
        oversold: Seuil de survente (défaut: 20)

    Returns:
        Dictionnaire contenant:
        - 'oversold': Booléen - en zone de survente
        - 'overbought': Booléen - en zone de surachat
        - 'bullish_cross': Booléen - %K croise %D à la hausse
        - 'bearish_cross': Booléen - %K croise %D à la baisse
        - 'bullish_exit_oversold': Booléen - sortie de survente vers le haut
        - 'bearish_exit_overbought': Booléen - sortie de surachat vers le bas

    Example:
        >>> k, d = stochastic_oscillator(high, low, close)
        >>> signals = stochastic_signals(k, d)
        >>> long_entry = signals['bullish_exit_oversold']
    """
    n = len(k_values)

    # Zones de surachat/survente
    in_oversold = k_values < oversold
    in_overbought = k_values > overbought

    # Croisements %K / %D
    k_above_d = k_values > d_values
    k_above_d_prev = np.roll(k_above_d, 1)
    k_above_d_prev[0] = False

    bullish_cross = (~k_above_d_prev) & k_above_d  # %K croise %D vers le haut
    bearish_cross = k_above_d_prev & (~k_above_d)  # %K croise %D vers le bas

    # Sorties de zones
    was_oversold = np.roll(in_oversold, 1)
    was_oversold[0] = False
    was_overbought = np.roll(in_overbought, 1)
    was_overbought[0] = False

    bullish_exit_oversold = was_oversold & (~in_oversold) & (k_values > d_values)
    bearish_exit_overbought = was_overbought & (~in_overbought) & (k_values < d_values)

    return {
        "oversold": in_oversold,
        "overbought": in_overbought,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "bullish_exit_oversold": bullish_exit_oversold,
        "bearish_exit_overbought": bearish_exit_overbought,
    }


def stochastic_dataframe(
    df: pd.DataFrame, settings: StochasticSettings | None = None
) -> pd.DataFrame:
    """
    Ajoute les colonnes Stochastic à un DataFrame OHLC.

    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close'
        settings: Configuration (défaut: StochasticSettings())

    Returns:
        DataFrame avec colonnes ajoutées:
        - 'stoch_k': %K
        - 'stoch_d': %D
        - 'stoch_oversold': Booléen
        - 'stoch_overbought': Booléen

    Example:
        >>> df_with_stoch = stochastic_dataframe(df)
        >>> df_with_stoch[['close', 'stoch_k', 'stoch_d']].tail()
    """
    if settings is None:
        settings = StochasticSettings()

    # Validation
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Calcul
    k, d = stochastic_oscillator(df["high"], df["low"], df["close"], settings)

    # Ajout colonnes
    df_result = df.copy()
    df_result["stoch_k"] = k
    df_result["stoch_d"] = d
    df_result["stoch_oversold"] = k < settings.oversold
    df_result["stoch_overbought"] = k > settings.overbought

    return df_result


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    "StochasticSettings",
    "stochastic_oscillator",
    "stochastic_signals",
    "stochastic_dataframe",
]
