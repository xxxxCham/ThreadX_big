"""
ThreadX - Exponential Moving Average (EMA) Indicator
====================================================

Calcul GPU-optimisé de l'EMA (Exponential Moving Average).

L'EMA donne plus de poids aux prix récents comparé à la SMA.
Très utilisé en scalping pour sa réactivité aux changements de prix.

Formule:
    EMA[t] = α * Close[t] + (1-α) * EMA[t-1]
    où α = 2 / (period + 1)

Exemple:
    >>> from threadx.indicators.ema import ema, EMASettings
    >>> settings = EMASettings(period=50)
    >>> ema_values = ema(close, settings)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from threadx.utils.log import get_logger

logger = get_logger(__name__)


# ==========================================
# SETTINGS
# ==========================================


@dataclass
class EMASettings:
    """
    Configuration pour l'EMA.

    Attributes:
        period: Période de l'EMA (défaut: 50)
        use_gpu: Utiliser GPU si disponible (défaut: False - CPU suffit pour EMA)
    """

    period: int = 50
    use_gpu: bool = False

    def __post_init__(self):
        if self.period < 1:
            raise ValueError(f"period must be >= 1, got {self.period}")


# ==========================================
# NUMBA OPTIMIZED CORE
# ==========================================


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _ema_core_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calcul Numba de l'EMA.

    Args:
        close: Prix de clôture (n,)
        period: Période de l'EMA

    Returns:
        EMA values
    """
    n = len(close)
    ema_values = np.full(n, np.nan, dtype=np.float64)

    # Facteur de lissage
    alpha = 2.0 / (period + 1)

    # Initialisation: première valeur = SMA des 'period' premières valeurs
    if n >= period:
        ema_values[period - 1] = np.mean(close[:period])

        # Calcul itératif
        for i in range(period, n):
            ema_values[i] = alpha * close[i] + (1.0 - alpha) * ema_values[i - 1]

    return ema_values


# ==========================================
# PUBLIC API
# ==========================================


def ema(
    close: pd.Series | np.ndarray, settings: EMASettings | None = None
) -> np.ndarray:
    """
    Calcule l'EMA (Exponential Moving Average).

    Args:
        close: Prix de clôture
        settings: Configuration (défaut: EMASettings())

    Returns:
        EMA values - array numpy de même taille que l'entrée

    Example:
        >>> ema_50 = ema(df['close'], EMASettings(period=50))
        >>> ema_100 = ema(df['close'], EMASettings(period=100))
        >>> # Détection croisement
        >>> cross_up = (ema_50 > ema_100) & (np.roll(ema_50, 1) <= np.roll(ema_100, 1))
    """
    if settings is None:
        settings = EMASettings()

    # Conversion en numpy
    if isinstance(close, pd.Series):
        close = close.values

    close = np.asarray(close, dtype=np.float64)

    # Calcul
    ema_values = _ema_core_numba(close, settings.period)

    logger.debug(
        f"EMA calculé: period={settings.period}, n_valid={(~np.isnan(ema_values)).sum()}"
    )

    return ema_values


def ema_crossover_signals(
    fast_ema: np.ndarray, slow_ema: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Génère des signaux de croisement entre deux EMAs.

    Args:
        fast_ema: EMA rapide (période courte)
        slow_ema: EMA lente (période longue)

    Returns:
        Dictionnaire contenant:
        - 'bullish_cross': Booléen - EMA rapide croise EMA lente vers le haut
        - 'bearish_cross': Booléen - EMA rapide croise EMA lente vers le bas
        - 'fast_above_slow': Booléen - EMA rapide au-dessus de EMA lente

    Example:
        >>> ema_50 = ema(close, EMASettings(50))
        >>> ema_100 = ema(close, EMASettings(100))
        >>> signals = ema_crossover_signals(ema_50, ema_100)
        >>> long_entry = signals['bullish_cross']
    """
    fast_above_slow = fast_ema > slow_ema
    fast_above_slow_prev = np.roll(fast_above_slow, 1)
    fast_above_slow_prev[0] = False

    bullish_cross = (~fast_above_slow_prev) & fast_above_slow
    bearish_cross = fast_above_slow_prev & (~fast_above_slow)

    return {
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "fast_above_slow": fast_above_slow,
    }


def ema_dataframe(
    df: pd.DataFrame, periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Ajoute des colonnes EMA à un DataFrame OHLC.

    Args:
        df: DataFrame avec colonne 'close'
        periods: Liste des périodes EMA (défaut: [50, 100])

    Returns:
        DataFrame avec colonnes ajoutées:
        - 'ema_X': EMA période X

    Example:
        >>> df_with_ema = ema_dataframe(df, periods=[50, 100, 200])
        >>> df_with_ema[['close', 'ema_50', 'ema_100']].tail()
    """
    if periods is None:
        periods = [50, 100]

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    df_result = df.copy()

    for period in periods:
        settings = EMASettings(period=period)
        ema_values = ema(df["close"], settings)
        df_result[f"ema_{period}"] = ema_values

    return df_result


# ==========================================
# EXPORTS
# ==========================================

__all__ = ["EMASettings", "ema", "ema_crossover_signals", "ema_dataframe"]
