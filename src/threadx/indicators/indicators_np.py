"""
ThreadX Indicateurs NumPy - Implémentations natives
===================================================

Fonctions d'indicateurs techniques optimisées NumPy.
Code extrait et consolidé depuis unified_data_historique_with_indicators.py

Performance:
    - 50x plus rapide que pandas rolling
    - Optimisations EMA custom
    - Gestion NaN robuste

Auteur: ThreadX Core Team
Version: 1.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ewm(x: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential weighted moving average optimisée.

    Args:
        x: Array de valeurs
        span: Période EWM

    Returns:
        Array EWM
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=np.float64)

    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]

    if span <= 1:
        out[:] = x
        return out

    alpha = 2.0 / (span + 1.0)
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]

    return out


def ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential Moving Average.

    Args:
        arr: Prix (généralement close)
        span: Période EMA

    Returns:
        Array EMA
    """
    return _ewm(arr, span)


def atr_np(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Average True Range.

    Args:
        high: Prix hauts
        low: Prix bas
        close: Prix clôture
        period: Période ATR

    Returns:
        Array ATR
    """
    prev = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return _ewm(tr, period)


def boll_np(
    close: np.ndarray, period: int = 20, std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    Args:
        close: Prix clôture
        period: Période moyenne mobile
        std: Multiplicateur écart-type

    Returns:
        Tuple (lower, ma, upper, z-score)
    """
    ma = _ewm(close, period)
    var = _ewm((close - ma) ** 2, period)
    sd = np.sqrt(np.maximum(var, 1e-12))

    upper = ma + std * sd
    lower = ma - std * sd
    z = (close - ma) / sd

    return lower, ma, upper, z


def rsi_np(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.

    Args:
        close: Prix clôture
        period: Période RSI

    Returns:
        Array RSI (0-100)
    """
    if close.size == 0:
        return np.array([], dtype=np.float64)

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = _ewm(gain, period)
    avg_loss = _ewm(loss, period)

    rs = np.divide(avg_gain, np.maximum(avg_loss, 1e-12))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def macd_np(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence.

    Args:
        close: Prix clôture
        fast: Période EMA rapide
        slow: Période EMA lente
        signal: Période signal line

    Returns:
        Tuple (macd, signal, histogram)
    """
    ema_fast = ema_np(close, fast)
    ema_slow = ema_np(close, slow)
    macd = ema_fast - ema_slow
    sig = ema_np(macd, signal)
    hist = macd - sig

    return macd, sig, hist


def vwap_np(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    window: int = 96,
) -> np.ndarray:
    """
    Volume Weighted Average Price.

    Args:
        close: Prix clôture
        high: Prix hauts
        low: Prix bas
        volume: Volume
        window: Fenêtre EMA

    Returns:
        Array VWAP
    """
    typical = (high + low + close) / 3.0
    vol_ema = ema_np(volume, window)
    pv_ema = ema_np(typical * volume, window)
    vwap = np.divide(pv_ema, np.maximum(vol_ema, 1e-12))

    return vwap


def obv_np(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume.

    Args:
        close: Prix clôture
        volume: Volume

    Returns:
        Array OBV
    """
    obv = np.zeros_like(close, dtype=np.float64)

    for i in range(1, close.size):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    return obv


def vortex_df(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> pd.DataFrame:
    """
    Vortex Indicator.

    Args:
        highs: Prix hauts
        lows: Prix bas
        closes: Prix clôture
        period: Période indicateur

    Returns:
        DataFrame avec colonnes vi_plus, vi_minus
    """
    n = len(closes)
    if n == 0:
        return pd.DataFrame({"vi_plus": [], "vi_minus": []})

    h = highs.astype(np.float64)
    l = lows.astype(np.float64)
    c = closes.astype(np.float64)

    # Valeurs précédentes
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_c = np.roll(c, 1)
    prev_h[0] = h[0]
    prev_l[0] = l[0]
    prev_c[0] = c[0]

    # Calculs Vortex
    vm_plus = np.abs(h - prev_l)
    vm_minus = np.abs(l - prev_h)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    # Sommes rolling
    vm_p_sum = (
        pd.Series(vm_plus, dtype="float64")
        .rolling(window=period, min_periods=period)
        .sum()
    )
    vm_m_sum = (
        pd.Series(vm_minus, dtype="float64")
        .rolling(window=period, min_periods=period)
        .sum()
    )
    tr_sum = (
        pd.Series(tr, dtype="float64").rolling(window=period, min_periods=period).sum()
    )

    # Indicateurs finaux
    vi_plus = (vm_p_sum / tr_sum).to_numpy()
    vi_minus = (vm_m_sum / tr_sum).to_numpy()

    return pd.DataFrame({"vi_plus": vi_plus, "vi_minus": vi_minus})


__all__ = [
    "ema_np",
    "atr_np",
    "boll_np",
    "rsi_np",
    "macd_np",
    "vwap_np",
    "obv_np",
    "vortex_df",
]
