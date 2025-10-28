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

from typing import Dict, Iterable, Tuple

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


def sma_np(values: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average (SMA) using cumulative sums for performance.

    Args:
        values: Array of prices.
        period: SMA window.

    Returns:
        Array of SMA values with NaN padding for the warm-up window.
    """
    values = np.asarray(values, dtype=np.float64)
    if period <= 0:
        raise ValueError("period must be positive")
    if values.size == 0:
        return np.array([], dtype=np.float64)

    result = np.full(values.shape, np.nan, dtype=np.float64)
    cumsum = np.cumsum(values, dtype=np.float64)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    result[period - 1 :] = cumsum[period - 1 :] / period
    return result


def standard_deviation_np(values: np.ndarray, period: int, ddof: int = 0) -> np.ndarray:
    """
    Rolling standard deviation.

    Args:
        values: Array of prices.
        period: Window length.
        ddof: Delta degrees of freedom (0 = population, 1 = sample).

    Returns:
        Array of rolling standard deviation values.
    """
    values = np.asarray(values, dtype=np.float64)
    if period <= 1 or values.size == 0:
        return np.full(values.shape, np.nan, dtype=np.float64)

    series = pd.Series(values, dtype="float64")
    return series.rolling(window=period, min_periods=period).std(ddof=ddof).to_numpy()


def momentum_np(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Price momentum (difference vs. previous closing price).

    Args:
        close: Close prices.
        period: Lookback for momentum.

    Returns:
        Array of momentum values.
    """
    close = np.asarray(close, dtype=np.float64)
    if period <= 0 or close.size == 0:
        return np.zeros_like(close, dtype=np.float64)

    out = np.full(close.shape, np.nan, dtype=np.float64)
    out[period:] = close[period:] - close[:-period]
    return out


def stochastic_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator (%K and %D).
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if min(high.size, low.size, close.size) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    highest = pd.Series(high).rolling(window=k_period, min_periods=k_period).max()
    lowest = pd.Series(low).rolling(window=k_period, min_periods=k_period).min()

    k = np.full(close.shape, np.nan, dtype=np.float64)
    denom = highest - lowest
    valid = denom != 0
    k[valid] = ((close[valid] - lowest[valid]) / denom[valid]) * 100.0

    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean().to_numpy()
    return k, d


def cci_np(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20
) -> np.ndarray:
    """
    Commodity Channel Index.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    sma_tp = tp_series.rolling(window=period, min_periods=period).mean()
    mad = tp_series.rolling(window=period, min_periods=period).apply(
        lambda arr: np.mean(np.abs(arr - arr.mean())), raw=True
    )

    cci = np.full(close.shape, np.nan, dtype=np.float64)
    valid = mad != 0
    cci[valid] = (typical_price[valid] - sma_tp[valid]) / (0.015 * mad[valid])
    cci[~valid] = 0.0
    return cci


def ichimoku_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> Dict[str, np.ndarray]:
    """
    Ichimoku Cloud components.
    """
    df = pd.DataFrame(
        {"high": high, "low": low, "close": close}, dtype="float64"
    )

    tenkan_line = (
        df["high"].rolling(window=tenkan, min_periods=tenkan).max()
        + df["low"].rolling(window=tenkan, min_periods=tenkan).min()
    ) / 2.0

    kijun_line = (
        df["high"].rolling(window=kijun, min_periods=kijun).max()
        + df["low"].rolling(window=kijun, min_periods=kijun).min()
    ) / 2.0

    senkou_span_a = ((tenkan_line + kijun_line) / 2.0).shift(displacement)

    senkou_span_b = (
        df["high"].rolling(window=senkou_b, min_periods=senkou_b).max()
        + df["low"].rolling(window=senkou_b, min_periods=senkou_b).min()
    ) / 2.0
    senkou_span_b = senkou_span_b.shift(displacement)

    chikou_span = df["close"].shift(-displacement)

    return {
        "tenkan": tenkan_line.to_numpy(),
        "kijun": kijun_line.to_numpy(),
        "senkou_a": senkou_span_a.to_numpy(),
        "senkou_b": senkou_span_b.to_numpy(),
        "chikou": chikou_span.to_numpy(),
    }


def parabolic_sar_np(
    high: np.ndarray,
    low: np.ndarray,
    step: float = 0.02,
    max_af: float = 0.2,
) -> np.ndarray:
    """
    Parabolic SAR implementation.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    length = len(high)
    if length == 0:
        return np.array([], dtype=np.float64)

    sar = np.zeros(length, dtype=np.float64)
    trend_up = True
    af = step
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, length):
        prev_sar = sar[i - 1]
        if trend_up:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_af)
            if low[i] < sar[i]:
                trend_up = False
                sar[i] = ep
                ep = low[i]
                af = step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[i])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_af)
            if high[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                ep = high[i]
                af = step

    return sar


def adx_np(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> Dict[str, np.ndarray]:
    """
    Average Directional Index along with +DI and -DI.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if len(close) == 0:
        return {"adx": np.array([], dtype=np.float64), "plus_di": np.array([], dtype=np.float64), "minus_di": np.array([], dtype=np.float64)}

    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]

    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0.0), 0.0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0.0), 0.0)
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])

    alpha = 1.0 / period
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    plus_di = 100.0 * np.divide(
        plus_dm_smooth,
        tr_smooth,
        out=np.zeros_like(plus_dm_smooth),
        where=tr_smooth != 0,
    )
    minus_di = 100.0 * np.divide(
        minus_dm_smooth,
        tr_smooth,
        out=np.zeros_like(minus_dm_smooth),
        where=tr_smooth != 0,
    )

    dx = 100.0 * np.divide(
        np.abs(plus_di - minus_di),
        plus_di + minus_di,
        out=np.zeros_like(plus_di),
        where=(plus_di + minus_di) != 0,
    )
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def mfi_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Money Flow Index.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume

    tp_diff = np.diff(typical_price, prepend=typical_price[0])
    positive_flow = np.where(tp_diff >= 0, money_flow, 0.0)
    negative_flow = np.where(tp_diff < 0, money_flow, 0.0)

    pos_sum = pd.Series(positive_flow).rolling(window=period, min_periods=period).sum()
    neg_sum = pd.Series(negative_flow).rolling(window=period, min_periods=period).sum()

    ratio = np.divide(
        pos_sum,
        neg_sum,
        out=np.zeros_like(pos_sum.to_numpy()),
        where=neg_sum.to_numpy() != 0,
    )
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    return mfi.to_numpy()


def volume_oscillator_np(
    volume: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, np.ndarray]:
    """
    Percentage Volume Oscillator (PVO) and signal.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if volume.size == 0:
        empty = np.array([], dtype=np.float64)
        return {"pvo": empty, "signal": empty, "histogram": empty}

    fast_ema = ema_np(volume, fast_period)
    slow_ema = ema_np(volume, slow_period)
    denominator = np.where(slow_ema != 0, slow_ema, np.nan)
    pvo = 100.0 * (fast_ema - slow_ema) / denominator
    signal = ema_np(pvo, signal_period)
    hist = pvo - signal

    return {"pvo": pvo, "signal": signal, "histogram": hist}


def fibonacci_levels_np(
    high: np.ndarray,
    low: np.ndarray,
    ratios: Iterable[float],
    lookback: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Fibonacci retracement levels based on rolling swing high/low.
    """
    high_series = pd.Series(high, dtype="float64")
    low_series = pd.Series(low, dtype="float64")

    if lookback and lookback > 1:
        swing_high = high_series.rolling(window=lookback, min_periods=lookback).max()
        swing_low = low_series.rolling(window=lookback, min_periods=lookback).min()
    else:
        swing_high = high_series.expanding().max()
        swing_low = low_series.expanding().min()

    span = swing_high - swing_low
    levels: Dict[str, np.ndarray] = {}
    for ratio in ratios:
        level = swing_high - span * ratio
        levels[f"fibo_{ratio:.3f}"] = level.to_numpy()
    return levels


def _infer_bars_per_day(index: Iterable[pd.Timestamp]) -> int:
    """Infer approximate number of bars per day from timestamps."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if len(index) < 2:
        return 1
    diffs = index.to_series().diff().dropna()
    median_seconds = diffs.dt.total_seconds().median()
    if median_seconds is None or median_seconds <= 0:
        return 1
    bars = max(int(round(86_400 / median_seconds)), 1)
    return bars


def pivot_points_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    index: Iterable[pd.Timestamp],
    timeframe: str = "daily",
) -> Dict[str, np.ndarray]:
    """
    Classic pivot points computed from the previous period.
    """
    timeframe_map = {"daily": "1D", "weekly": "1W", "monthly": "1M"}
    rule = timeframe_map.get(timeframe.lower(), "1D")

    df = pd.DataFrame({"high": high, "low": low, "close": close}, index=pd.Index(index))
    agg = df.resample(rule, label="right", closed="right").agg(
        {"high": "max", "low": "min", "close": "last"}
    )
    agg = agg.shift(1)  # use previous period values

    pivot = (agg["high"] + agg["low"] + agg["close"]) / 3.0
    r1 = 2 * pivot - agg["low"]
    s1 = 2 * pivot - agg["high"]
    range_ = agg["high"] - agg["low"]
    r2 = pivot + range_
    s2 = pivot - range_
    r3 = agg["high"] + 2 * (pivot - agg["low"])
    s3 = agg["low"] - 2 * (agg["high"] - pivot)

    def _expand(series: pd.Series) -> np.ndarray:
        return series.reindex(df.index, method="ffill").to_numpy()

    return {
        f"pivot_{timeframe.lower()}": _expand(pivot),
        f"pivot_r1_{timeframe.lower()}": _expand(r1),
        f"pivot_s1_{timeframe.lower()}": _expand(s1),
        f"pivot_r2_{timeframe.lower()}": _expand(r2),
        f"pivot_s2_{timeframe.lower()}": _expand(s2),
        f"pivot_r3_{timeframe.lower()}": _expand(r3),
        f"pivot_s3_{timeframe.lower()}": _expand(s3),
    }


def onchain_smoothing_np(
    volume: np.ndarray,
    index: Iterable[pd.Timestamp],
    smoothing_days: int = 7,
) -> np.ndarray:
    """
    On-chain style smoothing (EMA) applied to transaction volume.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if volume.size == 0:
        return np.array([], dtype=np.float64)

    bars_per_day = _infer_bars_per_day(index)
    window = max(int(smoothing_days * bars_per_day), 1)
    return ema_np(volume, window)


def crypto_fear_greed_np(
    close: np.ndarray,
    volume: np.ndarray,
    index: Iterable[pd.Timestamp],
    smoothing_days: int = 7,
) -> np.ndarray:
    """
    Synthetic crypto Fear & Greed index (0-100) derived from price, volume and volatility.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    if close.size == 0:
        return np.array([], dtype=np.float64)

    bars_per_day = _infer_bars_per_day(index)
    window = max(int(smoothing_days * bars_per_day), 1)

    returns = pd.Series(close).pct_change().fillna(0.0)
    momentum = ema_np(returns.to_numpy(), window)
    momentum_score = 50.0 + 50.0 * np.tanh(momentum * 10.0)

    volatility = returns.rolling(window=window, min_periods=2).std().to_numpy()
    vol_norm = volatility / (np.nanmean(volatility) + 1e-12)
    volatility_score = np.clip(100.0 - vol_norm * 50.0, 0.0, 100.0)

    vol_series = pd.Series(volume)
    vol_mean = vol_series.rolling(window=window, min_periods=2).mean()
    vol_std = vol_series.rolling(window=window, min_periods=2).std()
    volume_z = np.divide(
        vol_series - vol_mean,
        vol_std,
        out=np.zeros_like(vol_series.to_numpy()),
        where=vol_std.to_numpy() != 0,
    )
    volume_score = np.clip(50.0 + 10.0 * volume_z.to_numpy(), 0.0, 100.0)

    composite = (
        0.25 * momentum_score + 0.25 * volume_score + 0.25 * volatility_score + 0.25 * 50.0
    )
    return ema_np(composite, window)


def pi_cycle_np(
    close: np.ndarray,
    index: Iterable[pd.Timestamp],
    dma_111_days: int = 111,
    dma_350_days: int = 350,
) -> Dict[str, np.ndarray]:
    """
    Pi Cycle indicator components (111 DMA and 350 DMA * 2).
    """
    close = np.asarray(close, dtype=np.float64)
    bars_per_day = _infer_bars_per_day(index)
    period_111 = max(int(dma_111_days * bars_per_day), 1)
    period_350 = max(int(dma_350_days * bars_per_day), 1)

    dma_111 = sma_np(close, period_111)
    dma_350 = sma_np(close, period_350)
    signal = dma_111 - 2.0 * dma_350

    return {
        "pi_cycle_dma111": dma_111,
        "pi_cycle_dma350x2": 2.0 * dma_350,
        "pi_cycle_signal": signal,
    }


def amplitude_hunter_np(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    index: Iterable[pd.Timestamp],
    bb_period: int = 20,
    bb_std: float = 2.0,
    lookback: int = 20,
    bbwidth_percentile_threshold: float = 50.0,
    volume_zscore_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Amplitude Hunter composite score based on Bollinger Band width percentiles and volume z-score.
    """
    lower, mid, upper, _ = boll_np(close, bb_period, bb_std)
    bbwidth = np.divide(
        upper - lower, mid, out=np.zeros_like(mid), where=np.abs(mid) > 1e-12
    )

    width_series = pd.Series(bbwidth)

    def _percentile_of_last(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        last = valid[-1]
        return (np.sum(valid <= last) / valid.size) * 100.0

    width_percentile = width_series.rolling(
        window=lookback, min_periods=lookback
    ).apply(_percentile_of_last, raw=True)

    vol_series = pd.Series(volume, dtype="float64")
    rolling_mean = vol_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = vol_series.rolling(window=lookback, min_periods=lookback).std()
    volume_z = np.divide(
        vol_series - rolling_mean,
        rolling_std,
        out=np.zeros_like(vol_series.to_numpy()),
        where=rolling_std.to_numpy() != 0,
    )

    percentile_norm = (width_percentile / 100.0).to_numpy()
    volume_z_values = volume_z.to_numpy()

    threshold_pct = bbwidth_percentile_threshold / 100.0
    bbwidth_condition = np.where(percentile_norm <= threshold_pct, 1.0, 0.0)
    volume_condition = np.where(volume_z_values >= volume_zscore_threshold, 1.0, 0.0)
    score = np.clip(
        (1.0 - percentile_norm) * np.maximum(volume_z_values / (volume_zscore_threshold + 1e-9), 0.0),
        0.0,
        1.0,
    )

    return {
        "amplitude_bbwidth": bbwidth,
        "amplitude_width_percentile": percentile_norm,
        "amplitude_volume_zscore": volume_z_values,
        "amplitude_setup": np.minimum(bbwidth_condition, volume_condition),
        "amplitude_score": score,
    }


__all__ = [
    "ema_np",
    "atr_np",
    "boll_np",
    "sma_np",
    "standard_deviation_np",
    "momentum_np",
    "rsi_np",
    "macd_np",
    "vwap_np",
    "obv_np",
    "vortex_df",
    "stochastic_np",
    "cci_np",
    "ichimoku_np",
    "parabolic_sar_np",
    "adx_np",
    "mfi_np",
    "volume_oscillator_np",
    "fibonacci_levels_np",
    "pivot_points_np",
    "onchain_smoothing_np",
    "crypto_fear_greed_np",
    "pi_cycle_np",
    "amplitude_hunter_np",
]



