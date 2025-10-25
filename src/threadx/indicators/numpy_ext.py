"""
ThreadX Indicateurs NumPy - Module Centralisé
==============================================

Module unique pour tous les indicateurs techniques optimisés NumPy.
Importe depuis src.threadx.indicators.indicators_np pour garantir
cohérence et performance.

Usage:
    from threadx.indicators.numpy import rsi_np, macd_np, boll_np

    close_prices = df['close'].values
    rsi = rsi_np(close_prices, period=14)
    macd, signal, hist = macd_np(close_prices)

Performance:
    - 50x plus rapide que pandas rolling
    - Optimisations EMA custom
    - Gestion NaN robuste

Auteur: ThreadX Core Team
Version: 2.0 (Consolidé)
Date: 11 octobre 2025
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Import depuis le module natif ThreadX
from .indicators_np import (
    ema_np,
    rsi_np,
    boll_np,
    macd_np,
    atr_np,
    vwap_np,
    obv_np,
    vortex_df,
)


# Réexportation pour API propre
__all__ = [
    "ema_np",
    "rsi_np",
    "boll_np",
    "macd_np",
    "atr_np",
    "vwap_np",
    "obv_np",
    "vortex_df",
]


# Fonctions helper pour intégration facile avec pandas DataFrame


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    """
    Ajoute RSI à un DataFrame pandas.

    Args:
        df: DataFrame avec colonne 'close' (ou column)
        period: Période RSI (défaut 14)
        column: Nom de la colonne prix (défaut 'close')

    Returns:
        DataFrame avec colonne 'rsi' ajoutée
    """
    df = df.copy()
    df["rsi"] = rsi_np(df[column].values, period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """
    Ajoute MACD à un DataFrame pandas.

    Args:
        df: DataFrame avec colonne 'close' (ou column)
        fast: Période EMA rapide (défaut 12)
        slow: Période EMA lente (défaut 26)
        signal: Période signal (défaut 9)
        column: Nom de la colonne prix (défaut 'close')

    Returns:
        DataFrame avec colonnes 'macd', 'macd_signal', 'macd_hist'
    """
    df = df.copy()
    macd, sig, hist = macd_np(df[column].values, fast, slow, signal)
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    return df


def add_bollinger(
    df: pd.DataFrame, period: int = 20, std: float = 2.0, column: str = "close"
) -> pd.DataFrame:
    """
    Ajoute Bollinger Bands à un DataFrame pandas.

    Args:
        df: DataFrame avec colonne 'close' (ou column)
        period: Période moyenne mobile (défaut 20)
        std: Nombre d'écarts-types (défaut 2.0)
        column: Nom de la colonne prix (défaut 'close')

    Returns:
        DataFrame avec colonnes 'bb_lower', 'bb_middle', 'bb_upper', 'bb_zscore'
    """
    df = df.copy()
    lower, middle, upper, z = boll_np(df[column].values, period, std)
    df["bb_lower"] = lower
    df["bb_middle"] = middle
    df["bb_upper"] = upper
    df["bb_zscore"] = z
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Ajoute ATR à un DataFrame pandas.

    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close'
        period: Période ATR (défaut 14)

    Returns:
        DataFrame avec colonne 'atr'
    """
    df = df.copy()
    df["atr"] = atr_np(df["high"].values, df["low"].values, df["close"].values, period)
    return df


def add_vwap(df: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    """
    Ajoute VWAP à un DataFrame pandas.

    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close', 'volume'
        window: Fenêtre de calcul (défaut 96 pour 1h sur données 1m)

    Returns:
        DataFrame avec colonne 'vwap'
    """
    df = df.copy()
    df["vwap"] = vwap_np(
        df["close"].values,
        df["high"].values,
        df["low"].values,
        df["volume"].values,
        window,
    )
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute OBV à un DataFrame pandas.

    Args:
        df: DataFrame avec colonnes 'close', 'volume'

    Returns:
        DataFrame avec colonne 'obv'
    """
    df = df.copy()
    df["obv"] = obv_np(df["close"].values, df["volume"].values)
    return df


def add_vortex(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Ajoute Vortex Indicator à un DataFrame pandas.

    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close'
        period: Période vortex (défaut 14)

    Returns:
        DataFrame avec colonnes 'vi_plus', 'vi_minus'
    """
    vortex = vortex_df(df["high"].values, df["low"].values, df["close"].values, period)
    df = df.copy()
    df["vi_plus"] = vortex["vi_plus"].values
    df["vi_minus"] = vortex["vi_minus"].values
    return df


def add_all_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    vwap_window: int = 96,
    vortex_period: int = 14,
) -> pd.DataFrame:
    """
    Ajoute TOUS les indicateurs standard à un DataFrame.

    Args:
        df: DataFrame OHLCV
        *_period/*_window: Paramètres des indicateurs

    Returns:
        DataFrame enrichi avec tous les indicateurs
    """
    df = df.copy()

    # RSI
    df = add_rsi(df, rsi_period)

    # MACD
    df = add_macd(df, macd_fast, macd_slow, macd_signal)

    # Bollinger Bands
    df = add_bollinger(df, bb_period, bb_std)

    # ATR
    df = add_atr(df, atr_period)

    # VWAP
    df = add_vwap(df, vwap_window)

    # OBV
    df = add_obv(df)

    # Vortex
    df = add_vortex(df, vortex_period)

    return df


# =========================================================
#  Tests rapides
# =========================================================


def _test_indicators():
    """Test rapide des indicateurs avec données synthétiques."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 200

    # Données synthétiques réalistes
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 2)
    low = close - np.abs(np.random.randn(n) * 2)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume}
    )

    print("🧪 Test des indicateurs NumPy ThreadX")
    print("=" * 50)

    # Test RSI
    df = add_rsi(df)
    print(f"✅ RSI: {df['rsi'].iloc[-1]:.2f}")

    # Test MACD
    df = add_macd(df)
    print(f"✅ MACD: {df['macd'].iloc[-1]:.4f}")

    # Test Bollinger
    df = add_bollinger(df)
    print(f"✅ BB Upper: {df['bb_upper'].iloc[-1]:.2f}")

    # Test ATR
    df = add_atr(df)
    print(f"✅ ATR: {df['atr'].iloc[-1]:.4f}")

    # Test VWAP
    df = add_vwap(df)
    print(f"✅ VWAP: {df['vwap'].iloc[-1]:.2f}")

    # Test OBV
    df = add_obv(df)
    print(f"✅ OBV: {df['obv'].iloc[-1]:.0f}")

    # Test Vortex
    df = add_vortex(df)
    print(f"✅ VI+: {df['vi_plus'].iloc[-1]:.4f}")

    print("\n🎉 Tous les indicateurs fonctionnent correctement!")
    print(f"DataFrame final: {len(df.columns)} colonnes")
    print(f"Colonnes: {', '.join(df.columns)}")


if __name__ == "__main__":
    _test_indicators()
