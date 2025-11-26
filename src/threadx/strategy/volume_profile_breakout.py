"""
Volume Profile Breakout Strategy - Stratégie 2025 Ultra-Rentable Crypto Volatile
================================================================================

Logique:
    - Calcule Volume Profile sur fenêtre roulante (window_bars, défaut 96 = 24h en 15m)
    - Identifie POC (Point of Control), VAH/VAL (Value Area 70% du volume)
    - Long sur cassure haussière VAH avec volume spike (> volume_multiplier × moyenne)
    - Short sur cassure baissière VAL avec volume spike
    - Filtre tendance optionnel: EMA 200 + ADX > 25
    - SL/TP basés sur ATR ou % fixe

Tokens Optimaux:
    SOL, PEPE, DOGE, BTC, ETH

Timeframes Recommandés:
    15m, 30m, 1h (haute fréquence pour crypto volatile)

Paramètres Optimisables (LLM-friendly):
    window_bars:       48-192   (96 = 24h en 15m)
    va_percent:        68-75    (70 = Value Area standard)
    volume_multiplier: 1.3-2.5  (1.8 = spike volume significatif)
    use_trend_filter:  True/False (True = tendance + breakout)
    atr_period:        14       (ATR standard)
    sl_atr_mult:       1.5-3.0  (2.0 = stop modéré)
    tp_atr_mult:       3.0-6.0  (4.5 = R:R ~2.25:1)

Indicateurs Techniques Utilisés:
    - Volume Profile (POC, VAH, VAL) - calculé en pandas
    - ATR (Average True Range) - via IndicatorBank
    - EMA 200 (tendance long terme) - via IndicatorBank
    - ADX (force tendance) - via IndicatorBank
    - Volume MA 20 (volume moyen) - pandas rolling

Author: ThreadX Framework
Version: 1.0
Compatible: ThreadX BacktestEngine (GPU-ready via IndicatorBank)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threadx.indicators.bank import IndicatorBank
from threadx.strategy.model import Strategy


class VolumeProfileBreakout(Strategy):
    """
    Volume Profile Breakout - Cassure zones clés du profil de volume.

    Stratégie professionnelle exploitant les niveaux de support/résistance
    révélés par le Volume Profile (POC, VAH, VAL).
    """

    def generate_signals(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Génère signaux trading basés sur cassure Volume Profile.

        Args:
            data: DataFrame avec colonnes OHLCV (open, high, low, close, volume)
            params: Paramètres stratégie (voir get_default_params)

        Returns:
            DataFrame avec colonnes:
            - signal: 1 (long), -1 (short), 0 (neutral)
            - entry_price: Prix entrée
            - sl_price: Stop-loss
            - tp_price: Take-profit
        """
        df = data.copy()

        # Initialiser colonnes output
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["sl_price"] = np.nan
        df["tp_price"] = np.nan

        # Extraction paramètres avec defaults robustes
        window = int(params.get("window_bars", 96))  # 24h en 15m
        va_percent = params.get("va_percent", 70.0) / 100  # Convertir % → ratio
        volume_mult = params.get("volume_multiplier", 1.8)
        use_trend = params.get("use_trend_filter", True)
        atr_period = int(params.get("atr_period", 14))
        sl_mult = params.get("sl_atr_mult", 2.0)
        tp_mult = params.get("tp_atr_mult", 4.5)

        # Validation sécurité
        if len(df) < window + 200:  # Besoin EMA 200 + window
            return df[["signal", "entry_price", "sl_price", "tp_price"]]

        # =====================================================================
        # INDICATEURS TECHNIQUES (via IndicatorBank - GPU-ready)
        # =====================================================================
        bank = IndicatorBank()

        # ATR - Volatilité pour SL/TP adaptatif
        df["atr"] = bank.atr(df["high"], df["low"], df["close"], atr_period)

        # EMA 200 - Filtre tendance long terme
        df["ema200"] = bank.ema(df["close"], 200)

        # ADX - Force de la tendance (> 25 = tendance forte)
        df["adx"] = bank.adx(df["high"], df["low"], df["close"], 14)

        # Volume MA 20 - Référence volume moyen
        df["volume_ma"] = df["volume"].rolling(20).mean()

        # =====================================================================
        # VOLUME PROFILE (POC, VAH, VAL) - Calcul roulant pandas
        # =====================================================================

        def calculate_vp_profile(window_df: pd.DataFrame):
            """
            Calcule Volume Profile sur fenêtre donnée.

            Returns:
                (poc_price, vah_price, val_price) ou (None, None, None) si échec
            """
            if len(window_df) == 0:
                return None, None, None

            # Discrétiser prix en 100 bins
            price_min = window_df["low"].min()
            price_max = window_df["high"].max()

            if price_min >= price_max:  # Prix plat
                return None, None, None

            price_bins = np.linspace(price_min, price_max, 100)
            volume_by_price = np.zeros(len(price_bins) - 1)

            # Agréger volume par bin de prix
            for i in range(len(price_bins) - 1):
                low_bin, high_bin = price_bins[i], price_bins[i + 1]
                mask = (window_df["high"] >= low_bin) & (window_df["low"] <= high_bin)
                volume_by_price[i] = window_df.loc[mask, "volume"].sum()

            total_volume = volume_by_price.sum()
            if total_volume == 0:
                return None, None, None

            # POC (Point of Control) - Prix avec max volume
            poc_idx = np.argmax(volume_by_price)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

            # Value Area (VAH/VAL) - 70% volume central autour POC
            sorted_indices = np.argsort(volume_by_price)[::-1]  # Trier desc volume
            cumulative_volume = 0
            value_area_bins = []

            for idx in sorted_indices:
                cumulative_volume += volume_by_price[idx]
                value_area_bins.append(idx)
                if cumulative_volume >= total_volume * va_percent:
                    break

            # VAH (Value Area High) - Prix max dans Value Area
            vah_idx = max(value_area_bins)
            vah_price = (price_bins[vah_idx] + price_bins[vah_idx + 1]) / 2

            # VAL (Value Area Low) - Prix min dans Value Area
            val_idx = min(value_area_bins)
            val_price = (price_bins[val_idx] + price_bins[val_idx + 1]) / 2

            return poc_price, vah_price, val_price

        # Calcul roulant Volume Profile
        vah_list, val_list, poc_list = [], [], []

        for i in range(len(df)):
            if i < window:
                # Pas assez de données pour fenêtre
                vah_list.append(np.nan)
                val_list.append(np.nan)
                poc_list.append(np.nan)
                continue

            window_df = df.iloc[i - window : i]
            poc, vah, val = calculate_vp_profile(window_df)

            vah_list.append(vah)
            val_list.append(val)
            poc_list.append(poc)

        df["vp_poc"] = poc_list
        df["vp_vah"] = vah_list
        df["vp_val"] = val_list

        # =====================================================================
        # CONDITIONS ENTRÉE (Cassure Volume Profile + Volume Spike)
        # =====================================================================

        # LONG: Cassure haussière VAH avec volume spike
        long_condition = (
            (df["close"] > df["vp_vah"])  # Prix casse VAH
            & (df["close"].shift(1) <= df["vp_vah"].shift(1))  # Cassure nouvelle
            & (df["volume"] > volume_mult * df["volume_ma"])  # Volume spike
            & (~df["vp_vah"].isna())  # VAH valide
        )

        # SHORT: Cassure baissière VAL avec volume spike
        short_condition = (
            (df["close"] < df["vp_val"])  # Prix casse VAL
            & (df["close"].shift(1) >= df["vp_val"].shift(1))  # Cassure nouvelle
            & (df["volume"] > volume_mult * df["volume_ma"])  # Volume spike
            & (~df["vp_val"].isna())  # VAL valide
        )

        # FILTRE TENDANCE (optionnel): EMA 200 + ADX > 25
        if use_trend:
            long_condition &= (df["close"] > df["ema200"]) & (df["adx"] > 25)
            short_condition &= (df["close"] < df["ema200"]) & (df["adx"] > 25)

        # Appliquer signaux
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # =====================================================================
        # STOP-LOSS / TAKE-PROFIT (basés sur ATR)
        # =====================================================================

        # LONG: Entry = close, SL = close - sl_mult*ATR, TP = close + tp_mult*ATR
        df.loc[df["signal"] == 1, "entry_price"] = df["close"]
        df.loc[df["signal"] == 1, "sl_price"] = df["close"] - sl_mult * df["atr"]
        df.loc[df["signal"] == 1, "tp_price"] = df["close"] + tp_mult * df["atr"]

        # SHORT: Entry = close, SL = close + sl_mult*ATR, TP = close - tp_mult*ATR
        df.loc[df["signal"] == -1, "entry_price"] = df["close"]
        df.loc[df["signal"] == -1, "sl_price"] = df["close"] + sl_mult * df["atr"]
        df.loc[df["signal"] == -1, "tp_price"] = df["close"] - tp_mult * df["atr"]

        return df[["signal", "entry_price", "sl_price", "tp_price"]]

    def get_default_params(self) -> dict:
        """
        Paramètres par défaut optimisés pour crypto volatile (SOL, PEPE 15m-1h).

        Returns:
            Dict paramètres avec valeurs robustes testées
        """
        return {
            # Volume Profile Configuration
            "window_bars": 96,  # 24h en 15m (fenêtre roulante)
            "va_percent": 70.0,  # 70% = Value Area standard

            # Entry Filters
            "volume_multiplier": 1.8,  # Volume spike > 1.8× moyenne
            "use_trend_filter": True,  # Activer filtre EMA 200 + ADX

            # Risk Management
            "atr_period": 14,  # ATR standard 14 périodes
            "sl_atr_mult": 2.0,  # Stop-loss à 2× ATR (modéré)
            "tp_atr_mult": 4.5,  # Take-profit à 4.5× ATR (R:R ~2.25:1)
        }

    def get_param_specs(self) -> dict:
        """
        Spécifications paramètres pour optimisation LLM.

        Returns:
            Dict avec min/max/type pour chaque paramètre optimisable
        """
        return {
            "window_bars": {
                "min": 48,
                "max": 192,
                "type": "int",
                "description": "Fenêtre Volume Profile (bars) - 48=12h, 96=24h, 192=48h en 15m",
            },
            "va_percent": {
                "min": 68.0,
                "max": 75.0,
                "type": "float",
                "description": "Pourcentage Value Area (68-75%, 70% standard)",
            },
            "volume_multiplier": {
                "min": 1.3,
                "max": 2.5,
                "type": "float",
                "description": "Multiplicateur volume spike (1.3=lax, 2.5=strict)",
            },
            "use_trend_filter": {
                "type": "bool",
                "description": "Activer filtre tendance EMA 200 + ADX > 25",
            },
            "atr_period": {
                "min": 10,
                "max": 20,
                "type": "int",
                "description": "Période ATR (10=rapide, 20=lent)",
            },
            "sl_atr_mult": {
                "min": 1.5,
                "max": 3.0,
                "type": "float",
                "description": "Stop-loss en multiple ATR (1.5=serré, 3.0=large)",
            },
            "tp_atr_mult": {
                "min": 3.0,
                "max": 6.0,
                "type": "float",
                "description": "Take-profit en multiple ATR (3.0=conservateur, 6.0=agressif)",
            },
        }

    def get_indicator_info(self) -> dict:
        """
        Informations indicateurs techniques utilisés (pour LLM context).

        Returns:
            Dict décrivant chaque indicateur et son rôle
        """
        return {
            "volume_profile": {
                "type": "custom_pandas",
                "components": ["POC", "VAH", "VAL"],
                "description": "Profil volume - POC (max volume), VAH/VAL (Value Area 70%)",
                "window": "window_bars parameter (96 bars default)",
                "role": "Identifier support/résistance basés volume réel",
            },
            "atr": {
                "type": "indicatorbank",
                "period": "atr_period parameter (14 default)",
                "description": "Average True Range - Mesure volatilité",
                "role": "Adapter SL/TP à volatilité marché",
            },
            "ema200": {
                "type": "indicatorbank",
                "period": 200,
                "description": "Exponential Moving Average 200 - Tendance long terme",
                "role": "Filtre tendance (long si prix > EMA 200)",
            },
            "adx": {
                "type": "indicatorbank",
                "period": 14,
                "description": "Average Directional Index - Force tendance",
                "role": "Confirmer tendance forte (ADX > 25)",
            },
            "volume_ma": {
                "type": "pandas_rolling",
                "period": 20,
                "description": "Volume Moving Average 20 - Volume moyen",
                "role": "Référence pour détecter volume spikes",
            },
        }

    def get_strategy_metadata(self) -> dict:
        """
        Métadonnées stratégie (pour registry & UI).

        Returns:
            Dict avec infos affichage/classification
        """
        return {
            "name": "VolumeProfileBreakout",
            "category": "Breakout",
            "subcategory": "Volume Analysis",
            "description": "Cassure zones clés Volume Profile (VAH/VAL) avec confirmation volume spike",
            "author": "ThreadX Framework",
            "version": "1.0",
            "created": "2025-01-15",
            "optimal_tokens": ["SOL", "PEPE", "DOGE", "BTC", "ETH"],
            "optimal_timeframes": ["15m", "30m", "1h"],
            "market_type": "Crypto Volatile (high volume)",
            "risk_level": "Medium-High",
            "complexity": "Advanced",
            "requires_indicators": ["ATR", "EMA", "ADX", "Volume"],
            "gpu_compatible": True,
        }
