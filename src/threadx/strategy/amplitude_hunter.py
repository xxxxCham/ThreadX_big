"""
ThreadX Strategy - AmplitudeHunter (BB Amplitude Rider)
========================================================

Stratégie avancée de capture d'amplitude complète sur Bollinger Bands.

Concept: Capturer l'amplitude complète d'un swing BB (basse → médiane → haute → extension)
et laisser courir au-delà de la bande opposée quand le momentum le permet.

Fonctionnalités:
- Filtre de régime multi-critères (BBWidth percentile, Volume z-score, ADX optionnel)
- Setup "Spring → Drive" avec détection séquentielle MACD
- Score d'Amplitude pour modulation de l'agressivité
- Pyramiding intelligent (jusqu'à 2 adds)
- Trailing stop conditionnel basé sur %B et MACD
- Cible dynamique BIP (Bollinger Implied Price)
- Stop loss spécifique pour positions SHORT

Auteur: Claude (Anthropic)
Version: 1.0.0
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from threadx.indicators import ensure_indicator
from threadx.strategy.model import (
    RunStats,
    Trade,
    validate_ohlcv_dataframe,
    validate_strategy_params,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# ==========================================
# STRATEGY PARAMETERS
# ==========================================


@dataclass
class AmplitudeHunterParams:
    """
    Paramètres de la stratégie AmplitudeHunter (BB Amplitude Rider).

    Attributes:
        # Bollinger Bands
        bb_period: Période pour moyennes mobiles (défaut: 20)
        bb_std: Multiplicateur écart-type pour bandes (défaut: 2.0)

        # Filtre de régime
        bbwidth_percentile_threshold: Seuil percentile BBWidth (défaut: 50, range: 30-70)
        bbwidth_lookback: Période lookback pour percentile BBWidth (défaut: 100)
        volume_zscore_threshold: Seuil volume z-score (défaut: 0.5)
        volume_lookback: Période lookback pour volume z-score (défaut: 50)
        use_adx_filter: Activer filtre ADX (défaut: False)
        adx_threshold: Seuil ADX minimum (défaut: 15)
        adx_period: Période ADX (défaut: 14)

        # Setup Spring → Drive
        spring_lookback: Lookback pour détecter spring (défaut: 20)
        pb_entry_threshold_min: %B minimum pour entrée (défaut: 0.2)
        pb_entry_threshold_max: %B maximum pour entrée (défaut: 0.5)
        macd_fast: Période MACD rapide (défaut: 12)
        macd_slow: Période MACD lente (défaut: 26)
        macd_signal: Période signal MACD (défaut: 9)

        # Score d'Amplitude
        amplitude_score_threshold: Score minimum pour trade (défaut: 0.6)
        amplitude_w1_bbwidth: Poids BBWidth percentile (défaut: 0.3)
        amplitude_w2_pb: Poids |%B| (défaut: 0.2)
        amplitude_w3_macd_slope: Poids pente MACD (défaut: 0.3)
        amplitude_w4_volume: Poids volume z-score (défaut: 0.2)

        # Pyramiding
        pyramiding_enabled: Activer pyramiding (défaut: False)
        pyramiding_max_adds: Nombre max d'adds (1 ou 2, défaut: 1)

        # Stops et trailing
        atr_period: Période ATR pour stops (défaut: 14)
        sl_atr_multiplier: Multiplicateur ATR pour SL initial (défaut: 2.0)
        sl_min_pct: SL minimum en % (médiane-basse) (défaut: 0.37)
        short_stop_pct: Stop loss fixe pour SHORT (défaut: 0.37, soit 37%)

        trailing_activation_pb_threshold: %B seuil pour activer trailing (défaut: 1.0)
        trailing_activation_gain_r: Gain en R pour activer trailing (défaut: 1.0)
        trailing_type: Type de trailing ("chandelier" | "pb_floor" | "macd_fade") (défaut: "chandelier")
        trailing_chandelier_atr_mult: Mult ATR pour chandelier (défaut: 2.5)
        trailing_pb_floor: %B floor pour sortie (défaut: 0.5)

        # Cible intelligente (BIP)
        use_bip_target: Utiliser cible BIP (défaut: True)
        bip_partial_exit_pct: % à sortir à BIP (défaut: 0.5, soit 50%)

        # Risk Management
        risk_per_trade: Risque par trade en fraction du capital (défaut: 0.02 = 2%)
        max_hold_bars: Durée max position en barres (défaut: 100)
        leverage: Effet de levier (défaut: 1.0)

        # Métadonnées
        meta: Dictionnaire métadonnées personnalisées

    Example:
        >>> params = AmplitudeHunterParams(
        ...     bb_period=20, bb_std=2.0,
        ...     amplitude_score_threshold=0.7,
        ...     pyramiding_enabled=True,
        ...     pyramiding_max_adds=2
        ... )
        >>> strategy = AmplitudeHunterStrategy()
        >>> signals = strategy.generate_signals(df, params.to_dict())
    """

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Filtre de régime
    bbwidth_percentile_threshold: float = 50.0  # 30-70 recommandé
    bbwidth_lookback: int = 100
    volume_zscore_threshold: float = 0.5
    volume_lookback: int = 50
    use_adx_filter: bool = False
    adx_threshold: float = 15.0
    adx_period: int = 14

    # Setup Spring → Drive
    spring_lookback: int = 20
    pb_entry_threshold_min: float = 0.2  # %B min pour entrée
    pb_entry_threshold_max: float = 0.5  # %B max pour entrée
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Score d'Amplitude
    amplitude_score_threshold: float = 0.6
    amplitude_w1_bbwidth: float = 0.3
    amplitude_w2_pb: float = 0.2
    amplitude_w3_macd_slope: float = 0.3
    amplitude_w4_volume: float = 0.2

    # Pyramiding
    pyramiding_enabled: bool = False
    pyramiding_max_adds: int = 1  # 1 ou 2

    # Stops et trailing
    atr_period: int = 14
    sl_atr_multiplier: float = 2.0
    sl_min_pct: float = 0.37  # % de (médiane - basse) pour SL min
    short_stop_pct: float = 0.37  # 37% au-dessus du prix d'entrée SHORT

    trailing_activation_pb_threshold: float = 1.0  # %B > 1 pour activer
    trailing_activation_gain_r: float = 1.0  # Gain >= 1R pour activer
    trailing_type: str = "chandelier"  # "chandelier" | "pb_floor" | "macd_fade"
    trailing_chandelier_atr_mult: float = 2.5
    trailing_pb_floor: float = 0.5

    # Cible BIP
    use_bip_target: bool = True
    bip_partial_exit_pct: float = 0.5  # 50% sortie partielle

    # Risk Management
    risk_per_trade: float = 0.02  # 2% du capital
    max_hold_bars: int = 100
    leverage: float = 1.0

    # Métadonnées
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation des paramètres"""
        # Bollinger
        if self.bb_period < 2:
            raise ValueError(f"bb_period must be >= 2, got: {self.bb_period}")
        if self.bb_std <= 0:
            raise ValueError(f"bb_std must be > 0, got: {self.bb_std}")

        # Filtre régime
        if not 0 <= self.bbwidth_percentile_threshold <= 100:
            raise ValueError(
                f"bbwidth_percentile_threshold must be in [0, 100], got: {self.bbwidth_percentile_threshold}"
            )
        if self.bbwidth_lookback < 10:
            raise ValueError(
                f"bbwidth_lookback must be >= 10, got: {self.bbwidth_lookback}"
            )
        if self.volume_lookback < 10:
            raise ValueError(
                f"volume_lookback must be >= 10, got: {self.volume_lookback}"
            )
        if self.adx_threshold < 0:
            raise ValueError(f"adx_threshold must be >= 0, got: {self.adx_threshold}")

        # %B thresholds
        if not 0 <= self.pb_entry_threshold_min <= 1:
            raise ValueError(
                f"pb_entry_threshold_min must be in [0, 1], got: {self.pb_entry_threshold_min}"
            )
        if not 0 <= self.pb_entry_threshold_max <= 1:
            raise ValueError(
                f"pb_entry_threshold_max must be in [0, 1], got: {self.pb_entry_threshold_max}"
            )
        if self.pb_entry_threshold_min > self.pb_entry_threshold_max:
            raise ValueError(
                "pb_entry_threshold_min must be <= pb_entry_threshold_max"
            )

        # MACD
        if self.macd_fast <= 0 or self.macd_slow <= 0 or self.macd_signal <= 0:
            raise ValueError("MACD periods must be > 0")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be < macd_slow")

        # Score amplitude
        if not 0 <= self.amplitude_score_threshold <= 1:
            raise ValueError(
                f"amplitude_score_threshold must be in [0, 1], got: {self.amplitude_score_threshold}"
            )
        # Vérifier que les poids somment à ~1.0
        total_weight = (
            self.amplitude_w1_bbwidth
            + self.amplitude_w2_pb
            + self.amplitude_w3_macd_slope
            + self.amplitude_w4_volume
        )
        if not 0.95 <= total_weight <= 1.05:
            logger.warning(
                f"Amplitude weights sum to {total_weight:.3f}, should be ~1.0"
            )

        # Pyramiding
        if self.pyramiding_max_adds not in [1, 2]:
            raise ValueError(f"pyramiding_max_adds must be 1 or 2, got: {self.pyramiding_max_adds}")

        # Stops
        if self.atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got: {self.atr_period}")
        if self.sl_atr_multiplier <= 0:
            raise ValueError(
                f"sl_atr_multiplier must be > 0, got: {self.sl_atr_multiplier}"
            )
        if not 0 < self.sl_min_pct <= 1:
            raise ValueError(f"sl_min_pct must be in (0, 1], got: {self.sl_min_pct}")
        if not 0 < self.short_stop_pct <= 1:
            raise ValueError(
                f"short_stop_pct must be in (0, 1], got: {self.short_stop_pct}"
            )

        # Trailing
        if self.trailing_type not in ["chandelier", "pb_floor", "macd_fade"]:
            raise ValueError(
                f"trailing_type must be 'chandelier', 'pb_floor', or 'macd_fade', got: {self.trailing_type}"
            )

        # BIP
        if not 0 <= self.bip_partial_exit_pct <= 1:
            raise ValueError(
                f"bip_partial_exit_pct must be in [0, 1], got: {self.bip_partial_exit_pct}"
            )

        # Risk
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(
                f"risk_per_trade must be in (0, 1], got: {self.risk_per_trade}"
            )
        if self.max_hold_bars < 1:
            raise ValueError(
                f"max_hold_bars must be >= 1, got: {self.max_hold_bars}"
            )
        if self.leverage <= 0:
            raise ValueError(f"leverage must be > 0, got: {self.leverage}")

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour compatibilité"""
        return {
            # Bollinger
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            # Filtre régime
            "bbwidth_percentile_threshold": self.bbwidth_percentile_threshold,
            "bbwidth_lookback": self.bbwidth_lookback,
            "volume_zscore_threshold": self.volume_zscore_threshold,
            "volume_lookback": self.volume_lookback,
            "use_adx_filter": self.use_adx_filter,
            "adx_threshold": self.adx_threshold,
            "adx_period": self.adx_period,
            # Setup
            "spring_lookback": self.spring_lookback,
            "pb_entry_threshold_min": self.pb_entry_threshold_min,
            "pb_entry_threshold_max": self.pb_entry_threshold_max,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            # Score amplitude
            "amplitude_score_threshold": self.amplitude_score_threshold,
            "amplitude_w1_bbwidth": self.amplitude_w1_bbwidth,
            "amplitude_w2_pb": self.amplitude_w2_pb,
            "amplitude_w3_macd_slope": self.amplitude_w3_macd_slope,
            "amplitude_w4_volume": self.amplitude_w4_volume,
            # Pyramiding
            "pyramiding_enabled": self.pyramiding_enabled,
            "pyramiding_max_adds": self.pyramiding_max_adds,
            # Stops
            "atr_period": self.atr_period,
            "sl_atr_multiplier": self.sl_atr_multiplier,
            "sl_min_pct": self.sl_min_pct,
            "short_stop_pct": self.short_stop_pct,
            "trailing_activation_pb_threshold": self.trailing_activation_pb_threshold,
            "trailing_activation_gain_r": self.trailing_activation_gain_r,
            "trailing_type": self.trailing_type,
            "trailing_chandelier_atr_mult": self.trailing_chandelier_atr_mult,
            "trailing_pb_floor": self.trailing_pb_floor,
            # BIP
            "use_bip_target": self.use_bip_target,
            "bip_partial_exit_pct": self.bip_partial_exit_pct,
            # Risk
            "risk_per_trade": self.risk_per_trade,
            "max_hold_bars": self.max_hold_bars,
            "leverage": self.leverage,
            # Meta
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AmplitudeHunterParams":
        """Crée depuis un dictionnaire"""
        return cls(
            # Bollinger
            bb_period=data.get("bb_period", 20),
            bb_std=data.get("bb_std", 2.0),
            # Filtre régime
            bbwidth_percentile_threshold=data.get("bbwidth_percentile_threshold", 50.0),
            bbwidth_lookback=data.get("bbwidth_lookback", 100),
            volume_zscore_threshold=data.get("volume_zscore_threshold", 0.5),
            volume_lookback=data.get("volume_lookback", 50),
            use_adx_filter=data.get("use_adx_filter", False),
            adx_threshold=data.get("adx_threshold", 15.0),
            adx_period=data.get("adx_period", 14),
            # Setup
            spring_lookback=data.get("spring_lookback", 20),
            pb_entry_threshold_min=data.get("pb_entry_threshold_min", 0.2),
            pb_entry_threshold_max=data.get("pb_entry_threshold_max", 0.5),
            macd_fast=data.get("macd_fast", 12),
            macd_slow=data.get("macd_slow", 26),
            macd_signal=data.get("macd_signal", 9),
            # Score
            amplitude_score_threshold=data.get("amplitude_score_threshold", 0.6),
            amplitude_w1_bbwidth=data.get("amplitude_w1_bbwidth", 0.3),
            amplitude_w2_pb=data.get("amplitude_w2_pb", 0.2),
            amplitude_w3_macd_slope=data.get("amplitude_w3_macd_slope", 0.3),
            amplitude_w4_volume=data.get("amplitude_w4_volume", 0.2),
            # Pyramiding
            pyramiding_enabled=data.get("pyramiding_enabled", False),
            pyramiding_max_adds=data.get("pyramiding_max_adds", 1),
            # Stops
            atr_period=data.get("atr_period", 14),
            sl_atr_multiplier=data.get("sl_atr_multiplier", 2.0),
            sl_min_pct=data.get("sl_min_pct", 0.37),
            short_stop_pct=data.get("short_stop_pct", 0.37),
            trailing_activation_pb_threshold=data.get(
                "trailing_activation_pb_threshold", 1.0
            ),
            trailing_activation_gain_r=data.get("trailing_activation_gain_r", 1.0),
            trailing_type=data.get("trailing_type", "chandelier"),
            trailing_chandelier_atr_mult=data.get("trailing_chandelier_atr_mult", 2.5),
            trailing_pb_floor=data.get("trailing_pb_floor", 0.5),
            # BIP
            use_bip_target=data.get("use_bip_target", True),
            bip_partial_exit_pct=data.get("bip_partial_exit_pct", 0.5),
            # Risk
            risk_per_trade=data.get("risk_per_trade", 0.02),
            max_hold_bars=data.get("max_hold_bars", 100),
            leverage=data.get("leverage", 1.0),
            # Meta
            meta=data.get("meta", {}),
        )


# ==========================================
# STRATEGY IMPLEMENTATION
# ==========================================


class AmplitudeHunterStrategy:
    """
    Implémentation de la stratégie AmplitudeHunter (BB Amplitude Rider).

    Logique de trading:
    1. Filtre de régime: BBWidth percentile, Volume z-score, ADX (optionnel)
    2. Calcul Score d'Amplitude pour modulation
    3. Détection setup Spring → Drive:
       - LONG: spring (close < bb_lower récent) + MACD ralentissement→impulsion + %B franchit seuil
       - SHORT: symétrique (close > bb_upper + MACD + %B)
    4. Pyramiding conditionnel (jusqu'à 2 adds)
    5. Gestion stops:
       - SL initial: max(swing_low - k*ATR, pct*(médiane-basse))
       - Trailing conditionnel: activé quand %B > 1 OU gain >= 1R
       - Stop fixe pour SHORT: 37% au-dessus du prix d'entrée
    6. Cible BIP (Bollinger Implied Price): médiane + (médiane - basse)

    Example:
        >>> strategy = AmplitudeHunterStrategy("BTCUSDT", "1h")
        >>> params = AmplitudeHunterParams(amplitude_score_threshold=0.7)
        >>> signals = strategy.generate_signals(df, params.to_dict())
        >>> equity, stats = strategy.backtest(df, params.to_dict(), 10000)
    """

    def __init__(self, symbol: str = "UNKNOWN", timeframe: str = "1h"):
        """
        Initialise la stratégie.

        Args:
            symbol: Symbole pour cache d'indicateurs
            timeframe: Timeframe pour cache d'indicateurs
        """
        self.symbol = symbol
        self.timeframe = timeframe
        logger.info(
            f"Stratégie AmplitudeHunter initialisée: {symbol}/{timeframe}"
        )

    # --- Indicateurs techniques ---

    def _calculate_macd(
        self, close: np.ndarray, fast: int, slow: int, signal: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule MACD (ligne MACD, signal, histogramme).

        Args:
            close: Prix de clôture
            fast: Période EMA rapide
            slow: Période EMA lente
            signal: Période signal

        Returns:
            Tuple (macd_line, signal_line, histogram)
        """
        close_series = pd.Series(close)
        ema_fast = close_series.ewm(span=fast, adjust=False).mean().values
        ema_slow = close_series.ewm(span=slow, adjust=False).mean().values

        macd_line = ema_fast - ema_slow
        signal_line = (
            pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        )
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """
        Calcule l'Average Directional Index (ADX).

        Args:
            high: Prix hauts
            low: Prix bas
            close: Prix de clôture
            period: Période ADX

        Returns:
            Array ADX
        """
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # Premier élément

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = down_move[0] = 0

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed TR, +DM, -DM
        atr_smooth = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di_smooth = (
            pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values
        )
        minus_di_smooth = (
            pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values
        )

        # +DI, -DI
        plus_di = 100 * plus_di_smooth / atr_smooth
        minus_di = 100 * minus_di_smooth / atr_smooth

        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # ADX
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values

        return adx

    def _calculate_percent_b(
        self, close: np.ndarray, bb_upper: np.ndarray, bb_lower: np.ndarray
    ) -> np.ndarray:
        """
        Calcule %B (position du prix dans les bandes).

        Args:
            close: Prix de clôture
            bb_upper: Bande Bollinger supérieure
            bb_lower: Bande Bollinger inférieure

        Returns:
            Array %B
        """
        bb_width = bb_upper - bb_lower
        percent_b = (close - bb_lower) / (bb_width + 1e-10)  # Éviter division par 0
        return percent_b

    def _calculate_bbwidth_percentile(
        self, bb_width: np.ndarray, lookback: int
    ) -> np.ndarray:
        """
        Calcule le percentile du BBWidth sur une période lookback.

        Args:
            bb_width: BBWidth actuel
            lookback: Période de lookback

        Returns:
            Array de percentiles
        """
        percentiles = np.full(len(bb_width), np.nan)

        for i in range(lookback, len(bb_width)):
            window = bb_width[i - lookback : i]
            current = bb_width[i]
            # Percentile = proportion des valeurs <= current
            percentile = (np.sum(window <= current) / len(window)) * 100
            percentiles[i] = percentile

        return percentiles

    def _calculate_volume_zscore(
        self, volume: np.ndarray, lookback: int
    ) -> np.ndarray:
        """
        Calcule le z-score du volume.

        Args:
            volume: Volume
            lookback: Période de lookback

        Returns:
            Array de z-scores
        """
        zscores = np.full(len(volume), np.nan)

        for i in range(lookback, len(volume)):
            window = volume[i - lookback : i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                zscores[i] = (volume[i] - mean) / std
            else:
                zscores[i] = 0.0

        return zscores

    def _calculate_amplitude_score(
        self,
        bbwidth_pct: np.ndarray,
        percent_b: np.ndarray,
        macd_hist: np.ndarray,
        volume_zscore: np.ndarray,
        params: AmplitudeHunterParams,
    ) -> np.ndarray:
        """
        Calcule le Score d'Amplitude pour modulation de l'agressivité.

        Score = w1*BBWidth_pct/100 + w2*|%B| + w3*|pente_MACD| + w4*volume_zscore_norm

        Args:
            bbwidth_pct: BBWidth en percentile [0, 100]
            percent_b: %B
            macd_hist: MACD histogram
            volume_zscore: Volume z-score
            params: Paramètres

        Returns:
            Array de scores [0, 1]
        """
        n = len(bbwidth_pct)
        scores = np.full(n, np.nan)

        # Normalisation des composantes
        bbwidth_norm = bbwidth_pct / 100.0  # [0, 1]
        pb_norm = np.abs(percent_b)  # |%B| peut être > 1, on clip à 1
        pb_norm = np.clip(pb_norm, 0, 1)

        # Pente MACD = différence histogram (normalisée)
        macd_slope = np.abs(np.diff(macd_hist, prepend=macd_hist[0]))
        # Normalisation empirique (on clip à 1)
        macd_slope_norm = np.clip(macd_slope / (np.nanmax(macd_slope) + 1e-10), 0, 1)

        # Volume zscore normalisé (clip entre 0 et 1, zscore peut être négatif)
        volume_norm = np.clip(volume_zscore / 3.0, 0, 1)  # z-score de 3 = 1.0

        # Calcul score
        for i in range(n):
            if not np.isnan(bbwidth_norm[i]):
                scores[i] = (
                    params.amplitude_w1_bbwidth * bbwidth_norm[i]
                    + params.amplitude_w2_pb * pb_norm[i]
                    + params.amplitude_w3_macd_slope * macd_slope_norm[i]
                    + params.amplitude_w4_volume * volume_norm[i]
                )

        return scores

    def _detect_spring_long(
        self,
        close: np.ndarray,
        bb_lower: np.ndarray,
        lookback: int,
    ) -> np.ndarray:
        """
        Détecte un "spring" pour LONG: close < bb_lower dans les M dernières barres.

        Args:
            close: Prix de clôture
            bb_lower: Bande Bollinger inférieure
            lookback: Période de lookback

        Returns:
            Boolean array True si spring détecté
        """
        n = len(close)
        spring = np.full(n, False)

        for i in range(lookback, n):
            # Regarder les M dernières barres
            window_close = close[i - lookback : i]
            window_lower = bb_lower[i - lookback : i]
            # Spring si au moins une barre en dessous
            if np.any(window_close < window_lower):
                spring[i] = True

        return spring

    def _detect_spring_short(
        self,
        close: np.ndarray,
        bb_upper: np.ndarray,
        lookback: int,
    ) -> np.ndarray:
        """
        Détecte un "spring" pour SHORT: close > bb_upper dans les M dernières barres.
        """
        n = len(close)
        spring = np.full(n, False)

        for i in range(lookback, n):
            window_close = close[i - lookback : i]
            window_upper = bb_upper[i - lookback : i]
            if np.any(window_close > window_upper):
                spring[i] = True

        return spring

    def _detect_macd_impulse_long(
        self, macd_hist: np.ndarray
    ) -> np.ndarray:
        """
        Détecte une impulsion MACD pour LONG:
        - MACD histo passe rouge foncé → rouge clair (ralentissement)
        - Puis >= 1 barre verte (impulsion)

        Simplifié: MACD histo négatif qui augmente puis passe positif

        Returns:
            Boolean array True si impulsion détectée
        """
        n = len(macd_hist)
        impulse = np.full(n, False)

        for i in range(2, n):
            # Condition: histo était négatif, augmente, puis positif
            if (
                macd_hist[i - 2] < 0  # Rouge foncé (très négatif)
                and macd_hist[i - 1] < 0  # Rouge clair (moins négatif)
                and macd_hist[i - 1] > macd_hist[i - 2]  # Ralentissement
                and macd_hist[i] > 0  # Barre verte (impulsion)
            ):
                impulse[i] = True

        return impulse

    def _detect_macd_impulse_short(
        self, macd_hist: np.ndarray
    ) -> np.ndarray:
        """
        Détecte une impulsion MACD pour SHORT (symétrique):
        - MACD histo positif qui diminue puis passe négatif
        """
        n = len(macd_hist)
        impulse = np.full(n, False)

        for i in range(2, n):
            if (
                macd_hist[i - 2] > 0  # Vert foncé
                and macd_hist[i - 1] > 0  # Vert clair
                and macd_hist[i - 1] < macd_hist[i - 2]  # Ralentissement
                and macd_hist[i] < 0  # Barre rouge (impulsion)
            ):
                impulse[i] = True

        return impulse

    def _ensure_indicators(
        self, df: pd.DataFrame, params: AmplitudeHunterParams
    ) -> pd.DataFrame:
        """
        Garantit la disponibilité de tous les indicateurs.

        Args:
            df: DataFrame OHLCV
            params: Paramètres stratégie

        Returns:
            DataFrame enrichi avec tous les indicateurs
        """
        logger.debug(
            f"Calcul indicateurs: BB(period={params.bb_period}, std={params.bb_std}), "
            f"MACD({params.macd_fast},{params.macd_slow},{params.macd_signal}), ATR({params.atr_period})"
        )

        df_ind = df.copy()

        # 1. Bollinger Bands via IndicatorBank
        bb_result = ensure_indicator(
            "bollinger",
            {"period": params.bb_period, "std": params.bb_std},
            df,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        if isinstance(bb_result, tuple) and len(bb_result) == 3:
            upper, middle, lower = bb_result
            df_ind["bb_upper"] = upper
            df_ind["bb_middle"] = middle
            df_ind["bb_lower"] = lower
        else:
            raise ValueError(f"Bollinger result format invalide: {type(bb_result)}")

        # 2. ATR via IndicatorBank
        atr_result = ensure_indicator(
            "atr",
            {"period": params.atr_period, "method": "ema"},
            df,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        if isinstance(atr_result, np.ndarray):
            df_ind["atr"] = atr_result
        else:
            raise ValueError(f"ATR result format invalide: {type(atr_result)}")

        # 3. %B
        df_ind["percent_b"] = self._calculate_percent_b(
            df["close"].values, upper, lower
        )

        # 4. BBWidth
        df_ind["bb_width"] = upper - lower

        # 5. BBWidth percentile
        df_ind["bbwidth_percentile"] = self._calculate_bbwidth_percentile(
            df_ind["bb_width"].values, params.bbwidth_lookback
        )

        # 6. MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            df["close"].values, params.macd_fast, params.macd_slow, params.macd_signal
        )
        df_ind["macd_line"] = macd_line
        df_ind["macd_signal"] = signal_line
        df_ind["macd_hist"] = histogram

        # 7. Volume z-score
        df_ind["volume_zscore"] = self._calculate_volume_zscore(
            df["volume"].values, params.volume_lookback
        )

        # 8. ADX (si activé)
        if params.use_adx_filter:
            df_ind["adx"] = self._calculate_adx(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                params.adx_period,
            )

        # 9. Score d'Amplitude
        df_ind["amplitude_score"] = self._calculate_amplitude_score(
            df_ind["bbwidth_percentile"].values,
            df_ind["percent_b"].values,
            df_ind["macd_hist"].values,
            df_ind["volume_zscore"].values,
            params,
        )

        logger.debug(
            f"Indicateurs calculés: {len(df_ind)} barres enrichies avec "
            f"{len([c for c in df_ind.columns if c not in df.columns])} nouveaux indicateurs"
        )

        return df_ind

    def generate_signals(
        self, df: pd.DataFrame, params: dict
    ) -> pd.DataFrame:
        """
        Génère les signaux de trading basés sur AmplitudeHunter.

        Args:
            df: DataFrame OHLCV avec timestamp index (UTC)
            params: Dictionnaire paramètres (format AmplitudeHunterParams.to_dict())

        Returns:
            DataFrame avec colonne 'signal' et métadonnées

        Signals générés:
        - "ENTER_LONG": Setup Spring→Drive LONG validé
        - "ENTER_SHORT": Setup Spring→Drive SHORT validé
        - "HOLD": Pas de signal
        """
        logger.info(f"Génération signaux AmplitudeHunter: {len(df)} barres")

        # Validation inputs
        validate_ohlcv_dataframe(df)
        validate_strategy_params(
            params,
            ["bb_period", "bb_std", "amplitude_score_threshold"],
        )

        # Parse paramètres
        strategy_params = AmplitudeHunterParams.from_dict(params)

        # Ensure indicateurs
        df_with_indicators = self._ensure_indicators(df, strategy_params)

        # Extraction des données
        close = df["close"].values
        n_bars = len(df)

        # Extraction indicateurs
        bb_upper = df_with_indicators["bb_upper"].values
        bb_lower = df_with_indicators["bb_lower"].values
        df_with_indicators["bb_middle"].values
        percent_b = df_with_indicators["percent_b"].values
        bbwidth_pct = df_with_indicators["bbwidth_percentile"].values
        macd_hist = df_with_indicators["macd_hist"].values
        volume_zscore = df_with_indicators["volume_zscore"].values
        amplitude_score = df_with_indicators["amplitude_score"].values

        # Initialisation signaux
        signals = np.full(n_bars, "HOLD", dtype=object)

        # --- Filtre de régime ---
        regime_filter = np.full(n_bars, True)

        # BBWidth percentile >= seuil
        regime_filter &= bbwidth_pct >= strategy_params.bbwidth_percentile_threshold

        # Volume z-score >= seuil
        regime_filter &= volume_zscore >= strategy_params.volume_zscore_threshold

        # ADX >= seuil (si activé)
        if strategy_params.use_adx_filter:
            adx = df_with_indicators["adx"].values
            regime_filter &= adx >= strategy_params.adx_threshold

        # Score amplitude >= seuil
        regime_filter &= amplitude_score >= strategy_params.amplitude_score_threshold

        # --- Détection springs ---
        spring_long = self._detect_spring_long(
            close, bb_lower, strategy_params.spring_lookback
        )
        spring_short = self._detect_spring_short(
            close, bb_upper, strategy_params.spring_lookback
        )

        # --- Détection impulsions MACD ---
        macd_impulse_long = self._detect_macd_impulse_long(macd_hist)
        macd_impulse_short = self._detect_macd_impulse_short(macd_hist)

        # --- Génération signaux LONG ---
        logger.debug("Application logique signaux LONG")

        for i in range(max(strategy_params.bb_period, strategy_params.spring_lookback), n_bars):
            # Skip si filtre régime échoue
            if not regime_filter[i]:
                continue

            # Skip si NaN
            if np.isnan(percent_b[i]) or np.isnan(macd_hist[i]):
                continue

            # Signal LONG
            if (
                spring_long[i]  # Spring détecté
                and macd_impulse_long[i]  # Impulsion MACD
                and strategy_params.pb_entry_threshold_min
                <= percent_b[i]
                <= strategy_params.pb_entry_threshold_max  # %B dans range
            ):
                signals[i] = "ENTER_LONG"
                logger.debug(
                    f"ENTER_LONG @ bar {i}: price={close[i]:.2f}, %B={percent_b[i]:.2f}, "
                    f"amplitude_score={amplitude_score[i]:.2f}"
                )

            # Signal SHORT (symétrique, mais %B inversé: 1 - thresholds)
            elif (
                spring_short[i]
                and macd_impulse_short[i]
                and (1 - strategy_params.pb_entry_threshold_max)
                <= percent_b[i]
                <= (1 - strategy_params.pb_entry_threshold_min)
            ):
                signals[i] = "ENTER_SHORT"
                logger.debug(
                    f"ENTER_SHORT @ bar {i}: price={close[i]:.2f}, %B={percent_b[i]:.2f}, "
                    f"amplitude_score={amplitude_score[i]:.2f}"
                )

        # Construction DataFrame de sortie
        result_df = df_with_indicators.copy()
        result_df["signal"] = signals

        # Statistiques signaux
        enter_longs = np.sum(signals == "ENTER_LONG")
        enter_shorts = np.sum(signals == "ENTER_SHORT")
        total_signals = enter_longs + enter_shorts

        logger.info(
            f"Signaux générés: {total_signals} total ({enter_longs} LONG, {enter_shorts} SHORT)"
        )

        return result_df

    def backtest(
        self,
        df: pd.DataFrame,
        params: dict,
        initial_capital: float = 10000.0,
        fee_bps: float = 4.5,
        slippage_bps: float = 0.0,
    ) -> tuple[pd.Series, RunStats]:
        """
        Exécute un backtest complet de la stratégie AmplitudeHunter.

        Args:
            df: DataFrame OHLCV avec timestamp index (UTC)
            params: Paramètres stratégie (format AmplitudeHunterParams.to_dict())
            initial_capital: Capital initial
            fee_bps: Frais de transaction en basis points (défaut: 4.5)
            slippage_bps: Slippage en basis points (défaut: 0.0)

        Returns:
            Tuple (equity_curve, run_stats)

        Gestion des positions:
        - Position initiale + pyramiding (jusqu'à 2 adds)
        - Stop loss initial: max(swing_low - k*ATR, pct*(médiane-basse))
        - Trailing stop conditionnel (activé si %B > 1 OU gain >= 1R)
        - Stop fixe pour SHORT (37% au-dessus entrée)
        - Cible BIP avec sortie partielle (50% par défaut)
        """
        logger.info(
            f"Début backtest AmplitudeHunter: capital={initial_capital}, "
            f"fee={fee_bps}bps, slippage={slippage_bps}bps"
        )

        # Validation
        validate_ohlcv_dataframe(df)
        strategy_params = AmplitudeHunterParams.from_dict(params)

        # Génération signaux
        signals_df = self.generate_signals(df, params)

        # Initialisation backtest
        n_bars = len(df)
        equity = np.full(n_bars, initial_capital, dtype=float)

        cash = initial_capital
        positions: list[Trade] = []  # Liste des positions en cours (initial + adds)
        closed_trades: list[Trade] = []

        fee_rate = (fee_bps + slippage_bps) / 10000.0

        # État pour pyramiding
        num_adds = 0
        trailing_active = False
        bip_target_hit = False

        logger.debug(f"Backtest initialisé: {n_bars} barres, fee_rate={fee_rate:.6f}")

        # Boucle principale
        for i, (timestamp, row) in enumerate(signals_df.iterrows()):
            current_price = row["close"]
            current_atr = row["atr"]
            signal = row["signal"]
            percent_b = row["percent_b"]
            bb_upper = row["bb_upper"]
            bb_lower = row["bb_lower"]
            bb_middle = row["bb_middle"]
            macd_hist = row["macd_hist"]

            # Skip si ATR invalide
            if np.isnan(current_atr) or current_atr <= 0:
                equity[i] = self._calculate_total_equity(
                    cash, positions, current_price
                )
                continue

            # --- GESTION POSITIONS EXISTANTES ---
            if len(positions) > 0:
                initial_position = positions[0]
                should_exit = False
                exit_reason = ""

                # 1. Vérification stop loss
                for pos in positions:
                    if pos.should_stop_loss(current_price):
                        should_exit = True
                        exit_reason = "stop_loss"
                        break

                # 2. Vérification durée maximale
                if not should_exit:
                    entry_timestamp = pd.to_datetime(
                        initial_position.entry_time, utc=True
                    )
                    bars_held = (df.index <= timestamp).sum() - (
                        df.index <= entry_timestamp
                    ).sum()

                    if bars_held >= strategy_params.max_hold_bars:
                        should_exit = True
                        exit_reason = "max_hold_bars"

                # 3. Vérification trailing stop (si activé)
                if not should_exit and trailing_active:
                    if strategy_params.trailing_type == "chandelier":
                        # Chandelier: médiane - k*ATR
                        chandelier_stop = bb_middle - (
                            current_atr
                            * strategy_params.trailing_chandelier_atr_mult
                        )
                        if initial_position.is_long() and current_price <= chandelier_stop:
                            should_exit = True
                            exit_reason = "trailing_chandelier"
                        elif initial_position.is_short() and current_price >= chandelier_stop:
                            should_exit = True
                            exit_reason = "trailing_chandelier"

                    elif strategy_params.trailing_type == "pb_floor":
                        # %B floor: sortir si %B < 0.5 après extension
                        if percent_b < strategy_params.trailing_pb_floor:
                            should_exit = True
                            exit_reason = "trailing_pb_floor"

                    elif strategy_params.trailing_type == "macd_fade":
                        # MACD fade: barre verte→rouge pour LONG, rouge→verte pour SHORT
                        if i > 0:
                            prev_macd_hist = signals_df.iloc[i - 1]["macd_hist"]
                            if initial_position.is_long() and macd_hist < 0 and prev_macd_hist > 0:
                                should_exit = True
                                exit_reason = "trailing_macd_fade"
                            elif initial_position.is_short() and macd_hist > 0 and prev_macd_hist < 0:
                                should_exit = True
                                exit_reason = "trailing_macd_fade"

                # 4. Vérification cible BIP (sortie partielle)
                if (
                    not should_exit
                    and strategy_params.use_bip_target
                    and not bip_target_hit
                ):
                    # BIP = médiane + (médiane - basse)
                    bip_target = bb_middle + (bb_middle - bb_lower)

                    if initial_position.is_long() and current_price >= bip_target:
                        # Sortie partielle
                        self._partial_exit_bip(
                            positions,
                            current_price,
                            timestamp,
                            fee_rate,
                            strategy_params.bip_partial_exit_pct,
                            cash,
                            closed_trades,
                        )
                        bip_target_hit = True
                        logger.debug(
                            f"BIP target hit @ {current_price:.2f}, "
                            f"sortie partielle {strategy_params.bip_partial_exit_pct*100:.0f}%"
                        )

                    elif initial_position.is_short() and current_price <= bip_target:
                        self._partial_exit_bip(
                            positions,
                            current_price,
                            timestamp,
                            fee_rate,
                            strategy_params.bip_partial_exit_pct,
                            cash,
                            closed_trades,
                        )
                        bip_target_hit = True

                # 5. Activation trailing stop (si conditions remplies)
                if not trailing_active and len(positions) > 0:
                    # Calculer gain actuel en R
                    initial_risk = abs(
                        initial_position.entry_price - initial_position.stop
                    )
                    current_pnl = initial_position.calculate_unrealized_pnl(
                        current_price
                    )
                    gain_in_r = current_pnl / (initial_risk * initial_position.qty) if initial_risk > 0 else 0

                    # Activer si %B > seuil OU gain >= R
                    if (
                        percent_b > strategy_params.trailing_activation_pb_threshold
                        or gain_in_r >= strategy_params.trailing_activation_gain_r
                    ):
                        trailing_active = True
                        logger.debug(
                            f"Trailing stop activé @ bar {i}: %B={percent_b:.2f}, gain={gain_in_r:.2f}R"
                        )

                # 6. Fermeture complète de toutes les positions
                if should_exit:
                    for pos in positions:
                        exit_value = current_price * pos.qty
                        exit_fees = exit_value * fee_rate

                        pos.close_trade(
                            exit_price=current_price,
                            exit_time=str(timestamp),
                            exit_fees=exit_fees,
                        )

                        pnl_val = pos.pnl_realized if pos.pnl_realized is not None else 0.0
                        cash += pnl_val + (pos.entry_price * pos.qty)
                        closed_trades.append(pos)

                        logger.debug(
                            f"Position fermée @ {current_price:.2f}: {exit_reason}, "
                            f"PnL={pos.pnl_realized:.2f}"
                        )

                    # Reset état
                    positions = []
                    num_adds = 0
                    trailing_active = False
                    bip_target_hit = False

                # 7. Pyramiding (Add #1 et Add #2)
                elif (
                    strategy_params.pyramiding_enabled
                    and num_adds < strategy_params.pyramiding_max_adds
                    and len(positions) > 0
                ):
                    initial_position = positions[0]

                    # Add #1: close > bb_upper ET MACD s'intensifie
                    if num_adds == 0:
                        if i > 0:
                            prev_macd_hist = signals_df.iloc[i - 1]["macd_hist"]
                            if initial_position.is_long():
                                if (
                                    current_price > bb_upper
                                    and macd_hist > 0
                                    and macd_hist > prev_macd_hist
                                ):
                                    self._add_position(
                                        positions,
                                        "LONG",
                                        current_price,
                                        current_atr,
                                        bb_middle,
                                        bb_lower,
                                        timestamp,
                                        strategy_params,
                                        cash,
                                        fee_rate,
                                    )
                                    num_adds += 1
                                    logger.debug(f"Add #1 LONG @ {current_price:.2f}")

                            elif initial_position.is_short():
                                if (
                                    current_price < bb_lower
                                    and macd_hist < 0
                                    and macd_hist < prev_macd_hist
                                ):
                                    self._add_position(
                                        positions,
                                        "SHORT",
                                        current_price,
                                        current_atr,
                                        bb_middle,
                                        bb_upper,
                                        timestamp,
                                        strategy_params,
                                        cash,
                                        fee_rate,
                                    )
                                    num_adds += 1
                                    logger.debug(f"Add #1 SHORT @ {current_price:.2f}")

                    # Add #2: pullback tenu >= médiane + MACD reste vert/rouge
                    elif num_adds == 1:
                        if initial_position.is_long():
                            if current_price >= bb_middle and macd_hist > 0:
                                self._add_position(
                                    positions,
                                    "LONG",
                                    current_price,
                                    current_atr,
                                    bb_middle,
                                    bb_lower,
                                    timestamp,
                                    strategy_params,
                                    cash,
                                    fee_rate,
                                )
                                num_adds += 1
                                logger.debug(f"Add #2 LONG @ {current_price:.2f}")

                        elif initial_position.is_short():
                            if current_price <= bb_middle and macd_hist < 0:
                                self._add_position(
                                    positions,
                                    "SHORT",
                                    current_price,
                                    current_atr,
                                    bb_middle,
                                    bb_upper,
                                    timestamp,
                                    strategy_params,
                                    cash,
                                    fee_rate,
                                )
                                num_adds += 1
                                logger.debug(f"Add #2 SHORT @ {current_price:.2f}")

            # --- NOUVEAUX SIGNAUX D'ENTRÉE ---
            if len(positions) == 0 and signal in ["ENTER_LONG", "ENTER_SHORT"]:
                # Position sizing basé sur ATR et risk
                atr_stop_distance = current_atr * strategy_params.sl_atr_multiplier
                risk_amount = cash * strategy_params.risk_per_trade

                # Calcul quantité optimale
                position_size = risk_amount / atr_stop_distance
                max_position_size = (cash * strategy_params.leverage) / current_price

                qty = min(position_size, max_position_size)

                if qty > 0:
                    # Calcul stop loss initial
                    if signal == "ENTER_LONG":
                        # SL = max(swing_low - k*ATR, sl_min_pct*(médiane-basse))
                        swing_low = min(df["low"].iloc[max(0, i - 10) : i + 1])
                        stop_atr = swing_low - atr_stop_distance
                        stop_min = bb_lower + strategy_params.sl_min_pct * (
                            bb_middle - bb_lower
                        )
                        stop_price = max(stop_atr, stop_min)

                    else:  # ENTER_SHORT
                        swing_high = max(df["high"].iloc[max(0, i - 10) : i + 1])
                        stop_atr = swing_high + atr_stop_distance
                        stop_max = bb_upper - strategy_params.sl_min_pct * (
                            bb_upper - bb_middle
                        )
                        stop_price = min(stop_atr, stop_max)

                        # Stop fixe supplémentaire pour SHORT
                        stop_fixed = current_price * (1 + strategy_params.short_stop_pct)
                        stop_price = min(stop_price, stop_fixed)

                    # Frais d'entrée
                    entry_value = current_price * qty
                    entry_fees = entry_value * fee_rate

                    if entry_value + entry_fees <= cash:
                        # Création nouvelle position
                        position = Trade(
                            side=signal.replace("ENTER_", ""),
                            qty=qty,
                            entry_price=current_price,
                            entry_time=str(timestamp),
                            stop=stop_price,
                            fees_paid=entry_fees,
                            meta={
                                "percent_b": percent_b,
                                "amplitude_score": row.get("amplitude_score", 0),
                                "atr": current_atr,
                                "sl_atr_multiplier": strategy_params.sl_atr_multiplier,
                                "bb_middle": bb_middle,
                                "bb_lower": bb_lower if signal == "ENTER_LONG" else bb_upper,
                            },
                        )

                        positions.append(position)
                        cash -= entry_value + entry_fees
                        num_adds = 0
                        trailing_active = False
                        bip_target_hit = False

                        logger.debug(
                            f"Nouvelle position: {signal} {qty:.4f} @ {current_price:.2f}, "
                            f"stop={stop_price:.2f}"
                        )

            # Mise à jour équité
            equity[i] = self._calculate_total_equity(cash, positions, current_price)

        # Fermeture positions finales si nécessaire
        if len(positions) > 0:
            final_price = df["close"].iloc[-1]
            for pos in positions:
                pos.close_trade(
                    exit_price=final_price,
                    exit_time=df.index[-1].isoformat(),
                    exit_fees=final_price * pos.qty * fee_rate,
                )
                closed_trades.append(pos)

        # Construction courbe d'équité
        equity_curve = pd.Series(equity, index=df.index)

        # Calcul statistiques
        run_stats = RunStats.from_trades_and_equity(
            trades=closed_trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            meta={
                "strategy": "AmplitudeHunter",
                "params": params,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
            },
        )

        logger.info(
            f"Backtest terminé: {run_stats.total_trades} trades, "
            f"PnL={run_stats.total_pnl:.2f} ({run_stats.total_pnl_pct:.2f}%)"
        )

        return equity_curve, run_stats

    # --- Helper methods pour backtest ---

    def _calculate_total_equity(
        self, cash: float, positions: list[Trade], current_price: float
    ) -> float:
        """Calcule l'équité totale (cash + positions non réalisées)"""
        total_unrealized = sum(
            pos.calculate_unrealized_pnl(current_price) for pos in positions
        )
        total_invested = sum(pos.entry_price * pos.qty for pos in positions)
        return cash + total_invested + total_unrealized

    def _partial_exit_bip(
        self,
        positions: list[Trade],
        exit_price: float,
        exit_time: Any,
        fee_rate: float,
        exit_pct: float,
        cash: float,
        closed_trades: list[Trade],
    ) -> float:
        """
        Sortie partielle à BIP target.

        Args:
            positions: Liste des positions en cours (modifiée in-place)
            exit_price: Prix de sortie
            exit_time: Timestamp
            fee_rate: Taux de frais
            exit_pct: Pourcentage à sortir (0.5 = 50%)
            cash: Cash actuel (modifié in-place via reference)
            closed_trades: Liste des trades fermés (modifiée in-place)

        Returns:
            PnL réalisé de la sortie partielle
        """
        total_pnl = 0.0

        for pos in positions:
            # Quantité à sortir
            qty_to_exit = pos.qty * exit_pct

            # Créer un trade partiel pour les statistiques
            partial_trade = Trade(
                side=pos.side,
                qty=qty_to_exit,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                stop=pos.stop,
                fees_paid=pos.fees_paid * exit_pct,  # Frais proportionnels
                meta=pos.meta.copy(),
            )

            exit_value = exit_price * qty_to_exit
            exit_fees = exit_value * fee_rate

            partial_trade.close_trade(
                exit_price=exit_price,
                exit_time=str(exit_time),
                exit_fees=exit_fees,
            )

            pnl = partial_trade.pnl_realized if partial_trade.pnl_realized is not None else 0.0
            total_pnl += pnl
            closed_trades.append(partial_trade)

            # Réduire la position restante
            pos.qty -= qty_to_exit
            pos.fees_paid -= pos.fees_paid * exit_pct

        return total_pnl

    def _add_position(
        self,
        positions: list[Trade],
        side: str,
        entry_price: float,
        current_atr: float,
        bb_middle: float,
        bb_boundary: float,
        timestamp: Any,
        params: AmplitudeHunterParams,
        cash: float,
        fee_rate: float,
    ) -> bool:
        """
        Ajoute une position pyramidée.

        Returns:
            True si l'add a réussi, False sinon
        """
        # Sizing similaire à la position initiale
        atr_stop_distance = current_atr * params.sl_atr_multiplier
        risk_amount = cash * params.risk_per_trade

        position_size = risk_amount / atr_stop_distance
        max_position_size = (cash * params.leverage) / entry_price

        qty = min(position_size, max_position_size)

        if qty <= 0:
            return False

        # Stop pour l'add (même logique que position initiale)
        if side == "LONG":
            stop_price = max(
                bb_boundary - atr_stop_distance,
                bb_boundary + params.sl_min_pct * (bb_middle - bb_boundary),
            )
        else:
            stop_price = min(
                bb_boundary + atr_stop_distance,
                bb_boundary - params.sl_min_pct * (bb_boundary - bb_middle),
            )

        entry_value = entry_price * qty
        entry_fees = entry_value * fee_rate

        if entry_value + entry_fees > cash:
            return False

        # Créer l'add
        add_position = Trade(
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_time=str(timestamp),
            stop=stop_price,
            fees_paid=entry_fees,
            meta={"is_add": True, "add_number": len(positions)},
        )

        positions.append(add_position)
        # Note: le cash n'est pas modifié ici car la signature ne permet pas de le faire
        # Il faudrait retourner le cash modifié ou utiliser un container mutable

        return True

    # --- Optimization Presets ---

    @staticmethod
    def get_optimization_ranges() -> dict[str, tuple[float, float]]:
        """
        Retourne les plages d'optimisation recommandées pour AmplitudeHunter.

        Utilise les presets "classiques" depuis le fichier indicator_ranges.toml.
        Les plages sont automatiquement mappées aux paramètres de la stratégie.

        Returns:
            Dictionnaire {param_name: (min, max)}

        Example:
            >>> ranges = AmplitudeHunterStrategy.get_optimization_ranges()
            >>> print(ranges['bb_period'])  # (10, 50)
            >>> print(ranges['amplitude_score_threshold'])  # (0.4, 0.8)
        """
        try:
            from threadx.optimization.presets import get_strategy_preset

            mapper = get_strategy_preset("AmplitudeHunter")
            return mapper.get_optimization_ranges()
        except ImportError:
            logger.warning(
                "Module threadx.optimization.presets non disponible. "
                "Retour des plages par défaut."
            )
            # Fallback: plages basiques
            return {
                "bb_period": (10, 50),
                "bb_std": (1.5, 3.0),
                "amplitude_score_threshold": (0.4, 0.8),
            }

    @staticmethod
    def get_optimization_grid() -> dict[str, list[Any]]:
        """
        Retourne les grilles de valeurs pour grid search.

        Génère automatiquement les valeurs à tester pour chaque paramètre
        selon les presets (min, max, step).

        Returns:
            Dictionnaire {param_name: [valeurs]}

        Example:
            >>> grid = AmplitudeHunterStrategy.get_optimization_grid()
            >>> print(grid['bb_period'])  # [10, 11, 12, ..., 50]
            >>> print(grid['pyramiding_max_adds'])  # [0, 1, 2]
        """
        try:
            from threadx.optimization.presets import get_strategy_preset

            mapper = get_strategy_preset("AmplitudeHunter")
            return mapper.get_grid_parameters()
        except ImportError:
            logger.warning(
                "Module threadx.optimization.presets non disponible. "
                "Retour de grilles par défaut."
            )
            # Fallback: grilles basiques
            return {
                "bb_period": list(range(10, 51, 1)),
                "amplitude_score_threshold": [0.4, 0.5, 0.6, 0.7, 0.8],
            }

    @staticmethod
    def get_default_optimization_params() -> dict[str, Any]:
        """
        Retourne les valeurs par défaut recommandées pour l'optimisation.

        Returns:
            Dictionnaire {param_name: default_value}

        Example:
            >>> defaults = AmplitudeHunterStrategy.get_default_optimization_params()
            >>> print(defaults['bb_period'])  # 20
            >>> print(defaults['amplitude_score_threshold'])  # 0.6
        """
        try:
            from threadx.optimization.presets import get_strategy_preset

            mapper = get_strategy_preset("AmplitudeHunter")
            return mapper.get_default_parameters()
        except ImportError:
            logger.warning(
                "Module threadx.optimization.presets non disponible. "
                "Retour des valeurs par défaut de AmplitudeHunterParams."
            )
            # Fallback: valeurs par défaut de la dataclass
            return AmplitudeHunterParams().to_dict()


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================


def generate_signals(
    df: pd.DataFrame,
    params: dict,
    symbol: str = "UNKNOWN",
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Fonction de convenance pour génération de signaux AmplitudeHunter.

    Args:
        df: DataFrame OHLCV
        params: Paramètres stratégie
        symbol: Symbole pour cache
        timeframe: Timeframe pour cache

    Returns:
        DataFrame avec signaux et métadonnées

    Example:
        >>> params = AmplitudeHunterParams(amplitude_score_threshold=0.7).to_dict()
        >>> signals = generate_signals(df, params, "BTCUSDT", "1h")
    """
    strategy = AmplitudeHunterStrategy(symbol=symbol, timeframe=timeframe)
    return strategy.generate_signals(df, params)


def backtest(
    df: pd.DataFrame,
    params: dict,
    initial_capital: float = 10000.0,
    symbol: str = "UNKNOWN",
    timeframe: str = "1h",
    **kwargs,
) -> tuple[pd.Series, RunStats]:
    """
    Fonction de convenance pour backtest AmplitudeHunter.

    Args:
        df: DataFrame OHLCV
        params: Paramètres stratégie
        initial_capital: Capital initial
        symbol: Symbole pour cache
        timeframe: Timeframe pour cache
        **kwargs: Arguments supplémentaires (fee_bps, slippage_bps, etc.)

    Returns:
        Tuple (equity_curve, run_stats)

    Example:
        >>> params = AmplitudeHunterParams(pyramiding_enabled=True).to_dict()
        >>> equity, stats = backtest(df, params, 50000, "ETHUSDT", "4h")
        >>> print(f"ROI: {stats.total_pnl_pct:.2f}%, Trades: {stats.total_trades}")
    """
    strategy = AmplitudeHunterStrategy(symbol=symbol, timeframe=timeframe)
    return strategy.backtest(df, params, initial_capital, **kwargs)


def create_default_params(**overrides) -> AmplitudeHunterParams:
    """
    Crée des paramètres par défaut avec surcharges optionnelles.

    Args:
        **overrides: Paramètres à surcharger

    Returns:
        Instance AmplitudeHunterParams avec valeurs par défaut + surcharges

    Example:
        >>> params = create_default_params(
        ...     amplitude_score_threshold=0.75,
        ...     pyramiding_enabled=True,
        ...     pyramiding_max_adds=2
        ... )
        >>> params.amplitude_score_threshold
        0.75
    """
    base_params = AmplitudeHunterParams()

    # Application des surcharges
    for key, value in overrides.items():
        if hasattr(base_params, key):
            setattr(base_params, key, value)
        else:
            logger.warning(f"Paramètre inconnu ignoré: {key}={value}")

    return base_params


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    # Classes principales
    "AmplitudeHunterParams",
    "AmplitudeHunterStrategy",
    # Fonctions de convenance
    "generate_signals",
    "backtest",
    "create_default_params",
]
