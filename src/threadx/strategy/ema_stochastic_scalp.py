"""
ThreadX - EMA 50/100 + Stochastic Scalping Strategy
===================================================

Stratégie de scalping crypto 1-minute basée sur:
- Croisement EMA 50/100 pour la direction de tendance
- Stochastic pour le timing d'entrée (confirmation zones survente/surachat)
- Pullback vers les EMAs pour meilleure entrée
- Volume en hausse pour validation

Win rate attendu: 65-80% selon documentation 2025
Utilisée sur BTC/USDT, ETH/USDT, SOL/USDT avec levier 10-20x

Règles d'Entrée LONG:
1. EMA 50 croise au-dessus EMA 100
2. Prix fait pullback vers les EMAs
3. Stochastic sort de zone survente (<20) ou croise haussier dans 20-40
4. Volume en hausse sur 2-3 bougies

Règles d'Entrée SHORT:
1. EMA 50 croise en-dessous EMA 100
2. Prix fait pullback vers les EMAs
3. Stochastic sort de zone surachat (>80) ou croise baissier dans 60-80
4. Volume en hausse

Gestion de Position:
- Take Profit: 0.4-0.8% (BTC/ETH) ou 1-2% (altcoins)
- Stop Loss: 0.4-0.6% maximum
- Risk-Reward: 1:1.5 à 1:2
- Max hold bars: 20-50 (éviter positions stagnantes)

Exemple:
    >>> from threadx.strategy.ema_stochastic_scalp import EMAStochScalpStrategy, EMAStochScalpParams
    >>> params = EMAStochScalpParams(
    ...     fast_ema=50, slow_ema=100,
    ...     stoch_k=14, stoch_d=3,
    ...     stop_loss_pct=0.5, take_profit_pct=0.7
    ... )
    >>> strategy = EMAStochScalpStrategy()
    >>> equity, stats = strategy.backtest(df, params.to_dict(), initial_capital=10000)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from threadx.strategy.model import RunStats, Trade, validate_ohlcv_dataframe
from threadx.utils.log import get_logger

logger = get_logger(__name__)


# ==========================================
# NUMBA OPTIMIZED BACKTEST LOOP
# ==========================================


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def _backtest_loop_numba(
    close_vals: np.ndarray,
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    signal_vals: np.ndarray,  # 0=HOLD, 1=LONG, 2=SHORT
    initial_capital: float,
    fee_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    risk_per_trade: float,
    leverage: float,
    max_hold_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boucle de backtest EMA+Stochastic optimisée Numba.

    Args:
        close_vals: Prix de clôture (n_bars,)
        high_vals: Prix high (n_bars,)
        low_vals: Prix low (n_bars,)
        signal_vals: Signaux encodés (n_bars,) - 0:HOLD, 1:LONG, 2:SHORT
        initial_capital: Capital de départ
        fee_rate: Taux de frais total (fee + slippage)
        stop_loss_pct: Stop loss en % (ex: 0.5 = -0.5%)
        take_profit_pct: Take profit en % (ex: 0.7 = +0.7%)
        risk_per_trade: Fraction capital risquée par trade
        leverage: Levier autorisé
        max_hold_bars: Durée max position

    Returns:
        Tuple (equity_curve, trade_results)
    """
    n_bars = len(close_vals)
    equity = np.full(n_bars, initial_capital, dtype=np.float64)

    # Pré-allocation résultats trades
    trade_results = np.zeros((n_bars, 10), dtype=np.float64)
    trade_count = 0

    cash = initial_capital

    # State position
    has_position = False
    pos_side = 0  # 1=LONG, 2=SHORT
    pos_qty = 0.0
    pos_entry_price = 0.0
    pos_stop = 0.0
    pos_take_profit = 0.0
    pos_entry_bar = 0
    pos_entry_fees = 0.0

    # Boucle principale
    for i in range(n_bars):
        current_price = close_vals[i]
        current_high = high_vals[i]
        current_low = low_vals[i]
        signal = signal_vals[i]

        # === GESTION POSITION EXISTANTE ===
        if has_position:
            should_exit = False
            exit_price = current_price

            # 1. Stop loss check (utilise high/low pour précision intrabar)
            if pos_side == 1:  # LONG
                if current_low <= pos_stop:
                    should_exit = True
                    exit_price = pos_stop  # Sortie au stop exact
            else:  # SHORT
                if current_high >= pos_stop:
                    should_exit = True
                    exit_price = pos_stop

            # 2. Take profit check
            if not should_exit:
                if pos_side == 1 and current_high >= pos_take_profit:
                    should_exit = True
                    exit_price = pos_take_profit
                elif pos_side == 2 and current_low <= pos_take_profit:
                    should_exit = True
                    exit_price = pos_take_profit

            # 3. Signal inverse
            if not should_exit:
                if pos_side == 1 and signal == 2:  # LONG → SHORT signal
                    should_exit = True
                    exit_price = current_price
                elif pos_side == 2 and signal == 1:  # SHORT → LONG signal
                    should_exit = True
                    exit_price = current_price

            # 4. Max hold bars
            if not should_exit:
                bars_held = i - pos_entry_bar
                if bars_held >= max_hold_bars:
                    should_exit = True
                    exit_price = current_price

            # === FERMETURE POSITION ===
            if should_exit:
                exit_value = exit_price * pos_qty
                exit_fees = exit_value * fee_rate

                # Calcul PnL
                if pos_side == 1:  # LONG
                    pnl = (
                        (exit_price - pos_entry_price) * pos_qty
                        - pos_entry_fees
                        - exit_fees
                    )
                else:  # SHORT
                    pnl = (
                        (pos_entry_price - exit_price) * pos_qty
                        - pos_entry_fees
                        - exit_fees
                    )

                # Enregistrer trade
                trade_results[trade_count, 0] = pos_entry_bar
                trade_results[trade_count, 1] = i
                trade_results[trade_count, 2] = pos_side
                trade_results[trade_count, 3] = pos_qty
                trade_results[trade_count, 4] = pos_entry_price
                trade_results[trade_count, 5] = exit_price
                trade_results[trade_count, 6] = pos_entry_fees
                trade_results[trade_count, 7] = exit_fees
                trade_results[trade_count, 8] = pnl
                trade_results[trade_count, 9] = pos_stop
                trade_count += 1

                # Mise à jour cash
                cash += pnl + (pos_entry_price * pos_qty)

                # Reset position
                has_position = False
                pos_side = 0
                pos_qty = 0.0

        # === NOUVEAU SIGNAL D'ENTRÉE ===
        if not has_position and (signal == 1 or signal == 2):
            # Position sizing basé sur risque
            stop_distance_pct = stop_loss_pct / 100.0
            risk_amount = cash * risk_per_trade
            position_size = risk_amount / (current_price * stop_distance_pct)

            # Limite par levier
            max_position_size = (cash * leverage) / current_price
            qty = min(position_size, max_position_size)

            if qty > 0:
                # Calcul stop et TP
                if signal == 1:  # LONG
                    stop_price = current_price * (1.0 - stop_distance_pct)
                    tp_price = current_price * (1.0 + take_profit_pct / 100.0)
                else:  # SHORT
                    stop_price = current_price * (1.0 + stop_distance_pct)
                    tp_price = current_price * (1.0 - take_profit_pct / 100.0)

                # Frais entrée
                entry_value = current_price * qty
                entry_fees = entry_value * fee_rate

                if entry_value + entry_fees <= cash:
                    # Ouvrir position
                    has_position = True
                    pos_side = signal
                    pos_qty = qty
                    pos_entry_price = current_price
                    pos_stop = stop_price
                    pos_take_profit = tp_price
                    pos_entry_bar = i
                    pos_entry_fees = entry_fees

                    # Déduire cash
                    cash -= entry_value + entry_fees

        # === MISE À JOUR ÉQUITÉ ===
        if has_position:
            if pos_side == 1:  # LONG
                unrealized = (current_price - pos_entry_price) * pos_qty
            else:  # SHORT
                unrealized = (pos_entry_price - current_price) * pos_qty
            equity[i] = cash + unrealized + (pos_entry_price * pos_qty)
        else:
            equity[i] = cash

    # Fermeture position finale si nécessaire
    if has_position:
        final_price = close_vals[-1]
        exit_value = final_price * pos_qty
        exit_fees = exit_value * fee_rate

        if pos_side == 1:
            pnl = (
                (final_price - pos_entry_price) * pos_qty - pos_entry_fees - exit_fees
            )
        else:
            pnl = (
                (pos_entry_price - final_price) * pos_qty - pos_entry_fees - exit_fees
            )

        trade_results[trade_count, 0] = pos_entry_bar
        trade_results[trade_count, 1] = n_bars - 1
        trade_results[trade_count, 2] = pos_side
        trade_results[trade_count, 3] = pos_qty
        trade_results[trade_count, 4] = pos_entry_price
        trade_results[trade_count, 5] = final_price
        trade_results[trade_count, 6] = pos_entry_fees
        trade_results[trade_count, 7] = exit_fees
        trade_results[trade_count, 8] = pnl
        trade_results[trade_count, 9] = pos_stop
        trade_count += 1

        equity[-1] = cash + pnl + (pos_entry_price * pos_qty)

    return equity, trade_results[:trade_count]


# ==========================================
# STRATEGY PARAMETERS
# ==========================================


@dataclass
class EMAStochScalpParams:
    """
    Paramètres de la stratégie EMA + Stochastic Scalping.

    Attributes:
        fast_ema: Période EMA rapide (défaut: 50)
        slow_ema: Période EMA lente (défaut: 100)
        stoch_k: Période Stochastic %K (défaut: 14)
        stoch_d: Période Stochastic %D (défaut: 3)
        stoch_smooth: Période de lissage %K (défaut: 3)
        stoch_oversold: Seuil survente (défaut: 20)
        stoch_overbought: Seuil surachat (défaut: 80)
        require_pullback: Exiger pullback vers EMAs (défaut: True)
        pullback_tolerance_pct: Distance max EMAs en % (défaut: 0.3%)
        volume_ma_period: Période SMA volume (défaut: 10)
        volume_threshold: Multiplicateur volume (défaut: 1.2 = +20%)
        stop_loss_pct: Stop loss en % (défaut: 0.5%)
        take_profit_pct: Take profit en % (défaut: 0.7%)
        risk_per_trade: Risque par trade (défaut: 0.01 = 1%)
        leverage: Levier (défaut: 15.0)
        max_hold_bars: Durée max position (défaut: 30 barres)
        fee_bps: Frais en basis points (défaut: 4.5)
        slippage_bps: Slippage en basis points (défaut: 2.0)
        meta: Métadonnées personnalisées
    """

    # Indicateurs
    fast_ema: int = 50
    slow_ema: int = 100
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # Filtres d'entrée
    require_pullback: bool = True
    pullback_tolerance_pct: float = 0.3  # 0.3% de distance max aux EMAs
    volume_ma_period: int = 10
    volume_threshold: float = 1.2  # Volume > 1.2x moyenne

    # Risk Management
    stop_loss_pct: float = 0.5  # 0.5% stop loss (scalping serré)
    take_profit_pct: float = 0.7  # 0.7% take profit
    risk_per_trade: float = 0.01  # 1% du capital par trade

    # Position Management
    leverage: float = 15.0  # Levier scalping (10-20x recommandé)
    max_hold_bars: int = 30  # ~30 minutes en 1min

    # Frais et slippage
    fee_bps: float = 4.5  # 4.5 bps
    slippage_bps: float = 2.0  # Slippage scalping

    # Métadonnées
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertit les paramètres en dictionnaire"""
        return {
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "stoch_k": self.stoch_k,
            "stoch_d": self.stoch_d,
            "stoch_smooth": self.stoch_smooth,
            "stoch_oversold": self.stoch_oversold,
            "stoch_overbought": self.stoch_overbought,
            "require_pullback": self.require_pullback,
            "pullback_tolerance_pct": self.pullback_tolerance_pct,
            "volume_ma_period": self.volume_ma_period,
            "volume_threshold": self.volume_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_per_trade": self.risk_per_trade,
            "leverage": self.leverage,
            "max_hold_bars": self.max_hold_bars,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EMAStochScalpParams":
        """Crée les paramètres depuis un dictionnaire"""
        return cls(
            fast_ema=data.get("fast_ema", 50),
            slow_ema=data.get("slow_ema", 100),
            stoch_k=data.get("stoch_k", 14),
            stoch_d=data.get("stoch_d", 3),
            stoch_smooth=data.get("stoch_smooth", 3),
            stoch_oversold=data.get("stoch_oversold", 20.0),
            stoch_overbought=data.get("stoch_overbought", 80.0),
            require_pullback=data.get("require_pullback", True),
            pullback_tolerance_pct=data.get("pullback_tolerance_pct", 0.3),
            volume_ma_period=data.get("volume_ma_period", 10),
            volume_threshold=data.get("volume_threshold", 1.2),
            stop_loss_pct=data.get("stop_loss_pct", 0.5),
            take_profit_pct=data.get("take_profit_pct", 0.7),
            risk_per_trade=data.get("risk_per_trade", 0.01),
            leverage=data.get("leverage", 15.0),
            max_hold_bars=data.get("max_hold_bars", 30),
            fee_bps=data.get("fee_bps", 4.5),
            slippage_bps=data.get("slippage_bps", 2.0),
            meta=data.get("meta", {}),
        )


# ==========================================
# STRATEGY IMPLEMENTATION
# ==========================================


class EMAStochScalpStrategy:
    """
    Stratégie EMA 50/100 + Stochastic pour scalping crypto 1-minute.

    Cette stratégie combine:
    - Tendance: Croisement EMA 50/100
    - Timing: Stochastic pour zones survente/surachat
    - Confirmation: Pullback + Volume

    Win rate attendu: 65-80% avec discipline stricte
    Meilleur sur BTC/USDT, ETH/USDT, SOL/USDT en haute volatilité

    Example:
        >>> strategy = EMAStochScalpStrategy(symbol="BTCUSDT", timeframe="1m")
        >>> params = EMAStochScalpParams(stop_loss_pct=0.5, take_profit_pct=0.7)
        >>> equity, stats = strategy.backtest(df, params.to_dict())
        >>> print(f"Win Rate: {stats.win_rate_pct:.1f}%")
    """

    def __init__(
        self,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m",
        indicator_bank: Any = None,
    ):
        """
        Initialise la stratégie EMA + Stochastic Scalp.

        Args:
            symbol: Symbole (ex: BTCUSDT)
            timeframe: Timeframe (recommandé: 1m, 2m, 3m)
            indicator_bank: Instance IndicatorBank partagée (optionnel)
        """
        self.name = "EMA_Stochastic_Scalp"
        self.version = "1.0.0"
        self.symbol = symbol
        self.timeframe = timeframe
        self.indicator_bank = indicator_bank
        logger.info(
            f"Initialisation {self.name} v{self.version} ({symbol}/{timeframe})"
        )

    def _calculate_indicators(
        self, df: pd.DataFrame, params: EMAStochScalpParams
    ) -> pd.DataFrame:
        """
        Calcule tous les indicateurs nécessaires.

        Args:
            df: DataFrame OHLCV
            params: Paramètres de stratégie

        Returns:
            DataFrame avec indicateurs
        """
        df_ind = df.copy()

        # Import des indicateurs ici pour éviter imports circulaires
        from threadx.indicators.ema import ema, EMASettings
        from threadx.indicators.stochastic import (
            stochastic_oscillator,
            StochasticSettings,
        )

        # EMA 50 et 100
        ema_fast = ema(df["close"], EMASettings(period=params.fast_ema))
        ema_slow = ema(df["close"], EMASettings(period=params.slow_ema))

        df_ind["ema_fast"] = ema_fast
        df_ind["ema_slow"] = ema_slow

        # Stochastic
        stoch_settings = StochasticSettings(
            k_period=params.stoch_k,
            d_period=params.stoch_d,
            smooth_k=params.stoch_smooth,
            oversold=params.stoch_oversold,
            overbought=params.stoch_overbought,
        )

        stoch_k, stoch_d = stochastic_oscillator(
            df["high"], df["low"], df["close"], stoch_settings
        )

        df_ind["stoch_k"] = stoch_k
        df_ind["stoch_d"] = stoch_d

        # Volume SMA
        df_ind["volume_sma"] = (
            df["volume"].rolling(window=params.volume_ma_period).mean()
        )

        return df_ind

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Génère les signaux de trading EMA + Stochastic.

        Args:
            df: DataFrame OHLCV
            params: Paramètres de stratégie

        Returns:
            DataFrame avec colonne 'signal'
        """
        validate_ohlcv_dataframe(df)
        p = EMAStochScalpParams.from_dict(params)

        # Calcul indicateurs
        df_signals = self._calculate_indicators(df, p)

        # Initialisation signaux
        df_signals["signal"] = "HOLD"

        # Détection croisements EMA
        ema_fast = df_signals["ema_fast"].values
        ema_slow = df_signals["ema_slow"].values
        stoch_k = df_signals["stoch_k"].values
        stoch_d = df_signals["stoch_d"].values
        close = df_signals["close"].values
        volume = df_signals["volume"].values
        volume_sma = df_signals["volume_sma"].values

        # Tendance EMA (pas seulement croisement instantané)
        fast_above_slow = ema_fast > ema_slow

        # Conditions Stochastic
        stoch_oversold = stoch_k < p.stoch_oversold
        stoch_overbought = stoch_k > p.stoch_overbought

        # Sortie de zone + croisement %K/%D
        was_oversold = np.roll(stoch_oversold, 1)
        was_oversold[0] = False
        was_overbought = np.roll(stoch_overbought, 1)
        was_overbought[0] = False

        # Stochastic haussier: sortie de survente OU croisement %K > %D dans zone 20-40
        k_cross_d_up = (stoch_k > stoch_d) & (np.roll(stoch_k, 1) <= np.roll(stoch_d, 1))
        stoch_bullish = (
            (was_oversold & ~stoch_oversold)
            | (k_cross_d_up & (stoch_k >= 20) & (stoch_k <= 50))
            | ((stoch_k > stoch_d) & (stoch_k < 40))  # %K au-dessus %D en zone basse
        )

        # Stochastic baissier: sortie de surachat OU croisement %K < %D dans zone 60-80
        k_cross_d_down = (stoch_k < stoch_d) & (np.roll(stoch_k, 1) >= np.roll(stoch_d, 1))
        stoch_bearish = (
            (was_overbought & ~stoch_overbought)
            | (k_cross_d_down & (stoch_k >= 50) & (stoch_k <= 80))
            | ((stoch_k < stoch_d) & (stoch_k > 60))  # %K en-dessous %D en zone haute
        )

        # Pullback check (prix proche des EMAs)
        if p.require_pullback:
            avg_ema = (ema_fast + ema_slow) / 2.0
            distance_pct = np.abs((close - avg_ema) / avg_ema) * 100.0
            near_emas = distance_pct <= p.pullback_tolerance_pct
        else:
            near_emas = np.ones(len(df), dtype=bool)

        # Volume en hausse (comparé à SMA)
        volume_ok = volume > (volume_sma * p.volume_threshold)

        # === SIGNAUX LONG ===
        # Tendance haussière (EMA fast > slow) + Stochastic haussier + Volume + Pullback optionnel
        long_signal = fast_above_slow & stoch_bullish & volume_ok & near_emas

        # === SIGNAUX SHORT ===
        # Tendance baissière (EMA fast < slow) + Stochastic baissier + Volume + Pullback optionnel
        short_signal = (~fast_above_slow) & stoch_bearish & volume_ok & near_emas

        # Application des signaux
        df_signals.loc[long_signal, "signal"] = "ENTER_LONG"
        df_signals.loc[short_signal, "signal"] = "ENTER_SHORT"

        logger.debug(
            f"Signaux générés: {long_signal.sum()} LONG, {short_signal.sum()} SHORT"
        )

        return df_signals

    def backtest(
        self,
        df: pd.DataFrame,
        params: dict,
        initial_capital: float = 10000.0,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        precomputed_indicators: dict | None = None,
    ) -> tuple[pd.Series, RunStats]:
        """
        Exécute un backtest complet de la stratégie.

        Args:
            df: DataFrame OHLCV
            params: Paramètres de stratégie
            initial_capital: Capital initial
            fee_bps: Frais en basis points (override params)
            slippage_bps: Slippage en basis points (override params)
            precomputed_indicators: Indicateurs précalculés (non utilisé)

        Returns:
            Tuple (equity_curve, stats)
        """
        logger.info(
            f"Début backtest {self.name} sur {len(df)} barres, capital={initial_capital}"
        )

        validate_ohlcv_dataframe(df)
        p = EMAStochScalpParams.from_dict(params)

        # Override frais si fournis
        if fee_bps is not None:
            p.fee_bps = fee_bps
        if slippage_bps is not None:
            p.slippage_bps = slippage_bps

        # Génération signaux
        df_signals = self.generate_signals(df, params)

        # Encodage signaux pour Numba
        signal_map = {"HOLD": 0, "ENTER_LONG": 1, "ENTER_SHORT": 2}
        signal_vals = (
            df_signals["signal"].map(signal_map).fillna(0).astype(np.int32).values
        )

        # Préparation données Numba
        close_vals = df["close"].values.astype(np.float64)
        high_vals = df["high"].values.astype(np.float64)
        low_vals = df["low"].values.astype(np.float64)
        fee_rate = (p.fee_bps + p.slippage_bps) / 10000.0

        # Exécution backtest Numba
        equity_curve, trade_results = _backtest_loop_numba(
            close_vals=close_vals,
            high_vals=high_vals,
            low_vals=low_vals,
            signal_vals=signal_vals,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            stop_loss_pct=p.stop_loss_pct,
            take_profit_pct=p.take_profit_pct,
            risk_per_trade=p.risk_per_trade,
            leverage=p.leverage,
            max_hold_bars=p.max_hold_bars,
        )

        # Conversion résultats en objets Trade
        trades = []
        for row in trade_results:
            entry_bar = int(row[0])
            exit_bar = int(row[1])
            side = "LONG" if row[2] == 1 else "SHORT"

            trade = Trade(
                side=side,
                qty=row[3],
                entry_price=row[4],
                entry_time=df.index[entry_bar].isoformat(),
                exit_price=row[5],
                exit_time=df.index[exit_bar].isoformat(),
                stop=row[9],
                pnl_realized=row[8],
                fees_paid=row[6] + row[7],
                meta={"strategy": self.name, "params": params},
            )
            trades.append(trade)

        # Création série équité
        equity_series = pd.Series(equity_curve, index=df.index)

        # Calcul statistiques
        stats = RunStats.from_trades_and_equity(
            trades=trades,
            equity_curve=equity_series,
            initial_capital=initial_capital,
            meta={"strategy": self.name, "params": params, "fee_bps": p.fee_bps},
        )

        logger.info(
            f"Backtest terminé: {stats.total_trades} trades, "
            f"Win Rate={stats.win_rate_pct:.1f}%, "
            f"PnL={stats.total_pnl:.2f} ({stats.total_pnl_pct:.2f}%)"
        )

        return equity_series, stats


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    "EMAStochScalpStrategy",
    "EMAStochScalpParams",
]
