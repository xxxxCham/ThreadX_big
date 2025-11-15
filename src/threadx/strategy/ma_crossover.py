"""
ThreadX - MA Crossover Strategy (Validation/Test)
==================================================

Stratégie simple Moving Average Crossover pour tester le moteur de backtest.

Objectif: Validation système, pas optimisation performance
- Règles claires et facilement vérifiables
- Stops et TP fixes (% du prix)
- Pas de levier par défaut
- Position sizing simple

Utilisation:
    >>> from threadx.strategy.ma_crossover import MACrossoverStrategy
    >>> strategy = MACrossoverStrategy()
    >>> params = {"fast_period": 10, "slow_period": 20, "stop_pct": 2.0}
    >>> equity, stats = strategy.backtest(df, params)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from threadx.strategy.model import (
    RunStats,
    Trade,
    validate_ohlcv_dataframe,
    validate_strategy_params,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# ==========================================
# NUMBA OPTIMIZED BACKTEST LOOP
# ==========================================


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def _backtest_loop_numba(
    close_vals: np.ndarray,
    signal_vals: np.ndarray,  # 0=HOLD, 1=ENTER_LONG, 2=ENTER_SHORT, 3=EXIT
    initial_capital: float,
    fee_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    risk_per_trade: float,
    leverage: float,
    max_hold_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boucle de backtest MA Crossover optimisée Numba.

    Args:
        close_vals: Prix de clôture (n_bars,)
        signal_vals: Signaux encodés (n_bars,) - 0:HOLD, 1:LONG, 2:SHORT, 3:EXIT
        initial_capital: Capital de départ
        fee_rate: Taux de frais total (fee + slippage)
        stop_loss_pct: Stop loss en % (ex: 2.0 = -2%)
        take_profit_pct: Take profit en % (ex: 4.0 = +4%)
        risk_per_trade: Fraction capital risquée par trade
        leverage: Levier autorisé
        max_hold_bars: Durée max position

    Returns:
        Tuple (equity_curve, trade_results) où:
        - equity_curve: np.ndarray(n_bars,) équité à chaque barre
        - trade_results: np.ndarray(max_trades, 10) résultats trades
          Colonnes: [entry_bar, exit_bar, side, qty, entry_price, exit_price,
                     entry_fees, exit_fees, pnl, stop_price]
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
        signal = signal_vals[i]

        # === GESTION POSITION EXISTANTE ===
        if has_position:
            should_exit = False

            # 1. Stop loss check
            if pos_side == 1:  # LONG
                if current_price <= pos_stop:
                    should_exit = True
            else:  # SHORT
                if current_price >= pos_stop:
                    should_exit = True

            # 2. Take profit check
            if not should_exit:
                if pos_side == 1 and current_price >= pos_take_profit:
                    should_exit = True
                elif pos_side == 2 and current_price <= pos_take_profit:
                    should_exit = True

            # 3. Signal inverse (croisement opposé)
            if not should_exit:
                if pos_side == 1 and signal == 2:  # LONG → SHORT signal
                    should_exit = True
                elif pos_side == 2 and signal == 1:  # SHORT → LONG signal
                    should_exit = True

            # 4. Max hold bars
            if not should_exit:
                bars_held = i - pos_entry_bar
                if bars_held >= max_hold_bars:
                    should_exit = True

            # === FERMETURE POSITION ===
            if should_exit:
                exit_value = current_price * pos_qty
                exit_fees = exit_value * fee_rate

                # Calcul PnL
                if pos_side == 1:  # LONG
                    pnl = (
                        (current_price - pos_entry_price) * pos_qty
                        - pos_entry_fees
                        - exit_fees
                    )
                else:  # SHORT
                    pnl = (
                        (pos_entry_price - current_price) * pos_qty
                        - pos_entry_fees
                        - exit_fees
                    )

                # Enregistrer trade
                trade_results[trade_count, 0] = pos_entry_bar
                trade_results[trade_count, 1] = i
                trade_results[trade_count, 2] = pos_side
                trade_results[trade_count, 3] = pos_qty
                trade_results[trade_count, 4] = pos_entry_price
                trade_results[trade_count, 5] = current_price
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
            pnl = (final_price - pos_entry_price) * pos_qty - pos_entry_fees - exit_fees
        else:
            pnl = (pos_entry_price - final_price) * pos_qty - pos_entry_fees - exit_fees

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

        # Mise à jour équité finale
        equity[-1] = cash + pnl + (pos_entry_price * pos_qty)

    # Retourner seulement les trades valides
    return equity, trade_results[:trade_count]


# ==========================================
# STRATEGY PARAMETERS
# ==========================================


@dataclass
class MACrossoverParams:
    """
    Paramètres de la stratégie MA Crossover.

    Attributes:
        fast_period: Période SMA rapide (défaut: 10)
        slow_period: Période SMA lente (défaut: 30)
        stop_loss_pct: Stop loss en % (défaut: 2.0%)
        take_profit_pct: Take profit en % (défaut: 4.0%)
        risk_per_trade: Risque par trade en fraction du capital (défaut: 0.01 = 1%)
        leverage: Effet de levier (défaut: 1.0)
        max_hold_bars: Durée max position en barres (défaut: 100)
        fee_bps: Frais en basis points (défaut: 4.5)
        slippage_bps: Slippage en basis points (défaut: 0)
        meta: Métadonnées personnalisées

    Example:
        >>> params = MACrossoverParams(fast_period=10, slow_period=30, stop_loss_pct=2.0)
        >>> strategy = MACrossoverStrategy()
        >>> equity, stats = strategy.backtest(df, params.to_dict())
    """

    # Moving Averages
    fast_period: int = 10
    slow_period: int = 30

    # Risk Management
    stop_loss_pct: float = 2.0  # 2% stop loss
    take_profit_pct: float = 4.0  # 4% take profit
    risk_per_trade: float = 0.01  # 1% du capital par trade

    # Position Management
    leverage: float = 1.0  # Pas de levier par défaut
    max_hold_bars: int = 100  # ~4 jours en 1h

    # Frais et slippage
    fee_bps: float = 4.5  # 4.5 bps
    slippage_bps: float = 0.0

    # Métadonnées
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convertit les paramètres en dictionnaire"""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
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
    def from_dict(cls, data: dict) -> "MACrossoverParams":
        """Crée les paramètres depuis un dictionnaire"""
        return cls(
            fast_period=data.get("fast_period", 10),
            slow_period=data.get("slow_period", 30),
            stop_loss_pct=data.get("stop_loss_pct", 2.0),
            take_profit_pct=data.get("take_profit_pct", 4.0),
            risk_per_trade=data.get("risk_per_trade", 0.01),
            leverage=data.get("leverage", 1.0),
            max_hold_bars=data.get("max_hold_bars", 100),
            fee_bps=data.get("fee_bps", 4.5),
            slippage_bps=data.get("slippage_bps", 0.0),
            meta=data.get("meta", {}),
        )


# ==========================================
# STRATEGY IMPLEMENTATION
# ==========================================


class MACrossoverStrategy:
    """
    Stratégie Moving Average Crossover pour validation système.

    Règles:
    - LONG: SMA rapide croise au-dessus SMA lente
    - SHORT: SMA rapide croise en-dessous SMA lente
    - EXIT: Signal inverse, stop loss, take profit, ou max hold

    Example:
        >>> strategy = MACrossoverStrategy()
        >>> params = {"fast_period": 10, "slow_period": 30}
        >>> signals = strategy.generate_signals(df, params)
        >>> equity, stats = strategy.backtest(df, params, initial_capital=10000)
    """

    def __init__(
        self,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
        indicator_bank: Any = None,
    ):
        """
        Initialise la stratégie MA Crossover.

        Args:
            symbol: Symbole pour cache d'indicateurs (non utilisé pour MA simple)
            timeframe: Timeframe pour cache d'indicateurs
            indicator_bank: Instance IndicatorBank partagée (optionnel, non utilisé pour SMA)
        """
        self.name = "MA_Crossover"
        self.version = "1.0.0"
        self.symbol = symbol
        self.timeframe = timeframe
        self.indicator_bank = indicator_bank
        logger.info(f"Initialisation {self.name} v{self.version} ({symbol}/{timeframe})")

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Génère les signaux MA Crossover.

        Args:
            df: DataFrame OHLCV
            params: Paramètres de stratégie

        Returns:
            DataFrame avec colonne 'signal'
        """
        validate_ohlcv_dataframe(df)
        validate_strategy_params(params, ["fast_period", "slow_period"])

        p = MACrossoverParams.from_dict(params)
        df_signals = df.copy()

        # Calcul des moyennes mobiles
        fast_sma = df["close"].rolling(window=p.fast_period, min_periods=p.fast_period).mean()
        slow_sma = df["close"].rolling(window=p.slow_period, min_periods=p.slow_period).mean()

        # Détection croisements
        df_signals["signal"] = "HOLD"

        # Crossover up: LONG
        cross_up = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
        df_signals.loc[cross_up, "signal"] = "ENTER_LONG"

        # Crossover down: SHORT
        cross_down = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
        df_signals.loc[cross_down, "signal"] = "ENTER_SHORT"

        # Métadonnées pour analyse
        df_signals["fast_sma"] = fast_sma
        df_signals["slow_sma"] = slow_sma

        logger.debug(
            f"Signaux générés: {(df_signals['signal'] == 'ENTER_LONG').sum()} LONG, "
            f"{(df_signals['signal'] == 'ENTER_SHORT').sum()} SHORT"
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
            precomputed_indicators: Indicateurs précalculés (non utilisé pour MA simple)

        Returns:
            Tuple (equity_curve, stats)
        """
        logger.info(
            f"Début backtest {self.name} sur {len(df)} barres, capital={initial_capital}"
        )

        # Validation
        validate_ohlcv_dataframe(df)
        p = MACrossoverParams.from_dict(params)

        # Override frais si fournis
        if fee_bps is not None:
            p.fee_bps = fee_bps
        if slippage_bps is not None:
            p.slippage_bps = slippage_bps

        # Génération signaux
        df_signals = self.generate_signals(df, params)

        # Encodage signaux pour Numba
        signal_map = {"HOLD": 0, "ENTER_LONG": 1, "ENTER_SHORT": 2, "EXIT": 3}
        signal_vals = df_signals["signal"].map(signal_map).fillna(0).astype(np.int32).values

        # Préparation données Numba
        close_vals = df["close"].values.astype(np.float64)
        fee_rate = (p.fee_bps + p.slippage_bps) / 10000.0

        # Exécution backtest Numba
        equity_curve, trade_results = _backtest_loop_numba(
            close_vals=close_vals,
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
            f"Backtest terminé: {stats.total_trades} trades, PnL={stats.total_pnl:.2f} ({stats.total_pnl_pct:.2f}%)"
        )

        return equity_series, stats


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    "MACrossoverStrategy",
    "MACrossoverParams",
]
