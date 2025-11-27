"""
ThreadX - EMA Cross Strategy
============================

Stratégie EMA Crossover simple.
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

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def _backtest_loop_numba(
    close_vals: np.ndarray,
    signal_vals: np.ndarray,
    initial_capital: float,
    fee_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    risk_per_trade: float,
    leverage: float,
    max_hold_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Reuse logic from MA Crossover or similar
    # For brevity, I will implement a simplified version or copy from ma_crossover.py
    # Since I cannot import from another strategy file easily in Numba without code duplication or shared utils
    # I will duplicate the loop for now to ensure independence.
    
    n_bars = len(close_vals)
    equity = np.full(n_bars, initial_capital, dtype=np.float64)
    trade_results = np.zeros((n_bars, 10), dtype=np.float64)
    trade_count = 0
    cash = initial_capital
    has_position = False
    pos_side = 0
    pos_qty = 0.0
    pos_entry_price = 0.0
    pos_stop = 0.0
    pos_take_profit = 0.0
    pos_entry_bar = 0
    pos_entry_fees = 0.0

    for i in range(n_bars):
        current_price = close_vals[i]
        signal = signal_vals[i]

        if has_position:
            should_exit = False
            if pos_side == 1:
                if current_price <= pos_stop: should_exit = True
                elif current_price >= pos_take_profit: should_exit = True
                elif signal == 2: should_exit = True
            else:
                if current_price >= pos_stop: should_exit = True
                elif current_price <= pos_take_profit: should_exit = True
                elif signal == 1: should_exit = True
            
            if not should_exit and (i - pos_entry_bar >= max_hold_bars):
                should_exit = True

            if should_exit:
                exit_value = current_price * pos_qty
                exit_fees = exit_value * fee_rate
                if pos_side == 1:
                    pnl = (current_price - pos_entry_price) * pos_qty - pos_entry_fees - exit_fees
                else:
                    pnl = (pos_entry_price - current_price) * pos_qty - pos_entry_fees - exit_fees
                
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
                cash += pnl + (pos_entry_price * pos_qty)
                has_position = False
                pos_side = 0
                pos_qty = 0.0

        if not has_position and (signal == 1 or signal == 2):
            stop_distance_pct = stop_loss_pct / 100.0
            risk_amount = cash * risk_per_trade
            position_size = risk_amount / (current_price * stop_distance_pct)
            max_position_size = (cash * leverage) / current_price
            qty = min(position_size, max_position_size)

            if qty > 0:
                if signal == 1:
                    stop_price = current_price * (1.0 - stop_distance_pct)
                    tp_price = current_price * (1.0 + take_profit_pct / 100.0)
                else:
                    stop_price = current_price * (1.0 + stop_distance_pct)
                    tp_price = current_price * (1.0 - take_profit_pct / 100.0)
                
                entry_value = current_price * qty
                entry_fees = entry_value * fee_rate
                if entry_value + entry_fees <= cash:
                    has_position = True
                    pos_side = signal
                    pos_qty = qty
                    pos_entry_price = current_price
                    pos_stop = stop_price
                    pos_take_profit = tp_price
                    pos_entry_bar = i
                    pos_entry_fees = entry_fees
                    cash -= entry_value + entry_fees

        if has_position:
            if pos_side == 1: unrealized = (current_price - pos_entry_price) * pos_qty
            else: unrealized = (pos_entry_price - current_price) * pos_qty
            equity[i] = cash + unrealized + (pos_entry_price * pos_qty)
        else:
            equity[i] = cash

    return equity, trade_results[:trade_count]

@dataclass
class EMACrossParams:
    fast_window: int = 12
    slow_window: int = 26
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    risk_per_trade: float = 0.01
    leverage: float = 1.0
    max_hold_bars: int = 100
    fee_bps: float = 4.5
    slippage_bps: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "EMACrossParams":
        return cls(
            fast_window=data.get("fast_window", 12),
            slow_window=data.get("slow_window", 26),
            stop_loss_pct=data.get("stop_loss_pct", 2.0),
            take_profit_pct=data.get("take_profit_pct", 4.0),
            risk_per_trade=data.get("risk_per_trade", 0.01),
            leverage=data.get("leverage", 1.0),
            max_hold_bars=data.get("max_hold_bars", 100),
            fee_bps=data.get("fee_bps", 4.5),
            slippage_bps=data.get("slippage_bps", 0.0),
            meta=data.get("meta", {}),
        )

class EMACrossStrategy:
    def __init__(self, symbol: str = "UNKNOWN", timeframe: str = "15m", indicator_bank: Any = None):
        self.name = "EMA_Cross"
        self.symbol = symbol
        self.timeframe = timeframe
        self.indicator_bank = indicator_bank

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        p = EMACrossParams.from_dict(params)
        df_signals = df.copy()
        
        # EMA Calculation
        fast_ema = df["close"].ewm(span=p.fast_window, adjust=False).mean()
        slow_ema = df["close"].ewm(span=p.slow_window, adjust=False).mean()
        
        df_signals["signal"] = "HOLD"
        cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        df_signals.loc[cross_up, "signal"] = "ENTER_LONG"
        cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        df_signals.loc[cross_down, "signal"] = "ENTER_SHORT"
        
        return df_signals

    def backtest(self, df: pd.DataFrame, params: dict, initial_capital: float = 10000.0, fee_bps: float = None, slippage_bps: float = None, precomputed_indicators: dict = None) -> tuple[pd.Series, RunStats]:
        p = EMACrossParams.from_dict(params)
        if fee_bps is not None: p.fee_bps = fee_bps
        if slippage_bps is not None: p.slippage_bps = slippage_bps
        
        df_signals = self.generate_signals(df, params)
        signal_map = {"HOLD": 0, "ENTER_LONG": 1, "ENTER_SHORT": 2, "EXIT": 3}
        signal_vals = df_signals["signal"].map(signal_map).fillna(0).astype(np.int32).values
        close_vals = df["close"].values.astype(np.float64)
        fee_rate = (p.fee_bps + p.slippage_bps) / 10000.0
        
        equity_curve, trade_results = _backtest_loop_numba(
            close_vals, signal_vals, initial_capital, fee_rate,
            p.stop_loss_pct, p.take_profit_pct, p.risk_per_trade,
            p.leverage, p.max_hold_bars
        )
        
        trades = []
        for row in trade_results:
            entry_bar, exit_bar = int(row[0]), int(row[1])
            side = "LONG" if row[2] == 1 else "SHORT"
            trades.append(Trade(
                side=side, qty=row[3], entry_price=row[4], entry_time=df.index[entry_bar].isoformat(),
                exit_price=row[5], exit_time=df.index[exit_bar].isoformat(), stop=row[9],
                pnl_realized=row[8], fees_paid=row[6]+row[7], meta={"strategy": self.name, "params": params}
            ))
            
        equity_series = pd.Series(equity_curve, index=df.index)
        stats = RunStats.from_trades_and_equity(
            trades, 
            equity_series, 
            initial_capital, 
            meta={
                "strategy": self.name, 
                "trades": [t.to_dict() for t in trades]
            }
        )
        return equity_series, stats
