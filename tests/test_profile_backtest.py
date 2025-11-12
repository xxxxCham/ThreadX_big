"""
Profilage détaillé du backtest BBAtrStrategy.

Identifie EXACTEMENT quelle partie du backtest prend le plus de temps.
"""

import time
import pandas as pd
import numpy as np
from threadx.strategy.bb_atr import BBAtrStrategy, BBAtrParams
from threadx.indicators.bank import IndicatorBank

# Préparation données
print("=" * 80)
print("🔬 PROFILAGE DÉTAILLÉ - BACKTEST BBATR STRATEGY")
print("=" * 80)
print()

print("📊 Génération données de test...")
dates = pd.date_range(start="2024-01-01", periods=5000, freq="1h")
df = pd.DataFrame(
    {
        "open": 50000 + np.random.randn(5000) * 100,
        "high": 50100 + np.random.randn(5000) * 100,
        "low": 49900 + np.random.randn(5000) * 100,
        "close": 50000 + np.random.randn(5000) * 100,
        "volume": np.random.randint(1000, 10000, 5000),
    },
    index=dates,
)
print(f"✅ {len(df)} barres générées")
print()

# Préparation indicateurs
print("🔧 Précomputation indicateurs...")
bank = IndicatorBank()

start = time.perf_counter()
bb_result = bank.ensure("bollinger", {"period": 20, "std": 2.0}, df, "TEST", "1h")
atr_result = bank.ensure("atr", {"period": 14, "method": "ema"}, df, "TEST", "1h")
precomputed_time = (time.perf_counter() - start) * 1000

import json

bb_key = json.dumps({"period": 20, "std": 2.0}, sort_keys=True, separators=(",", ":"))
atr_key = json.dumps(
    {"method": "ema", "period": 14}, sort_keys=True, separators=(",", ":")
)

precomputed = {"bollinger": {bb_key: bb_result}, "atr": {atr_key: atr_result}}
print(f"✅ Indicateurs prêts en {precomputed_time:.2f}ms")
print()

# Paramètres
params = {
    "bb_period": 20,
    "bb_std": 2.0,
    "entry_z": 1.0,
    "atr_period": 14,
    "atr_multiplier": 1.5,
}

# Stratégie
strategy = BBAtrStrategy(symbol="TEST", timeframe="1h")

print("=" * 80)
print("🧪 PROFILAGE PAR PHASE")
print("=" * 80)
print()

# ========================================
# PHASE 1: generate_signals()
# ========================================
print("📝 PHASE 1: generate_signals()")
print("-" * 80)

start = time.perf_counter()
signals_df = strategy.generate_signals(df, params, precomputed_indicators=precomputed)
signals_time = (time.perf_counter() - start) * 1000

print(f"   ⏱️  Temps: {signals_time:.2f}ms")
print(f"   📊 Signaux: {len(signals_df)} lignes générées")
print()

# ========================================
# PHASE 2: backtest() - CHRONOMÉTRAGE INTERNE
# ========================================
print("📝 PHASE 2: backtest() - Décomposition")
print("-" * 80)

# Injection de chronométrages pour profiler le backtest
import threadx.strategy.bb_atr as bb_atr_module

# Sauvegarder la méthode originale
original_backtest = strategy.backtest


def profiled_backtest(
    self,
    df,
    params,
    initial_capital=10000.0,
    fee_bps=4.5,
    slippage_bps=0.0,
    precomputed_indicators=None,
):
    """Backtest avec profilage détaillé"""
    timings = {}

    # Phase 1: Génération signaux
    t0 = time.perf_counter()
    signals_df = self.generate_signals(
        df, params, precomputed_indicators=precomputed_indicators
    )
    timings["generate_signals"] = (time.perf_counter() - t0) * 1000

    # Phase 2: Initialisation
    t0 = time.perf_counter()
    from threadx.strategy.model import (
        validate_ohlcv_dataframe,
        validate_strategy_params,
        RunStats,
        Trade,
    )

    validate_ohlcv_dataframe(df)
    strategy_params = BBAtrParams.from_dict(params)

    n_bars = len(df)
    equity = np.full(n_bars, initial_capital, dtype=float)
    cash = initial_capital
    position = None
    trades = []
    fee_rate = (fee_bps + slippage_bps) / 10000.0

    close_vals = signals_df["close"].values
    atr_vals = signals_df["atr"].values
    signal_vals = signals_df["signal"].values
    bb_middle_vals = signals_df["bb_middle"].values
    bb_z_vals = signals_df["bb_z"].values
    timestamps = signals_df.index.values
    has_tz = df.index.tz is not None
    timings["initialization"] = (time.perf_counter() - t0) * 1000

    # Phase 3: Boucle principale
    t0 = time.perf_counter()
    position_updates = 0
    entry_signals = 0
    exit_signals = 0

    for i in range(n_bars):
        current_price = close_vals[i]
        current_atr = atr_vals[i]
        signal = signal_vals[i]
        timestamp = (
            pd.Timestamp(timestamps[i], tz=df.index.tz)
            if has_tz
            else pd.Timestamp(timestamps[i])
        )

        if np.isnan(current_atr) or current_atr <= 0:
            equity[i] = (
                cash
                if position is None
                else cash + position.calculate_unrealized_pnl(current_price)
            )
            continue

        # Gestion position existante
        if position is not None:
            position_updates += 1
            should_exit = False
            exit_reason = ""

            # Vérifications stops
            if position.should_stop_loss(current_price):
                should_exit = True
                exit_reason = "stop_loss"
            elif position.is_long() and current_price >= bb_middle_vals[i]:
                should_exit = True
                exit_reason = "take_profit_bb_middle"
            elif position.is_short() and current_price <= bb_middle_vals[i]:
                should_exit = True
                exit_reason = "take_profit_bb_middle"

            entry_bar_index = position.meta.get("entry_bar_index", 0)
            bars_held = i - entry_bar_index
            if bars_held >= strategy_params.max_hold_bars:
                should_exit = True
                exit_reason = "max_hold_bars"

            # Trailing stop
            if strategy_params.trailing_stop and not should_exit:
                if position.is_long():
                    new_stop = current_price - (
                        current_atr * strategy_params.atr_multiplier
                    )
                    if new_stop > position.stop:
                        position.stop = new_stop
                else:
                    new_stop = current_price + (
                        current_atr * strategy_params.atr_multiplier
                    )
                    if new_stop < position.stop:
                        position.stop = new_stop

            # Fermeture
            if should_exit:
                exit_signals += 1
                exit_value = current_price * position.qty
                exit_fees = exit_value * fee_rate
                position.close_trade(
                    exit_price=current_price,
                    exit_time=(
                        str(timestamp)
                        if hasattr(timestamp, "isoformat")
                        else str(timestamp)
                    ),
                    exit_fees=exit_fees,
                )

                pnl_val = (
                    position.pnl_realized if position.pnl_realized is not None else 0.0
                )
                pnl_pct = abs(pnl_val / (position.entry_price * position.qty)) * 100
                if pnl_pct >= strategy_params.min_pnl_pct:
                    cash += pnl_val + (position.entry_price * position.qty)
                    trades.append(position)

                position = None

        # Nouveau signal
        if position is None and signal in ["ENTER_LONG", "ENTER_SHORT"]:
            entry_signals += 1
            atr_stop_distance = current_atr * strategy_params.atr_multiplier
            risk_amount = cash * strategy_params.risk_per_trade
            position_size = risk_amount / atr_stop_distance
            max_position_size = (cash * strategy_params.leverage) / current_price
            qty = min(position_size, max_position_size)

            if qty > 0:
                stop_price = (
                    current_price - atr_stop_distance
                    if signal == "ENTER_LONG"
                    else current_price + atr_stop_distance
                )
                entry_value = current_price * qty
                entry_fees = entry_value * fee_rate

                if entry_value + entry_fees <= cash:
                    position = Trade(
                        side=signal.replace("ENTER_", ""),
                        qty=qty,
                        entry_price=current_price,
                        entry_time=(
                            str(timestamp)
                            if hasattr(timestamp, "isoformat")
                            else str(timestamp)
                        ),
                        stop=stop_price,
                        fees_paid=entry_fees,
                        meta={
                            "bb_z": bb_z_vals[i],
                            "atr": current_atr,
                            "atr_multiplier": strategy_params.atr_multiplier,
                            "risk_per_trade": strategy_params.risk_per_trade,
                            "entry_bar_index": i,
                        },
                    )
                    cash -= entry_value + entry_fees

        # MAJ équité
        if position is not None:
            equity[i] = cash + position.calculate_unrealized_pnl(current_price)
        else:
            equity[i] = cash

    timings["main_loop"] = (time.perf_counter() - t0) * 1000

    # Phase 4: Finalisation
    t0 = time.perf_counter()
    if position is not None:
        final_price = df["close"].iloc[-1]
        position.close_trade(
            exit_price=final_price,
            exit_time=df.index[-1].isoformat(),
            exit_fees=final_price * position.qty * fee_rate,
        )
        pnl_val = position.pnl_realized if position.pnl_realized is not None else 0.0
        pnl_pct = abs(pnl_val / (position.entry_price * position.qty)) * 100
        if pnl_pct >= strategy_params.min_pnl_pct:
            trades.append(position)

    equity_curve = pd.Series(equity, index=df.index)
    run_stats = RunStats.from_trades_and_equity(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        meta={
            "strategy": "BBAtr",
            "params": params,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        },
    )
    timings["finalization"] = (time.perf_counter() - t0) * 1000

    # Affichage
    total_time = sum(timings.values())
    print(
        f"   ⏱️  generate_signals: {timings['generate_signals']:.2f}ms ({timings['generate_signals']/total_time*100:.1f}%)"
    )
    print(
        f"   ⏱️  initialization: {timings['initialization']:.2f}ms ({timings['initialization']/total_time*100:.1f}%)"
    )
    print(
        f"   ⏱️  main_loop: {timings['main_loop']:.2f}ms ({timings['main_loop']/total_time*100:.1f}%)"
    )
    print(
        f"   ⏱️  finalization: {timings['finalization']:.2f}ms ({timings['finalization']/total_time*100:.1f}%)"
    )
    print(f"   ⏱️  TOTAL: {total_time:.2f}ms")
    print()
    print(f"   📊 Statistiques boucle:")
    print(f"      - {n_bars} itérations")
    print(f"      - {position_updates} mises à jour de position")
    print(f"      - {entry_signals} signaux d'entrée")
    print(f"      - {exit_signals} sorties")
    print(f"      - {len(trades)} trades validés")
    print()

    return equity_curve, run_stats


# Patch temporaire
strategy.backtest = lambda *args, **kwargs: profiled_backtest(strategy, *args, **kwargs)

# Exécution
equity, stats = strategy.backtest(
    df=df,
    params=params,
    initial_capital=10000.0,
    fee_bps=4.5,
    slippage_bps=0.0,
    precomputed_indicators=precomputed,
)

print("=" * 80)
print("✅ PROFILAGE TERMINÉ")
print("=" * 80)
