"""Script de débogage de la stratégie EMA + Stochastic"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.strategy.ema_stochastic_scalp import (
    EMAStochScalpParams,
    EMAStochScalpStrategy,
)


def generate_trending_data(n_bars: int = 500) -> pd.DataFrame:
    """Génère des données avec une tendance claire pour forcer des signaux"""
    start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start_time, periods=n_bars, freq="1min")

    # Tendance haussière puis baissière
    trend = np.concatenate(
        [
            np.linspace(50000, 52000, n_bars // 2),  # Hausse
            np.linspace(52000, 50000, n_bars // 2),  # Baisse
        ]
    )

    # Ajout de volatilité
    noise = np.random.normal(0, 100, n_bars)
    close_prices = trend + noise

    high_prices = close_prices * 1.002
    low_prices = close_prices * 0.998
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    volume = 100 + np.random.uniform(0, 200, n_bars)

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=timestamps,
    )


def main():
    print("\n=== DÉBOGAGE STRATÉGIE EMA + STOCHASTIC ===\n")

    # Données avec tendance claire
    df = generate_trending_data(500)

    # Paramètres très permissifs
    params = EMAStochScalpParams(
        fast_ema=10,
        slow_ema=20,
        stoch_k=8,
        stoch_d=3,
        require_pullback=False,
        volume_threshold=0.5,
        stoch_oversold=40.0,
        stoch_overbought=60.0,
    )

    strategy = EMAStochScalpStrategy()
    df_signals = strategy.generate_signals(df, params.to_dict())

    # Analyse des indicateurs
    print("INDICATEURS:")
    print(f"  EMA Fast valides: {(~df_signals['ema_fast'].isna()).sum()}/{len(df)}")
    print(f"  EMA Slow valides: {(~df_signals['ema_slow'].isna()).sum()}/{len(df)}")
    print(f"  Stoch K valides: {(~df_signals['stoch_k'].isna()).sum()}/{len(df)}")
    print(f"  Stoch D valides: {(~df_signals['stoch_d'].isna()).sum()}/{len(df)}")
    print()

    # Croisements EMA
    ema_fast = df_signals["ema_fast"].values
    ema_slow = df_signals["ema_slow"].values
    valid = ~(np.isnan(ema_fast) | np.isnan(ema_slow))

    cross_up = (
        (ema_fast[1:] > ema_slow[1:])
        & (ema_fast[:-1] <= ema_slow[:-1])
        & valid[1:]
        & valid[:-1]
    )
    cross_down = (
        (ema_fast[1:] < ema_slow[1:])
        & (ema_fast[:-1] >= ema_slow[:-1])
        & valid[1:]
        & valid[:-1]
    )

    print(f"CROISEMENTS EMA:")
    print(f"  Croisements haussiers: {cross_up.sum()}")
    print(f"  Croisements baissiers: {cross_down.sum()}")
    print()

    # Stochastic
    stoch_k = df_signals["stoch_k"].values
    stoch_valid = ~np.isnan(stoch_k)
    print(f"STOCHASTIC:")
    print(f"  Min K: {np.nanmin(stoch_k):.2f}")
    print(f"  Max K: {np.nanmax(stoch_k):.2f}")
    print(f"  Moyenne K: {np.nanmean(stoch_k):.2f}")
    print(f"  K < 40: {(stoch_k[stoch_valid] < 40).sum()}")
    print(f"  K > 60: {(stoch_k[stoch_valid] > 60).sum()}")
    print()

    # Signaux
    long_count = (df_signals["signal"] == "ENTER_LONG").sum()
    short_count = (df_signals["signal"] == "ENTER_SHORT").sum()

    print(f"SIGNAUX GÉNÉRÉS:")
    print(f"  LONG: {long_count}")
    print(f"  SHORT: {short_count}")
    print()

    if long_count + short_count > 0:
        # Afficher quelques exemples de signaux
        signal_rows = df_signals[df_signals["signal"] != "HOLD"].head(5)
        print("EXEMPLES DE SIGNAUX:")
        print(
            signal_rows[
                ["close", "ema_fast", "ema_slow", "stoch_k", "stoch_d", "signal"]
            ]
        )
        print()

        # Backtest
        equity, stats = strategy.backtest(df, params.to_dict(), initial_capital=10000)
        print("RÉSULTATS BACKTEST:")
        print(f"  Total Trades: {stats.total_trades}")
        print(f"  Win Rate: {stats.win_rate_pct:.1f}%")
        print(f"  PnL: ${stats.total_pnl:.2f} ({stats.total_pnl_pct:+.2f}%)")
    else:
        print("⚠ Aucun signal généré - vérifier la logique de la stratégie")
        print("\nÉchantillon des données (premières lignes après période de warm-up):")
        sample = df_signals.iloc[50:60][
            ["close", "ema_fast", "ema_slow", "stoch_k", "stoch_d", "volume_sma"]
        ]
        print(sample)


if __name__ == "__main__":
    main()
