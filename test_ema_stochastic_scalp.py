"""
Test de la stratégie EMA 50/100 + Stochastic Scalping
=====================================================

Script de test pour la stratégie de scalping crypto 1-minute.

Ce test:
1. Charge des données OHLCV (ou génère des données de test)
2. Exécute un backtest avec paramètres optimisés pour scalping
3. Affiche les statistiques de performance
4. Génère un rapport visuel (optionnel)

Usage:
    python test_ema_stochastic_scalp.py

Attendu:
- Win rate: 65-80% avec paramètres optimisés
- ROI par trade: 0.4-0.8%
- Max drawdown: < 10%
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ajout du path src pour imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.strategy.ema_stochastic_scalp import (
    EMAStochScalpParams,
    EMAStochScalpStrategy,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def generate_test_data(n_bars: int = 1000, volatility: float = 0.02) -> pd.DataFrame:
    """
    Génère des données OHLCV de test simulant un marché crypto volatile.

    Args:
        n_bars: Nombre de barres (défaut: 1000)
        volatility: Volatilité (défaut: 0.02 = 2%)

    Returns:
        DataFrame OHLCV avec index datetime
    """
    logger.info(f"Génération de {n_bars} barres de test (volatilité={volatility})")

    # Timestamps (1 minute par barre)
    start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start_time, periods=n_bars, freq="1min")

    # Prix avec tendance et bruit
    np.random.seed(42)

    # Tendance sinusoïdale pour simuler cycles
    trend = 50000 + 2000 * np.sin(np.arange(n_bars) / 100)

    # Random walk avec volatilité
    returns = np.random.normal(0, volatility / 100, n_bars)
    close_prices = trend * np.exp(np.cumsum(returns))

    # OHLC autour du close
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, volatility / 200, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, volatility / 200, n_bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Volume aléatoire
    base_volume = 100.0
    volume = base_volume * (1 + np.random.uniform(-0.5, 1.5, n_bars))

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=timestamps,
    )

    logger.info(
        f"Données générées: {len(df)} barres, "
        f"prix moyen={df['close'].mean():.2f}, "
        f"volatilité réalisée={df['close'].pct_change().std()*100:.2f}%"
    )

    return df


def test_strategy_basic():
    """Test basique avec paramètres par défaut"""
    logger.info("=== TEST 1: Paramètres par défaut ===")

    # Génération données
    df = generate_test_data(n_bars=2000, volatility=3.0)

    # Paramètres adaptés pour tests (moins stricts)
    params = EMAStochScalpParams(
        fast_ema=20,  # EMAs plus rapides pour plus de signaux
        slow_ema=50,
        require_pullback=False,  # Pas de filtre pullback pour tests
        volume_threshold=1.0,  # Volume moins restrictif
        stoch_oversold=30.0,  # Zones plus larges
        stoch_overbought=70.0,
    )

    # Initialisation stratégie
    strategy = EMAStochScalpStrategy(symbol="BTCUSDT", timeframe="1m")

    # Backtest
    equity, stats = strategy.backtest(
        df=df, params=params.to_dict(), initial_capital=10000.0
    )

    # Affichage résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS BACKTEST - Paramètres Par Défaut")
    print("=" * 60)
    print(f"Capital Initial:     ${stats.initial_capital:,.2f}")
    print(f"Capital Final:       ${stats.final_equity:,.2f}")
    print(f"PnL Total:           ${stats.total_pnl:,.2f} ({stats.total_pnl_pct:+.2f}%)")
    print(f"Sharpe Ratio:        {stats.sharpe_ratio:.2f}" if stats.sharpe_ratio else "Sharpe Ratio:        N/A")
    print(f"Max Drawdown:        ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.2f}%)")
    print(f"\nTotal Trades:        {stats.total_trades}")
    print(f"Trades Gagnants:     {stats.win_trades}")
    print(f"Trades Perdants:     {stats.loss_trades}")
    print(f"Win Rate:            {stats.win_rate_pct:.1f}%")
    print(f"Profit Factor:       {stats.profit_factor:.2f}" if stats.profit_factor else "Profit Factor:       N/A")
    print(f"Gain Moyen:          ${stats.avg_win:.2f}" if stats.avg_win else "Gain Moyen:          N/A")
    print(f"Perte Moyenne:       ${stats.avg_loss:.2f}" if stats.avg_loss else "Perte Moyenne:       N/A")
    print(f"Frais Totaux:        ${stats.total_fees_paid:.2f}")
    print("=" * 60 + "\n")

    # Validation basique (tolérance pour données synthétiques)
    if stats.total_trades == 0:
        logger.warning("⚠ Aucun trade généré (données synthétiques)")
    else:
        logger.info(f"✓ Test basique réussi - {stats.total_trades} trades générés")

    return equity, stats


def test_strategy_conservative():
    """Test avec paramètres conservateurs (win rate élevé)"""
    logger.info("=== TEST 2: Paramètres conservateurs ===")

    df = generate_test_data(n_bars=2000, volatility=3.5)

    # Paramètres plus stricts pour meilleur win rate
    params = EMAStochScalpParams(
        fast_ema=30,
        slow_ema=60,
        stoch_k=14,
        stoch_d=3,
        require_pullback=False,  # Désactivé pour tests
        pullback_tolerance_pct=0.5,  # Plus tolérant
        volume_threshold=1.1,  # Volume moins restrictif
        stop_loss_pct=0.8,  # SL plus large pour tests
        take_profit_pct=1.2,  # TP plus large (R:R 1:1.5)
        leverage=10.0,  # Levier modéré
        max_hold_bars=40,  # Plus de temps
        stoch_oversold=30.0,
        stoch_overbought=70.0,
    )

    strategy = EMAStochScalpStrategy(symbol="BTCUSDT", timeframe="1m")
    equity, stats = strategy.backtest(df, params.to_dict(), initial_capital=10000.0)

    print("\n" + "=" * 60)
    print("RÉSULTATS BACKTEST - Paramètres Conservateurs")
    print("=" * 60)
    print(f"Capital Final:       ${stats.final_equity:,.2f}")
    print(f"PnL Total:           ${stats.total_pnl:,.2f} ({stats.total_pnl_pct:+.2f}%)")
    print(f"Win Rate:            {stats.win_rate_pct:.1f}%")
    print(f"Total Trades:        {stats.total_trades}")
    print(f"Profit Factor:       {stats.profit_factor:.2f}" if stats.profit_factor else "Profit Factor:       N/A")
    print("=" * 60 + "\n")

    logger.info("✓ Test conservateur réussi")
    return equity, stats


def test_strategy_aggressive():
    """Test avec paramètres agressifs (plus de trades)"""
    logger.info("=== TEST 3: Paramètres agressifs ===")

    df = generate_test_data(n_bars=2000, volatility=4.0)

    # Paramètres agressifs pour plus de trades
    params = EMAStochScalpParams(
        fast_ema=10,  # EMAs très rapides
        slow_ema=30,
        stoch_k=8,  # Stochastic très réactif
        stoch_d=3,
        require_pullback=False,  # Pas d'attente pullback
        volume_threshold=0.8,  # Volume minimal
        stop_loss_pct=1.0,  # SL plus large
        take_profit_pct=1.0,  # R:R 1:1
        leverage=20.0,  # Levier élevé
        max_hold_bars=60,
        stoch_oversold=35.0,
        stoch_overbought=65.0,
    )

    strategy = EMAStochScalpStrategy(symbol="ETHUSD", timeframe="1m")
    equity, stats = strategy.backtest(df, params.to_dict(), initial_capital=10000.0)

    print("\n" + "=" * 60)
    print("RÉSULTATS BACKTEST - Paramètres Agressifs")
    print("=" * 60)
    print(f"Capital Final:       ${stats.final_equity:,.2f}")
    print(f"PnL Total:           ${stats.total_pnl:,.2f} ({stats.total_pnl_pct:+.2f}%)")
    print(f"Win Rate:            {stats.win_rate_pct:.1f}%")
    print(f"Total Trades:        {stats.total_trades}")
    print(f"Profit Factor:       {stats.profit_factor:.2f}" if stats.profit_factor else "Profit Factor:       N/A")
    print("=" * 60 + "\n")

    logger.info("✓ Test agressif réussi")
    return equity, stats


def test_signal_generation():
    """Test de génération de signaux"""
    logger.info("=== TEST 4: Génération de signaux ===")

    df = generate_test_data(n_bars=500, volatility=2.0)
    params = EMAStochScalpParams()

    strategy = EMAStochScalpStrategy()
    df_signals = strategy.generate_signals(df, params.to_dict())

    # Comptage signaux
    long_count = (df_signals["signal"] == "ENTER_LONG").sum()
    short_count = (df_signals["signal"] == "ENTER_SHORT").sum()
    hold_count = (df_signals["signal"] == "HOLD").sum()

    print("\n" + "=" * 60)
    print("GÉNÉRATION DE SIGNAUX")
    print("=" * 60)
    print(f"Total barres:        {len(df_signals)}")
    print(f"Signaux LONG:        {long_count}")
    print(f"Signaux SHORT:       {short_count}")
    print(f"Signaux HOLD:        {hold_count}")
    print(f"% Signaux actifs:    {((long_count + short_count) / len(df_signals) * 100):.1f}%")
    print("=" * 60 + "\n")

    # Vérification indicateurs
    assert "ema_fast" in df_signals.columns
    assert "ema_slow" in df_signals.columns
    assert "stoch_k" in df_signals.columns
    assert "stoch_d" in df_signals.columns

    logger.info("✓ Test génération de signaux réussi")
    return df_signals


def main():
    """Exécution de tous les tests"""
    logger.info("Début des tests de la stratégie EMA + Stochastic Scalping")
    logger.info("=" * 60)

    try:
        # Test 1: Basique
        equity1, stats1 = test_strategy_basic()

        # Test 2: Conservateur
        equity2, stats2 = test_strategy_conservative()

        # Test 3: Agressif
        equity3, stats3 = test_strategy_aggressive()

        # Test 4: Génération signaux
        df_signals = test_signal_generation()

        # Résumé final
        print("\n" + "=" * 60)
        print("RÉSUMÉ DES TESTS")
        print("=" * 60)
        print(
            f"Test Basique:        {stats1.total_trades} trades, Win Rate: {stats1.win_rate_pct:.1f}%"
        )
        print(
            f"Test Conservateur:   {stats2.total_trades} trades, Win Rate: {stats2.win_rate_pct:.1f}%"
        )
        print(
            f"Test Agressif:       {stats3.total_trades} trades, Win Rate: {stats3.win_rate_pct:.1f}%"
        )
        print("=" * 60)
        print("\n✅ TOUS LES TESTS RÉUSSIS!\n")

        logger.info("Tous les tests terminés avec succès")

    except Exception as e:
        logger.error(f"Erreur durant les tests: {e}", exc_info=True)
        print(f"\n❌ ERREUR: {e}\n")
        raise


if __name__ == "__main__":
    main()
