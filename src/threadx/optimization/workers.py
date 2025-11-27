"""
ThreadX Optimization Workers - Fonctions picklables pour ProcessPoolExecutor
===========================================================================

Fonctions workers séparées pour éviter conflits pickle avec Streamlit reload.
Sur Windows, multiprocessing utilise 'spawn' mode qui réimporte les modules.
En séparant les workers, on garantit qu'ils sont toujours picklables.

Author: ThreadX Framework
"""

import pandas as pd

# Variables globales pour process-level caching (initialisées par _init_process_globals)
_G_INDICATORS = None
_G_REAL_DATA = None
_G_SYMBOL = None
_G_TIMEFRAME = None
_G_STRATEGY_NAME = None


def _init_process_globals(
    computed_indicators: dict,
    real_data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
) -> None:
    """
    Initialise les variables globales du process worker.

    Appelée une fois au démarrage de chaque process via ProcessPoolExecutor(initializer=...).
    Permet d'éviter de re-sérialiser les DataFrames lourds à chaque appel.

    Args:
        computed_indicators: Dict des indicateurs pré-calculés
        real_data: DataFrame OHLCV complet
        symbol: Symbole tradé (ex: "BTC")
        timeframe: Timeframe (ex: "1h")
        strategy_name: Nom de la stratégie
    """
    global _G_INDICATORS, _G_REAL_DATA, _G_SYMBOL, _G_TIMEFRAME, _G_STRATEGY_NAME
    _G_INDICATORS = computed_indicators
    _G_REAL_DATA = real_data
    _G_SYMBOL = symbol
    _G_TIMEFRAME = timeframe
    _G_STRATEGY_NAME = strategy_name


def _evaluate_combo_worker(
    combo: dict,
    computed_indicators: dict | None,
    real_data: pd.DataFrame | None,
    symbol: str | None,
    timeframe: str | None,
    strategy_name: str = "Bollinger_Breakout",
) -> dict:
    """
    Worker function pour évaluation combo (picklable pour ProcessPoolExecutor).

    Utilise les variables globales si disponibles (initialisées par _init_process_globals),
    sinon utilise les arguments passés (mode ThreadPool ou fallback).

    Args:
        combo: Paramètres de la stratégie à tester
        computed_indicators: Dict des indicateurs (None si globale initialisée)
        real_data: DataFrame OHLCV (None si globale initialisée)
        symbol: Symbole tradé (None si globale initialisée)
        timeframe: Timeframe (None si globale initialisée)
        strategy_name: Nom de la stratégie

    Returns:
        dict: Résultats du backtest avec métriques
    """
    try:
        # Import local pour éviter overhead dans process principal
        from threadx.indicators.bank import IndicatorBank, IndicatorSettings
        from threadx.strategy import (
            BollingerDualStrategy,
            MACrossoverStrategy,
            EMACrossStrategy,
            ATRChannelStrategy,
        )

        # Utiliser variables globales si disponibles (ProcessPool), sinon args (ThreadPool)
        global _G_INDICATORS, _G_REAL_DATA, _G_SYMBOL, _G_TIMEFRAME, _G_STRATEGY_NAME

        if _G_INDICATORS is not None:
            # Mode ProcessPool avec initializer : utiliser globales
            computed_indicators = _G_INDICATORS
            real_data = _G_REAL_DATA
            symbol = _G_SYMBOL
            timeframe = _G_TIMEFRAME
            if strategy_name == "Bollinger_Breakout":  # fallback
                strategy_name = _G_STRATEGY_NAME
        else:
            # Mode ThreadPool ou sans initializer : utiliser args
            if computed_indicators is None or real_data is None:
                raise ValueError("Données manquantes et globales non initialisées")

        # Créer IndicatorBank dans ce process worker
        settings = IndicatorSettings(use_gpu=True)
        indicator_bank = IndicatorBank(settings)

        # Mapping stratégie → classe
        strategy_classes = {
            "Bollinger_Dual": BollingerDualStrategy,
            "MA_Crossover": MACrossoverStrategy,
            "EMA_Cross": EMACrossStrategy,
            "ATR_Channel": ATRChannelStrategy,
        }

        strategy_class = strategy_classes.get(strategy_name)
        if strategy_class is None:
            raise ValueError(f"Stratégie inconnue: {strategy_name}")

        # Créer instance de stratégie avec combo de paramètres
        strategy = strategy_class(
            indicator_bank=indicator_bank,
            **combo
        )

        # Exécuter backtest
        trades = strategy.run_backtest(
            data=real_data,
            symbol=symbol,
            timeframe=timeframe,
            computed_indicators=computed_indicators,
        )

        # Calculer métriques de performance
        from threadx.backtest.performance import PerformanceCalculator

        perf_calc = PerformanceCalculator(trades=trades, initial_capital=10000.0)
        metrics = perf_calc.calculate_all_metrics()

        # Ajouter les paramètres au résultat
        result = {
            "params": combo.copy(),
            **metrics,
        }

        return result

    except Exception as e:
        # Retourner dict avec erreur pour éviter crash du process
        return {
            "params": combo.copy(),
            "error": str(e),
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }
