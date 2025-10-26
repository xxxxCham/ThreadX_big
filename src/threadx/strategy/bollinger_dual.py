"""
ThreadX - Bollinger Dual Strategy
===================================

Stratégie basée sur les bandes de Bollinger avec double condition d'entrée:
- Signal d'entrée: Franchissement de bande + Franchissement de MA
- Trailing stop dynamique à partir de la médiane
- Stop loss fixe configurable pour les shorts

Author: ThreadX Framework
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import logging

from threadx.configuration.settings import S
from threadx.utils.log import get_logger
from threadx.strategy.model import (
    Strategy,
    Trade,
    RunStats,
    validate_ohlcv_dataframe,
    validate_strategy_params,
)
from threadx.indicators import ensure_indicator

logger = get_logger(__name__)


@dataclass
class BollingerDualParams:
    """
    Paramètres de la stratégie Bollinger Dual.

    Attributes:
        # Bollinger Bands
        bb_window: Période pour Bollinger SMA (défaut: 20)
        bb_std: Multiplicateur écart-type pour bandes (défaut: 2.0)

        # Moving Average pour franchissement
        ma_window: Période MA pour signal franchissement (défaut: 10)
        ma_type: Type de MA - 'sma' ou 'ema' (défaut: 'sma')

        # Trailing Stop
        trailing_pct: Pourcentage pour trailing stop (0.8 = 80%) (défaut: 0.8)
        median_activated: Activer trailing uniquement après médiane (défaut: True)

        # Stop Loss fixe pour SHORT
        short_stop_pct: Stop loss fixe SHORT en % au-dessus entrée (défaut: 0.37 = 37%)

        # Risk Management
        risk_per_trade: Risque par trade en fraction du capital (défaut: 0.02 = 2%)
        max_hold_bars: Durée max position en barres (défaut: 100)

        # Métadonnées
        meta: Dictionnaire métadonnées personnalisées
    """

    # Bollinger Bands
    bb_window: int = 20
    bb_std: float = 2.0

    # Moving Average
    ma_window: int = 10
    ma_type: str = 'sma'

    # Trailing Stop
    trailing_pct: float = 0.8  # 80% entre bande et médiane
    median_activated: bool = True

    # Stop Loss SHORT
    short_stop_pct: float = 0.37  # 37%

    # Risk Management
    risk_per_trade: float = 0.02
    max_hold_bars: int = 100

    # Métadonnées
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation des paramètres"""
        if self.bb_window < 2:
            raise ValueError(f"bb_window must be >= 2, got: {self.bb_window}")
        if self.bb_std <= 0:
            raise ValueError(f"bb_std must be > 0, got: {self.bb_std}")
        if self.ma_window < 1:
            raise ValueError(f"ma_window must be >= 1, got: {self.ma_window}")
        if self.ma_type not in ['sma', 'ema']:
            raise ValueError(f"ma_type must be 'sma' or 'ema', got: {self.ma_type}")
        if not 0 < self.trailing_pct <= 1:
            raise ValueError(f"trailing_pct must be in (0, 1], got: {self.trailing_pct}")
        if self.short_stop_pct < 0:
            raise ValueError(f"short_stop_pct must be >= 0, got: {self.short_stop_pct}")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(f"risk_per_trade must be in (0, 1], got: {self.risk_per_trade}")
        if self.max_hold_bars < 1:
            raise ValueError(f"max_hold_bars must be >= 1, got: {self.max_hold_bars}")

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour compatibilité"""
        return {
            "bb_window": self.bb_window,
            "bb_std": self.bb_std,
            "ma_window": self.ma_window,
            "ma_type": self.ma_type,
            "trailing_pct": self.trailing_pct,
            "median_activated": self.median_activated,
            "short_stop_pct": self.short_stop_pct,
            "risk_per_trade": self.risk_per_trade,
            "max_hold_bars": self.max_hold_bars,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BollingerDualParams":
        """Crée depuis un dictionnaire"""
        return cls(
            bb_window=data.get("bb_window", 20),
            bb_std=data.get("bb_std", 2.0),
            ma_window=data.get("ma_window", 10),
            ma_type=data.get("ma_type", 'sma'),
            trailing_pct=data.get("trailing_pct", 0.8),
            median_activated=data.get("median_activated", True),
            short_stop_pct=data.get("short_stop_pct", 0.37),
            risk_per_trade=data.get("risk_per_trade", 0.02),
            max_hold_bars=data.get("max_hold_bars", 100),
            meta=data.get("meta", {}),
        )


class BollingerDualStrategy:
    """
    Implémentation de la stratégie Bollinger Dual.

    Logique de trading:
    1. LONG: Prix < Bande Basse + Franchissement haussier MA
    2. SHORT: Prix > Bande Haute + Franchissement baissier MA
    3. Trailing Stop après médiane:
       - LONG: stop = bande_basse + (médiane - bande_basse) * trailing_pct
       - SHORT: stop = médiane + (bande_haute - médiane) * trailing_pct
    4. Stop Loss fixe SHORT: entry_price * (1 + short_stop_pct)
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
        logger.info(f"Stratégie Bollinger Dual initialisée: {symbol}/{timeframe}")

    def _ensure_indicators(
        self, df: pd.DataFrame, params: BollingerDualParams
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Garantit la disponibilité des indicateurs via IndicatorBank.

        Args:
            df: DataFrame OHLCV
            params: Paramètres stratégie

        Returns:
            Tuple (df_with_indicators, ma_array)
        """
        logger.debug(
            f"Calcul indicateurs: BB(window={params.bb_window}, std={params.bb_std}), MA(window={params.ma_window}, type={params.ma_type})"
        )

        # Bollinger Bands
        bb_result = ensure_indicator(
            "bollinger",
            {"period": params.bb_window, "std": params.bb_std},
            df,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        if isinstance(bb_result, tuple) and len(bb_result) == 3:
            upper, middle, lower = bb_result
            df_bb = df.copy()
            df_bb["bb_upper"] = upper
            df_bb["bb_middle"] = middle
            df_bb["bb_lower"] = lower
        else:
            raise ValueError(f"Bollinger result format invalide: {type(bb_result)}")

        # Moving Average pour franchissement
        ma_indicator = "sma" if params.ma_type == 'sma' else "ema"
        ma_result = ensure_indicator(
            ma_indicator,
            {"period": params.ma_window},
            df,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        if isinstance(ma_result, np.ndarray):
            ma_array = ma_result
        else:
            raise ValueError(f"MA result format invalide: {type(ma_result)}")

        logger.debug(
            f"Indicateurs calculés: BB range [{lower.min():.2f}, {upper.max():.2f}], MA dernier {ma_array[-1]:.2f}"
        )

        return df_bb, ma_array

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Génère les signaux de trading basés sur Bollinger Dual.

        Args:
            df: DataFrame OHLCV avec timestamp index (UTC)
            params: Dictionnaire paramètres (format BollingerDualParams.to_dict())

        Returns:
            DataFrame avec colonne 'signal' et métadonnées

        Signals générés:
        - "ENTER_LONG": Prix < Bande Basse + Franchissement haussier MA
        - "ENTER_SHORT": Prix > Bande Haute + Franchissement baissier MA
        - "HOLD": Maintenir position actuelle
        """
        logger.info(f"Génération signaux Bollinger Dual: {len(df)} barres")

        # Validation inputs
        validate_ohlcv_dataframe(df)
        validate_strategy_params(params, ["bb_window", "bb_std", "ma_window"])

        # Parse paramètres
        strategy_params = BollingerDualParams.from_dict(params)

        # Ensure indicateurs
        df_with_indicators, ma_array = self._ensure_indicators(df, strategy_params)

        # Extraction des données
        close = df["close"].values
        bb_upper = df_with_indicators["bb_upper"].values
        bb_lower = df_with_indicators["bb_lower"].values
        bb_middle = df_with_indicators["bb_middle"].values

        # Initialisation signaux
        n_bars = len(df)
        signals = np.full(n_bars, "HOLD", dtype=object)

        # Logique de signaux
        logger.debug(
            f"Application logique signaux: BB({strategy_params.bb_window}), MA({strategy_params.ma_window}, {strategy_params.ma_type})"
        )

        for i in range(max(strategy_params.bb_window, strategy_params.ma_window) + 1, n_bars):
            # Skip si données invalides
            if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(ma_array[i]):
                continue

            # Détection franchissement MA
            ma_cross_up = close[i] > ma_array[i] and close[i-1] <= ma_array[i-1]
            ma_cross_down = close[i] < ma_array[i] and close[i-1] >= ma_array[i-1]

            # LONG: Prix < Bande Basse + Franchissement haussier MA
            if close[i] < bb_lower[i] and ma_cross_up:
                signals[i] = "ENTER_LONG"
                logger.debug(
                    f"ENTER_LONG @ bar {i}: price={close[i]:.2f}, bb_lower={bb_lower[i]:.2f}, ma={ma_array[i]:.2f}"
                )

            # SHORT: Prix > Bande Haute + Franchissement baissier MA
            elif close[i] > bb_upper[i] and ma_cross_down:
                signals[i] = "ENTER_SHORT"
                logger.debug(
                    f"ENTER_SHORT @ bar {i}: price={close[i]:.2f}, bb_upper={bb_upper[i]:.2f}, ma={ma_array[i]:.2f}"
                )

        # Construction DataFrame de sortie
        result_df = pd.DataFrame(index=df.index)
        result_df["signal"] = signals
        result_df["bb_upper"] = bb_upper
        result_df["bb_middle"] = bb_middle
        result_df["bb_lower"] = bb_lower
        result_df["ma"] = ma_array
        result_df["close"] = close

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
    ) -> Tuple[pd.Series, RunStats]:
        """
        Exécute un backtest complet de la stratégie Bollinger Dual.

        Args:
            df: DataFrame OHLCV avec timestamp index (UTC)
            params: Paramètres stratégie (format BollingerDualParams.to_dict())
            initial_capital: Capital initial
            fee_bps: Frais de transaction en basis points (défaut: 4.5)
            slippage_bps: Slippage en basis points (défaut: 0.0)

        Returns:
            Tuple (equity_curve, run_stats)

        Gestion des positions:
        - Trailing stop à 80% (bande ↔ médiane) après franchissement médiane
        - Stop loss fixe SHORT: +37%
        - Sortie forcée après max_hold_bars
        """
        logger.info(
            f"Début backtest Bollinger Dual: capital={initial_capital}, fee={fee_bps}bps"
        )

        # Validation
        validate_ohlcv_dataframe(df)
        strategy_params = BollingerDualParams.from_dict(params)

        # Génération signaux
        signals_df = self.generate_signals(df, params)

        # Initialisation backtest
        n_bars = len(df)
        equity = np.full(n_bars, initial_capital, dtype=float)

        cash = initial_capital
        position = None
        trades: List[Trade] = []
        median_reached = False  # Flag pour tracking médiane

        fee_rate = (fee_bps + slippage_bps) / 10000.0

        logger.debug(f"Backtest initialisé: {n_bars} barres, fee_rate={fee_rate:.6f}")

        # Boucle principale
        for i, (timestamp, row) in enumerate(signals_df.iterrows()):
            current_price = row["close"]
            signal = row["signal"]
            bb_upper = row["bb_upper"]
            bb_middle = row["bb_middle"]
            bb_lower = row["bb_lower"]

            # Skip si données invalides
            if np.isnan(current_price) or np.isnan(bb_middle):
                equity[i] = cash if position is None else cash + position.calculate_unrealized_pnl(current_price)
                continue

            # Gestion position existante
            if position is not None:
                should_exit = False
                exit_reason = ""

                # 1. Vérifier si médiane atteinte (pour activer trailing)
                if strategy_params.median_activated and not median_reached:
                    if position.is_long() and current_price >= bb_middle:
                        median_reached = True
                        logger.debug(f"LONG: Médiane atteinte @ {current_price:.2f}")
                    elif position.is_short() and current_price <= bb_middle:
                        median_reached = True
                        logger.debug(f"SHORT: Médiane atteinte @ {current_price:.2f}")

                # 2. Mise à jour trailing stop si médiane atteinte
                if median_reached:
                    if position.is_long():
                        # Trailing LONG: stop = bande_basse + (médiane - bande_basse) * trailing_pct
                        new_stop = bb_lower + (bb_middle - bb_lower) * strategy_params.trailing_pct
                        if new_stop > position.stop:  # Trail vers le haut seulement
                            position.stop = new_stop
                            logger.debug(f"LONG trailing stop ajusté: {new_stop:.2f}")
                    else:
                        # Trailing SHORT: stop = médiane + (bande_haute - médiane) * trailing_pct
                        new_stop = bb_middle + (bb_upper - bb_middle) * strategy_params.trailing_pct
                        if new_stop < position.stop:  # Trail vers le bas seulement
                            position.stop = new_stop
                            logger.debug(f"SHORT trailing stop ajusté: {new_stop:.2f}")

                # 3. Vérifier stop loss
                if position.should_stop_loss(current_price):
                    should_exit = True
                    exit_reason = "stop_loss"

                # 4. Vérifier durée max
                entry_timestamp = pd.to_datetime(position.entry_time, utc=True)
                bars_held = (df.index <= timestamp).sum() - (df.index <= entry_timestamp).sum()

                if bars_held >= strategy_params.max_hold_bars:
                    should_exit = True
                    exit_reason = "max_hold_bars"

                # Fermeture position
                if should_exit:
                    exit_value = current_price * position.qty
                    exit_fees = exit_value * fee_rate

                    position.close_trade(
                        exit_price=current_price,
                        exit_time=str(timestamp) if hasattr(timestamp, "isoformat") else str(timestamp),
                        exit_fees=exit_fees,
                    )

                    pnl_val = position.pnl_realized if position.pnl_realized is not None else 0.0
                    cash += pnl_val + (position.entry_price * position.qty)
                    trades.append(position)
                    logger.debug(
                        f"Position fermée @ {current_price:.2f}: {exit_reason}, PnL={position.pnl_realized:.2f}"
                    )

                    position = None
                    median_reached = False

            # Nouveau signal d'entrée
            if position is None and signal in ["ENTER_LONG", "ENTER_SHORT"]:
                # Position sizing simple (basé sur risk_per_trade)
                risk_amount = cash * strategy_params.risk_per_trade
                position_size = risk_amount / current_price

                if position_size > 0:
                    # Calcul prix stop
                    if signal == "ENTER_LONG":
                        # LONG: stop initial = bande basse
                        stop_price = bb_lower
                    else:
                        # SHORT: stop fixe = entry + 37%
                        stop_price = current_price * (1 + strategy_params.short_stop_pct)

                    # Frais d'entrée
                    entry_value = current_price * position_size
                    entry_fees = entry_value * fee_rate

                    if entry_value + entry_fees <= cash:
                        # Création nouveau trade
                        position = Trade(
                            side=signal.replace("ENTER_", ""),
                            qty=position_size,
                            entry_price=current_price,
                            entry_time=str(timestamp) if hasattr(timestamp, "isoformat") else str(timestamp),
                            stop=stop_price,
                            fees_paid=entry_fees,
                            meta={
                                "bb_upper": bb_upper,
                                "bb_middle": bb_middle,
                                "bb_lower": bb_lower,
                                "ma": row["ma"],
                                "trailing_pct": strategy_params.trailing_pct,
                            },
                        )

                        cash -= entry_value + entry_fees
                        median_reached = False

                        logger.debug(
                            f"Nouvelle position: {signal} {position_size:.4f} @ {current_price:.2f}, stop={stop_price:.2f}"
                        )

            # Mise à jour équité
            if position is not None:
                equity[i] = cash + position.calculate_unrealized_pnl(current_price)
            else:
                equity[i] = cash

        # Fermeture position finale si nécessaire
        if position is not None:
            final_price = df["close"].iloc[-1]
            position.close_trade(
                exit_price=final_price,
                exit_time=df.index[-1].isoformat(),
                exit_fees=final_price * position.qty * fee_rate,
            )
            trades.append(position)

        # Construction courbe d'équité
        equity_curve = pd.Series(equity, index=df.index)

        # Calcul statistiques
        run_stats = RunStats.from_trades_and_equity(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            meta={
                "strategy": "BollingerDual",
                "params": params,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
            },
        )

        logger.info(
            f"Backtest terminé: {run_stats.total_trades} trades, PnL={run_stats.total_pnl:.2f} ({run_stats.total_pnl_pct:.2f}%)"
        )

        return equity_curve, run_stats


def create_default_params(**overrides) -> BollingerDualParams:
    """
    Crée des paramètres par défaut avec surcharges optionnelles.

    Args:
        **overrides: Paramètres à surcharger

    Returns:
        Instance BollingerDualParams avec valeurs par défaut + surcharges
    """
    base_params = BollingerDualParams()

    for key, value in overrides.items():
        if hasattr(base_params, key):
            setattr(base_params, key, value)
        else:
            logger.warning(f"Paramètre inconnu ignoré: {key}={value}")

    return base_params


__all__ = [
    "BollingerDualParams",
    "BollingerDualStrategy",
    "create_default_params",
]
