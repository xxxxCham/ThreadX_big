"""
ThreadX Phase 4 - BB+ATR Strategy Implementation
===============================================

Stratégie Bollinger Bands + ATR avec gestion avancée du risque.

Fonctionnalités:
- Signaux basés sur Bollinger Bands avec filtrage Z-score
- Stops dynamiques basés sur ATR avec multiplicateur configurable
- Filtrage des trades (min PnL, durée, espacement)
- Intégration complète avec Phase 3 Indicators Layer
- Backtest déterministe avec seed reproductible
- Gestion des positions longues et courtes

Améliorations vs TradXPro:
- atr_multiplier paramétrable (défaut 1.5) pour stops adaptatifs
- Filtrage min_pnl_pct optionnel (désactivé par défaut pour éviter sur-filtrage)
- Trailing stop ATR plus robuste
- Intégration native avec IndicatorBank (cache TTL)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from threadx.configuration.settings import S
from threadx.utils.log import get_logger
from threadx.strategy.model import (
    Strategy,
    Trade,
    RunStats,
    validate_ohlcv_dataframe,
    validate_strategy_params,
)
from threadx.indicators import ensure_indicator, batch_ensure_indicators

logger = get_logger(__name__)

# ==========================================
# STRATEGY PARAMETERS
# ==========================================


@dataclass
class BBAtrParams:
    """
    Paramètres de la stratégie Bollinger Bands + ATR.

    Attributes:
        # Bollinger Bands
        bb_period: Période pour moyennes mobiles (défaut: 20)
        bb_std: Multiplicateur écart-type pour bandes (défaut: 2.0)
        entry_z: Seuil Z-score pour déclenchement signal (défaut: 1.0)
        entry_logic: Logique d'entrée "AND"|"OR" (défaut: "AND")

        # ATR et gestion risque
        atr_period: Période ATR (défaut: 14)
        atr_multiplier: Multiplicateur ATR pour stops (défaut: 1.5)
        trailing_stop: Activer trailing stop ATR (défaut: True)

        # Risk Management
        risk_per_trade: Risque par trade en fraction du capital (défaut: 0.01 = 1%)
        min_pnl_pct: PnL minimum requis pour valider trade (défaut: 0.0% = désactivé)

        # Positions et timing
        leverage: Effet de levier (défaut: 1.0)
        max_hold_bars: Durée max position en barres (défaut: 72)
        spacing_bars: Espacement min entre trades (défaut: 6)

        # Filtrage optionnel
        trend_period: Période EMA tendance (0=désactivé, défaut: 0)

        # Métadonnées
        meta: Dictionnaire métadonnées personnalisées

    Example:
        >>> params = BBAtrParams(
        ...     bb_period=20, bb_std=2.0, entry_z=1.5,
        ...     atr_multiplier=2.0, risk_per_trade=0.02
        ... )
        >>> # Utilisation avec stratégie
        >>> strategy = BBAtrStrategy()
        >>> signals = strategy.generate_signals(df, params.to_dict())
    """

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    entry_z: float = 1.0
    entry_logic: str = "AND"

    # ATR et stops
    atr_period: int = 14
    atr_multiplier: float = 1.5  # Amélioration: multiplicateur configurable
    trailing_stop: bool = True

    # Risk management
    risk_per_trade: float = 0.01  # 1% du capital par trade
    min_pnl_pct: float = (
        0.0  # FIX: Désactivé par défaut (0.01% filtrait TOUS les trades)
    )

    # Position management
    leverage: float = 1.0
    max_hold_bars: int = 72  # 3 jours en 1h
    spacing_bars: int = 6  # 6h entre trades

    # Filtres optionnels
    trend_period: int = 0  # 0 = pas de filtre tendance

    # Métadonnées
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation des paramètres"""
        if self.bb_period < 2:
            raise ValueError(f"bb_period must be >= 2, got: {self.bb_period}")

        if self.bb_std <= 0:
            raise ValueError(f"bb_std must be > 0, got: {self.bb_std}")

        if self.entry_z <= 0:
            raise ValueError(f"entry_z must be > 0, got: {self.entry_z}")

        if self.entry_logic not in ["AND", "OR"]:
            raise ValueError(
                f"entry_logic must be 'AND' or 'OR', got: {self.entry_logic}"
            )

        if self.atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got: {self.atr_period}")

        if self.atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got: {self.atr_multiplier}")

        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(
                f"risk_per_trade must be in (0, 1], got: {self.risk_per_trade}"
            )

        if self.min_pnl_pct < 0:
            raise ValueError(f"min_pnl_pct must be >= 0, got: {self.min_pnl_pct}")

        if self.leverage <= 0:
            raise ValueError(f"leverage must be > 0, got: {self.leverage}")

        if self.max_hold_bars < 1:
            raise ValueError(f"max_hold_bars must be >= 1, got: {self.max_hold_bars}")

        if self.spacing_bars < 0:
            raise ValueError(f"spacing_bars must be >= 0, got: {self.spacing_bars}")

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour compatibilité"""
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "entry_z": self.entry_z,
            "entry_logic": self.entry_logic,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "trailing_stop": self.trailing_stop,
            "risk_per_trade": self.risk_per_trade,
            "min_pnl_pct": self.min_pnl_pct,
            "leverage": self.leverage,
            "max_hold_bars": self.max_hold_bars,
            "spacing_bars": self.spacing_bars,
            "trend_period": self.trend_period,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BBAtrParams":
        """Crée depuis un dictionnaire"""
        return cls(
            bb_period=data.get("bb_period", 20),
            bb_std=data.get("bb_std", 2.0),
            entry_z=data.get("entry_z", 1.0),
            entry_logic=data.get("entry_logic", "AND"),
            atr_period=data.get("atr_period", 14),
            atr_multiplier=data.get("atr_multiplier", 1.5),
            trailing_stop=data.get("trailing_stop", True),
            risk_per_trade=data.get("risk_per_trade", 0.01),
            min_pnl_pct=data.get("min_pnl_pct", 0.0),  # FIX: 0.0 par défaut
            leverage=data.get("leverage", 1.0),
            max_hold_bars=data.get("max_hold_bars", 72),
            spacing_bars=data.get("spacing_bars", 6),
            trend_period=data.get("trend_period", 0),
            meta=data.get("meta", {}),
        )


# ==========================================
# STRATEGY IMPLEMENTATION
# ==========================================


class BBAtrStrategy:
    """
    Implémentation de la stratégie Bollinger Bands + ATR.

    Logique de trading:
    1. Calcul indicateurs via IndicatorBank (cache Phase 3)
    2. Génération signaux basés sur:
       - Z-score Bollinger > entry_z pour ENTER_SHORT
       - Z-score Bollinger < -entry_z pour ENTER_LONG
       - Stops dynamiques ATR * atr_multiplier
       - Filtrage tendance optionnel (EMA)
    3. Gestion positions:
       - Risk sizing basé sur ATR
       - Trailing stops ATR
       - Filtrage min PnL et espacement

    Améliorations vs TradXPro:
    - atr_multiplier paramétrable vs fixe
    - Filtrage min_pnl_pct évite micro-trades
    - Intégration native cache Phase 3
    - Code plus lisible et testable

    Example:
        >>> strategy = BBAtrStrategy()
        >>> params = BBAtrParams(bb_period=20, atr_multiplier=2.0)
        >>> signals = strategy.generate_signals(df, params.to_dict())
        >>> equity, stats = strategy.backtest(df, params.to_dict(), 10000)
    """

    def __init__(self, symbol: str = "UNKNOWN", timeframe: str = "15m"):
        """
        Initialise la stratégie.

        Args:
            symbol: Symbole pour cache d'indicateurs
            timeframe: Timeframe pour cache d'indicateurs
        """
        self.symbol = symbol
        self.timeframe = timeframe
        logger.debug(f"Stratégie BB+ATR initialisée: {symbol}/{timeframe}")

    def _ensure_indicators(
        self,
        df: pd.DataFrame,
        params: BBAtrParams,
        precomputed_indicators: Optional[Dict] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Garantit la disponibilité des indicateurs via IndicatorBank.

        Args:
            df: DataFrame OHLCV
            params: Paramètres stratégie
            precomputed_indicators: Dictionnaire {key: result} d'indicateurs pré-calculés (optionnel)

        Returns:
            Tuple (df_with_bollinger, atr_array)
        """
        import json

        # 🚀 OPTIMISATION: Utiliser le même format de clé que _params_to_key() dans engine.py
        bb_key = json.dumps(
            {"period": params.bb_period, "std": params.bb_std},
            sort_keys=True,
            separators=(",", ":"),
        )
        atr_key = json.dumps(
            {"method": "ema", "period": params.atr_period},
            sort_keys=True,
            separators=(",", ":"),
        )

        # Debug: Voir ce qui est fourni (désactivé pour performance)
        # if precomputed_indicators:
        #     logger.info(
        #         f"🔍 DEBUG precomputed keys: bollinger={list(precomputed_indicators.get('bollinger', {}).keys())}, atr={list(precomputed_indicators.get('atr', {}).keys())}, wanted_bb={bb_key}, wanted_atr={atr_key}"
        #     )

        if (
            precomputed_indicators
            and "bollinger" in precomputed_indicators
            and "atr" in precomputed_indicators
            and bb_key in precomputed_indicators["bollinger"]
            and atr_key in precomputed_indicators["atr"]
        ):
            # logger.info(f"⚡ FAST PATH: Réutilisation BB({bb_key}), ATR({atr_key})")
            pass  # Log désactivé pour performance

            # Récupération Bollinger pré-calculé
            bb_result = precomputed_indicators["bollinger"][bb_key]
            if isinstance(bb_result, tuple) and len(bb_result) == 3:
                upper, middle, lower = bb_result
                df_bb = df.copy()
                df_bb["bb_upper"] = upper
                df_bb["bb_middle"] = middle
                df_bb["bb_lower"] = lower

                # Calcul Z-score
                close = df["close"].values
                bb_std_dev = (upper - lower) / (4 * params.bb_std)
                df_bb["bb_z"] = (close - middle) / bb_std_dev
            else:
                raise ValueError(f"Bollinger format invalide: {type(bb_result)}")

            # Récupération ATR pré-calculé
            atr_array = precomputed_indicators["atr"][atr_key]
            if not isinstance(atr_array, np.ndarray):
                raise ValueError(f"ATR format invalide: {type(atr_array)}")

            return df_bb, atr_array

        # Sinon, calcul classique via IndicatorBank
        logger.debug(
            f"Calcul indicateurs: BB(period={params.bb_period}, std={params.bb_std}), ATR(period={params.atr_period})"
        )

        # Bollinger Bands via IndicatorBank
        bb_result = ensure_indicator(
            "bollinger",
            {"period": params.bb_period, "std": params.bb_std},
            df,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        if isinstance(bb_result, tuple) and len(bb_result) == 3:
            # Format (upper, middle, lower)
            upper, middle, lower = bb_result
            df_bb = df.copy()
            df_bb["bb_upper"] = upper
            df_bb["bb_middle"] = middle
            df_bb["bb_lower"] = lower

            # Calcul Z-score manual (pas dans cache)
            close = df["close"].values
            bb_std_dev = (upper - lower) / (4 * params.bb_std)  # Approximation
            df_bb["bb_z"] = (close - middle) / bb_std_dev

        else:
            raise ValueError(f"Bollinger result format invalide: {type(bb_result)}")

        # ATR via IndicatorBank
        atr_result = ensure_indicator(
            "atr",
            {"period": params.atr_period, "method": "ema"},  # Plus réactif que SMA
            df,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        if isinstance(atr_result, np.ndarray):
            atr_array = atr_result
        else:
            raise ValueError(f"ATR result format invalide: {type(atr_result)}")

        logger.debug(
            f"Indicateurs calculés: BB Z-score range [{df_bb['bb_z'].min():.2f}, {df_bb['bb_z'].max():.2f}], ATR moyen {atr_array.mean():.4f}"
        )

        return df_bb, atr_array

    def _calculate_trend_filter(
        self, close: np.ndarray, trend_period: int
    ) -> Optional[np.ndarray]:
        """
        Calcule le filtre de tendance EMA optionnel.

        Args:
            close: Prix de clôture
            trend_period: Période EMA (0=désactivé)

        Returns:
            Array EMA ou None si désactivé
        """
        if trend_period <= 0:
            return None

        # EMA simple via pandas (plus efficace que implémentation manuelle)
        close_series = pd.Series(close)
        ema = close_series.ewm(span=trend_period, adjust=False).mean().values

        logger.debug(
            f"Filtre tendance calculé: EMA({trend_period}), dernière valeur {ema[-1]:.2f}"
        )
        return np.array(ema) if ema is not None else None

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: dict,
        precomputed_indicators: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Génère les signaux de trading basés sur Bollinger+ATR.

        Args:
            df: DataFrame OHLCV avec timestamp index (UTC)
            params: Dictionnaire paramètres (format BBAtrParams.to_dict())
            precomputed_indicators: Dictionnaire {key: result} d'indicateurs pré-calculés (optionnel)

        Returns:
            DataFrame avec colonne 'signal' et métadonnées

        Signals générés:
        - "ENTER_LONG": Z-score < -entry_z (prix en dessous bande basse)
        - "ENTER_SHORT": Z-score > entry_z (prix au dessus bande haute)
        - "EXIT": Conditions de sortie (stop, take profit, durée)
        - "HOLD": Maintenir position actuelle
        """
        logger.debug(f"Génération signaux BB+ATR: {len(df)} barres")

        # Validation inputs
        validate_ohlcv_dataframe(df)
        validate_strategy_params(params, ["bb_period", "bb_std", "entry_z"])

        # Parse paramètres
        strategy_params = BBAtrParams.from_dict(params)

        # Ensure indicateurs (utilise pré-calculés si fournis)
        df_with_indicators, atr_array = self._ensure_indicators(
            df, strategy_params, precomputed_indicators=precomputed_indicators
        )

        # Extraction des données
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        bb_z = df_with_indicators["bb_z"].values
        bb_upper = df_with_indicators["bb_upper"].values
        bb_lower = df_with_indicators["bb_lower"].values
        bb_middle = df_with_indicators["bb_middle"].values

        # Filtre tendance optionnel
        trend_ema = self._calculate_trend_filter(
            np.array(close), strategy_params.trend_period
        )

        # Initialisation signaux
        n_bars = len(df)
        signals = np.full(n_bars, "HOLD", dtype=object)

        # Logique de signaux
        logger.debug(
            f"Application logique signaux: entry_z=±{strategy_params.entry_z}, logic={strategy_params.entry_logic}"
        )

        # Conditions d'entrée
        enter_long_condition = np.array(bb_z) < -strategy_params.entry_z
        enter_short_condition = np.array(bb_z) > strategy_params.entry_z

        # Filtre tendance si activé
        if trend_ema is not None:
            if strategy_params.entry_logic == "AND":
                # AND: tendance doit confirmer signal
                enter_long_condition = enter_long_condition & (close > trend_ema)
                enter_short_condition = enter_short_condition & (close < trend_ema)
            else:
                # OR: tendance ou Bollinger peut déclencher
                enter_long_condition = enter_long_condition | (close > trend_ema)
                enter_short_condition = enter_short_condition | (close < trend_ema)

        # Application des signaux avec espacement
        last_signal_bar = -strategy_params.spacing_bars - 1

        for i in range(strategy_params.bb_period, n_bars):  # Skip période de warmup
            # Vérification espacement minimum
            if i - last_signal_bar < strategy_params.spacing_bars:
                continue

            # Filtrage NaN (indicateurs pas encore stables)
            if np.isnan(bb_z[i]) or np.isnan(atr_array[i]):
                continue

            # Signal ENTER_LONG
            if (
                enter_long_condition[i] and not enter_long_condition[i - 1]
            ):  # Nouveau signal
                signals[i] = "ENTER_LONG"
                last_signal_bar = i
                logger.debug(
                    f"ENTER_LONG @ bar {i}: price={close[i]:.2f}, z={bb_z[i]:.2f}, atr={atr_array[i]:.4f}"
                )

            # Signal ENTER_SHORT
            elif (
                enter_short_condition[i] and not enter_short_condition[i - 1]
            ):  # Nouveau signal
                signals[i] = "ENTER_SHORT"
                last_signal_bar = i
                logger.debug(
                    f"ENTER_SHORT @ bar {i}: price={close[i]:.2f}, z={bb_z[i]:.2f}, atr={atr_array[i]:.4f}"
                )

        # Construction DataFrame de sortie
        result_df = pd.DataFrame(index=df.index)
        result_df["signal"] = signals

        # Métadonnées pour chaque barre
        result_df["bb_z"] = bb_z
        result_df["bb_upper"] = bb_upper
        result_df["bb_middle"] = bb_middle
        result_df["bb_lower"] = bb_lower
        result_df["atr"] = atr_array
        result_df["close"] = close

        if trend_ema is not None:
            result_df["trend_ema"] = trend_ema

        # Statistiques signaux
        enter_longs = np.sum(signals == "ENTER_LONG")
        enter_shorts = np.sum(signals == "ENTER_SHORT")
        total_signals = enter_longs + enter_shorts

        logger.debug(
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
        precomputed_indicators: Optional[Dict] = None,
    ) -> Tuple[pd.Series, RunStats]:
        """
        Exécute un backtest complet de la stratégie BB+ATR.

        Args:
            df: DataFrame OHLCV avec timestamp index (UTC)
            params: Paramètres stratégie (format BBAtrParams.to_dict())
            initial_capital: Capital initial
            fee_bps: Frais de transaction en basis points (défaut: 4.5)
            slippage_bps: Slippage en basis points (défaut: 0.0)
            precomputed_indicators: Dictionnaire {key: result} d'indicateurs pré-calculés (optionnel)
                                   Permet de skip ensure_indicator() et réutiliser calculs batch

        Returns:
            Tuple (equity_curve, run_stats) avec:
            - equity_curve: Série temporelle de l'équité
            - run_stats: Statistiques complètes du run

        Gestion des positions:
        - Size basé sur ATR et risk_per_trade
        - Stop loss dynamique: entry_price ± (ATR * atr_multiplier)
        - Trailing stop si activé
        - Sortie forcée après max_hold_bars
        - Filtrage trades avec PnL < min_pnl_pct
        """
        logger.debug(
            f"Début backtest BB+ATR: capital={initial_capital}, fee={fee_bps}bps, slippage={slippage_bps}bps"
        )

        # Validation
        validate_ohlcv_dataframe(df)
        strategy_params = BBAtrParams.from_dict(params)

        # Génération signaux (avec indicateurs pré-calculés si disponibles)
        signals_df = self.generate_signals(
            df, params, precomputed_indicators=precomputed_indicators
        )

        # Initialisation backtest
        n_bars = len(df)
        equity = np.full(n_bars, initial_capital, dtype=float)

        cash = initial_capital
        position = None  # Trade actuel ou None
        trades: List[Trade] = []

        fee_rate = (fee_bps + slippage_bps) / 10000.0

        logger.debug(f"Backtest initialisé: {n_bars} barres, fee_rate={fee_rate:.6f}")

        # Pré-extraction des colonnes en numpy arrays (3-4x plus rapide que iterrows)
        close_vals = signals_df["close"].values
        atr_vals = signals_df["atr"].values
        signal_vals = signals_df["signal"].values
        bb_middle_vals = signals_df["bb_middle"].values
        bb_z_vals = signals_df["bb_z"].values
        timestamps = signals_df.index.values

        # Détecter si l'index du DataFrame a un timezone
        has_tz = df.index.tz is not None

        # Boucle principale
        for i in range(n_bars):
            current_price = close_vals[i]
            current_atr = atr_vals[i]
            signal = signal_vals[i]
            # Créer timestamp en respectant le timezone de l'index pour éviter les erreurs
            timestamp = (
                pd.Timestamp(timestamps[i], tz=df.index.tz)
                if has_tz
                else pd.Timestamp(timestamps[i])
            )

            # Skip si ATR invalide
            if np.isnan(current_atr) or current_atr <= 0:
                equity[i] = (
                    cash
                    if position is None
                    else cash + position.calculate_unrealized_pnl(current_price)
                )
                continue

            # Gestion position existante
            if position is not None:
                # Vérification stops et conditions de sortie
                should_exit = False
                exit_reason = ""

                # 1. Stop loss ATR
                if position.should_stop_loss(current_price):
                    should_exit = True
                    exit_reason = "stop_loss"

                # 2. Take profit (retour vers BB middle)
                elif position.is_long() and current_price >= bb_middle_vals[i]:
                    should_exit = True
                    exit_reason = "take_profit_bb_middle"

                elif position.is_short() and current_price <= bb_middle_vals[i]:
                    should_exit = True
                    exit_reason = "take_profit_bb_middle"

                # 3. Durée maximale
                # Calcul O(1) au lieu de O(n) : utiliser l'index de barre
                entry_bar_index = position.meta.get("entry_bar_index", 0)
                bars_held = i - entry_bar_index

                if bars_held >= strategy_params.max_hold_bars:
                    should_exit = True
                    exit_reason = "max_hold_bars"

                # 4. Trailing stop ATR si activé
                if strategy_params.trailing_stop and not should_exit:
                    # Mise à jour trailing stop
                    new_stop = None
                    if position.is_long():
                        new_stop = current_price - (
                            current_atr * strategy_params.atr_multiplier
                        )
                        if new_stop > position.stop:  # Trail vers le haut seulement
                            position.stop = new_stop
                    else:
                        new_stop = current_price + (
                            current_atr * strategy_params.atr_multiplier
                        )
                        if new_stop < position.stop:  # Trail vers le bas seulement
                            position.stop = new_stop

                # Fermeture position
                if should_exit:
                    # Calcul frais de sortie
                    exit_value = current_price * position.qty
                    exit_fees = exit_value * fee_rate

                    # Fermeture trade
                    position.close_trade(
                        exit_price=current_price,
                        exit_time=(
                            str(timestamp)
                            if hasattr(timestamp, "isoformat")
                            else str(timestamp)
                        ),
                        exit_fees=exit_fees,
                    )

                    # Filtrage min PnL
                    pnl_val = (
                        position.pnl_realized
                        if position.pnl_realized is not None
                        else 0.0
                    )
                    pnl_pct = abs(pnl_val / (position.entry_price * position.qty)) * 100
                    if pnl_pct >= strategy_params.min_pnl_pct:
                        # Trade valide: mise à jour cash
                        pnl_val = (
                            position.pnl_realized
                            if position.pnl_realized is not None
                            else 0.0
                        )
                        cash += pnl_val + (position.entry_price * position.qty)
                        trades.append(position)
                        logger.debug(
                            f"Position fermée @ {current_price:.2f}: {exit_reason}, PnL={position.pnl_realized:.2f}"
                        )
                    else:
                        # Trade filtré: PnL trop faible
                        logger.debug(
                            f"Trade filtré (PnL {pnl_pct:.4f}% < {strategy_params.min_pnl_pct}%)"
                        )

                    position = None

            # Nouveau signal d'entrée (si pas de position)
            if position is None and signal in ["ENTER_LONG", "ENTER_SHORT"]:
                # Position sizing basé sur ATR et risk
                atr_stop_distance = current_atr * strategy_params.atr_multiplier
                risk_amount = cash * strategy_params.risk_per_trade

                # Calcul quantité optimale
                position_size = risk_amount / atr_stop_distance
                max_position_size = (cash * strategy_params.leverage) / current_price

                qty = min(position_size, max_position_size)

                if qty > 0:
                    # Calcul prix stop
                    if signal == "ENTER_LONG":
                        stop_price = current_price - atr_stop_distance
                    else:
                        stop_price = current_price + atr_stop_distance

                    # Frais d'entrée
                    entry_value = current_price * qty
                    entry_fees = entry_value * fee_rate

                    if entry_value + entry_fees <= cash:
                        # Création nouveau trade
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
                                "entry_bar_index": i,  # Stocker l'index pour calcul O(1) de bars_held
                            },
                        )

                        # Mise à jour cash
                        cash -= entry_value + entry_fees

                        logger.debug(
                            f"Nouvelle position: {signal} {qty:.4f} @ {current_price:.2f}, stop={stop_price:.2f}"
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

            # Application filtrage min PnL
            pnl_val = (
                position.pnl_realized if position.pnl_realized is not None else 0.0
            )
            pnl_pct = abs(pnl_val / (position.entry_price * position.qty)) * 100
            if pnl_pct >= strategy_params.min_pnl_pct:
                trades.append(position)

        # Construction courbe d'équité
        equity_curve = pd.Series(equity, index=df.index)

        # Calcul statistiques
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

        logger.debug(
            f"Backtest terminé: {run_stats.total_trades} trades, PnL={run_stats.total_pnl:.2f} ({run_stats.total_pnl_pct:.2f}%)"
        )

        return equity_curve, run_stats


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================


def generate_signals(
    df: pd.DataFrame, params: dict, symbol: str = "UNKNOWN", timeframe: str = "15m"
) -> pd.DataFrame:
    """
    Fonction de convenance pour génération de signaux BB+ATR.

    Args:
        df: DataFrame OHLCV
        params: Paramètres stratégie
        symbol: Symbole pour cache
        timeframe: Timeframe pour cache

    Returns:
        DataFrame avec signaux et métadonnées

    Example:
        >>> params = {'bb_period': 20, 'bb_std': 2.0, 'entry_z': 1.5}
        >>> signals = generate_signals(df, params, "BTCUSDT", "1h")
    """
    strategy = BBAtrStrategy(symbol=symbol, timeframe=timeframe)
    return strategy.generate_signals(df, params)


def backtest(
    df: pd.DataFrame,
    params: dict,
    initial_capital: float = 10000.0,
    symbol: str = "UNKNOWN",
    timeframe: str = "15m",
    **kwargs,
) -> Tuple[pd.Series, RunStats]:
    """
    Fonction de convenance pour backtest BB+ATR.

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
        >>> params = BBAtrParams(bb_period=50, atr_multiplier=2.0).to_dict()
        >>> equity, stats = backtest(df, params, 50000, "ETHUSDT", "4h")
        >>> print(f"ROI: {stats.total_pnl_pct:.2f}%, Trades: {stats.total_trades}")
    """
    strategy = BBAtrStrategy(symbol=symbol, timeframe=timeframe)
    return strategy.backtest(df, params, initial_capital, **kwargs)


def create_default_params(**overrides) -> BBAtrParams:
    """
    Crée des paramètres par défaut avec surcharges optionnelles.

    Args:
        **overrides: Paramètres à surcharger

    Returns:
        Instance BBAtrParams avec valeurs par défaut + surcharges

    Example:
        >>> params = create_default_params(bb_period=50, atr_multiplier=2.5)
        >>> params.bb_period
        50
        >>> params.bb_std  # Valeur par défaut conservée
        2.0
    """
    base_params = BBAtrParams()

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
    "BBAtrParams",
    "BBAtrStrategy",
    # Fonctions de convenance
    "generate_signals",
    "backtest",
    "create_default_params",
]
