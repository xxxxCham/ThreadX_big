"""
ThreadX Phase 4 - Model Layer
=============================

Types de données et structures pour les stratégies de trading.

Modules:
- Trade: Structure de transaction
- RunStats: Statistiques de performance
- Strategy Protocol: Interface standardisée
- JSON serialization/désérialization

Caractéristiques:
- TypedDict et dataclasses pour performances et validation
- Sérialisation JSON complète
- Protocol Pattern pour extensibilité
- Validation intégrée des paramètres
"""

from dataclasses import dataclass, field, asdict
from typing import Protocol, Dict, Any, Optional, List, Tuple, Union
from typing_extensions import TypedDict, NotRequired
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from threadx.config.settings import S
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# ==========================================
# TRADE STRUCTURES
# ==========================================


class TradeDict(TypedDict):
    """TypedDict pour trade optimisé en performance"""

    side: str  # "LONG" | "SHORT"
    qty: float  # Quantité
    entry_price: float  # Prix d'entrée
    entry_time: str  # Timestamp ISO UTC
    exit_price: NotRequired[float]  # Prix sortie (optionnel)
    exit_time: NotRequired[str]  # Timestamp sortie (optionnel)
    stop: float  # Stop loss
    take_profit: NotRequired[float]  # Take profit (optionnel)
    pnl_realized: NotRequired[float]  # PnL réalisé (optionnel)
    pnl_unrealized: NotRequired[float]  # PnL non réalisé (optionnel)
    fees_paid: NotRequired[float]  # Frais payés
    meta: NotRequired[Dict[str, Any]]  # Métadonnées


@dataclass
class Trade:
    """
    Représentation complète d'une transaction.

    Attributes:
        side: Direction ("LONG", "SHORT")
        qty: Quantité en nombre d'unités
        entry_price: Prix d'entrée
        entry_time: Timestamp d'entrée (UTC)
        stop: Prix de stop loss
        exit_price: Prix de sortie (None si pas encore fermé)
        exit_time: Timestamp de sortie (None si pas encore fermé)
        take_profit: Prix de take profit (optionnel)
        pnl_realized: PnL réalisé après fermeture
        pnl_unrealized: PnL actuel si position ouverte
        fees_paid: Total des frais payés
        meta: Dictionnaire de métadonnées (indicateurs, contexte, etc.)

    Example:
        >>> trade = Trade(
        ...     side="LONG",
        ...     qty=1.5,
        ...     entry_price=50000.0,
        ...     entry_time="2024-01-15T10:30:00Z",
        ...     stop=48000.0,
        ...     meta={"bb_z": -2.1, "atr": 1200.5}
        ... )
        >>> trade.is_open()
        True
        >>> trade.calculate_unrealized_pnl(51000.0)
        1500.0
    """

    side: str
    qty: float
    entry_price: float
    entry_time: str
    stop: float
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    take_profit: Optional[float] = None
    pnl_realized: Optional[float] = None
    pnl_unrealized: Optional[float] = None
    fees_paid: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation des paramètres après initialisation"""
        if self.side not in ["LONG", "SHORT"]:
            raise ValueError(f"Side must be 'LONG' or 'SHORT', got: {self.side}")

        if self.qty <= 0:
            raise ValueError(f"Quantity must be positive, got: {self.qty}")

        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got: {self.entry_price}")

        # Validation timestamps
        try:
            pd.to_datetime(self.entry_time, utc=True)
            if self.exit_time:
                pd.to_datetime(self.exit_time, utc=True)
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")

    def is_open(self) -> bool:
        """Vérifie si la position est encore ouverte"""
        return self.exit_price is None

    def is_long(self) -> bool:
        """Vérifie si c'est une position longue"""
        return self.side == "LONG"

    def is_short(self) -> bool:
        """Vérifie si c'est une position courte"""
        return self.side == "SHORT"

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calcule le PnL non réalisé à un prix donné.

        Args:
            current_price: Prix actuel du marché

        Returns:
            PnL non réalisé (positif = gain, négatif = perte)
        """
        if not self.is_open():
            return self.pnl_realized or 0.0

        if self.is_long():
            pnl = (current_price - self.entry_price) * self.qty
        else:
            pnl = (self.entry_price - current_price) * self.qty

        self.pnl_unrealized = pnl - self.fees_paid
        return self.pnl_unrealized

    def close_trade(self, exit_price: float, exit_time: str, exit_fees: float = 0.0):
        """
        Ferme la position et calcule le PnL réalisé.

        Args:
            exit_price: Prix de sortie
            exit_time: Timestamp de sortie (UTC)
            exit_fees: Frais de sortie supplémentaires
        """
        if not self.is_open():
            logger.warning(f"Tentative de fermeture d'une position déjà fermée")
            return

        self.exit_price = exit_price
        self.exit_time = exit_time
        self.fees_paid += exit_fees

        # Calcul PnL réalisé
        if self.is_long():
            gross_pnl = (exit_price - self.entry_price) * self.qty
        else:
            gross_pnl = (self.entry_price - exit_price) * self.qty

        self.pnl_realized = gross_pnl - self.fees_paid
        self.pnl_unrealized = None

        logger.debug(
            f"Position fermée: {self.side} {self.qty} @ {exit_price}, PnL: {self.pnl_realized:.2f}"
        )

    def should_stop_loss(self, current_price: float) -> bool:
        """Vérifie si le stop loss doit être déclenché"""
        if not self.is_open():
            return False

        if self.is_long():
            return current_price <= self.stop
        else:
            return current_price >= self.stop

    def should_take_profit(self, current_price: float) -> bool:
        """Vérifie si le take profit doit être déclenché"""
        if not self.is_open() or not self.take_profit:
            return False

        if self.is_long():
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit

    def duration_minutes(self) -> Optional[float]:
        """Calcule la durée du trade en minutes"""
        if not self.exit_time:
            return None

        entry_dt = pd.to_datetime(self.entry_time, utc=True)
        exit_dt = pd.to_datetime(self.exit_time, utc=True)
        return (exit_dt - entry_dt).total_seconds() / 60.0

    def roi_percent(self) -> Optional[float]:
        """Calcule le ROI en pourcentage sur la mise engagée"""
        if self.pnl_realized is None:
            return None

        invested = self.entry_price * self.qty
        return (self.pnl_realized / invested) * 100.0 if invested > 0 else 0.0

    def to_dict(self) -> TradeDict:
        """Convertit en dictionnaire pour JSON"""
        return TradeDict(
            side=self.side,
            qty=self.qty,
            entry_price=self.entry_price,
            entry_time=self.entry_time,
            stop=self.stop,
            exit_price=self.exit_price or 0.0,  # Valeur par défaut pour None
            exit_time=self.exit_time or "",  # Valeur par défaut pour None
            take_profit=self.take_profit or 0.0,
            pnl_realized=self.pnl_realized or 0.0,
            pnl_unrealized=self.pnl_unrealized or 0.0,
            fees_paid=self.fees_paid,
            meta=self.meta,
        )

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], TradeDict]) -> "Trade":
        """Crée un Trade depuis un dictionnaire"""
        return cls(
            side=data["side"],
            qty=data["qty"],
            entry_price=data["entry_price"],
            entry_time=data["entry_time"],
            stop=data["stop"],
            exit_price=data.get("exit_price"),
            exit_time=data.get("exit_time"),
            take_profit=data.get("take_profit"),
            pnl_realized=data.get("pnl_realized"),
            pnl_unrealized=data.get("pnl_unrealized"),
            fees_paid=data.get("fees_paid", 0.0),
            meta=data.get("meta", {}),
        )


# ==========================================
# RUN STATISTICS
# ==========================================


class RunStatsDict(TypedDict):
    """TypedDict pour statistiques de run"""

    final_equity: float
    initial_capital: float
    total_pnl: float
    total_pnl_pct: float
    sharpe_ratio: NotRequired[float]
    sortino_ratio: NotRequired[float]
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: NotRequired[int]
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate_pct: float
    avg_win: NotRequired[float]
    avg_loss: NotRequired[float]
    profit_factor: NotRequired[float]
    avg_trade_duration_minutes: NotRequired[float]
    total_fees_paid: float
    start_time: str
    end_time: str
    bars_analyzed: int
    meta: NotRequired[Dict[str, Any]]


@dataclass
class RunStats:
    """
    Statistiques complètes d'un run de backtesting.

    Attributes:
        final_equity: Capital final
        initial_capital: Capital initial
        total_pnl: PnL total réalisé
        sharpe_ratio: Ratio de Sharpe (rendement/risque)
        sortino_ratio: Ratio de Sortino (rendement/downside risk)
        max_drawdown: Drawdown maximum absolu
        max_drawdown_pct: Drawdown maximum en %
        total_trades: Nombre total de trades
        win_trades: Nombre de trades gagnants
        loss_trades: Nombre de trades perdants
        win_rate_pct: Taux de réussite en %
        avg_win: Gain moyen par trade gagnant
        avg_loss: Perte moyenne par trade perdant
        profit_factor: Facteur de profit (total gains / total pertes)
        avg_trade_duration_minutes: Durée moyenne des trades
        total_fees_paid: Total des frais payés
        start_time: Timestamp début analyse
        end_time: Timestamp fin analyse
        bars_analyzed: Nombre de barres analysées
        meta: Métadonnées (paramètres, configuration, etc.)

    Example:
        >>> equity_curve = pd.Series([10000, 10500, 9800, 11200])
        >>> trades = [Trade(...), Trade(...)]
        >>> stats = RunStats.from_trades_and_equity(trades, equity_curve, 10000)
        >>> print(f"ROI: {stats.total_pnl_pct:.2f}%, Win Rate: {stats.win_rate_pct:.1f}%")
    """

    final_equity: float
    initial_capital: float
    total_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate_pct: float
    total_fees_paid: float
    start_time: str
    end_time: str
    bars_analyzed: int
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown_duration_bars: Optional[int] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_trade_duration_minutes: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pnl_pct(self) -> float:
        """ROI total en pourcentage"""
        return (
            (self.total_pnl / self.initial_capital) * 100.0
            if self.initial_capital > 0
            else 0.0
        )

    @property
    def is_profitable(self) -> bool:
        """Vérifie si le run est profitable"""
        return self.total_pnl > 0

    @property
    def has_trades(self) -> bool:
        """Vérifie si des trades ont été générés"""
        return self.total_trades > 0

    def risk_reward_ratio(self) -> Optional[float]:
        """Calcule le ratio risque/récompense"""
        if not self.avg_win or not self.avg_loss or self.avg_loss >= 0:
            return None
        return abs(self.avg_win / self.avg_loss)

    def expectancy(self) -> Optional[float]:
        """Calcule l'espérance de gain par trade"""
        if not self.has_trades:
            return None

        win_prob = self.win_rate_pct / 100.0
        loss_prob = 1.0 - win_prob

        if self.avg_win and self.avg_loss:
            return (win_prob * self.avg_win) + (loss_prob * self.avg_loss)

        return self.total_pnl / self.total_trades

    def to_dict(self) -> RunStatsDict:
        """Convertit en dictionnaire pour JSON"""
        return RunStatsDict(
            final_equity=self.final_equity,
            initial_capital=self.initial_capital,
            total_pnl=self.total_pnl,
            total_pnl_pct=self.total_pnl_pct,
            sharpe_ratio=self.sharpe_ratio or 0.0,
            sortino_ratio=self.sortino_ratio or 0.0,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=self.max_drawdown_pct,
            max_drawdown_duration_bars=self.max_drawdown_duration_bars or 0,
            total_trades=self.total_trades,
            win_trades=self.win_trades,
            loss_trades=self.loss_trades,
            win_rate_pct=self.win_rate_pct,
            avg_win=self.avg_win or 0.0,
            avg_loss=self.avg_loss or 0.0,
            profit_factor=self.profit_factor or 0.0,
            avg_trade_duration_minutes=self.avg_trade_duration_minutes or 0.0,
            total_fees_paid=self.total_fees_paid,
            start_time=self.start_time,
            end_time=self.end_time,
            bars_analyzed=self.bars_analyzed,
            meta=self.meta,
        )

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], RunStatsDict]) -> "RunStats":
        """Crée RunStats depuis un dictionnaire"""
        return cls(
            final_equity=data["final_equity"],
            initial_capital=data["initial_capital"],
            total_pnl=data["total_pnl"],
            max_drawdown=data["max_drawdown"],
            max_drawdown_pct=data["max_drawdown_pct"],
            total_trades=data["total_trades"],
            win_trades=data["win_trades"],
            loss_trades=data["loss_trades"],
            win_rate_pct=data["win_rate_pct"],
            total_fees_paid=data["total_fees_paid"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            bars_analyzed=data["bars_analyzed"],
            sharpe_ratio=data.get("sharpe_ratio"),
            sortino_ratio=data.get("sortino_ratio"),
            max_drawdown_duration_bars=data.get("max_drawdown_duration_bars"),
            avg_win=data.get("avg_win"),
            avg_loss=data.get("avg_loss"),
            profit_factor=data.get("profit_factor"),
            avg_trade_duration_minutes=data.get("avg_trade_duration_minutes"),
            meta=data.get("meta", {}),
        )

    @classmethod
    def from_trades_and_equity(
        cls,
        trades: List[Trade],
        equity_curve: pd.Series,
        initial_capital: float,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "RunStats":
        """
        Calcule les statistiques depuis une liste de trades et courbe d'équité.

        Args:
            trades: Liste des trades exécutés
            equity_curve: Série temporelle de l'équité
            initial_capital: Capital initial
            start_time: Timestamp de début (optionnel)
            end_time: Timestamp de fin (optionnel)
            meta: Métadonnées supplémentaires

        Returns:
            RunStats calculé avec toutes les métriques
        """
        logger.debug(
            f"Calcul statistiques depuis {len(trades)} trades et equity_curve de {len(equity_curve)} points"
        )

        # Statistiques de base
        final_equity = (
            float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else initial_capital
        )
        total_pnl = final_equity - initial_capital

        # Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = equity_curve - running_max
        max_drawdown = float(drawdown.min())
        max_drawdown_pct = (
            (max_drawdown / initial_capital) * 100.0 if initial_capital > 0 else 0.0
        )

        # Durée drawdown maximum
        is_at_max = equity_curve == running_max
        drawdown_periods = []
        current_period = 0
        for at_max in is_at_max:
            if at_max:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
            else:
                current_period += 1
        if current_period > 0:
            drawdown_periods.append(current_period)

        max_drawdown_duration_bars = max(drawdown_periods) if drawdown_periods else 0

        # Statistiques trades
        closed_trades = [t for t in trades if not t.is_open()]
        total_trades = len(closed_trades)

        if total_trades > 0:
            wins = [t for t in closed_trades if t.pnl_realized and t.pnl_realized > 0]
            losses = [
                t for t in closed_trades if t.pnl_realized and t.pnl_realized <= 0
            ]

            win_trades = len(wins)
            loss_trades = len(losses)
            win_rate_pct = (win_trades / total_trades) * 100.0

            # Filtrer les None pour les calculs numpy
            win_pnls = [t.pnl_realized for t in wins if t.pnl_realized is not None]
            loss_pnls = [t.pnl_realized for t in losses if t.pnl_realized is not None]

            avg_win = np.mean(win_pnls) if win_pnls else None
            avg_loss = np.mean(loss_pnls) if loss_pnls else None

            total_wins = sum(win_pnls) if win_pnls else 0
            total_losses = abs(sum(loss_pnls)) if loss_pnls else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else None

            # Durée moyenne des trades
            durations = [
                t.duration_minutes()
                for t in closed_trades
                if t.duration_minutes() is not None
            ]
            # Filtrer les None pour la durée
            valid_durations = [d for d in durations if d is not None]
            avg_trade_duration_minutes = (
                np.mean(valid_durations) if valid_durations else None
            )

        else:
            win_trades = loss_trades = 0
            win_rate_pct = 0.0
            avg_win = avg_loss = profit_factor = avg_trade_duration_minutes = None

        # Ratios de Sharpe et Sortino
        sharpe_ratio = sortino_ratio = None
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))

                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0 and negative_returns.std() > 0:
                    sortino_ratio = float(
                        returns.mean() / negative_returns.std() * np.sqrt(252)
                    )

        # Frais totaux
        total_fees_paid = sum(t.fees_paid for t in trades)

        # Timestamps
        if not start_time and len(equity_curve) > 0:
            start_time = (
                equity_curve.index[0].isoformat()
                if hasattr(equity_curve.index[0], "isoformat")
                else str(equity_curve.index[0])
            )
        if not end_time and len(equity_curve) > 0:
            end_time = (
                equity_curve.index[-1].isoformat()
                if hasattr(equity_curve.index[-1], "isoformat")
                else str(equity_curve.index[-1])
            )

        return cls(
            final_equity=final_equity,
            initial_capital=initial_capital,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration_bars=max_drawdown_duration_bars,
            total_trades=total_trades,
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate_pct=win_rate_pct,
            avg_win=float(avg_win) if avg_win is not None else None,
            avg_loss=float(avg_loss) if avg_loss is not None else None,
            profit_factor=profit_factor,
            avg_trade_duration_minutes=(
                float(avg_trade_duration_minutes)
                if avg_trade_duration_minutes is not None
                else None
            ),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_fees_paid=total_fees_paid,
            start_time=start_time or "",
            end_time=end_time or "",
            bars_analyzed=len(equity_curve),
            meta=meta or {},
        )


# ==========================================
# STRATEGY PROTOCOL
# ==========================================


class Strategy(Protocol):
    """
    Protocol définissant l'interface standardisée pour les stratégies de trading.

    Toute stratégie doit implémenter ces méthodes pour être compatible
    avec le framework ThreadX.

    Example:
        >>> class MyStrategy:
        ...     def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        ...         # Implémentation des signaux
        ...         pass
        ...
        ...     def backtest(self, df: pd.DataFrame, params: dict, initial_capital: float) -> Tuple[pd.Series, RunStats]:
        ...         # Implémentation du backtest
        ...         pass
        >>>
        >>> # La stratégie respecte automatiquement le Protocol
        >>> strategy: Strategy = MyStrategy()
    """

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Génère les signaux de trading pour un DataFrame OHLCV.

        Args:
            df: DataFrame avec colonnes OHLCV + timestamp index (UTC)
            params: Dictionnaire des paramètres de stratégie

        Returns:
            DataFrame avec colonne 'signal' contenant les signaux:
            - "ENTER_LONG": Entrer en position longue
            - "ENTER_SHORT": Entrer en position courte
            - "EXIT": Sortir de position
            - "HOLD": Maintenir position actuelle

        Raises:
            ValueError: Si les paramètres sont invalides
            KeyError: Si des colonnes OHLCV sont manquantes
        """
        ...

    def backtest(
        self, df: pd.DataFrame, params: dict, initial_capital: float = 10000.0
    ) -> Tuple[pd.Series, RunStats]:
        """
        Exécute un backtest complet de la stratégie.

        Args:
            df: DataFrame OHLCV avec timestamp index (UTC)
            params: Paramètres de stratégie (dépendants de l'implémentation)
            initial_capital: Capital initial en devise de base

        Returns:
            Tuple contenant:
            - pd.Series: Courbe d'équité indexée par timestamp
            - RunStats: Statistiques complètes du run

        Raises:
            ValueError: Si les paramètres ou données sont invalides
        """
        ...


# ==========================================
# JSON SERIALIZATION
# ==========================================


class ThreadXJSONEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour les types ThreadX"""

    def default(self, o):  # Nom correct pour JSONEncoder
        if isinstance(o, (Trade, RunStats)):
            return o.to_dict()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, pd.Timestamp):
            return o.isoformat()
        elif isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def save_run_results(
    trades: List[Trade],
    stats: RunStats,
    equity_curve: pd.Series,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Sauvegarde les résultats d'un run en JSON.

    Args:
        trades: Liste des trades exécutés
        stats: Statistiques du run
        equity_curve: Courbe d'équité
        output_path: Chemin de sauvegarde
        metadata: Métadonnées supplémentaires (paramètres, config, etc.)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Conversion equity curve pour JSON
    equity_data = {
        "timestamps": [
            ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            for ts in equity_curve.index
        ],
        "values": equity_curve.values.tolist(),
    }

    run_data = {
        "metadata": metadata or {},
        "stats": stats.to_dict(),
        "trades": [trade.to_dict() for trade in trades],
        "equity_curve": equity_data,
        "generated_at": datetime.utcnow().isoformat(),
        "threadx_version": "4.0.0",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, cls=ThreadXJSONEncoder, indent=2, ensure_ascii=False)

    logger.info(
        f"Résultats sauvés: {output_path} ({len(trades)} trades, {stats.total_trades} fermés)"
    )


def load_run_results(
    file_path: Union[str, Path],
) -> Tuple[List[Trade], RunStats, pd.Series]:
    """
    Charge les résultats d'un run depuis JSON.

    Args:
        file_path: Chemin du fichier JSON

    Returns:
        Tuple contenant (trades, stats, equity_curve)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruction des objets
    trades = [Trade.from_dict(trade_data) for trade_data in data["trades"]]
    stats = RunStats.from_dict(data["stats"])

    # Reconstruction equity curve
    equity_data = data["equity_curve"]
    timestamps = pd.to_datetime(equity_data["timestamps"], utc=True)
    equity_curve = pd.Series(equity_data["values"], index=timestamps)

    logger.info(
        f"Résultats chargés: {file_path} ({len(trades)} trades, {stats.total_trades} fermés)"
    )
    return trades, stats, equity_curve


# ==========================================
# VALIDATION UTILITIES
# ==========================================


def validate_ohlcv_dataframe(df: pd.DataFrame) -> None:
    """
    Valide qu'un DataFrame contient les colonnes OHLCV requises.

    Args:
        df: DataFrame à valider

    Raises:
        ValueError: Si le DataFrame est invalide
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    required_columns = {"open", "high", "low", "close", "volume"}
    actual_columns = {col.lower() for col in df.columns}
    missing = required_columns - actual_columns

    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    # Validation index timestamp
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    # Validation valeurs numériques
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")

    logger.debug(
        f"DataFrame OHLCV validé: {len(df)} barres, {df.index[0]} à {df.index[-1]}"
    )


def validate_strategy_params(params: dict, required_keys: List[str]) -> None:
    """
    Valide les paramètres de stratégie.

    Args:
        params: Dictionnaire des paramètres
        required_keys: Clés requises

    Raises:
        ValueError: Si des paramètres sont manquants ou invalides
    """
    if not isinstance(params, dict):
        raise ValueError("Strategy params must be a dictionary")

    missing = set(required_keys) - set(params.keys())
    if missing:
        raise ValueError(f"Missing required strategy parameters: {missing}")

    logger.debug(f"Paramètres stratégie validés: {list(params.keys())}")


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    # Types de données
    "Trade",
    "TradeDict",
    "RunStats",
    "RunStatsDict",
    # Protocol
    "Strategy",
    # JSON utilities
    "ThreadXJSONEncoder",
    "save_run_results",
    "load_run_results",
    # Validation
    "validate_ohlcv_dataframe",
    "validate_strategy_params",
]
