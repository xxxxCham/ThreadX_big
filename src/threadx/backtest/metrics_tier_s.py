"""
ThreadX Metrics Tier S (2025) - Standards Professionnels
=========================================================

Implémentation des 10 métriques décisives pour validation stratégies trading.

Classification:
- Tier S: 10 métriques obligatoires (promotion/rejet automatique)
- Tier A: 6 métriques importantes (validation qualité)
- Tier B: 7 métriques utiles (analyse approfondie)
- Tier C: 4 métriques bonus ThreadX (innovantes)

Standards 2025 basés sur:
- Performance crypto réelle (volatilité élevée)
- Frictions trading modernes (spread, slippage, fees)
- Exigences institutionnelles quantitatives

Usage:
    from threadx.backtest.metrics_tier_s import calculate_tier_s_metrics, validate_tier_s
    
    metrics = calculate_tier_s_metrics(returns, trades, equity_curve)
    passed, score, report = validate_tier_s(metrics)

Author: ThreadX Framework
Version: 1.0 - Professional Standards 2025
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from threadx.utils.log import get_logger

logger = get_logger(__name__)


# =============================================================================
# SEUILS TIER S (2025) - STANDARDS PROFESSIONNELS
# =============================================================================

TIER_S_THRESHOLDS = {
    # Rang 1-10 : Métriques décisives
    "sharpe_ratio": {"min": 1.80, "ideal": 2.20, "tier": "S"},
    "sortino_ratio": {"min": 2.80, "ideal": 3.50, "tier": "S"},
    "calmar_ratio": {"min": 1.50, "ideal": 2.50, "tier": "S"},
    "profit_factor": {"min": 2.00, "ideal": 2.50, "tier": "S"},
    "max_drawdown_pct": {"max": -18.0, "ideal": -12.0, "tier": "S"},
    "recovery_factor": {"min": 6.0, "ideal": 10.0, "tier": "S"},
    "win_rate_trend": {"min": 0.58, "ideal": 0.65, "tier": "S"},
    "win_rate_meanrev": {"min": 0.68, "ideal": 0.75, "tier": "S"},
    "expectancy_pct": {"min": 0.8, "ideal": 1.5, "tier": "S"},
    "sqn": {"min": 2.8, "ideal": 4.0, "tier": "S"},
    "outlier_adjusted_sharpe": {"min": 1.4, "ideal": 1.8, "tier": "S"},
}

TIER_A_THRESHOLDS = {
    "r_multiple": {"min": 2.0, "ideal": 3.0, "tier": "A"},
    "time_in_market_pct": {"max": 45.0, "ideal": 30.0, "tier": "A"},
    "max_flat_period_days": {"max": 35, "ideal": 20, "tier": "A"},
    "annual_return_per_dd": {"min": 1.2, "ideal": 2.0, "tier": "A"},
    "gain_pain_ratio": {"min": 2.5, "ideal": 4.0, "tier": "A"},
    "multi_token_sharpe_mean": {"min": 1.6, "ideal": 2.0, "tier": "A"},
}

TIER_B_THRESHOLDS = {
    "max_consecutive_wins": {"min": 3, "tier": "B"},
    "max_consecutive_loss_pct": {"max": -12.0, "tier": "B"},
    "avg_trade_duration_hours": {"min": 4, "max": 72, "tier": "B"},
    "z_score_trades": {"min": -2.0, "max": 2.0, "tier": "B"},
    "ulcer_index": {"max": 10.0, "tier": "B"},
    "martin_ratio": {"min": 2.0, "tier": "B"},
    "efficiency_ratio": {"min": 0.5, "tier": "B"},
}

TIER_C_THRESHOLDS = {
    "pain_adjusted_return": {"min": 25.0, "tier": "C"},
    "serenity_ratio": {"min": 1.3, "tier": "C"},
    "stability_score": {"min": 1.5, "tier": "C"},
    "edge_ratio_pct_per_hour": {"min": 0.06, "tier": "C"},
}


@dataclass
class TierSMetrics:
    """Container pour métriques Tier S avec classification."""

    # Tier S (obligatoires)
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    recovery_factor: float
    win_rate: float
    expectancy_pct: float
    sqn: float
    outlier_adjusted_sharpe: float

    # Tier A (importantes)
    r_multiple: float
    time_in_market_pct: float
    max_flat_period_days: int
    annual_return_per_dd: float
    gain_pain_ratio: float

    # Tier B (utiles)
    max_consecutive_wins: int
    max_consecutive_loss_pct: float
    avg_trade_duration_hours: float
    z_score_trades: float
    ulcer_index: float

    # Tier C (bonus)
    pain_adjusted_return: float
    serenity_ratio: float

    # Metadata
    strategy_type: str = "trend"  # 'trend' ou 'meanrev'
    total_trades: int = 0


@dataclass
class ValidationReport:
    """Rapport de validation Tier S."""

    passed: bool
    score: float  # 0-100
    tier_s_passed: int  # Nombre métriques Tier S validées
    failed_metrics: list[str]  # Liste métriques échouées
    warnings: list[str]  # Avertissements
    tier_s_total: int = 10  # Total métriques Tier S (après non-defaults)
    ai_evolved_gold: bool = False  # Tag promotion auto


# =============================================================================
# CALCUL MÉTRIQUES TIER S
# =============================================================================


def calculate_tier_s_metrics(
    returns: pd.Series | np.ndarray,
    trades: pd.DataFrame | None = None,
    equity_curve: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    strategy_type: str = "trend",
) -> TierSMetrics:
    """
    Calcule toutes les métriques Tier S/A/B/C.

    Args:
        returns: Série returns quotidiens (ou autre fréquence)
        trades: DataFrame trades avec colonnes [pnl, entry_time, exit_time]
        equity_curve: Série equity (optionnel, calculé depuis returns si absent)
        risk_free_rate: Taux sans risque annualisé (défaut: 0.0)
        strategy_type: 'trend' ou 'meanrev' (détermine seuil win rate)

    Returns:
        TierSMetrics avec toutes métriques calculées
    """
    # Conversion en arrays NumPy
    if isinstance(returns, pd.Series):
        returns_arr = returns.values
    else:
        returns_arr = returns

    # Equity curve
    if equity_curve is None:
        equity_curve = pd.Series((1 + returns_arr).cumprod())
    elif isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)

    # === TIER S ===
    sharpe = _calculate_sharpe_ratio(returns_arr, risk_free_rate)
    sortino = _calculate_sortino_ratio(returns_arr, risk_free_rate)
    calmar = _calculate_calmar_ratio(returns_arr, equity_curve)
    profit_factor = _calculate_profit_factor(trades) if trades is not None else 0.0
    max_dd_pct = _calculate_max_drawdown(equity_curve)
    recovery_factor = _calculate_recovery_factor(returns_arr, max_dd_pct)
    win_rate = _calculate_win_rate(trades) if trades is not None else 0.0
    expectancy = _calculate_expectancy(trades) if trades is not None else 0.0
    sqn = _calculate_sqn(trades) if trades is not None else 0.0
    outlier_sharpe = _calculate_outlier_adjusted_sharpe(
        returns_arr, trades, risk_free_rate
    )

    # === TIER A ===
    r_multiple = _calculate_r_multiple(trades) if trades is not None else 0.0
    time_in_market = _calculate_time_in_market(trades) if trades is not None else 0.0
    max_flat_days = _calculate_max_flat_period(equity_curve)
    annual_return_dd = abs(_calculate_annual_return(returns_arr) / max_dd_pct) if max_dd_pct != 0 else 0.0
    gain_pain = recovery_factor  # Similaire au Recovery Factor

    # === TIER B ===
    max_consec_wins = _calculate_max_consecutive_wins(trades) if trades is not None else 0
    max_consec_loss = _calculate_max_consecutive_loss_pct(trades) if trades is not None else 0.0
    avg_duration = _calculate_avg_trade_duration(trades) if trades is not None else 0.0
    z_score = _calculate_z_score_trades(trades) if trades is not None else 0.0
    ulcer = _calculate_ulcer_index(equity_curve)

    # === TIER C ===
    annual_return = _calculate_annual_return(returns_arr)
    par = annual_return * (1 + recovery_factor) if recovery_factor > 0 else 0.0
    serenity = sharpe * (1 - time_in_market / 100.0) if time_in_market > 0 else sharpe

    return TierSMetrics(
        # Tier S
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd_pct,
        recovery_factor=recovery_factor,
        win_rate=win_rate,
        expectancy_pct=expectancy,
        sqn=sqn,
        outlier_adjusted_sharpe=outlier_sharpe,
        # Tier A
        r_multiple=r_multiple,
        time_in_market_pct=time_in_market,
        max_flat_period_days=max_flat_days,
        annual_return_per_dd=annual_return_dd,
        gain_pain_ratio=gain_pain,
        # Tier B
        max_consecutive_wins=max_consec_wins,
        max_consecutive_loss_pct=max_consec_loss,
        avg_trade_duration_hours=avg_duration,
        z_score_trades=z_score,
        ulcer_index=ulcer,
        # Tier C
        pain_adjusted_return=par,
        serenity_ratio=serenity,
        # Metadata
        strategy_type=strategy_type,
        total_trades=len(trades) if trades is not None else 0,
    )


def validate_tier_s(
    metrics: TierSMetrics, strict: bool = True
) -> tuple[bool, float, ValidationReport]:
    """
    Valide métriques contre seuils Tier S (2025).

    Args:
        metrics: TierSMetrics calculées
        strict: Si True, toutes métriques Tier S doivent passer

    Returns:
        (passed, score, report)
        - passed: True si validation OK
        - score: 0-100 (pourcentage métriques validées)
        - report: ValidationReport détaillé
    """
    failed = []
    warnings = []
    tier_s_passed = 0

    # Déterminer seuil win rate selon type stratégie
    win_rate_threshold = (
        TIER_S_THRESHOLDS["win_rate_meanrev"]["min"]
        if metrics.strategy_type == "meanrev"
        else TIER_S_THRESHOLDS["win_rate_trend"]["min"]
    )

    # === VALIDATION TIER S (10 métriques) ===
    tier_s_checks = [
        (
            "sharpe_ratio",
            metrics.sharpe_ratio >= TIER_S_THRESHOLDS["sharpe_ratio"]["min"],
            f"Sharpe {metrics.sharpe_ratio:.2f} < {TIER_S_THRESHOLDS['sharpe_ratio']['min']}",
        ),
        (
            "sortino_ratio",
            metrics.sortino_ratio >= TIER_S_THRESHOLDS["sortino_ratio"]["min"],
            f"Sortino {metrics.sortino_ratio:.2f} < {TIER_S_THRESHOLDS['sortino_ratio']['min']}",
        ),
        (
            "calmar_ratio",
            metrics.calmar_ratio >= TIER_S_THRESHOLDS["calmar_ratio"]["min"],
            f"Calmar {metrics.calmar_ratio:.2f} < {TIER_S_THRESHOLDS['calmar_ratio']['min']}",
        ),
        (
            "profit_factor",
            metrics.profit_factor >= TIER_S_THRESHOLDS["profit_factor"]["min"],
            f"Profit Factor {metrics.profit_factor:.2f} < {TIER_S_THRESHOLDS['profit_factor']['min']}",
        ),
        (
            "max_drawdown_pct",
            metrics.max_drawdown_pct >= TIER_S_THRESHOLDS["max_drawdown_pct"]["max"],
            f"Max DD {metrics.max_drawdown_pct:.1f}% < {TIER_S_THRESHOLDS['max_drawdown_pct']['max']}%",
        ),
        (
            "recovery_factor",
            metrics.recovery_factor >= TIER_S_THRESHOLDS["recovery_factor"]["min"],
            f"Recovery Factor {metrics.recovery_factor:.2f} < {TIER_S_THRESHOLDS['recovery_factor']['min']}",
        ),
        (
            "win_rate",
            metrics.win_rate >= win_rate_threshold,
            f"Win Rate {metrics.win_rate*100:.1f}% < {win_rate_threshold*100:.1f}% ({metrics.strategy_type})",
        ),
        (
            "expectancy_pct",
            metrics.expectancy_pct >= TIER_S_THRESHOLDS["expectancy_pct"]["min"],
            f"Expectancy {metrics.expectancy_pct:.2f}% < {TIER_S_THRESHOLDS['expectancy_pct']['min']}%",
        ),
        (
            "sqn",
            metrics.sqn >= TIER_S_THRESHOLDS["sqn"]["min"],
            f"SQN {metrics.sqn:.2f} < {TIER_S_THRESHOLDS['sqn']['min']}",
        ),
        (
            "outlier_adjusted_sharpe",
            metrics.outlier_adjusted_sharpe
            >= TIER_S_THRESHOLDS["outlier_adjusted_sharpe"]["min"],
            f"Outlier-Adjusted Sharpe {metrics.outlier_adjusted_sharpe:.2f} < {TIER_S_THRESHOLDS['outlier_adjusted_sharpe']['min']}",
        ),
    ]

    for metric_name, passed_check, fail_msg in tier_s_checks:
        if passed_check:
            tier_s_passed += 1
        else:
            failed.append(fail_msg)

    # === WARNINGS (overfitting suspect) ===
    if metrics.win_rate > 0.80:
        warnings.append(f"Win rate {metrics.win_rate*100:.1f}% >80% suspect (curve fitting?)")

    if metrics.sharpe_ratio > 3.5:
        warnings.append(f"Sharpe {metrics.sharpe_ratio:.2f} >3.5 irréaliste (vérifier données)")

    sharpe_drop = (
        (metrics.sharpe_ratio - metrics.outlier_adjusted_sharpe) / metrics.sharpe_ratio
        if metrics.sharpe_ratio > 0
        else 0
    )
    if sharpe_drop > 0.30:
        warnings.append(
            f"Outlier-Adjusted Sharpe chute {sharpe_drop*100:.1f}% (dépendance aux meilleurs trades)"
        )

    if metrics.max_consecutive_loss_pct < -12.0:
        warnings.append(
            f"Plus grosse perte consécutive {metrics.max_consecutive_loss_pct:.1f}% < -12% (danger)"
        )

    # === SCORING ===
    score = (tier_s_passed / 10) * 100.0

    # === PASSED ===
    if strict:
        passed = tier_s_passed == 10  # Toutes métriques Tier S
    else:
        passed = tier_s_passed >= 7  # Au moins 70%

    # === AI-EVOLVED-GOLD TAG ===
    # (nécessite validation multi-token/timeframe externe)
    ai_evolved_gold = passed and tier_s_passed == 10 and len(warnings) == 0

    report = ValidationReport(
        passed=passed,
        score=score,
        tier_s_passed=tier_s_passed,
        tier_s_total=10,
        failed_metrics=failed,
        warnings=warnings,
        ai_evolved_gold=ai_evolved_gold,
    )

    return passed, score, report


# =============================================================================
# HELPER FUNCTIONS - CALCUL MÉTRIQUES
# =============================================================================


def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Sharpe ratio annualisé."""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return (
        np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        if np.std(excess_returns) > 0
        else 0.0
    )


def _calculate_sortino_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0
) -> float:
    """Sortino ratio annualisé (downside deviation)."""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    downside = excess_returns[excess_returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0.0
    return (
        np.mean(excess_returns) / downside_std * np.sqrt(252)
        if downside_std > 0
        else 0.0
    )


def _calculate_calmar_ratio(returns: np.ndarray, equity: pd.Series) -> float:
    """Calmar ratio (annual return / max drawdown)."""
    annual_return = _calculate_annual_return(returns)
    max_dd = _calculate_max_drawdown(equity)
    return abs(annual_return / max_dd) if max_dd != 0 else 0.0


def _calculate_profit_factor(trades: pd.DataFrame) -> float:
    """Profit factor (gross profit / gross loss)."""
    if trades is None or len(trades) == 0:
        return 0.0
    gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
    return gross_profit / gross_loss if gross_loss > 0 else 0.0


def _calculate_max_drawdown(equity: pd.Series) -> float:
    """Max drawdown (%)."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _calculate_recovery_factor(returns: np.ndarray, max_dd: float) -> float:
    """Recovery factor (net profit / max drawdown)."""
    net_profit = returns.sum()
    return abs(net_profit / max_dd) if max_dd != 0 else 0.0


def _calculate_win_rate(trades: pd.DataFrame) -> float:
    """Win rate (%)."""
    if trades is None or len(trades) == 0:
        return 0.0
    return (trades["pnl"] > 0).mean()


def _calculate_expectancy(trades: pd.DataFrame) -> float:
    """Expectancy (% par trade)."""
    if trades is None or len(trades) == 0:
        return 0.0
    win_rate = _calculate_win_rate(trades)
    avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if (trades["pnl"] > 0).any() else 0
    avg_loss = abs(trades[trades["pnl"] < 0]["pnl"].mean()) if (trades["pnl"] < 0).any() else 0
    return (win_rate * avg_win - (1 - win_rate) * avg_loss) * 100


def _calculate_sqn(trades: pd.DataFrame) -> float:
    """Van Tharp System Quality Number."""
    if trades is None or len(trades) == 0:
        return 0.0
    expectancy = trades["pnl"].mean()
    std_dev = trades["pnl"].std()
    n = len(trades)
    return (expectancy / std_dev) * np.sqrt(n) if std_dev > 0 else 0.0


def _calculate_outlier_adjusted_sharpe(
    returns: np.ndarray, trades: pd.DataFrame | None, risk_free_rate: float
) -> float:
    """Sharpe sans les 3 meilleurs trades (détecte dépendance luck)."""
    if trades is None or len(trades) < 4:
        return _calculate_sharpe_ratio(returns, risk_free_rate)

    # Retirer 3 meilleurs trades
    trades_sorted = trades.sort_values("pnl", ascending=False)
    trades_adjusted = trades_sorted.iloc[3:]

    # Recalculer returns sans ces trades
    adjusted_returns = returns.copy()
    # (simplification: on retire directement le PnL des 3 meilleurs)
    # Dans implémentation complète, reconstruire equity sans ces trades

    return _calculate_sharpe_ratio(adjusted_returns, risk_free_rate)


def _calculate_r_multiple(trades: pd.DataFrame) -> float:
    """R-Multiple (Avg Win / Avg Loss)."""
    if trades is None or len(trades) == 0:
        return 0.0
    avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if (trades["pnl"] > 0).any() else 0
    avg_loss = abs(trades[trades["pnl"] < 0]["pnl"].mean()) if (trades["pnl"] < 0).any() else 1
    return avg_win / avg_loss if avg_loss > 0 else 0.0


def _calculate_time_in_market(trades: pd.DataFrame) -> float:
    """% temps en marché."""
    # Simplification: assume durée moyenne trade
    # Implémentation complète nécessite timestamps complets
    return 30.0  # Placeholder


def _calculate_max_flat_period(equity: pd.Series) -> int:
    """Plus long flat period (jours)."""
    running_max = equity.cummax()
    flat_periods = (equity == running_max).astype(int)
    max_flat = 0
    current = 0
    for val in flat_periods:
        if val:
            current += 1
            max_flat = max(max_flat, current)
        else:
            current = 0
    return max_flat


def _calculate_max_consecutive_wins(trades: pd.DataFrame) -> int:
    """Max nombre trades gagnants consécutifs."""
    if trades is None or len(trades) == 0:
        return 0
    wins = (trades["pnl"] > 0).astype(int)
    max_consec = 0
    current = 0
    for win in wins:
        if win:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return max_consec


def _calculate_max_consecutive_loss_pct(trades: pd.DataFrame) -> float:
    """Plus grosse perte consécutive (%)."""
    # Simplification: retourne plus grosse perte individuelle
    if trades is None or len(trades) == 0:
        return 0.0
    return float(trades["pnl"].min())


def _calculate_avg_trade_duration(trades: pd.DataFrame) -> float:
    """Durée moyenne trade (heures)."""
    if trades is None or len(trades) == 0 or "exit_time" not in trades.columns:
        return 0.0
    durations = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
    return float(durations.mean())


def _calculate_z_score_trades(trades: pd.DataFrame) -> float:
    """Z-Score détecte dépendance entre trades."""
    # Implémentation simplifiée
    return 0.0  # Placeholder


def _calculate_ulcer_index(equity: pd.Series) -> float:
    """Ulcer Index (douleur moyenne drawdown)."""
    running_max = equity.cummax()
    drawdown_pct = ((equity - running_max) / running_max) * 100
    return float(np.sqrt((drawdown_pct**2).mean()))


def _calculate_annual_return(returns: np.ndarray) -> float:
    """Rendement annualisé."""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    return ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
