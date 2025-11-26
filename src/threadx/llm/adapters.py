"""
LLM Adapters - Connecteurs BacktestEngine ↔ Agents LLM.

Convertit structures de données ThreadX en formats analysables par LLM et vice-versa.

Features:
- RunResult → JSON structuré pour Analyst
- Propositions LLM → Params valides pour BacktestEngine
- Validation contraintes StrategyRegistry
- Métriques enrichies pour analyse qualitative

Author: ThreadX Framework
Version: 1.0 - Multi-Agent Integration
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from threadx.backtest.engine import RunResult
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def backtest_result_to_llm_json(result: RunResult) -> dict[str, Any]:
    """
    Convertit RunResult en JSON analysable par agents LLM.
    
    IMPORTANT: Utilise directement result.metrics (calculées par performance.summarize)
    qui contient TOUTES les métriques Tier S/A/B/C + validation automatique.
    
    Le LLM reçoit les métriques déjà calculées - il n'a PAS à les recalculer.

    Args:
        result: Résultat backtest ThreadX (avec .metrics enrichi)

    Returns:
        dict avec:
        - metrics: Métriques performance complètes (Tier S incluses)
        - trades_stats: Statistiques trades détaillées
        - returns_stats: Statistiques returns série temporelle
        - quality_indicators: Indicateurs qualité analyse
        - tier_s_validation: Validation standards 2025 (si disponible)
        - tier_s_thresholds: Seuils référence pour LLM
    """
    # Métriques calculées par performance.summarize() - DÉJÀ DISPONIBLES
    metrics = result.metrics.copy() if result.metrics else {}
    
    # Si metrics vide (ancien code), fallback sur calcul manuel
    if not metrics:
        logger.warning("RunResult.metrics empty, using fallback calculation")
        metrics = _legacy_metrics_fallback(result)
    
    # Statistiques trades (enrichissement depuis trades DataFrame)
    trades_stats = {}
    if result.trades is not None and len(result.trades) > 0:
        trades_df = pd.DataFrame(result.trades)

        trades_stats = {
            "total_trades": len(trades_df),
            "winning_trades": int((trades_df["pnl"] > 0).sum()),
            "losing_trades": int((trades_df["pnl"] < 0).sum()),
            "win_rate": float((trades_df["pnl"] > 0).mean()),
            "avg_win": float(trades_df[trades_df["pnl"] > 0]["pnl"].mean())
            if (trades_df["pnl"] > 0).any()
            else 0.0,
            "avg_loss": float(trades_df[trades_df["pnl"] < 0]["pnl"].mean())
            if (trades_df["pnl"] < 0).any()
            else 0.0,
            "max_consecutive_wins": _max_consecutive(trades_df["pnl"] > 0),
            "max_consecutive_losses": _max_consecutive(trades_df["pnl"] < 0),
            "avg_holding_period": float(
                (trades_df["exit_time"] - trades_df["entry_time"]).mean()
            )
            if "exit_time" in trades_df.columns and "entry_time" in trades_df.columns
            else 0.0,
        }
    else:
        trades_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_holding_period": 0.0,
        }

    # Statistiques returns
    returns_stats = {}
    if result.returns is not None:
        returns_arr = (
            result.returns.values
            if isinstance(result.returns, pd.Series)
            else result.returns
        )

        returns_stats = {
            "mean": float(np.mean(returns_arr)),
            "std": float(np.std(returns_arr)),
            "min": float(np.min(returns_arr)),
            "max": float(np.max(returns_arr)),
            "skewness": float(_calculate_skewness(returns_arr)),
            "kurtosis": float(_calculate_kurtosis(returns_arr)),
            "positive_periods": int((returns_arr > 0).sum()),
            "negative_periods": int((returns_arr < 0).sum()),
            "zero_periods": int((returns_arr == 0).sum()),
        }
    else:
        returns_stats = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "positive_periods": 0,
            "negative_periods": 0,
            "zero_periods": 0,
        }

    # Indicateurs qualité pour LLM (basés sur Tier S si disponible)
    tier_s_validation = metrics.get("tier_s_validation", {})
    
    if tier_s_validation:
        # Utiliser validation Tier S si disponible
        quality_indicators = {
            "sample_size": "SUFFICIENT"
            if trades_stats["total_trades"] >= 100
            else "MARGINAL"
            if trades_stats["total_trades"] >= 50
            else "INSUFFICIENT",
            "sharpe_quality": _classify_metric(
                metrics.get("sharpe_ratio", 0),
                tier="sharpe_ratio"
            ),
            "sortino_quality": _classify_metric(
                metrics.get("sortino_ratio", 0),
                tier="sortino_ratio"
            ),
            "calmar_quality": _classify_metric(
                metrics.get("calmar_ratio", 0),
                tier="calmar_ratio"
            ),
            "drawdown_quality": _classify_metric(
                abs(metrics.get("max_drawdown", 0)),
                tier="max_drawdown_pct",
                invert=True
            ),
            "tier_s_score": tier_s_validation.get("score", 0),
            "tier_s_passed": f"{tier_s_validation.get('tier_s_passed', 0)}/10",
            "ai_evolved_gold": tier_s_validation.get("ai_evolved_gold", False),
        }
    else:
        # Fallback ancien système
        quality_indicators = {
            "sample_size": "SUFFICIENT"
            if trades_stats["total_trades"] >= 100
            else "MARGINAL"
            if trades_stats["total_trades"] >= 50
            else "INSUFFICIENT",
            "sharpe_quality": "EXCELLENT"
            if metrics.get("sharpe_ratio", 0) >= 1.5
            else "GOOD"
            if metrics.get("sharpe_ratio", 0) >= 1.0
            else "AVERAGE"
            if metrics.get("sharpe_ratio", 0) >= 0.5
            else "POOR",
            "drawdown_quality": "EXCELLENT"
            if abs(metrics.get("max_drawdown", 0)) <= 0.10
            else "GOOD"
            if abs(metrics.get("max_drawdown", 0)) <= 0.20
            else "AVERAGE"
            if abs(metrics.get("max_drawdown", 0)) <= 0.30
            else "POOR",
            "consistency": "HIGH"
            if trades_stats.get("max_consecutive_losses", 0) <= 3
            else "MEDIUM"
            if trades_stats.get("max_consecutive_losses", 0) <= 5
            else "LOW",
        }

    return {
        "metrics": metrics,  # TOUTES métriques (Tier S incluses)
        "trades_stats": trades_stats,
        "returns_stats": returns_stats,
        "quality_indicators": quality_indicators,
        "tier_s_validation": tier_s_validation,  # Validation complète
        "tier_s_thresholds": metrics.get("tier_s_thresholds", {}),  # Seuils référence
    }


def proposals_to_registry_params(
    proposals: list[dict[str, Any]], param_constraints: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Valide et filtre propositions LLM contre contraintes StrategyRegistry.

    Args:
        proposals: Liste propositions depuis Strategist
        param_constraints: Contraintes params depuis registry
            Format: {"param_name": {"min": val, "max": val, "type": type}}

    Returns:
        Liste propositions validées (params seulement)
    """
    validated = []

    for proposal in proposals:
        modifications = proposal.get("modifications", {})

        # Valider chaque paramètre
        valid = True
        validated_params = {}

        for param_name, param_value in modifications.items():
            if param_name not in param_constraints:
                logger.warning(
                    f"Proposal {proposal.get('id')}: Unknown param '{param_name}'"
                )
                valid = False
                break

            constraints = param_constraints[param_name]

            # Vérifier type
            expected_type = constraints.get("type", type(param_value))
            if not isinstance(param_value, expected_type):
                logger.warning(
                    f"Proposal {proposal.get('id')}: Param '{param_name}' "
                    f"type mismatch (expected {expected_type}, got {type(param_value)})"
                )
                valid = False
                break

            # Vérifier min/max
            if "min" in constraints and param_value < constraints["min"]:
                logger.warning(
                    f"Proposal {proposal.get('id')}: Param '{param_name}' "
                    f"below min ({param_value} < {constraints['min']})"
                )
                valid = False
                break

            if "max" in constraints and param_value > constraints["max"]:
                logger.warning(
                    f"Proposal {proposal.get('id')}: Param '{param_name}' "
                    f"above max ({param_value} > {constraints['max']})"
                )
                valid = False
                break

            validated_params[param_name] = param_value

        # Ajouter si tous params valides
        if valid:
            validated.append(validated_params)
        else:
            logger.info(f"Proposal {proposal.get('id')} rejected (constraint violation)")

    logger.info(f"Validated {len(validated)}/{len(proposals)} proposals")
    return validated


def enrich_metrics_for_llm(metrics: dict[str, float]) -> dict[str, Any]:
    """
    Enrichit métriques brutes avec contexte qualitatif pour LLM.

    Args:
        metrics: Métriques numériques brutes

    Returns:
        dict avec métriques + labels qualitatifs + benchmarks
    """
    sharpe = metrics.get("sharpe_ratio", 0.0)
    drawdown = abs(metrics.get("max_drawdown", 0.0))
    win_rate = metrics.get("win_rate", 0.0)

    enriched = metrics.copy()

    # Labels qualitatifs
    enriched["sharpe_label"] = (
        "EXCELLENT (>1.5)"
        if sharpe >= 1.5
        else "GOOD (1.0-1.5)"
        if sharpe >= 1.0
        else "AVERAGE (0.5-1.0)"
        if sharpe >= 0.5
        else "POOR (<0.5)"
    )

    enriched["drawdown_label"] = (
        "EXCELLENT (<10%)"
        if drawdown <= 0.10
        else "GOOD (10-20%)"
        if drawdown <= 0.20
        else "AVERAGE (20-30%)"
        if drawdown <= 0.30
        else "POOR (>30%)"
    )

    enriched["win_rate_label"] = (
        "EXCELLENT (>60%)"
        if win_rate >= 0.60
        else "GOOD (50-60%)"
        if win_rate >= 0.50
        else "AVERAGE (40-50%)"
        if win_rate >= 0.40
        else "POOR (<40%)"
    )

    # Benchmarks comparatifs
    enriched["vs_market_benchmark"] = {
        "sharpe_spy": 0.8,  # S&P 500 historique
        "sharpe_strategy": sharpe,
        "outperformance": sharpe - 0.8,
    }

    # Flags warnings
    enriched["warnings"] = []
    if sharpe > 3.0:
        enriched["warnings"].append("Sharpe >3.0 unusually high (overfitting risk)")
    if win_rate > 0.80:
        enriched["warnings"].append("Win rate >80% suspect (curve fitting?)")
    if drawdown < 0.05 and sharpe > 2.0:
        enriched["warnings"].append("Too perfect metrics (check data integrity)")

    return enriched


def format_memory_for_llm(
    memory_iterations: list[dict[str, Any]], n: int = 5
) -> str:
    """
    Formate historique mémoire pour prompt LLM.

    Args:
        memory_iterations: Iterations depuis OptimizationMemory
        n: Nombre itérations récentes à inclure

    Returns:
        str formatted pour inclusion dans prompt
    """
    recent = memory_iterations[-n:] if len(memory_iterations) > n else memory_iterations

    formatted_lines = ["**Historique récent (dernières itérations):**\n"]

    for it in recent:
        formatted_lines.append(f"- Iteration {it['iteration']}:")
        formatted_lines.append(f"  - Params: {it['params']}")
        formatted_lines.append(f"  - Score: {it['score']:.3f}")
        formatted_lines.append("")

    return "\n".join(formatted_lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _max_consecutive(boolean_series: pd.Series) -> int:
    """Calcule longueur max de séquence True consécutive."""
    if len(boolean_series) == 0:
        return 0

    max_count = 0
    current_count = 0

    for val in boolean_series:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


def _calculate_skewness(arr: np.ndarray) -> float:
    """Calcule coefficient asymétrie (skewness)."""
    if len(arr) < 3:
        return 0.0

    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return 0.0

    return np.mean(((arr - mean) / std) ** 3)


def _calculate_kurtosis(arr: np.ndarray) -> float:
    """Calcule coefficient aplatissement (kurtosis)."""
    if len(arr) < 4:
        return 0.0

    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return 0.0

    return np.mean(((arr - mean) / std) ** 4) - 3.0  # Excess kurtosis


def _classify_metric(value: float, tier: str, invert: bool = False) -> str:
    """
    Classifie métrique selon seuils Tier S.
    
    Args:
        value: Valeur métrique
        tier: Nom métrique (ex: "sharpe_ratio")
        invert: Si True, inverse logique (ex: drawdown où moins = mieux)
    
    Returns:
        "EXCELLENT", "GOOD", "AVERAGE", ou "POOR"
    """
    try:
        from threadx.backtest.metrics_tier_s import TIER_S_THRESHOLDS, TIER_A_THRESHOLDS
        
        thresholds = {**TIER_S_THRESHOLDS, **TIER_A_THRESHOLDS}.get(tier, {})
        
        if not thresholds:
            return "UNKNOWN"
        
        ideal = thresholds.get("ideal", thresholds.get("min", 0))
        minimum = thresholds.get("min", thresholds.get("max", 0))
        
        if invert:
            # Pour drawdown: plus petit = mieux
            if abs(value) <= abs(ideal):
                return "EXCELLENT"
            elif abs(value) <= abs(minimum):
                return "GOOD"
            elif abs(value) <= abs(minimum) * 1.5:
                return "AVERAGE"
            else:
                return "POOR"
        else:
            # Pour sharpe/sortino: plus grand = mieux
            if value >= ideal:
                return "EXCELLENT"
            elif value >= minimum:
                return "GOOD"
            elif value >= minimum * 0.7:
                return "AVERAGE"
            else:
                return "POOR"
                
    except ImportError:
        return "UNKNOWN"


def _legacy_metrics_fallback(result: RunResult) -> dict[str, Any]:
    """Fallback si RunResult.metrics vide (ancien code)."""
    logger.warning("Using legacy metrics calculation - update BacktestEngine to use performance.summarize()")
    
    # Calcul basique
    from threadx.backtest.performance import (
        sharpe_ratio,
        max_drawdown,
        profit_factor,
        win_rate,
    )
    
    return {
        "sharpe_ratio": sharpe_ratio(result.returns) if result.returns is not None else 0.0,
        "max_drawdown": max_drawdown(result.equity) if result.equity is not None else 0.0,
        "profit_factor": profit_factor(result.trades) if result.trades is not None else 0.0,
        "win_rate": win_rate(result.trades) if result.trades is not None else 0.0,
        "total_trades": len(result.trades) if result.trades is not None else 0,
    }

