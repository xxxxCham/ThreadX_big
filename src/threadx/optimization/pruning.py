"""
ThreadX Pareto Pruning - Early Stop Optimization
===============================================

Algorithme de pruning Pareto soft pour optimisation paramétrique :
- Maintient une frontière Pareto approximative
- Early stopping basé sur patience et quantiles
- Métriques multiples avec seuils adaptatifs

Author: ThreadX Framework
Version: Phase 10 - Pareto Pruning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time

from threadx.utils.log import get_logger

logger = get_logger(__name__)


def pareto_soft_prune(
    df: pd.DataFrame,
    metrics: Tuple[str, ...] = ("pnl", "max_drawdown", "sharpe"),
    patience: int = 200,
    quantile: float = 0.85,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pruning Pareto soft avec early stopping adaptatif.

    Maintient une frontière Pareto approximative et stoppe l'optimisation
    si aucune configuration n'entre dans le top quantile sur ≥ 2 métriques
    pendant les 'patience' derniers essais.

    Args:
        df: DataFrame des résultats avec colonnes de métriques
        metrics: Tuple des noms de métriques à considérer
        patience: Nombre d'essais sans amélioration avant arrêt
        quantile: Seuil de quantile pour considération (0.85 = top 15%)

    Returns:
        Tuple (df_kept, metadata) où:
        - df_kept: DataFrame des configurations conservées
        - metadata: Dict avec statistiques de pruning

    Example:
        >>> results_df = pd.DataFrame({
        ...     'pnl': [100, 150, 80, 200, 120],
        ...     'max_drawdown': [0.1, 0.15, 0.08, 0.12, 0.09],
        ...     'sharpe': [1.2, 1.8, 0.9, 2.1, 1.4]
        ... })
        >>> kept_df, meta = pareto_soft_prune(results_df, patience=3)
    """
    start_time = time.time()

    if df.empty:
        return df, _empty_prune_metadata()

    logger.info(
        f"Pruning Pareto soft: {len(df)} configurations, "
        f"métriques={metrics}, patience={patience}, quantile={quantile}"
    )

    # Validation des métriques
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        logger.warning(f"Métriques manquantes: {missing_metrics}")
        available_metrics = tuple(m for m in metrics if m in df.columns)
        if not available_metrics:
            return df, _empty_prune_metadata()
        metrics = available_metrics

    # Initialisation
    n_original = len(df)
    df_work = df.copy().reset_index(drop=True)

    # Calcul des seuils de quantiles pour chaque métrique
    quantile_thresholds = {}
    for metric in metrics:
        # Détermination du sens d'optimisation
        is_maximize = _is_maximize_metric(metric)
        threshold = df_work[metric].quantile(quantile if is_maximize else 1 - quantile)
        quantile_thresholds[metric] = threshold

        logger.debug(
            f"Métrique {metric}: seuil {quantile:.2%} = {threshold:.4f} "
            f"({'max' if is_maximize else 'min'})"
        )

    # Algorithme de pruning itératif
    pareto_front: List[Tuple[int, pd.Series]] = []  # noqa: SPELL
    pruned_count = 0
    stagnation_counter = 0
    last_improvement_iter = 0

    for i, row in df_work.iterrows():
        # Évaluation Pareto
        is_pareto_candidate = _evaluate_pareto_candidate(
            row, pareto_front, metrics, quantile_thresholds
        )

        if is_pareto_candidate:
            # Ajout à la frontière
            pareto_front.append((int(i), row))
            stagnation_counter = 0
            last_improvement_iter = int(i)

            # Nettoyage de la frontière (suppression des dominés)
            pareto_front = _clean_pareto_front(pareto_front, metrics)

        else:
            # Configuration non-Pareto
            pruned_count += 1
            stagnation_counter += 1

            # Vérification early stopping
            if stagnation_counter >= patience:
                logger.info(
                    f"Early stopping activé à l'itération {i} "
                    f"(stagnation: {stagnation_counter}/{patience})"
                )
                break

    # Construction du DataFrame final
    if pareto_front:
        kept_indices = [idx for idx, _ in pareto_front]
        df_kept = df_work.iloc[kept_indices].copy()
    else:
        # Fallback : garder les meilleures selon métrique principale
        primary_metric = metrics[0]
        is_maximize = _is_maximize_metric(primary_metric)
        df_kept = (
            df_work.nlargest(1, primary_metric)
            if is_maximize
            else df_work.nsmallest(1, primary_metric)
        )

    # Métadonnées
    execution_time = time.time() - start_time
    time_saved_estimate = (n_original - len(df_kept)) * 0.1  # Estimation basique

    metadata = {
        "original_count": n_original,
        "kept_count": len(df_kept),
        "pruned_count": pruned_count,
        "pruning_rate": pruned_count / n_original if n_original > 0 else 0.0,
        "early_stop_triggered": stagnation_counter >= patience,
        "stagnation_counter": stagnation_counter,
        "last_improvement_iter": last_improvement_iter,
        "quantile_thresholds": quantile_thresholds,
        "pareto_front_size": len(pareto_front),
        "execution_time": execution_time,
        "estimated_time_saved": time_saved_estimate,
        "metrics_used": metrics,
        "patience": patience,
        "quantile": quantile,
    }

    logger.info(
        f"Pruning terminé: {len(df_kept)}/{n_original} configurations conservées "
        f"({metadata['pruning_rate']:.1%} pruned), "
        f"temps épargné estimé: {time_saved_estimate:.1f}s"
    )

    return df_kept, metadata


def _is_maximize_metric(metric: str) -> bool:
    """Détermine si une métrique doit être maximisée."""
    maximize_metrics = {
        "pnl",
        "profit",
        "return",
        "returns",
        "sharpe",
        "sharpe_ratio",
        "sortino",
        "sortino_ratio",
        "calmar",
        "win_rate",
        "success_rate",
    }

    minimize_metrics = {
        "drawdown",
        "max_drawdown",
        "max_dd",
        "volatility",
        "var",
        "cvar",
        "loss",
        "risk",
        "downside_deviation",
    }

    metric_lower = metric.lower()

    if metric_lower in maximize_metrics:
        return True
    elif metric_lower in minimize_metrics:
        return False
    else:
        # Heuristique : si contient "drawdown", "loss", "risk" -> minimize
        minimize_keywords = ["drawdown", "loss", "risk", "error", "deviation"]
        for keyword in minimize_keywords:
            if keyword in metric_lower:
                return False

        # Par défaut : maximiser
        return True


def _evaluate_pareto_candidate(
    candidate: pd.Series,
    pareto_front: List[Tuple[int, pd.Series]],
    metrics: Tuple[str, ...],
    quantile_thresholds: Dict[str, float],
) -> bool:
    """
    Évalue si un candidat doit être ajouté à la frontière Pareto.

    Critères :
    1. Non dominé par la frontière existante
    2. Atteint le seuil de quantile sur au moins 2 métriques
    """
    # Critère 1 : Vérification des seuils de quantile
    metrics_above_threshold = 0

    for metric in metrics:
        threshold = quantile_thresholds[metric]
        is_maximize = _is_maximize_metric(metric)

        if is_maximize and candidate[metric] >= threshold:
            metrics_above_threshold += 1
        elif not is_maximize and candidate[metric] <= threshold:
            metrics_above_threshold += 1

    if metrics_above_threshold < 2:
        return False

    # Critère 2 : Non-domination par la frontière existante
    for _, front_member in pareto_front:
        if _dominates(front_member, candidate, metrics):
            return False

    return True


def _dominates(
    solution_a: pd.Series, solution_b: pd.Series, metrics: Tuple[str, ...]
) -> bool:
    """
    Vérifie si solution_a domine solution_b selon les métriques Pareto.

    Domination : A domine B si A est au moins aussi bon que B sur toutes
    les métriques ET strictement meilleur sur au moins une métrique.
    """
    at_least_as_good = True
    strictly_better = False

    for metric in metrics:
        is_maximize = _is_maximize_metric(metric)

        value_a = solution_a[metric]
        value_b = solution_b[metric]

        if is_maximize:
            if value_a < value_b:
                at_least_as_good = False
                break
            elif value_a > value_b:
                strictly_better = True
        else:
            if value_a > value_b:
                at_least_as_good = False
                break
            elif value_a < value_b:
                strictly_better = True

    return at_least_as_good and strictly_better


def _clean_pareto_front(
    pareto_front: List[Tuple[int, pd.Series]], metrics: Tuple[str, ...]
) -> List[Tuple[int, pd.Series]]:
    """
    Nettoie la frontière Pareto en supprimant les solutions dominées.
    """
    if len(pareto_front) <= 1:
        return pareto_front

    cleaned_front = []

    for i, (idx_a, solution_a) in enumerate(pareto_front):
        is_dominated = False

        for j, (idx_b, solution_b) in enumerate(pareto_front):
            if i != j and _dominates(solution_b, solution_a, metrics):
                is_dominated = True
                break

        if not is_dominated:
            cleaned_front.append((idx_a, solution_a))

    return cleaned_front


def _empty_prune_metadata() -> Dict[str, Any]:
    """Retourne des métadonnées vides pour cas d'erreur."""
    return {
        "original_count": 0,
        "kept_count": 0,
        "pruned_count": 0,
        "pruning_rate": 0.0,
        "early_stop_triggered": False,
        "stagnation_counter": 0,
        "last_improvement_iter": 0,
        "quantile_thresholds": {},
        "pareto_front_size": 0,
        "execution_time": 0.0,
        "estimated_time_saved": 0.0,
        "metrics_used": (),
        "patience": 0,
        "quantile": 0.0,
    }


# === Utilitaires additionnels ===


def analyze_pareto_front(df: pd.DataFrame, metrics: Tuple[str, ...]) -> Dict[str, Any]:
    """
    Analyse la frontière Pareto d'un ensemble de résultats.

    Args:
        df: DataFrame des résultats
        metrics: Métriques à analyser

    Returns:
        Dict avec statistiques de la frontière Pareto
    """
    if df.empty:
        return {"pareto_solutions": [], "front_size": 0, "coverage": 0.0}

    # Identification des solutions Pareto
    pareto_solutions = []

    for i, row in df.iterrows():
        is_pareto = True

        for j, other_row in df.iterrows():
            if i != j and _dominates(other_row, row, metrics):
                is_pareto = False
                break

        if is_pareto:
            pareto_solutions.append(i)

    coverage = len(pareto_solutions) / len(df) if len(df) > 0 else 0.0

    return {
        "pareto_solutions": pareto_solutions,
        "front_size": len(pareto_solutions),
        "coverage": coverage,
        "total_solutions": len(df),
    }


if __name__ == "__main__":
    # Test rapide
    test_df = pd.DataFrame(
        {
            "pnl": [100, 150, 80, 200, 120, 90, 160],
            "max_drawdown": [0.1, 0.15, 0.08, 0.12, 0.09, 0.20, 0.11],
            "sharpe": [1.2, 1.8, 0.9, 2.1, 1.4, 0.8, 1.6],
        }
    )

    kept_df, metadata = pareto_soft_prune(test_df, patience=3, quantile=0.7)
    print(f"Test pruning: {len(kept_df)}/{len(test_df)} configurations conservées")
    print(
        f"Métadonnées: {metadata['pruning_rate']:.1%} pruned, "
        f"early_stop={metadata['early_stop_triggered']}"
    )
