"""
ThreadX Optimization Reporting - Distribution Analysis & Heatmaps
===============================================================

Génération de rapports quantitatifs pour sweeps paramétriques :
- Analyse des distributions (mean/std/percentiles)
- Heatmaps 2D pour visualisation des paramètres
- Export CSV/Parquet avec manifests JSON
- Métriques de performance consolidées

Author: ThreadX Framework
Version: Phase 10 - Optimization Reporting
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime

from threadx.utils.log import get_logger

logger = get_logger(__name__)


def summarize_distribution(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse des distributions de métriques de performance.

    Args:
        results_df: DataFrame des résultats avec métriques

    Returns:
        Dict avec statistiques descriptives par métrique

    Example:
        >>> results = pd.DataFrame({
        ...     'pnl': [100, 150, 80, 200, 120],
        ...     'sharpe': [1.2, 1.8, 0.9, 2.1, 1.4],
        ...     'max_drawdown': [0.1, 0.15, 0.08, 0.12, 0.09]
        ... })
        >>> stats = summarize_distribution(results)
        >>> stats['pnl']['mean']  # 130.0
    """
    if results_df.empty:
        return {}

    logger.info(f"Analyse des distributions: {len(results_df)} résultats")

    summary = {}

    # Identification des colonnes numériques (métriques)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = results_df[col].dropna()

        if len(series) == 0:
            continue

        # Statistiques de base
        stats = {
            "count": len(series),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            # Percentiles
            "p5": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            # Métriques de forme
            "skewness": float(series.skew()) if len(series) > 2 else 0.0,
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0.0,
            # Dispersion
            "var": float(series.var()),
            "cv": (
                float(series.std() / series.mean())
                if series.mean() != 0
                else float("inf")
            ),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            # Ratios de performance (si applicable)
            "success_rate": (
                float((series > 0).mean())
                if "pnl" in col.lower() or "return" in col.lower()
                else None
            ),
            "negative_rate": (
                float((series < 0).mean())
                if "pnl" in col.lower() or "return" in col.lower()
                else None
            ),
        }

        # Calculs spécialisés selon le type de métrique
        if "sharpe" in col.lower():
            stats["above_1"] = float((series > 1.0).mean())
            stats["above_2"] = float((series > 2.0).mean())
        elif "sortino" in col.lower():
            stats["above_1"] = float((series > 1.0).mean())
            stats["above_1_5"] = float((series > 1.5).mean())
        elif "drawdown" in col.lower():
            stats["below_5pct"] = float((series < 0.05).mean())
            stats["below_10pct"] = float((series < 0.10).mean())
            stats["above_20pct"] = float((series > 0.20).mean())

        summary[col] = stats

    # Statistiques globales
    if summary:
        summary["_meta"] = {
            "total_configurations": len(results_df),
            "total_metrics": len(summary) - 1,  # -1 pour _meta
            "analysis_timestamp": datetime.now().isoformat(),
            "missing_data_pct": float(
                results_df.isnull().sum().sum()
                / (len(results_df) * len(results_df.columns))
            ),
        }

    logger.info(f"Distributions analysées: {len(summary) - 1} métriques")

    return summary


def build_heatmaps(results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Construction de heatmaps 2D pour analyse des paramètres.

    Args:
        results_df: DataFrame avec paramètres et métriques

    Returns:
        Dict[metric_name, DataFrame] des heatmaps par métrique

    Example:
        >>> results = pd.DataFrame({
        ...     'bb_period': [10, 20, 10, 20, 30],
        ...     'bb_std': [1.5, 1.5, 2.0, 2.0, 1.5],
        ...     'pnl': [100, 150, 80, 200, 120]
        ... })
        >>> heatmaps = build_heatmaps(results)
        >>> 'pnl' in heatmaps
    """
    if results_df.empty:
        return {}

    logger.info(f"Construction des heatmaps: {len(results_df)} résultats")

    # Identification des colonnes de paramètres et métriques
    param_cols = _identify_parameter_columns(results_df)
    metric_cols = _identify_metric_columns(results_df)

    if len(param_cols) < 2:
        logger.warning(
            f"Nombre insuffisant de paramètres pour heatmaps: {len(param_cols)}"
        )
        return {}

    heatmaps = {}

    # Génération des heatmaps pour chaque métrique
    for metric in metric_cols:
        metric_heatmaps = _build_metric_heatmaps(results_df, param_cols, metric)
        heatmaps.update(metric_heatmaps)

    logger.info(f"Heatmaps construites: {len(heatmaps)} combinaisons param/métrique")

    return heatmaps


def _identify_parameter_columns(df: pd.DataFrame) -> List[str]:
    """Identifie les colonnes de paramètres."""
    param_indicators = [
        "period",
        "std",
        "window",
        "length",
        "threshold",
        "alpha",
        "beta",
        "gamma",
        "lambda",
        "factor",
        "multiplier",
        "ratio",
        "size",
        "step",
    ]

    param_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # Colonnes numériques avec indicateurs de paramètres
        if df[col].dtype in [np.int64, np.float64, int, float]:
            if any(indicator in col_lower for indicator in param_indicators):
                param_cols.append(col)
            # Ou colonnes avec préfixes typiques
            elif any(
                col_lower.startswith(prefix)
                for prefix in ["bb_", "atr_", "rsi_", "ma_", "ema_"]
            ):
                param_cols.append(col)

    return param_cols


def _identify_metric_columns(df: pd.DataFrame) -> List[str]:
    """Identifie les colonnes de métriques."""
    metric_indicators = [
        "pnl",
        "profit",
        "return",
        "sharpe",
        "sortino",
        "calmar",
        "drawdown",
        "volatility",
        "var",
        "win_rate",
        "success_rate",
    ]

    metric_cols = []

    for col in df.columns:
        col_lower = col.lower()

        if df[col].dtype in [np.int64, np.float64, int, float]:
            if any(indicator in col_lower for indicator in metric_indicators):
                metric_cols.append(col)

    return metric_cols


def _build_metric_heatmaps(
    df: pd.DataFrame, param_cols: List[str], metric: str
) -> Dict[str, pd.DataFrame]:
    """Construit les heatmaps pour une métrique donnée."""
    heatmaps = {}

    # Combinaisons de paramètres 2 à 2
    for i in range(len(param_cols)):
        for j in range(i + 1, len(param_cols)):
            param_x = param_cols[i]
            param_y = param_cols[j]

            heatmap_key = f"{metric}_{param_x}_vs_{param_y}"

            try:
                # Pivot table pour créer la heatmap
                heatmap_df = df.pivot_table(
                    values=metric,
                    index=param_y,
                    columns=param_x,
                    aggfunc="mean",  # Moyenne si plusieurs valeurs par case
                    fill_value=np.nan,
                )

                # Tri des axes pour cohérence
                heatmap_df = heatmap_df.sort_index().sort_index(axis=1)

                heatmaps[heatmap_key] = heatmap_df

            except Exception as e:
                logger.warning(f"Erreur construction heatmap {heatmap_key}: {e}")
                continue

    return heatmaps


def write_reports(
    results_df: pd.DataFrame,
    out_dir: str,
    *,
    seeds: Optional[List[int]] = None,
    devices: Optional[List[str]] = None,
    gpu_ratios: Optional[Dict[str, float]] = None,
    min_samples: Optional[int] = None,
) -> Dict[str, str]:
    """
    Écrit les rapports d'optimisation sur disque.

    Args:
        results_df: DataFrame des résultats
        out_dir: Répertoire de sortie
        seeds: Liste des seeds utilisés
        devices: Liste des devices utilisés
        gpu_ratios: Ratios GPU utilisés
        min_samples: Seuil minimum pour GPU

    Returns:
        Dict[file_type, file_path] des fichiers créés

    Example:
        >>> results = pd.DataFrame({'pnl': [100, 150], 'sharpe': [1.2, 1.8]})
        >>> files = write_reports(results, "artifacts/reports")
        >>> 'csv' in files and 'parquet' in files and 'manifest' in files
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Écriture des rapports: {len(results_df)} résultats → {out_path}")

    created_files = {}

    # 1. Export CSV
    csv_path = out_path / f"optimization_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    created_files["csv"] = str(csv_path.resolve())

    # 2. Export Parquet
    parquet_path = out_path / f"optimization_results_{timestamp}.parquet"
    results_df.to_parquet(parquet_path, index=False)
    created_files["parquet"] = str(parquet_path.resolve())

    # 3. Analyse des distributions
    distribution_stats = summarize_distribution(results_df)
    stats_path = out_path / f"distribution_stats_{timestamp}.json"
    with stats_path.open("w") as f:
        json.dump(distribution_stats, f, indent=2, ensure_ascii=False)
    created_files["distribution_stats"] = str(stats_path.resolve())

    # 4. Heatmaps
    heatmaps = build_heatmaps(results_df)
    if heatmaps:
        heatmaps_dir = out_path / f"heatmaps_{timestamp}"
        heatmaps_dir.mkdir(exist_ok=True)

        for heatmap_name, heatmap_df in heatmaps.items():
            heatmap_path = heatmaps_dir / f"{heatmap_name}.csv"
            heatmap_df.to_csv(heatmap_path)

        created_files["heatmaps_dir"] = str(heatmaps_dir.resolve())
        created_files["heatmaps_count"] = len(heatmaps)

    # 5. Manifest JSON
    manifest = {
        "generation_info": {
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results_df),
            "output_directory": str(out_path.resolve()),
        },
        "execution_context": {
            "seeds": seeds or [],
            "devices": devices or [],
            "gpu_ratios": gpu_ratios or {},
            "min_samples_threshold": min_samples,
        },
        "files_created": created_files,
        "data_summary": {
            "columns": list(results_df.columns),
            "numeric_columns": list(
                results_df.select_dtypes(include=[np.number]).columns
            ),
            "row_count": len(results_df),
            "missing_values": results_df.isnull().sum().to_dict(),
        },
        "performance_summary": _quick_performance_summary(results_df),
    }

    manifest_path = out_path / f"manifest_{timestamp}.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    created_files["manifest"] = str(manifest_path.resolve())

    logger.info(f"Rapports créés: {len(created_files)} fichiers")

    return created_files


def _quick_performance_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Résumé rapide des performances pour le manifest."""
    if results_df.empty:
        return {}

    summary = {}

    # Métriques clés si disponibles
    key_metrics = ["pnl", "sharpe", "max_drawdown", "win_rate", "profit", "return"]

    for metric in key_metrics:
        matching_cols = [
            col for col in results_df.columns if metric.lower() in col.lower()
        ]

        if matching_cols:
            col = matching_cols[0]  # Prendre la première correspondance
            series = results_df[col].dropna()

            if len(series) > 0:
                summary[metric] = {
                    "best": float(
                        series.max() if metric not in ["max_drawdown"] else series.min()
                    ),
                    "worst": float(
                        series.min() if metric not in ["max_drawdown"] else series.max()
                    ),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                }

    return summary


# === Utilitaires de validation ===


def validate_results_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valide la structure du DataFrame de résultats.

    Returns:
        Dict avec statut de validation et détails
    """
    validation = {"is_valid": True, "issues": [], "warnings": [], "info": {}}

    # Vérifications de base
    if df.empty:
        validation["is_valid"] = False
        validation["issues"].append("DataFrame vide")
        return validation

    # Colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        validation["is_valid"] = False
        validation["issues"].append("Aucune colonne numérique trouvée")

    # Valeurs manquantes
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_pct > 0.5:
        validation["warnings"].append(
            f"Taux élevé de valeurs manquantes: {missing_pct:.1%}"
        )

    # Doublons
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation["warnings"].append(f"{duplicates} lignes dupliquées détectées")

    # Informations
    validation["info"] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": len(numeric_cols),
        "missing_percentage": missing_pct,
        "duplicate_rows": duplicates,
    }

    return validation


if __name__ == "__main__":
    # Test rapide
    test_df = pd.DataFrame(
        {
            "bb_period": [10, 20, 30, 10, 20],
            "bb_std": [1.5, 1.5, 1.5, 2.0, 2.0],
            "pnl": [100, 150, 120, 80, 200],
            "sharpe": [1.2, 1.8, 1.4, 0.9, 2.1],
            "max_drawdown": [0.1, 0.15, 0.12, 0.08, 0.13],
        }
    )

    # Test des fonctions
    stats = summarize_distribution(test_df)
    heatmaps = build_heatmaps(test_df)

    print(
        f"Test reporting: {len(stats)} métriques analysées, "
        f"{len(heatmaps)} heatmaps générées"
    )
