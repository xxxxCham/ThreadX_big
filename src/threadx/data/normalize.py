"""
ThreadX Data Normalization Module
==================================

Module de normalisation automatique des données OHLCV pour ThreadX.
Détecte et corrige automatiquement les problèmes de structure des données.

Features:
- Détection automatique de la structure des fichiers
- Normalisation des colonnes (time/timestamp/datetime → index)
- Conversion des types (int → float pour OHLC)
- Gestion des timezones (UTC)
- Suppression des duplicates
- Validation OHLC (high >= low, etc.)
- Rapports détaillés de transformation

Author: ThreadX Framework
Version: 1.0.0 - Data Normalization Module
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import numpy as np

from .schemas import (
    OHLCVSchema,
    NormalizationConfig,
    NormalizationReport,
    DEFAULT_OHLCV_SCHEMA,
    DEFAULT_NORMALIZATION_CONFIG,
)

logger = logging.getLogger(__name__)


def normalize_ohlcv(
    df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
    schema: Optional[OHLCVSchema] = None,
) -> Tuple[pd.DataFrame, NormalizationReport]:
    """
    Normalise un DataFrame OHLCV vers le format standard ThreadX.

    Cette fonction détecte et corrige automatiquement:
    - Colonnes time/timestamp/datetime → index DatetimeIndex
    - Noms de colonnes en minuscules
    - Types numériques pour OHLC
    - Timezone UTC
    - Duplicates et tri chronologique
    - Validation OHLC (optionnel)

    Args:
        df: DataFrame à normaliser
        config: Configuration de normalisation (None = défaut)
        schema: Schéma OHLCV attendu (None = défaut)

    Returns:
        Tuple (DataFrame normalisé, Rapport de normalisation)

    Examples:
        >>> df_norm, report = normalize_ohlcv(df)
        >>> print(report)
        >>> if report.success:
        ...     print(f"Normalisé: {report.original_shape} → {report.final_shape}")
    """
    if config is None:
        config = DEFAULT_NORMALIZATION_CONFIG

    if schema is None:
        schema = DEFAULT_OHLCV_SCHEMA

    # Initialiser le rapport
    original_shape = df.shape
    transformations = []
    warnings = []
    errors = []
    metadata = {}

    # Copier si pas inplace
    if not config.inplace:
        df = df.copy()

    try:
        # 1. Normaliser les noms de colonnes
        if config.fix_column_names:
            df, trans = _fix_column_names(df)
            if trans:
                transformations.extend(trans)

        # 2. Détecter et fixer l'index timestamp
        if config.fix_timestamp:
            df, trans, warns = _fix_timestamp_index(df, schema)
            transformations.extend(trans)
            warnings.extend(warns)

        # 3. Assurer timezone UTC
        if config.fix_timezone:
            df, trans = _fix_timezone(df)
            if trans:
                transformations.append(trans)

        # 4. Convertir les types numériques
        df, trans = _fix_numeric_types(df, schema)
        if trans:
            transformations.extend(trans)

        # 5. Supprimer les duplicates
        if config.remove_duplicates:
            df, trans, meta = _remove_duplicates(df)
            if trans:
                transformations.append(trans)
                metadata.update(meta)

        # 6. Trier par index
        if config.sort_index:
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
                transformations.append("Index trié par ordre chronologique")

        # 7. Valider cohérence OHLC
        if config.validate_ohlc:
            df, trans, warns, meta = _validate_ohlc_logic(
                df, drop_invalid=config.drop_invalid
            )
            transformations.extend(trans)
            warnings.extend(warns)
            metadata.update(meta)

        # 8. Remplir valeurs manquantes (optionnel)
        if config.fill_missing:
            df, trans, meta = _fill_missing_values(df)
            if trans:
                transformations.extend(trans)
                metadata.update(meta)

        # Rapport final
        final_shape = df.shape
        success = True

    except Exception as e:
        logger.error(f"Erreur lors de la normalisation: {e}")
        errors.append(str(e))
        final_shape = df.shape
        success = False

    report = NormalizationReport(
        success=success,
        original_shape=original_shape,
        final_shape=final_shape,
        transformations=transformations,
        warnings=warnings,
        errors=errors,
        metadata=metadata,
    )

    return df, report


def _fix_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normalise les noms de colonnes en minuscules."""
    transformations = []

    # Mapping de noms de colonnes
    original_cols = df.columns.tolist()
    new_cols = {col: col.lower().strip() for col in original_cols}

    if new_cols != {col: col for col in original_cols}:
        df = df.rename(columns=new_cols)
        transformations.append(
            f"Noms de colonnes normalisés: {len(original_cols)} colonnes"
        )

    return df, transformations


def _fix_timestamp_index(
    df: pd.DataFrame, schema: OHLCVSchema
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Détecte et convertit la colonne timestamp en index DatetimeIndex."""
    transformations = []
    warnings = []

    # Vérifier si l'index est déjà un DatetimeIndex valide
    if isinstance(df.index, pd.DatetimeIndex):
        # Vérifier si timezone aware
        if df.index.tz is None:
            transformations.append("Index déjà DatetimeIndex (sans timezone)")
        else:
            transformations.append(
                f"Index déjà DatetimeIndex (timezone: {df.index.tz})"
            )
        return df, transformations, warnings

    # Chercher une colonne timestamp
    timestamp_col = None
    for col_name in schema.timestamp_columns:
        if col_name in df.columns:
            timestamp_col = col_name
            break

    if timestamp_col is None:
        # Pas de colonne timestamp trouvée
        # Vérifier si l'index peut être converti en datetime
        if df.index.dtype != "int64":
            try:
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                transformations.append("Index numérique converti en DatetimeIndex")
                return df, transformations, warnings
            except Exception as e:
                warnings.append(
                    f"Impossible de convertir l'index en datetime: {e}"
                )
                return df, transformations, warnings

        warnings.append(
            "Aucune colonne timestamp détectée et index non convertible"
        )
        return df, transformations, warnings

    # Convertir la colonne en datetime et la définir comme index
    try:
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], utc=True, errors="coerce"
        )
        df = df.set_index(timestamp_col)
        df.index.name = "timestamp"
        transformations.append(
            f"Colonne '{timestamp_col}' convertie en index DatetimeIndex"
        )

        # Compter les NaT (dates invalides)
        nat_count = df.index.isna().sum()
        if nat_count > 0:
            warnings.append(
                f"{nat_count} timestamps invalides (NaT) détectés"
            )

    except Exception as e:
        warnings.append(f"Erreur lors de la conversion du timestamp: {e}")

    return df, transformations, warnings


def _fix_timezone(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Assure que l'index a une timezone UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df, None

    if df.index.tz is None:
        # Localiser en UTC
        df.index = df.index.tz_localize("UTC")
        return df, "Timezone UTC ajoutée à l'index"
    elif str(df.index.tz) != "UTC":
        # Convertir vers UTC
        df.index = df.index.tz_convert("UTC")
        return df, f"Timezone convertie de {df.index.tz} vers UTC"

    return df, None


def _fix_numeric_types(
    df: pd.DataFrame, schema: OHLCVSchema
) -> Tuple[pd.DataFrame, List[str]]:
    """Convertit les colonnes OHLCV en types numériques."""
    transformations = []

    for col in schema.numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    transformations.append(f"Colonne '{col}' convertie en numérique")
                except Exception as e:
                    logger.warning(f"Impossible de convertir '{col}' en numérique: {e}")

    return df, transformations


def _remove_duplicates(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[str], Dict[str, int]]:
    """Supprime les lignes dupliquées basées sur l'index."""
    metadata = {}

    # Compter les duplicates sur l'index
    duplicates = df.index.duplicated()
    n_duplicates = duplicates.sum()

    if n_duplicates > 0:
        df = df[~duplicates]
        metadata["duplicates_removed"] = n_duplicates
        return df, f"{n_duplicates} lignes dupliquées supprimées", metadata

    return df, None, metadata


def _validate_ohlc_logic(
    df: pd.DataFrame, drop_invalid: bool = False
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, int]]:
    """
    Valide la cohérence logique des données OHLC.

    Vérifie:
    - high >= open, close, low
    - low <= open, close, high
    - volume >= 0 (si présent)
    """
    transformations = []
    warnings = []
    metadata = {}

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        warnings.append(
            f"Validation OHLC ignorée: colonnes manquantes {required_cols - set(df.columns)}"
        )
        return df, transformations, warnings, metadata

    # Masque pour lignes invalides
    invalid_mask = pd.Series(False, index=df.index)

    # Vérifier high >= low
    invalid_high_low = df["high"] < df["low"]
    invalid_mask |= invalid_high_low
    if invalid_high_low.sum() > 0:
        warnings.append(
            f"{invalid_high_low.sum()} lignes avec high < low détectées"
        )

    # Vérifier high >= open, close
    invalid_high_open = df["high"] < df["open"]
    invalid_high_close = df["high"] < df["close"]
    invalid_mask |= invalid_high_open | invalid_high_close
    if (invalid_high_open | invalid_high_close).sum() > 0:
        warnings.append(
            f"{(invalid_high_open | invalid_high_close).sum()} lignes avec high < open/close"
        )

    # Vérifier low <= open, close
    invalid_low_open = df["low"] > df["open"]
    invalid_low_close = df["low"] > df["close"]
    invalid_mask |= invalid_low_open | invalid_low_close
    if (invalid_low_open | invalid_low_close).sum() > 0:
        warnings.append(
            f"{(invalid_low_open | invalid_low_close).sum()} lignes avec low > open/close"
        )

    # Vérifier volume >= 0
    if "volume" in df.columns:
        invalid_volume = df["volume"] < 0
        invalid_mask |= invalid_volume
        if invalid_volume.sum() > 0:
            warnings.append(f"{invalid_volume.sum()} lignes avec volume < 0")

    # Total invalides
    n_invalid = invalid_mask.sum()
    metadata["invalid_rows"] = n_invalid

    if n_invalid > 0:
        if drop_invalid:
            df = df[~invalid_mask]
            transformations.append(f"{n_invalid} lignes invalides supprimées")
            metadata["invalid_rows_dropped"] = n_invalid
        else:
            warnings.append(
                f"{n_invalid} lignes invalides trouvées mais non supprimées (drop_invalid=False)"
            )

    return df, transformations, warnings, metadata


def _fill_missing_values(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Remplit les valeurs manquantes (NaN) dans les colonnes OHLCV."""
    transformations = []
    metadata = {}

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    total_filled = 0

    for col in ohlcv_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                # Forward fill puis backward fill
                df[col] = df[col].ffill().bfill()
                transformations.append(
                    f"{n_missing} valeurs manquantes remplies dans '{col}'"
                )
                total_filled += n_missing

    if total_filled > 0:
        metadata["missing_values_filled"] = total_filled

    return df, transformations, metadata


def detect_and_fix_structure(
    file_path: Path, config: Optional[NormalizationConfig] = None
) -> Tuple[pd.DataFrame, NormalizationReport]:
    """
    Détecte et corrige automatiquement la structure d'un fichier de données.

    Lit le fichier, détecte son format, applique les corrections nécessaires
    et retourne un DataFrame normalisé.

    Args:
        file_path: Chemin vers le fichier (parquet, csv, json)
        config: Configuration de normalisation

    Returns:
        Tuple (DataFrame normalisé, Rapport)

    Examples:
        >>> df, report = detect_and_fix_structure(Path("BTC_1h.parquet"))
        >>> print(report)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")

    # Lire le fichier
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Format non supporté: {suffix}")

    # Normaliser
    df_norm, report = normalize_ohlcv(df, config=config)

    # Ajouter metadata du fichier
    report.metadata["source_file"] = str(file_path)
    report.metadata["file_format"] = suffix

    return df_norm, report


def batch_normalize_directory(
    directory: Path,
    output_dir: Optional[Path] = None,
    pattern: str = "*.parquet",
    config: Optional[NormalizationConfig] = None,
    dry_run: bool = False,
) -> Dict[str, NormalizationReport]:
    """
    Normalise tous les fichiers d'un répertoire.

    Args:
        directory: Répertoire contenant les fichiers
        output_dir: Répertoire de sortie (None = écrase les fichiers)
        pattern: Pattern de fichiers à traiter
        config: Configuration de normalisation
        dry_run: Si True, ne sauvegarde pas les fichiers

    Returns:
        Dict {nom_fichier: rapport}

    Examples:
        >>> reports = batch_normalize_directory(
        ...     Path("data/crypto_data_parquet"),
        ...     pattern="*.parquet"
        ... )
        >>> for name, report in reports.items():
        ...     print(f"{name}: {report.success}")
    """
    if not directory.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {directory}")

    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    files = list(directory.glob(pattern))

    logger.info(f"Normalisation de {len(files)} fichiers dans {directory}")

    for file_path in files:
        try:
            logger.info(f"Traitement de {file_path.name}...")

            df_norm, report = detect_and_fix_structure(file_path, config=config)

            if not dry_run and report.success:
                # Déterminer chemin de sortie
                if output_dir:
                    output_path = output_dir / file_path.name
                else:
                    output_path = file_path

                # Sauvegarder
                if file_path.suffix.lower() == ".parquet":
                    df_norm.to_parquet(output_path)
                elif file_path.suffix.lower() == ".csv":
                    df_norm.to_csv(output_path)
                else:
                    logger.warning(f"Format non supporté pour sauvegarde: {file_path.suffix}")

                report.metadata["output_file"] = str(output_path)

            reports[file_path.name] = report

        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path.name}: {e}")
            reports[file_path.name] = NormalizationReport(
                success=False,
                original_shape=(0, 0),
                final_shape=(0, 0),
                transformations=[],
                warnings=[],
                errors=[str(e)],
                metadata={"source_file": str(file_path)},
            )

    return reports


# Export public API
__all__ = [
    "normalize_ohlcv",
    "detect_and_fix_structure",
    "batch_normalize_directory",
]
