"""
ThreadX Data Schemas
====================

Définit les schémas standard pour les données OHLCV et indicateurs techniques.
Utilisé pour validation et normalisation automatique des datasets.

Author: ThreadX Framework
Version: 1.0.0 - Data Normalization Module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any
from enum import Enum


class ColumnType(Enum):
    """Types de colonnes supportées."""

    TIMESTAMP = "timestamp"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    INDICATOR = "indicator"
    UNKNOWN = "unknown"


@dataclass
class OHLCVSchema:
    """
    Schéma standard pour données OHLCV.

    Attributes:
        required_columns: Colonnes obligatoires
        optional_columns: Colonnes optionnelles
        timestamp_columns: Noms possibles pour la colonne de temps
        numeric_columns: Colonnes qui doivent être numériques
        index_name: Nom attendu pour l'index
    """

    required_columns: Set[str] = None
    optional_columns: Set[str] = None
    timestamp_columns: List[str] = None
    numeric_columns: Set[str] = None
    index_name: str = "timestamp"

    def __post_init__(self):
        """Initialise les valeurs par défaut."""
        if self.required_columns is None:
            self.required_columns = {"open", "high", "low", "close"}

        if self.optional_columns is None:
            self.optional_columns = {"volume", "trades", "vwap"}

        if self.timestamp_columns is None:
            self.timestamp_columns = [
                "timestamp",
                "time",
                "datetime",
                "date",
                "ts",
                "date_time",
            ]

        if self.numeric_columns is None:
            self.numeric_columns = {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trades",
                "vwap",
            }

    def validate_columns(self, columns: List[str]) -> Dict[str, Any]:
        """
        Valide les colonnes d'un DataFrame.

        Args:
            columns: Liste des colonnes à valider

        Returns:
            Dict avec:
                - valid: bool
                - missing: Liste des colonnes manquantes
                - extra: Liste des colonnes supplémentaires
                - timestamp_col: Nom de la colonne timestamp détectée
        """
        columns_lower = {col.lower() for col in columns}

        missing = self.required_columns - columns_lower
        extra = columns_lower - (self.required_columns | self.optional_columns)

        # Détecter colonne timestamp
        timestamp_col = None
        for ts_name in self.timestamp_columns:
            if ts_name in columns_lower:
                timestamp_col = ts_name
                break

        return {
            "valid": len(missing) == 0,
            "missing": list(missing),
            "extra": list(extra),
            "timestamp_col": timestamp_col,
            "has_timestamp_col": timestamp_col is not None,
        }


# Instance par défaut
DEFAULT_OHLCV_SCHEMA = OHLCVSchema()


@dataclass
class NormalizationConfig:
    """
    Configuration pour la normalisation des données.

    Attributes:
        fix_column_names: Renommer les colonnes en minuscules
        fix_timestamp: Convertir timestamp en index datetime
        fix_timezone: Ajouter timezone UTC si manquante
        remove_duplicates: Supprimer les lignes dupliquées
        sort_index: Trier par index chronologique
        fill_missing: Remplir les valeurs manquantes
        validate_ohlc: Valider cohérence OHLC (high >= low, etc.)
        drop_invalid: Supprimer lignes invalides
        inplace: Modifier le DataFrame en place (plus rapide)
    """

    fix_column_names: bool = True
    fix_timestamp: bool = True
    fix_timezone: bool = True
    remove_duplicates: bool = True
    sort_index: bool = True
    fill_missing: bool = False
    validate_ohlc: bool = True
    drop_invalid: bool = False
    inplace: bool = False

    def __repr__(self) -> str:
        """Représentation lisible de la config."""
        enabled = [
            k for k, v in self.__dict__.items() if isinstance(v, bool) and v
        ]
        return f"NormalizationConfig(enabled={enabled})"


# Configuration par défaut (non destructive)
DEFAULT_NORMALIZATION_CONFIG = NormalizationConfig(
    fill_missing=False, drop_invalid=False, inplace=False
)

# Configuration stricte (pour production)
STRICT_NORMALIZATION_CONFIG = NormalizationConfig(
    fill_missing=True, drop_invalid=True, validate_ohlc=True, inplace=False
)


@dataclass
class NormalizationReport:
    """
    Rapport de normalisation d'un DataFrame.

    Attributes:
        success: Normalisation réussie
        original_shape: Forme originale (rows, cols)
        final_shape: Forme finale après normalisation
        transformations: Liste des transformations appliquées
        warnings: Liste des avertissements
        errors: Liste des erreurs
        metadata: Métadonnées supplémentaires
    """

    success: bool
    original_shape: tuple
    final_shape: tuple
    transformations: List[str]
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Format lisible du rapport."""
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        lines = [
            f"\n{'='*60}",
            f"Normalisation Report - {status}",
            f"{'='*60}",
            f"Shape: {self.original_shape} → {self.final_shape}",
            f"",
        ]

        if self.transformations:
            lines.append("Transformations appliquées:")
            for t in self.transformations:
                lines.append(f"  ✓ {t}")
            lines.append("")

        if self.warnings:
            lines.append("⚠️  Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        if self.errors:
            lines.append("❌ Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
            lines.append("")

        if self.metadata:
            lines.append("Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")

        lines.append("="*60)
        return "\n".join(lines)


# Export public API
__all__ = [
    "ColumnType",
    "OHLCVSchema",
    "NormalizationConfig",
    "NormalizationReport",
    "DEFAULT_OHLCV_SCHEMA",
    "DEFAULT_NORMALIZATION_CONFIG",
    "STRICT_NORMALIZATION_CONFIG",
]
