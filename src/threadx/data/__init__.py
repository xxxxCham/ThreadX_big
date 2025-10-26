"""
ThreadX Data Package
====================

Package pour la gestion, validation et normalisation des données OHLCV.

Modules:
- normalize: Normalisation automatique des données
- schemas: Schémas et configurations de normalisation
- validate: Validation des datasets (crypto_data_parquet/validate.py)
"""

from .normalize import (
    normalize_ohlcv,
    detect_and_fix_structure,
    batch_normalize_directory,
)

from .schemas import (
    OHLCVSchema,
    NormalizationConfig,
    NormalizationReport,
    DEFAULT_OHLCV_SCHEMA,
    DEFAULT_NORMALIZATION_CONFIG,
    STRICT_NORMALIZATION_CONFIG,
)

__all__ = [
    # Fonctions de normalisation
    "normalize_ohlcv",
    "detect_and_fix_structure",
    "batch_normalize_directory",
    # Schémas et configurations
    "OHLCVSchema",
    "NormalizationConfig",
    "NormalizationReport",
    "DEFAULT_OHLCV_SCHEMA",
    "DEFAULT_NORMALIZATION_CONFIG",
    "STRICT_NORMALIZATION_CONFIG",
]
