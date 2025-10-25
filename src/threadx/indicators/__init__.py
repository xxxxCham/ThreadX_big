"""
ThreadX Indicators Layer - Phase 3
==================================

Module d'indicateurs techniques vectorisés avec support GPU multi-carte.

Modules principaux:
- bollinger.py : Bandes de Bollinger vectorisées
- atr.py : Average True Range vectorisé
- bank.py : Cache centralisé d'indicateurs

Caractéristiques:
- Vectorisation NumPy/CuPy pour performances optimales
- Support GPU RTX 5090 (32GB) + RTX 2060 avec répartition 75%/25%
- Cache intelligent avec TTL et checksums
- Batch processing automatique (seuil: 100 paramètres)
- Fallback CPU transparent
"""

from .bollinger import BollingerBands, compute_bollinger_bands, compute_bollinger_batch

from .xatr import ATR, compute_atr, compute_atr_batch

from .bank import (
    IndicatorBank,
    IndicatorSettings,
    ensure_indicator,
    force_recompute_indicator,
    batch_ensure_indicators,
)

__version__ = "3.0.0"
__all__ = [
    # Bollinger Bands
    "BollingerBands",
    "compute_bollinger_bands",
    "compute_bollinger_batch",
    # ATR
    "ATR",
    "compute_atr",
    "compute_atr_batch",
    # Bank
    "IndicatorBank",
    "IndicatorSettings",
    "ensure_indicator",
    "force_recompute_indicator",
    "batch_ensure_indicators",
]
