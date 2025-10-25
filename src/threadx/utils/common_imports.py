"""
Module d'imports communs pour ThreadX.

Ce module centralise les imports fréquemment utilisés à travers le projet
pour réduire la duplication (DRY principle).

⚠️ IMPORTANT: Ce module ne doit PAS importer de modules ThreadX qui pourraient
créer des dépendances circulaires (bridge, optimization, backtest, etc.).
Seuls les imports standards et typing sont autorisés ici.

Usage:
    from threadx.utils.common_imports import pd, np, create_logger
    # ou:
    from threadx.utils.common_imports import *

Author: ThreadX Framework
Phase: Phase 2 Step 3.1 - DRY Refactoring
"""

# === Data Science Libraries ===
import pandas as pd
import numpy as np

# === Typing ===
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    TypeVar,
    Generic,
)

# === Logging (safe - no circular dependency) ===
from threadx.utils.log import get_logger

# === Exports pour from common_imports import * ===
__all__ = [
    # Data science
    "pd",
    "np",
    # Typing
    "Any",
    "Dict",
    "List",
    "Optional",
    "Tuple",
    "Union",
    "Callable",
    "TypeVar",
    "Generic",
    # Logging
    "get_logger",
    "create_logger",
]


def create_logger(name: str):
    """
    Helper pour créer un logger avec le bon nom.

    Args:
        name: Nom du module (ex: "threadx.data.loader")

    Returns:
        Logger configuré

    Examples:
        >>> logger = create_logger(__name__)
        >>> logger.info("Message")
    """
    return get_logger(name)


# === Default logger pour usage simple ===
logger = get_logger("threadx")
