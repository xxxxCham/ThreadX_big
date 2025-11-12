"""
Module d'imports communs pour ThreadX.

Ce module centralise les imports fréquemment utilisés à travers le projet
pour réduire la duplication (DRY principle).

⚠️ IMPORTANT: Ce module ne doit PAS importer de modules ThreadX qui pourraient
créer des dépendances circulaires (bridge, optimization, backtest, etc.).
Seuls les imports standards et typing sont autorisés ici.

Usage:
    from threadx.utils.common_imports import pd, np, get_logger
    # ou:
    from threadx.utils.common_imports import *

Author: ThreadX Framework
Phase: Phase 2 Step 3.1 - DRY Refactoring
"""

# === Data Science Libraries ===
# === Typing ===
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

# === Logging (safe - no circular dependency) ===
from threadx.utils.log import get_logger

# === Exports pour from common_imports import * ===
__all__ = [
    # Data science
    "pd",
    "np",
    # Typing
    "Any",
    "Optional",
    "Union",
    "Callable",
    "TypeVar",
    "Generic",
    # Logging
    "get_logger",
]


# === Default logger pour usage simple ===
logger = get_logger("threadx")
