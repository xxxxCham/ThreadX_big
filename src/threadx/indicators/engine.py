"""
ThreadX Indicators Engine - Moteur unifié de calcul d'indicateurs
================================================================
"""

import logging
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def enrich_indicators(
    df: pd.DataFrame, specs: List[Dict[str, Any]], **kwargs
) -> pd.DataFrame:
    """Enrichit un DataFrame avec des indicateurs techniques."""
    if df.empty:
        return df.copy()

    result_df = df.copy()
    for spec in specs:
        name = spec.get("name", "").upper()
        params = spec.get("params", {})
        outputs = spec.get("outputs", [name.lower()])

        if name == "SMA":
            window = params.get("window", 20)
            if "close" in df.columns:
                result_df[outputs[0]] = df["close"].rolling(window=window).mean()
        elif name == "EMA":
            span = params.get("span", 20)
            if "close" in df.columns:
                result_df[outputs[0]] = df["close"].ewm(span=span).mean()

    return result_df


def get_available_indicators():
    """Retourne la liste des indicateurs disponibles."""
    return ["SMA", "EMA", "RSI"]
