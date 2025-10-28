"""
ThreadX - Configuration centralisée des chemins de données
==========================================================

Définit tous les chemins utilisés par ThreadX et unified_data_historique.
Modifiez ce fichier pour changer l'organisation des dossiers.

Architecture:
    crypto_data_json/          ← Téléchargement OHLCV brut (Binance API)
    crypto_data_parquet/       ← Conversion OHLCV optimisée (lecture rapide)
    indicateurs_data_parquet/  ← Indicateurs techniques calculés
        └── {TOKEN}/
            └── {TIMEFRAME}/
                └── {TOKEN}_{TF}_{indicator}.parquet

Author: ThreadX Framework
"""

import os
import platform

IS_WINDOWS = platform.system() == "Windows"

# =========================================================
#  Chemins racine
# =========================================================

if IS_WINDOWS:
    THREADX_ROOT = r"D:\ThreadX_big\src\threadx"
    DATA_ROOT = os.path.join(THREADX_ROOT, "data")
else:
    THREADX_ROOT = "/home/user/threadx"
    DATA_ROOT = os.path.join(THREADX_ROOT, "data")

# =========================================================
#  Chemins données
# =========================================================

# OHLCV brut (JSON téléchargé depuis Binance)
JSON_DATA_DIR = os.path.join(DATA_ROOT, "crypto_data_json")

# OHLCV optimisé (Parquet pour lecture rapide)
PARQUET_DATA_DIR = os.path.join(DATA_ROOT, "crypto_data_parquet")

# Indicateurs techniques
INDICATORS_DB_ROOT = os.path.join(DATA_ROOT, "indicateurs_data_parquet")

# Cache stratégies et résultats
CACHE_DIR = os.path.join(THREADX_ROOT, "cache")
RESULTS_DIR = os.path.join(THREADX_ROOT, "results")

# =========================================================
#  Création automatique des dossiers
# =========================================================

for path in [
    JSON_DATA_DIR,
    PARQUET_DATA_DIR,
    INDICATORS_DB_ROOT,
    CACHE_DIR,
    RESULTS_DIR,
]:
    os.makedirs(path, exist_ok=True)

# =========================================================
#  Utilitaires
# =========================================================


def get_ohlcv_json_path(symbol: str, timeframe: str) -> str:
    """Retourne le chemin vers le fichier JSON OHLCV."""
    return os.path.join(JSON_DATA_DIR, f"{symbol}_{timeframe}.json")


def get_ohlcv_parquet_path(symbol: str, timeframe: str) -> str:
    """Retourne le chemin vers le fichier Parquet OHLCV."""
    return os.path.join(PARQUET_DATA_DIR, f"{symbol}_{timeframe}.parquet")


def get_indicator_path(symbol: str, timeframe: str, indicator_name: str) -> str:
    """Retourne le chemin vers un fichier d'indicateur.

    Args:
        symbol: Ex. 'BTCUSDC'
        timeframe: Ex. '1h'
        indicator_name: Ex. 'bollinger_period20_std2.0'

    Returns:
        Path: D:\\...\\indicateurs_data_parquet\\BTC\\1h\\BTC_1h_bollinger_period20_std2.0.parquet
    """
    token = symbol.replace("USDC", "")
    filename = f"{token}_{timeframe}_{indicator_name}.parquet"
    return os.path.join(INDICATORS_DB_ROOT, token, timeframe, filename)


def list_available_indicators(symbol: str, timeframe: str) -> list[str]:
    """Liste tous les indicateurs disponibles pour un symbol/timeframe."""
    token = symbol.replace("USDC", "")
    ind_dir = os.path.join(INDICATORS_DB_ROOT, token, timeframe)

    if not os.path.exists(ind_dir):
        return []

    return [f for f in os.listdir(ind_dir) if f.endswith(".parquet")]


def validate_indicator_exists(
    symbol: str, timeframe: str, indicator_name: str
) -> tuple[bool, str]:
    """Vérifie qu'un indicateur existe.

    Returns:
        (exists: bool, path: str)
    """
    path = get_indicator_path(symbol, timeframe, indicator_name)
    return os.path.exists(path), path
