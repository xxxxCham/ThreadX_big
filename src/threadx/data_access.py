import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

# Import du module de normalisation
try:
    from threadx.data.normalize import normalize_ohlcv
    from threadx.data.schemas import DEFAULT_NORMALIZATION_CONFIG

    NORMALIZATION_AVAILABLE = True
except ImportError:
    NORMALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


# Localisation robuste du dossier data
def _default_data_dir() -> Path:
    env = os.environ.get("THREADX_DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()

    # Prefer local snapshot folders used during development. If a `x_data`
    # folder was copied into `src/threadx/x_data` use it preferentially so
    # the UI can work out-of-the-box without extra env vars.
    try:
        repo_src = here.parents[2] if len(here.parents) >= 3 else None
        candidates = []
        if repo_src is not None:
            candidates.append(repo_src / "threadx" / "x_data")
            candidates.append(repo_src / "threadx" / "data")
        candidates.append(Path.cwd() / "src" / "threadx" / "x_data")
        candidates.append(Path.cwd() / "src" / "threadx" / "data")

        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return cand
    except Exception:
        pass

    # Conventional ancestor search for a `data/` folder (original behaviour).
    for ancestor in here.parents:
        data_root = ancestor / "data"
        if not data_root.exists():
            continue
        for child in data_root.iterdir():
            if child.is_dir() and "exploitable" in child.name.lower():
                return child
        return data_root

    return Path.cwd() / "data"


DATA_DIR = _default_data_dir()
EXTS = (".parquet", ".feather", ".csv", ".json")
DATA_FOLDERS = ("crypto_data_parquet", "crypto_data_json")


@lru_cache(maxsize=1)
def _iter_data_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for folder_name in DATA_FOLDERS:
        folder = DATA_DIR / folder_name
        if not folder.exists():
            continue
        for extension in EXTS:
            files.extend(folder.glob(f"*{extension}"))
    return tuple(files)


@lru_cache(maxsize=1)
def discover_tokens_and_timeframes() -> tuple[list[str], list[str]]:
    tokens, timeframes = set(), set()
    for file_path in _iter_data_files():
        parts = file_path.stem.split("_", 1)
        if len(parts) != 2:
            continue
        symbol, timeframe = parts
        tokens.add(symbol.upper())
        timeframes.add(timeframe)

    def _tf_key(value: str) -> tuple[int, int, str]:
        if not value:
            return (5, 0, value)
        unit = value[-1]
        amount_text = value[:-1]
        order = {"m": 0, "h": 1, "d": 2, "w": 3}.get(unit, 4)
        try:
            amount = int(amount_text)
        except ValueError:
            amount = 0
        return (order, amount, value)

    return sorted(tokens), sorted(timeframes, key=_tf_key)


def get_available_timeframes_for_token(symbol: str) -> list[str]:
    """Retourne les timeframes disponibles pour un token specifique."""
    symbol = symbol.upper()
    timeframes = set()

    for file_path in _iter_data_files():
        parts = file_path.stem.split("_", 1)
        if len(parts) != 2:
            continue
        file_symbol, timeframe = parts
        if file_symbol.upper() == symbol:
            timeframes.add(timeframe)

    def _tf_key(value: str) -> tuple[int, int, str]:
        if not value:
            return (5, 0, value)
        unit = value[-1]
        amount_text = value[:-1]
        order = {"m": 0, "h": 1, "d": 2, "w": 3}.get(unit, 4)
        try:
            amount = int(amount_text)
        except ValueError:
            amount = 0
        return (order, amount, value)

    return sorted(timeframes, key=_tf_key)


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported: {path}")


def _find_ohlcv_file(symbol: str, timeframe: str) -> Path | None:
    symbol = symbol.upper()
    target_prefix = f"{symbol}_{timeframe}"
    for file_path in _iter_data_files():
        if file_path.stem == target_prefix:
            return file_path
    return None


def load_ohlcv(symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
    file_path = _find_ohlcv_file(symbol, timeframe)
    if not file_path:
        raise FileNotFoundError(
            f"Fichier OHLCV introuvable pour {symbol}/{timeframe} dans {DATA_DIR}"
        )

    df = _read_any(file_path)

    # NORMALISATION AUTOMATIQUE
    if NORMALIZATION_AVAILABLE:
        # Utiliser le module de normalisation moderne
        df, report = normalize_ohlcv(df, config=DEFAULT_NORMALIZATION_CONFIG)

        if not report.success:
            logger.warning(
                f"Normalisation partielle pour {symbol}/{timeframe}: "
                f"{len(report.errors)} erreurs"
            )
            if report.errors:
                for error in report.errors:
                    logger.error(f"  - {error}")
        else:
            logger.debug(
                f"Normalisation réussie pour {symbol}/{timeframe}: "
                f"{len(report.transformations)} transformations"
            )
    else:
        # Fallback: ancienne méthode (compatibilité)
        logger.warning(
            "Module de normalisation non disponible, utilisation du fallback"
        )

        # Gerer differentes structures de fichiers
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.set_index("time")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")

        # Assurer que l'index est datetime avec timezone UTC
        if df.index.dtype != "datetime64[ns, UTC]":
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

        rename_map = {column: column.lower() for column in df.columns}
        df = df.rename(columns=rename_map).sort_index()

    # Filtrage par dates
    if start is not None:
        # Convertir date/datetime en Timestamp UTC au début du jour
        start_dt = pd.to_datetime(start).tz_localize(None).tz_localize("UTC")
        df = df[df.index >= start_dt]
    if end is not None:
        # Convertir date/datetime en Timestamp UTC à la fin du jour
        end_dt = pd.to_datetime(end).tz_localize(None).tz_localize("UTC")
        end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df.index <= end_dt]

    return df


# Chemin vers le fichier de référence des tokens par capitalisation
TOP_TOKENS_REFERENCE_FILE = Path(
    r"D:\my_soft\gestionnaire_telechargement_multi-timeframe\resultats_choix_des_100tokens.json"
)


def get_top_tokens_from_reference(
    n: int = 10, filter_available: bool = True, suffix: str = "USDC"
) -> list[dict[str, Any]]:
    """
    Retourne les N premiers tokens depuis le fichier de référence (classés par market cap).

    Args:
        n: Nombre de tokens à retourner
        filter_available: Si True, ne retourne que les tokens disponibles dans les données
        suffix: Suffixe à ajouter aux symboles (ex: "USDC" -> "BTCUSDC")

    Returns:
        Liste de dicts avec 'symbol', 'name', 'market_cap', 'market_cap_rank', 'volume', 'rank'
    """
    import json

    if not TOP_TOKENS_REFERENCE_FILE.exists():
        logger.warning(f"Fichier de référence non trouvé: {TOP_TOKENS_REFERENCE_FILE}")
        return []

    try:
        with open(TOP_TOKENS_REFERENCE_FILE, "r", encoding="utf-8") as f:
            all_tokens = json.load(f)
    except Exception as e:
        logger.error(f"Erreur lecture fichier de référence: {e}")
        return []

    # Tokens disponibles dans les données
    available_tokens, _ = discover_tokens_and_timeframes()
    available_set = set(available_tokens)

    # Filtrer les stablecoins et tokens non tradables
    excluded_symbols = {
        "USDT",
        "USDC",
        "USDS",
        "DAI",
        "FDUSD",
        "PYUSD",
        "USDE",
        "STETH",
        "WSTETH",
        "WBETH",
        "WBTC",
        "WEETH",
        "CBBTC",
        "RETH",
        "JITOSOL",
        "BNSOL",
        "WBNB",
        "WETH",
        "EUR",
        "AUD",
    }

    result = []
    rank = 0

    for token_data in all_tokens:
        symbol = token_data.get("symbol", "")

        # Exclure les stablecoins et tokens wrapped
        if symbol in excluded_symbols:
            continue

        # Construire le symbole avec suffixe
        full_symbol = f"{symbol}{suffix}"

        # Vérifier disponibilité si demandé
        if filter_available and full_symbol not in available_set:
            continue

        rank += 1
        result.append(
            {
                "symbol": full_symbol,
                "name": token_data.get("name", symbol),
                "market_cap": token_data.get("market_cap") or 0,
                "market_cap_rank": token_data.get("market_cap_rank") or 999,
                "volume_24h": token_data.get("volume") or 0,
                "rank": rank,
            }
        )

        if len(result) >= n:
            break

    return result


def get_top_tokens_by_volume(
    n: int = 10,
    timeframe: str | None = None,
    days: int = 7,
    prioritize_majors: bool = True,
    use_reference_file: bool = True,
) -> list[dict[str, Any]]:
    """
    Retourne les N tokens les plus actifs par volume/capitalisation.

    Args:
        n: Nombre de tokens à retourner
        timeframe: Timeframe à utiliser pour l'analyse (défaut: auto-detect best)
        days: Nombre de jours récents à analyser (défaut: 7)
        prioritize_majors: Si True, assure que BTC/ETH/SOL sont dans le top si disponibles
        use_reference_file: Si True, utilise le fichier de référence des 100 tokens

    Returns:
        Liste de dicts avec 'symbol', 'avg_volume', 'total_volume', 'bars_count', 'rank'
    """
    # Utiliser le fichier de référence si disponible
    if use_reference_file and TOP_TOKENS_REFERENCE_FILE.exists():
        ref_tokens = get_top_tokens_from_reference(n=n, filter_available=True)
        if ref_tokens:
            # Convertir au format attendu
            return [
                {
                    "symbol": t["symbol"],
                    "avg_volume": t["volume_24h"],
                    "total_volume": t["market_cap"],
                    "bars_count": 0,  # Non calculé
                    "rank": t["rank"],
                    "name": t["name"],
                    "market_cap_rank": t["market_cap_rank"],
                }
                for t in ref_tokens
            ]

    # Fallback: analyse des données locales
    from datetime import datetime, timedelta

    tokens, available_timeframes = discover_tokens_and_timeframes()

    # Priorité aux timeframes les plus fiables si non spécifié
    if timeframe is None:
        preferred_tfs = ["15m", "1h", "30m", "5m", "3m"]
        timeframe = next(
            (tf for tf in preferred_tfs if tf in available_timeframes),
            available_timeframes[0] if available_timeframes else "1h",
        )

    token_stats = []

    # Tokens majeurs connus (priorité haute)
    major_tokens = {
        "BTCUSDC",
        "ETHUSDC",
        "BNBUSDC",
        "SOLUSDC",
        "XRPUSDC",
        "ADAUSDC",
        "DOGEUSDC",
        "AVAXUSDC",
        "DOTUSDC",
        "MATICUSDC",
    }

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for symbol in tokens:
        try:
            file_path = _find_ohlcv_file(symbol, timeframe)
            if not file_path:
                # Essayer un autre timeframe si le principal n'existe pas
                available_tfs = get_available_timeframes_for_token(symbol)
                if available_tfs:
                    file_path = _find_ohlcv_file(symbol, available_tfs[0])

            if not file_path:
                continue

            # Lecture rapide sans normalisation complète pour la performance
            df = (
                pd.read_parquet(file_path)
                if file_path.suffix == ".parquet"
                else _read_any(file_path)
            )

            # Identifier la colonne de volume
            vol_col = None
            for col in ["volume", "Volume", "VOLUME", "vol"]:
                if col in df.columns:
                    vol_col = col
                    break

            if vol_col is None:
                continue

            # Identifier la colonne de temps pour filtrage
            time_col = None
            for col in ["time", "timestamp", "Time", "Timestamp", "date", "Date"]:
                if col in df.columns:
                    time_col = col
                    break

            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
                df = df[df[time_col] >= pd.Timestamp(start_date, tz="UTC")]

            if len(df) == 0:
                continue

            # Calculer le volume moyen (volume * prix pour avoir une approximation du $volume)
            close_col = None
            for col in ["close", "Close", "CLOSE"]:
                if col in df.columns:
                    close_col = col
                    break

            if close_col:
                # Volume en USD approximatif
                dollar_volume = df[vol_col].astype(float) * df[close_col].astype(float)
                avg_vol = dollar_volume.mean()
                total_vol = dollar_volume.sum()
            else:
                avg_vol = df[vol_col].astype(float).mean()
                total_vol = df[vol_col].astype(float).sum()

            token_stats.append(
                {
                    "symbol": symbol,
                    "avg_volume": avg_vol,
                    "total_volume": total_vol,
                    "bars_count": len(df),
                }
            )

        except Exception as e:
            logger.debug(f"Erreur analyse volume {symbol}: {e}")
            continue

    # Trier par volume moyen décroissant
    token_stats.sort(key=lambda x: x["avg_volume"], reverse=True)

    # Si prioritize_majors est activé, assurer que les majors sont inclus
    if prioritize_majors:
        result_symbols = {t["symbol"] for t in token_stats[:n]}
        all_stats_dict = {t["symbol"]: t for t in token_stats}

        # Ajouter les majors manquants s'ils existent dans les données
        majors_to_add = []
        for major in major_tokens:
            if major in all_stats_dict and major not in result_symbols:
                majors_to_add.append(all_stats_dict[major])

        if majors_to_add:
            # Trier les majors par volume
            majors_to_add.sort(key=lambda x: x["avg_volume"], reverse=True)
            # Remplacer les derniers tokens par les majors
            token_stats = token_stats[:n]
            for i, major_stat in enumerate(majors_to_add):
                if i < len(token_stats):
                    # Remplacer le dernier non-major
                    for j in range(len(token_stats) - 1, -1, -1):
                        if token_stats[j]["symbol"] not in major_tokens:
                            token_stats[j] = major_stat
                            break
            # Re-trier
            token_stats.sort(key=lambda x: x["avg_volume"], reverse=True)

    # Ajouter le rang
    for i, stats in enumerate(token_stats[:n]):
        stats["rank"] = i + 1

    return token_stats[:n]


def get_timeframe_performance_estimate(
    symbol: str, timeframes: list[str] | None = None, days: int = 30
) -> list[dict[str, Any]]:
    """
    Estime la qualité des données par timeframe pour un token.

    Args:
        symbol: Symbole du token
        timeframes: Liste de timeframes à évaluer (défaut: tous disponibles)
        days: Nombre de jours à analyser

    Returns:
        Liste de dicts avec 'timeframe', 'bars_count', 'completeness', 'volatility'
    """
    from datetime import datetime, timedelta

    import numpy as np

    if timeframes is None:
        timeframes = get_available_timeframes_for_token(symbol)

    results = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for tf in timeframes:
        try:
            df = load_ohlcv(symbol, tf, start=start_date, end=end_date)

            if len(df) < 10:
                continue

            # Calculer la complétude (barres attendues vs réelles)
            tf_minutes = _timeframe_to_minutes(tf)
            expected_bars = (days * 24 * 60) / tf_minutes
            completeness = min(1.0, len(df) / expected_bars)

            # Calculer la volatilité (proxy de l'activité)
            returns = df["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(
                252 * 24 * 60 / tf_minutes
            )  # Annualisée

            results.append(
                {
                    "timeframe": tf,
                    "bars_count": len(df),
                    "completeness": completeness,
                    "volatility": volatility,
                    "data_quality_score": completeness * 0.7
                    + min(1.0, volatility / 2) * 0.3,
                }
            )

        except Exception as e:
            logger.debug(f"Erreur analyse {symbol}/{tf}: {e}")
            continue

    # Trier par score de qualité
    results.sort(key=lambda x: x["data_quality_score"], reverse=True)
    return results


def _timeframe_to_minutes(tf: str) -> int:
    """Convertit un timeframe en minutes."""
    if not tf:
        return 60

    unit = tf[-1].lower()
    try:
        amount = int(tf[:-1])
    except ValueError:
        return 60

    multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    return amount * multipliers.get(unit, 60)
