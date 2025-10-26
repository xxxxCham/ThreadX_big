import os
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from functools import lru_cache


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
def _iter_data_files() -> Tuple[Path, ...]:
    files: List[Path] = []
    for folder_name in DATA_FOLDERS:
        folder = DATA_DIR / folder_name
        if not folder.exists():
            continue
        for extension in EXTS:
            files.extend(folder.glob(f"*{extension}"))
    return tuple(files)


@lru_cache(maxsize=1)
def discover_tokens_and_timeframes() -> Tuple[List[str], List[str]]:
    tokens, timeframes = set(), set()
    for file_path in _iter_data_files():
        parts = file_path.stem.split("_", 1)
        if len(parts) != 2:
            continue
        symbol, timeframe = parts
        tokens.add(symbol.upper())
        timeframes.add(timeframe)

    def _tf_key(value: str) -> Tuple[int, int, str]:
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


def _find_ohlcv_file(symbol: str, timeframe: str) -> Optional[Path]:
    """
    Trouve un fichier OHLCV en priorisant Parquet > Feather > CSV > JSON.
    Cette priorité assure les meilleures performances de lecture.
    """
    symbol = symbol.upper()
    target_prefix = f"{symbol}_{timeframe}"

    # Ordre de priorité pour les performances : Parquet est le plus rapide
    priority_extensions = [".parquet", ".feather", ".csv", ".json"]

    # Chercher d'abord par ordre de priorité
    for ext in priority_extensions:
        for file_path in _iter_data_files():
            if file_path.stem == target_prefix and file_path.suffix == ext:
                return file_path

    return None


def load_ohlcv(symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
    file_path = _find_ohlcv_file(symbol, timeframe)
    if not file_path:
        # Message d'erreur détaillé pour faciliter le débogage
        available_files = list(_iter_data_files())
        raise FileNotFoundError(
            f"Fichier OHLCV introuvable pour {symbol}/{timeframe}\n"
            f"Dossier de recherche : {DATA_DIR}\n"
            f"Sous-dossiers cherchés : {DATA_FOLDERS}\n"
            f"Formats supportés (par ordre de priorité) : .parquet > .feather > .csv > .json\n"
            f"Nombre de fichiers trouvés : {len(available_files)}\n"
            f"Nommage attendu : {symbol}_{timeframe}.[parquet|json|csv|feather]"
        )

    # Charger les données (Parquet est prioritaire pour les performances)
    df = _read_any(file_path)

    # Log pour confirmation du format utilisé (visible dans les logs Streamlit)
    print(f"✅ Chargé : {file_path.name} (format: {file_path.suffix}) - {len(df)} lignes")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.set_index("time")
    if df.index.dtype != "datetime64[ns, UTC]":
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    rename_map = {column: column.lower() for column in df.columns}
    df = df.rename(columns=rename_map).sort_index()

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    return df




