"""Module léger de validation des datasets.

Ce module fournit une fonction `validate_dataset(path)` minimale qui
permet à l'UI Streamlit et aux scripts de fonctionner même si le module
de validation complet (pandera, règles métier) est absent.

Ce fichier est un fallback sûr : il n'altère ni ne supprime de données.
Il réalise des vérifications passives (existence du dossier, liste de fichiers)
et renvoie une structure dict standardisée.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def validate_dataset(path: str) -> Dict[str, Any]:
    """Validation légère et non destructive d'un dataset.

    Args:
        path: chemin vers le dossier de données (fichier ou dossier)

    Returns:
        dict: rapport de validation avec clefs minimales ('ok', 'type', 'note', 'files')
    """
    p = Path(path)
    result = {
        "ok": True,
        "type": "fallback-light-validate",
        "note": "fallback validator in place; no destructive actions taken",
        "path": str(p),
        "files": [],
    }

    try:
        if not p.exists():
            result["ok"] = False
            result["note"] = "path does not exist"
            return result

        if p.is_file():
            result["files"] = [p.name]
            return result

        # Directory: list top-level files/folders (non-recursive)
        items = []
        for child in sorted(p.iterdir()):
            items.append(child.name)
        result["files"] = items
        return result

    except Exception as e:
        return {"ok": False, "type": "error", "note": str(e)}


"""Data validation utilities for ThreadX.

Provides `validate_dataset(path)` which returns a small report dict:
  { ok: bool, errors: list[str], type: 'candle'|'indicator'|'unknown', convertible: bool }

If pandera is installed it will be used for stricter validation. Otherwise a lightweight
fallback checks presence of OHLCV columns and numeric types.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema

    PANDERA_AVAILABLE = True
except Exception:
    PANDERA_AVAILABLE = False


OHLCV = ["open", "high", "low", "close", "volume"]


def _basic_check(df: pd.DataFrame) -> Dict[str, Any]:
    cols = [c.lower() for c in df.columns]
    has_ohlcv = all(c in cols for c in OHLCV)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return {
        "has_ohlcv": has_ohlcv,
        "numeric_columns": numeric_cols,
        "rows": len(df),
    }


def validate_dataset(path: str) -> Dict[str, Any]:
    """Validate dataset at path. Returns report dict.

    path may point to a parquet/json/csv file or a directory; when directory is provided
    the function inspects files inside and returns aggregated result (first successful file).
    """
    p = Path(path)
    report = {"ok": False, "errors": [], "type": "unknown", "convertible": False}

    candidates = []
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in (".parquet", ".json", ".csv"):
                candidates.append(f)
    elif p.is_file():
        candidates = [p]
    else:
        report["errors"].append(f"Path not found: {p}")
        return report

    for f in candidates:
        try:
            if f.suffix.lower() == ".parquet":
                df = pd.read_parquet(f)
            elif f.suffix.lower() == ".json":
                df = pd.read_json(f, orient="records", convert_dates=False)
            elif f.suffix.lower() == ".csv":
                df = pd.read_csv(f)
            else:
                continue

            basic = _basic_check(df)

            # Use pandera if available and if candidate looks like OHLCV
            if PANDERA_AVAILABLE and basic["has_ohlcv"]:
                schema = DataFrameSchema(
                    {
                        "open": Column(float, nullable=False),
                        "high": Column(float, nullable=False),
                        "low": Column(float, nullable=False),
                        "close": Column(float, nullable=False),
                        "volume": Column(float, nullable=True),
                    }
                )
                try:
                    schema.validate(df, lazy=True)
                    report.update({"ok": True, "type": "candle", "convertible": False})
                    return report
                except pa.errors.SchemaErrors as e:
                    report["errors"].append(str(e))
                    # fallthrough to basic handling

            # Basic handling: if OHLCV present -> candle
            if basic["has_ohlcv"]:
                report.update({"ok": True, "type": "candle", "convertible": False})
                return report

            # If numeric columns exist it's an indicator-like file
            if len(basic["numeric_columns"]) > 0 and basic["rows"] >= 1:
                report.update({"ok": True, "type": "indicator", "convertible": True})
                return report

        except Exception as exc:
            report["errors"].append(f"Failed reading {f.name}: {exc}")

    # If reached here no suitable file found
    report["ok"] = False
    if not report["errors"]:
        report["errors"].append("No supported data files found")
    return report



