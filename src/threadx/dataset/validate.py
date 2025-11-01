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

