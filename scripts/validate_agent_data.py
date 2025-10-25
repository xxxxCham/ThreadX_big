"""
Validate all data files under a root (D:/agent_data by default) using threadx.data.io.read_frame(normalize=True)
Outputs artifacts/data_validation_report.md
"""

from pathlib import Path
import traceback
from threadx.data.io import read_frame
from threadx.data.normalize import (
    validate_indicator,
    convert_indicator_df_to_parquet,
    guess_timestamp_column,
)
import pandas as pd

ROOT = Path(r"D:/agent_data")
OUT = Path("artifacts/data_validation_report.md")
OUT.parent.mkdir(parents=True, exist_ok=True)

files = sorted([p for p in ROOT.iterdir() if p.is_file()])

lines = ["# Data validation report\n", f"Root: {ROOT}\n", "\n"]

for p in files:
    lines.append(f"## {p.name}\n")
    try:
        # First attempt: try to read as OHLCV candle
        try:
            df = read_frame(p, normalize=True)
            nrows, ncols = df.shape
            cols = ", ".join(df.columns.tolist())
            monotonic = df.index.is_monotonic_increasing
            dupes = int(df.index.duplicated().sum())
            nulls = int(df.isnull().sum().sum())

            lines.append("- Type: candle\n")
            lines.append("- Status: OK\n")
            lines.append(f"- Rows: {nrows}, Columns: {ncols}\n")
            lines.append(f"- Columns: {cols}\n")
            lines.append(f"- Monotonic index: {monotonic}\n")
            lines.append(f"- Duplicate timestamps: {dupes}\n")
            lines.append(f"- Total nulls: {nulls}\n")
        except Exception as e_candle:
            # If candle read fails, try to read raw and validate as indicator
            try:
                # load raw dataframe with pandas (parquet/json) without normalization
                if p.suffix.lower() == ".parquet":
                    raw = pd.read_parquet(p)
                elif p.suffix.lower() == ".json":
                    raw = pd.read_json(p, orient="records", convert_dates=False)
                else:
                    raise ValueError(f"Unsupported extension: {p.suffix}")

                # Guess timestamp column
                ts = guess_timestamp_column(raw.columns)
                report = validate_indicator(raw, timestamp_col=ts)

                lines.append("- Type: indicator\n")
                lines.append(f"- Status: {report['status']}\n")
                for k, v in report.get("metrics", {}).items():
                    lines.append(f"- {k}: {v}\n")
                if report.get("details"):
                    for d in report["details"]:
                        lines.append(f"- Detail: {d}\n")

                # If invalid and parquet source, try conversion to canonical parquet and re-validate
                if report["status"] != "OK" and p.suffix.lower() == ".parquet":
                    try:
                        out_dir = Path("artifacts/converted_indicators")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / p.name
                        converted = convert_indicator_df_to_parquet(
                            raw, out_path, timestamp_col=ts
                        )
                        # re-validate converted file
                        raw2 = pd.read_parquet(converted)
                        report2 = validate_indicator(raw2)
                        lines.append("- Conversion attempt: succeeded\n")
                        lines.append(f"- Converted path: {converted}\n")
                        lines.append(f"- Post-conversion status: {report2['status']}\n")
                    except Exception as e_conv:
                        lines.append(f"- Conversion attempt: FAILED: {e_conv}\n")

            except Exception as e_ind:
                tb = traceback.format_exc()
                lines.append("- Type: unknown\n")
                lines.append(f"- Status: ERROR\n")
                lines.append(f"- Error: {type(e_ind).__name__}: {e_ind}\n")
                lines.append("```\n")
                lines.append(tb)
                lines.append("```\n")
    except Exception as e:
        tb = traceback.format_exc()
        lines.append("- Status: ERROR\n")
        lines.append(f"- Error: {type(e).__name__}: {e}\n")
        lines.append("```\n")
        lines.append(tb)
        lines.append("```\n")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote report to {OUT}")
