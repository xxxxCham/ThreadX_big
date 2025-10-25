"""
Import and reorganize a representative agent_data folder into ThreadX data layout.

Usage:
    python scripts/import_agent_data.py --source D:/agent_data --dest ./data --dry-run

The script will:
 - detect OHLCV candle files (readable by read_frame) and copy them to
   {dest}/processed/{symbol}/{timeframe}.parquet (symbol/timeframe guessed from filename)
 - detect indicator files (numeric columns) and convert them to parquet under
   {dest}/indicators/unassigned/{original_filename}.parquet (to be remapped manually)

Dry-run prints planned actions without writing by default.
"""

from pathlib import Path
import argparse
import re
from threadx.data.io import read_frame, write_frame
from threadx.data.normalize import convert_indicator_df_to_parquet
import pandas as pd


def guess_symbol_timeframe_from_name(name: str):
    """Try to parse symbol and timeframe from filename stem like 'BTCUSDC_1h'"""
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        tf = parts[-1]
        # timeframe pattern: digits + m/h/d (e.g. 1h, 15m, 1d)
        if re.match(r"^\d+[mhd]$", tf, flags=re.IGNORECASE):
            symbol = "_".join(parts[:-1])
            return symbol.upper(), tf.lower()
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=r"D:/agent_data")
    p.add_argument("--dest", default="./data/validated")
    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="If set, perform writes/copies. Otherwise run in dry-run mode",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy files instead of move",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files processed (0 => all)",
    )
    args = p.parse_args()

    src = Path(args.source)
    dest = Path(args.dest)
    if not src.exists():
        print(f"Source not found: {src}")
        return

    processed = 0
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        if args.limit and args.limit > 0 and processed >= args.limit:
            break

        print("---", f.name)

        # Try to read as candle (OHLCV)
        try:
            df = read_frame(f, normalize=True)
            # It's a candle file
            symbol, tf = guess_symbol_timeframe_from_name(f.name)
            if not symbol or not tf:
                target_dir = dest / "processed" / "unassigned"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / f.name
            else:
                target_dir = dest / "processed" / symbol
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / f"{tf}.parquet"

            print(f"Detected candle -> target: {target_path}")
            if args.apply:
                # use write_frame to enforce normalization and dtype casting
                write_frame(df, target_path, overwrite=True)
                if args.copy:
                    print(f"Wrote (copied) {target_path}")
                else:
                    print(f"Wrote and moving source {f} -> {target_path}")
                    f.unlink()

            continue

        except Exception:
            # Not a candle; try indicator
            pass

        # Try to load as parquet/json and validate as indicator
        try:
            if f.suffix.lower() == ".parquet":
                raw = pd.read_parquet(f)
            elif f.suffix.lower() == ".json":
                raw = pd.read_json(f, orient="records", convert_dates=False)
            else:
                print(f"Skipping unsupported extension: {f.suffix}")
                processed += 1
                continue

            # Keep numeric columns and convert
            numeric = raw.select_dtypes(include=["number"]).dropna(axis=1, how="all")
            if numeric.shape[1] == 0:
                print(f"No numeric columns found in {f.name}; skipping")
                processed += 1
                continue

            # Write converted indicators to dest/indicators/unassigned/
            out_dir = dest / "indicators" / "unassigned"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (f.stem + ".parquet")
            print(f"Detected indicator -> target: {out_path}")
            if args.apply:
                convert_indicator_df_to_parquet(raw, out_path)

        except Exception as exc:
            print(f"Failed to process {f.name}: {exc}")

        # mark processed regardless of success/failure for limit accounting
        processed += 1


if __name__ == "__main__":
    main()
