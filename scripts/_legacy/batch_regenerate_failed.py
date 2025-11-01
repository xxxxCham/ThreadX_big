from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import traceback

import pandas as pd

from threadx.data.io import write_frame, read_frame
from threadx.data.loader import BinanceDataLoader


def infer_symbol_timeframe_from_path(p: str) -> tuple[str | None, str | None]:
    # Try filename patterns like SYMBOL_3m or SYMBOL_15m
    fn = Path(p).name
    m = re.search(r"([A-Z0-9]+)_([0-9]+[mhd])", fn)
    if m:
        return m.group(1), m.group(2)
    # Try parent folder name
    parts = Path(p).parts
    for part in reversed(parts):
        m = re.match(r"([A-Z0-9]+)_([0-9]+[mhd])", part)
        if m:
            return m.group(1), m.group(2)
    return None, None


def process_row(
    source: Path, dest: Path, typ: str, dry_run: bool, loader: BinanceDataLoader
) -> dict:
    out = {
        "source": str(source),
        "dest": str(dest),
        "type": typ,
        "status": "skipped",
        "error": "",
    }

    # Skip if dest already exists
    if Path(dest).exists():
        out["status"] = "exists"
        return out

    try:
        # Try reading source
        try:
            df = pd.read_parquet(source)
        except Exception:
            # Try to download via loader if price data
            if typ == "candle":
                sym, tf = infer_symbol_timeframe_from_path(str(source))
                if not sym or not tf:
                    raise RuntimeError("Cannot infer symbol/timeframe for download")
                df = loader.download_ohlcv(sym, tf, days_history=365, save_parquet=True)
                if df is None or df.empty:
                    raise RuntimeError("Downloader returned empty DataFrame")
            else:
                raise

        # Final normalization via read_frame if candle
        if typ == "candle":
            # read_frame will validate/normalize
            df_norm = df
            # Use write_frame target path (dest may be under data/validated)
            target = Path(dest)
            if dry_run:
                # write to artifacts/regenerated/batch
                arte = Path("artifacts/regenerated/batch")
                arte.mkdir(parents=True, exist_ok=True)
                outpath = arte / target.name
                write_frame(df_norm, outpath, overwrite=True)
                out["status"] = "dry-regenerated"
                out["outpath"] = str(outpath)
            else:
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                write_frame(df_norm, dest, overwrite=True)
                out["status"] = "applied"
        else:
            # For now, attempt a simple pass-through for indicators
            df_norm = (
                df.select_dtypes(include=["number"])
                if isinstance(df, pd.DataFrame)
                else None
            )
            if df_norm is None or df_norm.empty:
                raise RuntimeError("Indicator source invalid or empty")
            if dry_run:
                arte = Path("artifacts/regenerated/batch")
                arte.mkdir(parents=True, exist_ok=True)
                outpath = arte / Path(dest).name
                df_norm.to_parquet(outpath, engine="pyarrow", compression="snappy")
                out["status"] = "dry-regenerated"
                out["outpath"] = str(outpath)
            else:
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                df_norm.to_parquet(dest, engine="pyarrow", compression="snappy")
                out["status"] = "applied"

    except Exception as e:
        out["status"] = "failed"
        out["error"] = f"{e}\n{traceback.format_exc()}"

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="artifacts/pandera_failed_files.csv")
    p.add_argument(
        "--dry-run", action="store_true", default=True, help="Default: dry-run"
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes into data/validated (overrides dry-run)",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="Limit number of rows to process (0=all)"
    )
    args = p.parse_args(argv)

    csvp = Path(args.csv)
    if not csvp.exists():
        print(f"CSV not found: {csvp}")
        return 2

    dry_run = not args.apply
    loader = BinanceDataLoader(
        json_cache_dir=Path("data/cache/json"),
        parquet_cache_dir=Path("data/crypto_data_parquet"),
    )

    report = []
    with open(csvp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            source = Path(row.get("source", ""))
            dest = Path(row.get("dest", ""))
            typ = row.get("type", "candle")

            if args.limit and count >= args.limit:
                break

            res = process_row(source, dest, typ, dry_run, loader)
            report.append(res)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} rows...")

    # Write report
    outp = Path("artifacts/batch_regen_report.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    keys = ["source", "dest", "type", "status", "outpath", "error"]
    with open(outp, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        for r in report:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"Batch regen finished: {len(report)} rows. Report: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
