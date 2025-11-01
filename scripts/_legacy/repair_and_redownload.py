#!/usr/bin/env python3
"""Repair and re-download pipeline for pandera failed files.

Behavior:
 - Reads artifacts/pandera_failed_files.csv (or artifacts/batch_regen_report.csv if preferred)
 - For each row:
    - If type == 'candle': if source parquet corrupt or dest invalid, optionally delete local copy and re-download via BinanceDataLoader.download_ohlcv
    - If type == 'indicator': try to regenerate via scripts/regenerate_indicator_from_price.py
 - Dry-run by default. Use --apply to actually perform deletes/downloads/regenerations.
 - Writes artifacts/repair_redownload_report.csv

Note: respects environment and uses BINANCE_API_KEY env var if needed.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import shutil
import traceback

ROOT = Path(__file__).resolve().parent.parent
REPORT_IN = Path("artifacts/pandera_failed_files.csv")
REPORT_OUT = Path("artifacts/repair_redownload_report.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--apply", action="store_true")
parser.add_argument("--source", default=str(REPORT_IN))
parser.add_argument(
    "--force-download", action="store_true", help="Force download even if source exists"
)
parser.add_argument(
    "--binance-api-key", default=None, help="Optional Binance API key for downloader"
)
args = parser.parse_args()

source = Path(args.source)
if not source.exists():
    print(f"Source fail list not found: {source}")
    sys.exit(1)

# Lazy import of project modules
sys.path.insert(0, str(ROOT))

from src.threadx.data.loader import BinanceDataLoader

loader = (
    BinanceDataLoader(api_key=args.binance_api_key)
    if args.binance_api_key
    else BinanceDataLoader()
)

rows_out = []
processed = 0

with source.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        if args.limit and processed >= args.limit:
            break
        processed += 1
        src = row.get("source", "")
        dest = row.get("dest", "")
        typ = row.get("type", "")
        out = {
            "source": src,
            "dest": dest,
            "type": typ,
            "action": "",
            "result": "",
            "error": "",
        }

        try:
            if typ == "candle":
                # source likely like data\crypto_data_parquet\last\SYMBOL_TF\... .parquet
                srcp = Path(src)
                # decide whether to delete and redownload
                if not srcp.exists() or args.force_download:
                    out["action"] = "download"
                    if not args.apply:
                        out["result"] = "dry-run-would-download"
                    else:
                        # parse symbol and timeframe from filename
                        name = srcp.stem
                        # expected format SYMBOL_TIMEFRAME (e.g. BTCUSDC_3m)
                        parts = name.rsplit("_", 1)
                        if len(parts) != 2:
                            out["result"] = "cannot-parse-symbol-tf"
                        else:
                            symbol, tf = parts
                            # build target folder for download
                            print(f"Downloading {symbol} {tf} ...")
                            # call loader.download_ohlcv(symbol, tf, ...) - use default args for days/history
                            try:
                                loader.download_ohlcv(
                                    symbol=symbol, interval=tf, days_history=30
                                )
                                out["result"] = "downloaded"
                            except Exception as e:
                                out["result"] = "download_failed"
                                out["error"] = str(e)
                else:
                    out["action"] = "skip"
                    out["result"] = "source_exists"

            elif typ == "indicator":
                # try to regenerate by calling the regenerate script
                out["action"] = "regenerate"
                # parse dest path to extract symbol, timeframe and indicator name
                destp = Path(dest)
                parts = destp.parts
                # expected ... data/indicators/SYMBOL/TF/indicator_file.parquet
                try:
                    idx = parts.index("indicators")
                    symbol = parts[idx + 1]
                    timeframe = parts[idx + 2]
                    filename = parts[-1]
                    # extract indicator type (e.g., ema_period20 -> ema, period20)
                    if not args.apply:
                        out["result"] = "dry-run-would-regenerate"
                    else:
                        cmd = [
                            sys.executable,
                            str(
                                ROOT / "scripts" / "regenerate_indicator_from_price.py"
                            ),
                            "--symbol",
                            symbol,
                            "--timeframe",
                            timeframe,
                            "--apply",
                        ]
                        print("Running:", " ".join(cmd))
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        if p.returncode == 0:
                            out["result"] = "regenerated"
                        else:
                            out["result"] = "regenerate_failed"
                            out["error"] = p.stderr[:2000] or p.stdout[:2000]
                except Exception as e:
                    out["result"] = "cannot-parse-dest"
                    out["error"] = str(e)
            else:
                out["action"] = "unknown"
                out["result"] = "skipped"
        except Exception as e:
            out["result"] = "exception"
            out["error"] = repr(e) + "\n" + traceback.format_exc()

        rows_out.append(out)

# write report
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
with REPORT_OUT.open("w", newline="", encoding="utf-8") as fh:
    fieldnames = ["source", "dest", "type", "action", "result", "error"]
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print(f"Processed: {processed}; wrote report: {REPORT_OUT}")
