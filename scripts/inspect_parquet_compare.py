#!/usr/bin/env python3
"""Inspect sample parquet files in validated/processed vs crypto_data_parquet and print schema, dtypes and sample rows."""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
symbols = ["BTCUSDC", "AAVEUSDC", "ADAUSDC", "1000CATUSDC"]


def find_validated(symbol):
    base = ROOT / "data" / "validated" / "processed" / symbol
    if not base.exists():
        return None
    # find any parquet file under this folder
    files = list(base.rglob("*.parquet"))
    return files[0] if files else None


def find_crypto(symbol):
    base = ROOT / "data" / "crypto_data_parquet"
    # check top-level file like SYMBOL_1h.parquet
    candidates = []
    if base.exists():
        for p in base.glob(f"{symbol}_*.parquet"):
            candidates.append(p)
        # also search under last/<symbol_tf>/<file>.parquet
        last = base / "last"
        if last.exists():
            for p in last.rglob(f"{symbol}_*.parquet"):
                candidates.append(p)
    return candidates[0] if candidates else None


def inspect(path):
    info = {}
    try:
        info["path"] = str(path)
        info["size"] = path.stat().st_size
        # read parquet schema via pyarrow
        pqf = pq.ParquetFile(str(path))
        info["schema"] = str(pqf.schema)
        # read small sample via pandas
        df = pd.read_parquet(path, columns=None)
        info["columns"] = list(df.columns)
        info["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
        info["nrows"] = len(df.index)
        info["head"] = df.head(3).to_dict(orient="list")
    except Exception as e:
        info["error"] = repr(e)
    return info


def print_info(title, info):
    print("===" + title + "===")
    if info is None:
        print("MISSING")
        return
    for k, v in info.items():
        print(f"{k}: {v}")
    print("\n")


def main():
    for s in symbols:
        print("\n****************************")
        print(f"Checking symbol: {s}")
        v = find_validated(s)
        c = find_crypto(s)
        print_info("validated", inspect(v) if v else None)
        print_info("crypto_data_parquet", inspect(c) if c else None)


if __name__ == "__main__":
    main()
