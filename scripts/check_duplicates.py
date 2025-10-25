#!/usr/bin/env python3
"""Scan repo for parquet files with duplicate basenames (stem) across key directories.
Writes artifacts/duplicates_report.csv and prints a short summary.
"""
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parent.parent
paths = [
    ROOT / "data" / "crypto_data_parquet",
    ROOT / "data" / "indicators",
    ROOT / "artifacts" / "regenerated",
]
filemap = {}
count = 0
for base in paths:
    if not base.exists():
        continue
    for p in base.rglob("*.parquet"):
        stem = p.stem
        entry = filemap.setdefault(stem, [])
        entry.append(str(p))
        count += 1

out = ROOT / "artifacts" / "duplicates_report.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["stem", "count", "paths"])
    dup_count = 0
    for stem, paths in sorted(filemap.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(paths) > 1:
            dup_count += 1
            writer.writerow([stem, len(paths), ";".join(paths)])

print(
    f"Scanned {count} parquet files; found {dup_count} stems with >1 path. Report: {out}"
)
if dup_count > 0:
    print("Top duplicates:")
    i = 0
    for stem, paths in sorted(filemap.items(), key=lambda x: -len(x[1])):
        if len(paths) > 1:
            print(f"{stem}: {len(paths)}")
            i += 1
            if i >= 10:
                break
