#!/usr/bin/env python3
"""Detect and optionally deduplicate parquet files by basename (stem).

Usage: python scripts/dedupe_parquets.py [--apply]

Behavior:
 - Scans these root dirs: data/crypto_data_parquet, data/indicators, artifacts/regenerated, artifacts/regenerated/batch
 - Builds map stem -> list(paths)
 - For stems with >1 path decides a keep-path using priority and size
 - In dry-run writes artifacts/duplicates_report.csv listing keep + to_move
 - With --apply moves duplicates (to_move) into artifacts/backup_duplicates/<timestamp>/ preserving directory structure
"""
from pathlib import Path
import argparse
import csv
import shutil
import time

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = [
    ROOT / "data" / "crypto_data_parquet",
    ROOT / "data" / "indicators",
    ROOT / "artifacts" / "regenerated",
]
OUT = ROOT / "artifacts" / "duplicates_report.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()

filemap = {}
for base in SCAN_DIRS:
    if not base.exists():
        continue
    for p in base.rglob("*.parquet"):
        stem = p.stem
        filemap.setdefault(stem, []).append(p)

rows = []
for stem, paths in sorted(filemap.items()):
    if len(paths) <= 1:
        continue

    # choose keep path by priority
    def priority(p: Path):
        s = str(p)
        if "/data/validated" in s or "\\data\\validated" in s:
            return 0
        if "/data/crypto_data_parquet" in s or "\\data\\crypto_data_parquet" in s:
            return 1
        if "/data/indicators" in s or "\\data\\indicators" in s:
            return 2
        if "/artifacts/regenerated" in s or "\\artifacts\\regenerated" in s:
            return 3
        return 4

    # sort by (priority, -size)
    paths_sorted = sorted(paths, key=lambda p: (priority(p), -p.stat().st_size))
    keep = paths_sorted[0]
    to_move = paths_sorted[1:]
    rows.append(
        {"stem": stem, "keep": str(keep), "to_move": ";".join(str(x) for x in to_move)}
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["stem", "keep", "to_move"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Found {len(rows)} duplicated stems. Report written to: {OUT}")
if args.apply:
    ts = time.strftime("%Y%m%dT%H%M%S")
    backup_root = ROOT / "artifacts" / "backup_duplicates" / ts
    backup_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for r in rows:
        to_move = r["to_move"].split(";") if r["to_move"] else []
        for p in to_move:
            src = Path(p)
            if not src.exists():
                print(f"Missing (skip): {src}")
                continue
            # preserve relative path under backup
            rel = src.relative_to(ROOT)
            dst = backup_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
    print(f"Moved {moved} duplicate files to {backup_root}")
else:
    print(
        "Dry-run only. Rerun with --apply to move duplicates to artifacts/backup_duplicates/<ts>/"
    )
