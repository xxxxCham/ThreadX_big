"""Migrate/copy datasets into `data/validated/timeline` and `data/validated/indicators`.

Dry-run by default. Usage:
  python scripts/migrate_to_validated.py --source ./data --dry-run

This script is non-destructive: it copies files. Use --apply to perform copies.
"""

from pathlib import Path
import argparse
import shutil
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="./data")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--apply", action="store_true", default=False)
    args = p.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"Source not found: {src}")
        sys.exit(1)

    validated_tl = src / "validated" / "timeline"
    validated_ind = src / "validated" / "indicators"

    print("Planned targets:")
    print(f" - Timeline: {validated_tl}")
    print(f" - Indicators: {validated_ind}")

    # Ensure parent directories exist in dry-run output
    if args.apply:
        validated_tl.mkdir(parents=True, exist_ok=True)
        validated_ind.mkdir(parents=True, exist_ok=True)

    # Copy processed candles into validated/timeline
    processed = src / "processed"
    if processed.exists():
        for sym_dir in sorted(processed.iterdir()):
            if not sym_dir.is_dir():
                continue
            for pf in sym_dir.glob("*.parquet"):
                dest = validated_tl / sym_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                dest_file = dest / pf.name
                print(f"Plan: copy {pf} -> {dest_file}")
                if args.apply:
                    shutil.copy2(pf, dest_file)

    # Copy indicator parquets found under src/indicators or top-level
    indicators_src = src / "indicators"
    if indicators_src.exists():
        for f in sorted(indicators_src.rglob("*.parquet")):
            dest_file = validated_ind / f.name
            print(f"Plan: copy {f} -> {dest_file}")
            if args.apply:
                validated_ind.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest_file)

    print("Done (dry-run mode)" if not args.apply else "Done (applied)")


if __name__ == "__main__":
    main()
