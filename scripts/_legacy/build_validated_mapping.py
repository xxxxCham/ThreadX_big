"""Scan data folder and produce artifacts/validated_mapping_plan.csv

Columns: source_path,dest_path,type,validated_by_pandera,rows,notes

The script will not perform copies; it only writes the mapping plan for manual review.
"""

from pathlib import Path
import csv
from threadx.data.validate import validate_dataset
import argparse


def guess_dest_for_file(f: Path, base: Path) -> tuple[str, str]:
    """Return (dest_path, type)"""
    stem = f.stem
    # try parse symbol_timeframe
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].lower().endswith(("m", "h", "d")):
        symbol = "_".join(parts[:-1]).upper()
        tf = parts[-1].lower()
        dest = base / "timeline" / symbol / f"{tf}.parquet"
        return str(dest), "candle"

    # Otherwise indicator target
    dest = (
        base
        / "indicators"
        / (f.name if f.suffix.lower() == ".parquet" else f.stem + ".parquet")
    )
    return str(dest), "indicator"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="./data")
    p.add_argument("--out", default="artifacts/validated_mapping_plan.csv")
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # collect candidate files
    for f in sorted(src.rglob("*.parquet")):
        # Skip validated folder itself
        if "validated" in f.parts:
            continue
        dest, ftype = guess_dest_for_file(f, src / "validated")

        # run lightweight validate_dataset on the file's parent (validate_dataset accepts file too)
        report = validate_dataset(str(f))
        validated = "yes" if report.get("ok") else "no"
        rows.append(
            {
                "source_path": str(f),
                "dest_path": dest,
                "type": ftype,
                "validated_by_pandera": validated,
                "rows": report.get("rows", "?"),
                "notes": "; ".join(report.get("errors", []))[:200],
            }
        )

    # write CSV
    with open(out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "source_path",
                "dest_path",
                "type",
                "validated_by_pandera",
                "rows",
                "notes",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote mapping plan to {out} ({len(rows)} entries)")


if __name__ == "__main__":
    main()
