#!/usr/bin/env python3
"""Apply batch regeneration dry-run results.
Reads artifacts/batch_regen_report.csv and moves files with status 'dry-regenerated'
from their outpath to the dest path under data/validated. Verifies parquet can be read.
Writes artifacts/batch_apply_report.csv with per-row result.

Usage: python scripts/apply_batch_regen.py --report artifacts/batch_regen_report.csv --apply
"""
import argparse
import csv
from pathlib import Path
import shutil
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--report", default="artifacts/batch_regen_report.csv")
parser.add_argument(
    "--apply", action="store_true", help="Actually move files (otherwise dry-run)"
)
args = parser.parse_args()

report_path = Path(args.report)
if not report_path.exists():
    print(f"Report not found: {report_path}")
    raise SystemExit(1)

out_rows = []
processed = 0
moved = 0
errors = 0

with report_path.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        processed += 1
        status = row.get("status", "").strip()
        outpath = row.get("outpath", "").strip()
        dest = row.get("dest", "").strip()
        rec = dict(row)
        rec["applied"] = ""
        rec["apply_error"] = ""
        rec["verify_ok"] = ""

        if status != "dry-regenerated":
            rec["applied"] = "skipped"
            out_rows.append(rec)
            continue

        if not outpath:
            rec["applied"] = "no_outpath"
            rec["apply_error"] = "missing outpath"
            errors += 1
            out_rows.append(rec)
            continue

        src = Path(outpath)
        if not src.exists():
            rec["applied"] = "missing_outfile"
            rec["apply_error"] = f"outpath does not exist: {src}"
            errors += 1
            out_rows.append(rec)
            continue

        dst = Path(dest)
        dst_parent = dst.parent
        try:
            dst_parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            rec["applied"] = "mkdir_failed"
            rec["apply_error"] = str(e)
            errors += 1
            out_rows.append(rec)
            continue

        if dst.exists():
            rec["applied"] = "dest_exists"
            # remove src to avoid duplicates? we'll not remove; just skip
            out_rows.append(rec)
            continue

        if args.apply:
            try:
                shutil.move(str(src), str(dst))
                moved += 1
                rec["applied"] = "moved"
            except Exception as e:
                rec["applied"] = "move_failed"
                rec["apply_error"] = str(e)
                errors += 1
                out_rows.append(rec)
                continue
        else:
            rec["applied"] = "would_move"

        # verify readable by pandas
        try:
            # attempt to read parquet (pandas will use pyarrow/fastparquet)
            pd.read_parquet(dst)
            rec["verify_ok"] = "yes"
        except Exception as e:
            rec["verify_ok"] = "no"
            rec["apply_error"] = f"verify_failed: {e}"
            errors += 1
            # if we actually moved the file and verification failed, move back to artifacts
            if args.apply and rec["applied"] == "moved":
                try:
                    fallback = Path("artifacts/need_manual_review")
                    fallback.mkdir(parents=True, exist_ok=True)
                    fallback_dst = fallback / src.name
                    shutil.move(str(dst), str(fallback_dst))
                    rec["applied"] = "moved_to_manual_review_due_verify_failed"
                except Exception as e2:
                    rec["applied"] = "moved_but_verify_and_fallback_failed"
                    rec["apply_error"] += f"; fallback_failed: {e2}"
        out_rows.append(rec)

print(f"Processed rows: {processed}, moved (attempts): {moved}, errors: {errors}")

out_path = Path("artifacts/batch_apply_report.csv")
with out_path.open("w", newline="", encoding="utf-8") as fh:
    if out_rows:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

print(f"Wrote apply report: {out_path}")

if errors > 0:
    print(
        "Some items failed or require manual review. See the report and artifacts/need_manual_review."
    )
else:
    print("All applied entries verified OK.")
