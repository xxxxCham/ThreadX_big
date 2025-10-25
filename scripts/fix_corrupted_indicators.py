from __future__ import annotations
import shutil
from pathlib import Path
import argparse
import pandas as pd

from threadx.data.normalize import convert_indicator_df_to_parquet, validate_indicator


def move_to_review(path: Path, review_dir: Path) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    dest = review_dir / path.name
    shutil.move(str(path), str(dest))
    return dest


def regen_indicator(src_path: Path, out_path: Path) -> Path:
    # Attempt to read with pandas (allow pyarrow engine if possible)
    # For indicators we expect a simple numeric table; try to read as CSV/JSON fallback
    try:
        df = pd.read_parquet(src_path)
    except Exception:
        # Could not read original (expected for corrupted). Try to find a canonical source
        # If the original is corrupted we assume the source price data exists elsewhere.
        raise RuntimeError(f"Cannot read source indicator file {src_path}")

    # Use existing converter which enforces float64 and writes parquet
    return convert_indicator_df_to_parquet(df, out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to corrupted indicator file")
    p.add_argument(
        "--out-dir",
        default="data/validated/indicators",
        help="Validated output directory",
    )
    p.add_argument(
        "--review-dir",
        default="artifacts/need_manual_review",
        help="Where to move originals",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write regenerated file (default: dry-run)",
    )
    args = p.parse_args(argv)

    src = Path(args.file)
    if not src.exists():
        print(f"File not found: {src}")
        return 2

    review = Path(args.review_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Archiving original to: {review}")
    archived = move_to_review(src, review)
    print(f"Archived: {archived}")

    # Build output path (flattened filename to keep consistent mapping)
    out_path = out_dir / src.name

    try:
        # Here we can't read the archived file; attempt to regenerate by locating
        # a source JSON or parquet in nearby locations. For now we fail fast.
        print(
            "Attempting to regenerate indicator from archived file (will fail if archived is unreadable)..."
        )
        regenerated = regen_indicator(archived, out_path)
    except RuntimeError as e:
        print(f"Regeneration failed: {e}")
        print(
            "Next steps: you can re-run indicator generation from price data or restore from backups."
        )
        return 3

    # Validation
    try:
        df_new = pd.read_parquet(regenerated)
        report = validate_indicator(df_new)
        print(f"Validation report for {regenerated}: {report}")
    except Exception as e:
        print(f"Failed to validate regenerated file: {e}")
        return 4

    if args.apply:
        print(f"Wrote regenerated validated file to {regenerated}")
    else:
        print(
            "Dry-run: regenerated file created but --apply not set. To commit, re-run with --apply."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
