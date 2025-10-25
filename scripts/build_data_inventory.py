"""Scan D:/agent_data (or provided root) and produce docs/DATA_INVENTORY.md

Usage: python scripts/build_data_inventory.py --root D:/agent_data
"""

from pathlib import Path
import argparse


def scan(root: Path):
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            out.append((str(rel), p.stat().st_size))
    return out


def build_md(entries, root: Path, outpath: Path):
    lines = [f"# Data inventory for {root}", "", "| path | size (bytes) |", "|---|---|"]
    for path, size in entries:
        lines.append(f"| {path} | {size} |")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:/agent_data")
    parser.add_argument("--out", default="docs/DATA_INVENTORY.md")
    args = parser.parse_args()
    root = Path(args.root)
    entries = scan(root)
    build_md(entries, root, Path(args.out))


if __name__ == "__main__":
    main()
