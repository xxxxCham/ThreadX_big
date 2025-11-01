"""Surveillance simple des suppressions dans le dossier `data/`.

Fonctionnement :
- Effectue un snapshot récursif des chemins (fichiers) sous `data/` au démarrage.
- Toutes les X secondes, refait un snapshot et détecte les fichiers disparus.
- Log non destructif dans `artifacts/delete_watch.log` si des suppressions sont détectées.

Utilisation :
    python scripts/watch_deletes.py --seconds 30 --interval 2

Ce script évite les dépendances externes et est sûr à exécuter.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Set


def snapshot_files(root: Path) -> Set[str]:
    files = set()
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file():
            try:
                files.add(str(p.resolve()))
            except Exception:
                # ignore files that can't be resolved
                files.add(str(p))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seconds", type=int, default=30, help="Durée totale de surveillance"
    )
    p.add_argument(
        "--interval", type=float, default=2.0, help="Intervalle de sondage (s)"
    )
    p.add_argument(
        "--root", type=str, default="data", help="Racine à surveiller (relatif au repo)"
    )
    args = p.parse_args()

    root = Path(args.root)
    out_log = Path("artifacts") / "delete_watch.log"
    out_log.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Watching deletions under: {root.resolve()} for {args.seconds}s (interval {args.interval}s)"
    )
    initial = snapshot_files(root)
    print(f"Initial files: {len(initial)}")

    start = time.time()
    last_seen = initial

    with out_log.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n--- Watch started {datetime.now().isoformat()} root={root} seconds={args.seconds} interval={args.interval}\n"
        )

        while time.time() - start < args.seconds:
            time.sleep(args.interval)
            current = snapshot_files(root)
            removed = last_seen - current
            if removed:
                ts = datetime.now().isoformat()
                for r in sorted(removed):
                    msg = f"{ts} REMOVED {r}\n"
                    fh.write(msg)
                    fh.flush()
                    print(msg.strip())
            last_seen = current

        fh.write(f"--- Watch ended {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
