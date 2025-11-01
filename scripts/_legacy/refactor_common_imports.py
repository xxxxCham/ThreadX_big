#!/usr/bin/env python3
"""
Script de refactorisation DRY - Phase 2 Step 3.1

Remplace les imports dupliqués par common_imports dans tous les fichiers.

Usage:
    python scripts/refactor_common_imports.py [--dry-run] [--files FILE1 FILE2 ...]
"""

import argparse
import re
from pathlib import Path
from typing import List, Set, Tuple


# Patterns d'imports à remplacer
PANDAS_PATTERN = r"^import pandas as pd\s*$"
NUMPY_PATTERN = r"^import numpy as np\s*$"
TYPING_PATTERN = r"^from typing import \(([^)]+)\)$"
LOGGER_PATTERN = r"^from threadx\.utils\.log import get_logger\s*$"
LOGGER_CREATION = r"^logger = get_logger\(__name__\)\s*$"


def find_python_files(root: Path) -> List[Path]:
    """Trouve tous les fichiers Python dans src/threadx/"""
    exclude_dirs = {"__pycache__", ".pytest_cache", "tests", ".git", "venv"}
    files = []

    for path in root.rglob("*.py"):
        if any(excl in path.parts for excl in exclude_dirs):
            continue
        if path.name == "common_imports.py":  # Skip le module lui-même
            continue
        files.append(path)

    return files


def analyze_file(filepath: Path) -> Tuple[bool, Set[str]]:
    """
    Analyse un fichier pour voir s'il a des imports à refactoriser.

    Returns:
        (needs_refactor, imports_found)
    """
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")

    imports_found = set()

    for line in lines:
        if re.match(PANDAS_PATTERN, line.strip()):
            imports_found.add("pd")
        if re.match(NUMPY_PATTERN, line.strip()):
            imports_found.add("np")
        if re.match(TYPING_PATTERN, line.strip()):
            imports_found.add("typing")
        if re.match(LOGGER_PATTERN, line.strip()):
            imports_found.add("logger")

    return len(imports_found) > 0, imports_found


def refactor_file(filepath: Path, dry_run: bool = False) -> bool:
    """
    Refactorise un fichier pour utiliser common_imports.

    Returns:
        True si des changements ont été faits
    """
    content = filepath.read_text(encoding="utf-8")
    original_content = content
    lines = content.split("\n")

    # 1. Identifier les imports existants
    has_pandas = any(re.match(PANDAS_PATTERN, line.strip()) for line in lines)
    has_numpy = any(re.match(NUMPY_PATTERN, line.strip()) for line in lines)
    has_logger = any(re.match(LOGGER_PATTERN, line.strip()) for line in lines)

    # Extract typing imports
    typing_imports = []
    for line in lines:
        match = re.match(TYPING_PATTERN, line.strip())
        if match:
            typing_imports.append(match.group(1).strip())

    # 2. Construire le nouvel import
    new_imports = []
    if has_pandas:
        new_imports.append("pd")
    if has_numpy:
        new_imports.append("np")
    if typing_imports:
        # Parse typing imports
        all_typing = []
        for imp in typing_imports:
            all_typing.extend([t.strip() for t in imp.split(",")])
        new_imports.extend(all_typing)
    if has_logger:
        new_imports.append("create_logger")

    if not new_imports:
        return False  # Rien à refactoriser

    # 3. Créer la ligne d'import
    import_line = f"from threadx.utils.common_imports import (\n"
    import_line += ",\n".join(f"    {imp}" for imp in new_imports)
    import_line += ",\n)"

    # 4. Remplacer les imports
    new_lines = []
    skip_next = False
    import_added = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Skip pandas import
        if re.match(PANDAS_PATTERN, line.strip()):
            if not import_added:
                new_lines.append(import_line)
                import_added = True
            continue

        # Skip numpy import
        if re.match(NUMPY_PATTERN, line.strip()):
            if not import_added:
                new_lines.append(import_line)
                import_added = True
            continue

        # Skip typing import (peut être multilignes)
        if re.match(r"^from typing import \(", line.strip()):
            # Find closing parenthesis
            j = i
            while j < len(lines) and ")" not in lines[j]:
                j += 1
            # Skip all these lines
            for k in range(i, j + 1):
                if k == i and not import_added:
                    new_lines.append(import_line)
                    import_added = True
            # Move to after closing paren
            skip_lines = j - i
            for _ in range(skip_lines):
                if i + 1 < len(lines):
                    lines.pop(i + 1)
            continue

        # Skip logger import
        if re.match(LOGGER_PATTERN, line.strip()):
            if not import_added:
                new_lines.append(import_line)
                import_added = True
            continue

        # Replace logger creation
        if re.match(LOGGER_CREATION, line.strip()):
            new_lines.append("logger = create_logger(__name__)")
            continue

        new_lines.append(line)

    new_content = "\n".join(new_lines)

    if new_content == original_content:
        return False

    if not dry_run:
        filepath.write_text(new_content, encoding="utf-8")
        print(f"✅ Refactorisé: {filepath.relative_to(Path.cwd())}")
    else:
        print(f"🔍 À refactoriser: {filepath.relative_to(Path.cwd())}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Refactorisation DRY - common_imports")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans modifier")
    parser.add_argument("--files", nargs="+", help="Fichiers spécifiques à traiter")
    args = parser.parse_args()

    root = Path("src/threadx")

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = find_python_files(root)

    print(f"🔍 Analyse de {len(files)} fichiers...")

    needs_refactor = []
    for filepath in files:
        needs, imports = analyze_file(filepath)
        if needs:
            needs_refactor.append((filepath, imports))

    print(f"\n📊 {len(needs_refactor)} fichiers à refactoriser")

    if args.dry_run:
        print("\n🔍 Mode dry-run - Aucun fichier ne sera modifié\n")
    else:
        print("\n✏️ Refactorisation en cours...\n")

    refactored_count = 0
    for filepath, imports in needs_refactor:
        if refactor_file(filepath, dry_run=args.dry_run):
            refactored_count += 1

    print(
        f"\n✅ Terminé: {refactored_count} fichiers {'analysés' if args.dry_run else 'refactorisés'}"
    )


if __name__ == "__main__":
    main()
