#!/usr/bin/env python3
"""
Checklist Développement ThreadX - Script Validation
====================================================

Usage:
  python3 check_directives.py                  # Vérifier structure repo
  python3 check_directives.py --fix            # Essayer auto-fix
  python3 check_directives.py --help           # Aide

Valide que repo respecte DIRECTIVES_DEV.md
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).parent
CRITICAL_FILES = {
    "DIRECTIVES_DEV.md": "Instructions obligatoires pour LLMs",
    "README.md": "Documentation principale",
    ".llmrc": "Instructions LLM",
    "requirements.txt": "Dépendances Python",
}

FORBIDDEN_FILES_PATTERNS = [
    "GUIDE_*.md",  # Guides dispersés
    "TUTORIAL_*.md",
    "*_INSTRUCTIONS.md",
    "*FRICTION*.md",  # Frictions doivent être dans DIRECTIVES_DEV.md
    "*EXECUTION*.py",  # realistic_execution.py interdit (dans engine.py)
]

REQUIRED_MODULES = [
    "src/threadx/backtest/engine.py",
    "src/threadx/indicators/",
    "src/threadx/strategy/",
    "src/threadx/utils/log.py",
]

# ============================================================================
# CHECKS
# ============================================================================


def check_critical_files() -> Tuple[bool, List[str]]:
    """Vérifier que fichiers critiques existent."""
    errors = []

    for filename, description in CRITICAL_FILES.items():
        filepath = REPO_ROOT / filename
        if not filepath.exists():
            errors.append(f"❌ MANQUANT: {filename} - {description}")
        else:
            size_kb = filepath.stat().st_size / 1024
            print(f"✅ {filename} ({size_kb:.0f}KB)")

    return len(errors) == 0, errors


def check_forbidden_patterns() -> Tuple[bool, List[str]]:
    """Vérifier qu'aucun fichier interdit n'existe."""
    errors = []

    for pattern in FORBIDDEN_FILES_PATTERNS:
        matches = list(REPO_ROOT.glob(f"**/{pattern}"))
        if matches:
            for match in matches:
                # Ignorer certains chemins
                if "__pycache__" in str(match) or ".git" in str(match):
                    continue
                relative = match.relative_to(REPO_ROOT)
                errors.append(f"❌ INTERDIT: {relative} (selon DIRECTIVES_DEV.md)")
                print(f"   Devrait être intégré à fichier existant")

    return len(errors) == 0, errors


def check_python_quality() -> Tuple[bool, List[str], List[str]]:
    """Vérifier qualité code Python."""
    errors: List[str] = []
    warnings: List[str] = []

    # Chercher fichiers Python sans type hints
    for pyfile in REPO_ROOT.glob("src/threadx/**/*.py"):
        if "__pycache__" in str(pyfile) or "_archive" in str(pyfile):
            continue

        content = pyfile.read_text()

        # Check type hints (pas strict, juste warn)
        if "def " in content and "->" not in content:
            warnings.append(f"⚠️ {pyfile.relative_to(REPO_ROOT)}: Ajouter type hints")

        # Check logging
        if "print(" in content and "threadx/config.py" not in str(pyfile):
            warnings.append(
                f"⚠️ {pyfile.relative_to(REPO_ROOT)}: "
                f"Utiliser logger.info() au lieu de print()"
            )

    return len(errors) == 0, errors, warnings


def check_module_structure() -> Tuple[bool, List[str]]:
    """Vérifier structure modules."""
    errors = []

    for module in REQUIRED_MODULES:
        path = REPO_ROOT / module
        if not path.exists():
            errors.append(f"❌ MANQUANT: {module}")
        else:
            print(f"✅ {module}")

    return len(errors) == 0, errors


def check_directives_updated() -> Tuple[bool, List[str]]:
    """Vérifier que DIRECTIVES_DEV.md a été mis à jour récemment."""
    directives = REPO_ROOT / "DIRECTIVES_DEV.md"

    if not directives.exists():
        return False, ["DIRECTIVES_DEV.md manquant"]

    # Check contenu clé
    content = directives.read_text()
    required_sections = [
        "PRINCIPE FONDAMENTAL",
        "RÈGLES DE CONSOLIDATION CODE",
        "ARCHITECTURE GÉNÉRALE",
        "CONVENTIONS DE NOMMAGE",
        "STACK TECHNOLOGIQUE",
        "FRICTIONS RÉALISTES",
        "NETDATA MCP BRIDGE",
    ]

    missing = []
    for section in required_sections:
        if section not in content:
            missing.append(f"Section '{section}' manquante")

    if missing:
        return False, missing

    return True, []


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run all checks."""
    print("=" * 70)
    print("🔍 CHECKLIST DIRECTIVES_DEV.md - ThreadX")
    print("=" * 70)
    print()

    all_pass = True
    all_errors = []

    # 1. Critical files
    print("1️⃣  FICHIERS CRITIQUES")
    print("-" * 70)
    ok, errors = check_critical_files()
    all_pass = all_pass and ok
    all_errors.extend(errors)
    print()

    # 2. Forbidden patterns
    print("2️⃣  FICHIERS INTERDITS (dispersés)")
    print("-" * 70)
    ok, errors = check_forbidden_patterns()
    if ok:
        print("✅ Aucun fichier interdit trouvé")
    all_pass = all_pass and ok
    all_errors.extend(errors)
    print()

    # 3. Module structure
    print("3️⃣  STRUCTURE MODULES")
    print("-" * 70)
    ok, errors = check_module_structure()
    all_pass = all_pass and ok
    all_errors.extend(errors)
    print()

    # 4. Directives content
    print("4️⃣  CONTENU DIRECTIVES_DEV.md")
    print("-" * 70)
    ok, errors = check_directives_updated()
    if ok:
        print("✅ Toutes les sections requises sont présentes")
    all_pass = all_pass and ok
    all_errors.extend(errors)
    print()

    # 5. Python quality
    print("5️⃣  QUALITÉ CODE PYTHON")
    print("-" * 70)
    ok, errors, warnings = check_python_quality()
    for w in warnings[:5]:  # Afficher max 5
        print(w)
    if not warnings:
        print("✅ Qualité OK")
    print()

    # Summary
    print("=" * 70)
    if all_pass and not all_errors:
        print("✅ TOUS LES CHECKS PASSENT - Repository OK!")
        print("=" * 70)
        return 0
    else:
        print("❌ ERREURS DÉTECTÉES:")
        print("=" * 70)
        for error in all_errors:
            print(f"  {error}")
        print()
        print("💡 Consulter DIRECTIVES_DEV.md pour les fixes")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
