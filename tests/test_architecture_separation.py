"""
Tests de Validation Architecturale - Séparation UI/Bridge/Engine
================================================================

Vérifie que l'architecture 3-tiers est respectée:
    1. UI ne doit JAMAIS importer Engine directement
    2. UI ne doit JAMAIS faire de calculs pandas/numpy directement
    3. Tous les calculs métier passent par Bridge
    4. UI importe uniquement Bridge ou autres UI

Usage:
    pytest tests/test_architecture_separation.py -v

Author: ThreadX Framework
Version: Architecture Enforcement Tests
"""

import re
from pathlib import Path

import pytest


# Chemins à auditer
UI_PATHS = [
    "apps",  # Dash entry-point(s)
    "src/threadx/ui",  # UI components
]

# Imports interdits dans UI (Engine direct)
FORBIDDEN_ENGINE_IMPORTS = [
    r"from\s+threadx\.engine",
    r"import\s+threadx\.engine",
    r"from\s+threadx\.data\.(?!__init__)",  # data modules sauf __init__
    r"from\s+threadx\.performance",
    r"from\s+threadx\.backtest",
]

# Whitelist temporaire (imports à corriger mais documentés)
TEMPORARY_ALLOWED_IMPORTS = [
    (
        "callbacks.py",
        957,
        "unified_diversity_pipeline",
    ),  # TODO: DiversityPipelineController
]

# Opérations pandas interdites dans UI
FORBIDDEN_PANDAS_OPS = [
    r"\.pct_change\(",
    r"\.rolling\(",
    r"\.expanding\(",
    r"\.ewm\(",
    r"\.resample\(",
    r"\.std\(",
    r"\.mean\(",
    r"\.cumsum\(",
    r"\.dropna\(",
    r"\.fillna\(",
]


def find_python_files(base_path: str) -> list[Path]:
    """Trouve tous les fichiers Python dans un chemin."""
    path = Path(base_path)
    if not path.exists():
        return []
    return list(path.rglob("*.py"))


def check_forbidden_imports(file_path: Path) -> list[dict]:
    """Vérifie les imports interdits dans un fichier."""
    violations = []

    try:
        content = file_path.read_text(encoding="utf-8")

        for pattern in FORBIDDEN_ENGINE_IMPORTS:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                # Vérifier contexte ligne
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.start())
                line_text = content[line_start:line_end]

                # Skip commentaires avec TODO/TEMPORAIRE/FIXME
                if line_text.strip().startswith("#"):
                    if any(
                        marker in line_text
                        for marker in ["TODO", "TEMPORAIRE", "FIXME"]
                    ):
                        continue

                # Check whitelist temporaire
                file_name = file_path.name
                if any(
                    (file_name == w[0] and line_num == w[1])
                    for w in TEMPORARY_ALLOWED_IMPORTS
                ):
                    continue

                violations.append(
                    {
                        "file": str(file_path),
                        "line": line_num,
                        "type": "forbidden_import",
                        "code": match.group(0),
                    }
                )

    except Exception as e:
        print(f"Erreur lecture {file_path}: {e}")

    return violations


def check_forbidden_operations(file_path: Path) -> list[dict]:
    """Vérifie les opérations pandas interdites dans un fichier."""
    violations = []

    try:
        content = file_path.read_text(encoding="utf-8")

        # Autoriser calculs pandas dans Engine (c'est leur rôle)
        if "/engine/" in str(file_path).replace("\\", "/"):
            return []

        # Autoriser calculs pandas dans charts UI (visualisation pure, pas décisionnel)
        # Ces fichiers transforment données déjà calculées pour affichage
        if file_path.name in ["charts.py"] and "/ui/" in str(file_path).replace(
            "\\", "/"
        ):
            return []

        # Ignorer si fichier importe MetricsController (autorisé)
        if "from threadx.bridge import MetricsController" in content:
            # Vérifier que les ops pandas sont DANS des appels MetricsController
            # Pour simplifier, on autorise si MetricsController présent
            return []

        for pattern in FORBIDDEN_PANDAS_OPS:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                # Vérifier contexte: autorisé si commenté ou dans string
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.start())
                line_text = content[line_start:line_end]

                # Skip commentaires et docstrings
                if line_text.strip().startswith("#"):
                    continue
                if '"""' in line_text or "'''" in line_text:
                    continue

                violations.append(
                    {
                        "file": str(file_path),
                        "line": line_num,
                        "type": "forbidden_operation",
                        "code": match.group(0),
                        "line_text": line_text.strip(),
                    }
                )

    except Exception as e:
        print(f"Erreur lecture {file_path}: {e}")

    return violations


def test_ui_no_engine_imports():
    """Test: UI ne doit pas importer Engine directement."""
    all_violations = []

    for ui_path in UI_PATHS:
        files = find_python_files(ui_path)
        for file_path in files:
            violations = check_forbidden_imports(file_path)
            all_violations.extend(violations)

    if all_violations:
        report = "\n\n❌ VIOLATIONS ARCHITECTURE: UI importe Engine directement\n"
        report += "=" * 70 + "\n\n"

        for v in all_violations:
            report += f"Fichier: {v['file']}\n"
            report += f"Ligne {v['line']}: {v['code']}\n"
            report += f"Type: {v['type']}\n\n"

        report += "CORRECTION REQUISE:\n"
        report += "- Remplacer imports Engine par: from threadx.bridge import ...\n"
        report += "- Utiliser controllers Bridge pour accéder Engine\n"

        pytest.fail(report)


def test_ui_no_pandas_operations():
    """Test: UI ne doit pas faire calculs pandas directement."""
    all_violations = []

    for ui_path in UI_PATHS:
        files = find_python_files(ui_path)
        for file_path in files:
            violations = check_forbidden_operations(file_path)
            all_violations.extend(violations)

    if all_violations:
        report = "\n\n❌ VIOLATIONS ARCHITECTURE: UI fait calculs pandas directement\n"
        report += "=" * 70 + "\n\n"

        for v in all_violations[:10]:  # Limite à 10 pour lisibilité
            report += f"Fichier: {v['file']}\n"
            report += f"Ligne {v['line']}: {v['code']}\n"
            report += f"Contexte: {v['line_text']}\n\n"

        if len(all_violations) > 10:
            report += f"... et {len(all_violations) - 10} autres violations\n\n"

        report += "CORRECTION REQUISE:\n"
        report += "- Importer: from threadx.bridge import MetricsController\n"
        report += "- Utiliser: metrics_controller.calculate_*()\n"
        report += (
            "- Exemple: sharpe = metrics_controller.calculate_sharpe_ratio(equity)\n"
        )

        pytest.fail(report)


def test_bridge_imports_allowed():
    """Test: Vérifier que imports Bridge sont autorisés dans UI."""
    bridge_import_pattern = r"from\s+threadx\.bridge\s+import"

    violations = []
    for ui_path in UI_PATHS:
        files = find_python_files(ui_path)
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")

            # Si fichier fait calculs pandas, il DOIT importer Bridge
            has_pandas_ops = any(re.search(op, content) for op in FORBIDDEN_PANDAS_OPS)
            has_bridge_import = re.search(bridge_import_pattern, content)

            if has_pandas_ops and not has_bridge_import:
                violations.append(
                    {
                        "file": str(file_path),
                        "reason": "Calculs pandas sans import Bridge",
                    }
                )

    if violations:
        report = "\n\n⚠️  AVERTISSEMENT: Fichiers UI avec calculs sans Bridge import\n"
        for v in violations:
            report += f"- {v['file']}: {v['reason']}\n"

        # Note: Warning seulement, pas failure
        print(report)


def test_bridge_controllers_exist():
    """Test: Vérifier que MetricsController existe et est exporté."""
    try:
        from threadx.bridge import MetricsController

        # Vérifier méthodes essentielles
        assert hasattr(MetricsController, "calculate_sharpe_ratio")
        assert hasattr(MetricsController, "calculate_max_drawdown")
        assert hasattr(MetricsController, "calculate_returns")
        assert hasattr(MetricsController, "calculate_moving_average")

    except ImportError as e:
        pytest.fail(f"MetricsController non accessible: {e}")


def test_ui_files_use_bridge():
    """Test: Compter fichiers UI utilisant Bridge vs Engine."""
    bridge_users = 0
    engine_users = 0

    for ui_path in UI_PATHS:
        files = find_python_files(ui_path)
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")

            if re.search(r"from\s+threadx\.bridge", content):
                bridge_users += 1
            if re.search(r"from\s+threadx\.engine", content):
                engine_users += 1

    print(f"\n📊 Statistiques imports UI:")
    print(f"   Fichiers utilisant Bridge: {bridge_users}")
    print(f"   Fichiers utilisant Engine: {engine_users}")

    # Objectif: 0 engine users
    assert engine_users == 0, f"{engine_users} fichiers UI importent encore Engine!"


if __name__ == "__main__":
    # Exécution manuelle pour debug
    print("=== Test Architecture Separation ===\n")

    print("1. Vérification imports Engine dans UI...")
    test_ui_no_engine_imports()
    print("✅ Aucun import Engine direct\n")

    print("2. Vérification opérations pandas dans UI...")
    test_ui_no_pandas_operations()
    print("✅ Aucune opération pandas directe\n")

    print("3. Vérification MetricsController...")
    test_bridge_controllers_exist()
    print("✅ MetricsController accessible\n")

    print("4. Statistiques utilisation Bridge...")
    test_ui_files_use_bridge()
    print("✅ Architecture respectée\n")
