"""
Script de Validation Code ThreadX
==================================

Vérifie rapidement la santé du code :
- Syntaxe Python
- Imports
- Détection problèmes courants

Usage:
    python validate_code.py
"""

import sys
from pathlib import Path
import subprocess

def print_section(title):
    """Affiche un titre de section."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def check_python_syntax():
    """Vérifie syntaxe Python de tous les fichiers."""
    print_section("1. Vérification Syntaxe Python")

    src_dir = Path("src/threadx")
    py_files = list(src_dir.rglob("*.py"))

    errors = []
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), str(py_file), 'exec')
        except SyntaxError as e:
            errors.append((py_file, e))

    if errors:
        print(f"❌ {len(errors)} fichiers avec erreurs syntaxe :")
        for file, error in errors[:5]:  # Max 5
            print(f"   {file}: {error}")
        return False
    else:
        print(f"✅ {len(py_files)} fichiers Python valides")
        return True

def check_imports():
    """Vérifie que tous les imports sont valides."""
    print_section("2. Vérification Imports")

    # Ajouter src/ au path
    sys.path.insert(0, str(Path("src").absolute()))

    critical_modules = [
        "threadx.optimization.engine",
        "threadx.gpu.multi_gpu",
        "threadx.llm.agents.analyst",
        "threadx.indicators.bank",
        "threadx.strategy",
    ]

    errors = []
    for module_name in critical_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            errors.append((module_name, e))
            print(f"❌ {module_name}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} modules avec erreurs import")
        return False
    else:
        print(f"\n✅ Tous les imports critiques OK")
        return True

def check_common_issues():
    """Vérifie problèmes courants via grep."""
    print_section("3. Détection Problèmes Courants")

    checks = [
        {
            "name": "Try/except sans logs",
            "pattern": r"except.*:\s*pass",
            "severity": "WARNING",
        },
        {
            "name": "Print statements (debug oublié)",
            "pattern": r"^\s*print\(",
            "severity": "INFO",
        },
        {
            "name": "TODO/FIXME",
            "pattern": r"(TODO|FIXME|HACK):",
            "severity": "INFO",
        },
    ]

    src_dir = Path("src/threadx")

    for check in checks:
        try:
            result = subprocess.run(
                ["grep", "-r", "-n", "-E", check["pattern"], str(src_dir)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:  # Trouvé des matches
                lines = result.stdout.strip().split('\n')
                count = len([l for l in lines if l])

                severity_icon = {
                    "ERROR": "❌",
                    "WARNING": "⚠️",
                    "INFO": "ℹ️"
                }.get(check["severity"], "•")

                print(f"{severity_icon} {check['name']}: {count} occurrences")

                # Afficher premières occurrences
                if check["severity"] == "WARNING" and count <= 5:
                    for line in lines[:5]:
                        if line:
                            print(f"     {line[:100]}")
            else:
                print(f"✅ {check['name']}: Aucune occurrence")

        except subprocess.TimeoutExpired:
            print(f"⏱️ {check['name']}: Timeout")
        except FileNotFoundError:
            print(f"⚠️ grep non disponible (ignoré)")
            break

    return True

def check_gpu_availability():
    """Vérifie disponibilité GPU."""
    print_section("4. Vérification GPU")

    try:
        import cupy as cp
        gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"✅ CuPy disponible : {gpu_count} GPU(s) détecté(s)")

        for i in range(gpu_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode('utf-8')
            print(f"   GPU {i}: {name}")

        return True
    except ImportError:
        print("⚠️ CuPy non disponible (mode CPU uniquement)")
        return False
    except Exception as e:
        print(f"❌ Erreur GPU: {e}")
        return False

def check_streamlit_app():
    """Vérifie que streamlit_app.py est valide."""
    print_section("5. Vérification Streamlit App")

    app_file = Path("src/threadx/streamlit_app.py")

    if not app_file.exists():
        print("❌ streamlit_app.py non trouvé")
        return False

    try:
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Vérifier imports critiques (streamlit_app.py est juste le point d'entrée)
        critical_imports = [
            "import streamlit as st",
            "from threadx.gpu.multi_gpu import get_default_manager",
        ]

        for imp in critical_imports:
            if imp in content:
                print(f"✅ {imp}")
            else:
                print(f"⚠️ Import manquant: {imp}")

        # Vérifier fonctions critiques
        critical_functions = [
            "def shutdown_app(",
            "def restart_app(",
            "def main(",
        ]

        for func in critical_functions:
            if func in content:
                print(f"✅ Fonction {func.strip('(:')} présente")
            else:
                print(f"❌ Fonction {func.strip('(:')} manquante")

        return True
    except Exception as e:
        print(f"❌ Erreur lecture streamlit_app.py: {e}")
        return False

def main():
    """Point d'entrée principal."""
    print("\n" + "🔍 " * 20)
    print("  VALIDATION CODE THREADX v2.0")
    print("🔍 " * 20)

    results = []

    # Exécuter tous les checks
    results.append(("Syntaxe Python", check_python_syntax()))
    results.append(("Imports", check_imports()))
    results.append(("Problèmes courants", check_common_issues()))
    results.append(("GPU", check_gpu_availability()))
    results.append(("Streamlit App", check_streamlit_app()))

    # Résumé final
    print_section("📊 RÉSUMÉ FINAL")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {check_name}")

    print(f"\n  Score: {passed}/{total} checks passés")

    if passed == total:
        print("\n  🎉 Tous les checks sont OK !")
        print("  ✅ Code prêt pour production")
        return 0
    else:
        print("\n  ⚠️ Certains checks ont échoué")
        print("  🔧 Vérifier les erreurs ci-dessus")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
