#!/usr/bin/env python3
"""
Examine en détail chaque module isolé pour décider : garder ou supprimer.

Pour chaque module isolé, on vérifie :
1. Taille du fichier (LOC)
2. Fonctions/classes exportées
3. Commentaires/docstrings expliquant l'usage
4. Potentiel d'utilisation future
"""

import json
from pathlib import Path
import ast

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "threadx"


def count_loc(file_path: Path) -> int:
    """Compte les lignes de code (sans blanks/comments)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
    except:
        return 0


def get_exports(file_path: Path) -> dict:
    """Extrait les classes/fonctions exportées."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        exports = {
            'classes': [],
            'functions': [],
            'constants': []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                exports['classes'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Skip private
                    exports['functions'].append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        exports['constants'].append(target.id)

        return exports
    except:
        return {'classes': [], 'functions': [], 'constants': []}


def get_module_docstring(file_path: Path) -> str:
    """Extrait la docstring du module."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        return ast.get_docstring(tree) or "Pas de docstring"
    except:
        return "Erreur lecture"


def analyze_module(module_name: str, file_path: Path) -> dict:
    """Analyse complète d'un module."""
    loc = count_loc(file_path)
    exports = get_exports(file_path)
    docstring = get_module_docstring(file_path)

    return {
        'module': module_name,
        'file': str(file_path.relative_to(PROJECT_ROOT)),
        'loc': loc,
        'exports': exports,
        'docstring': docstring[:200] + ('...' if len(docstring) > 200 else '')
    }


def categorize_module(module_name: str, analysis: dict) -> str:
    """
    Catégorise un module isolé :
    - ENTRY_POINT : Point d'entrée (garder)
    - ARCHIVE : Archive volontaire (garder)
    - DEAD_CODE : Code mort confirmé (supprimer)
    - TO_EXAMINE : À examiner manuellement
    """
    module = module_name.lower()

    # Points d'entrée
    if 'streamlit_app' in module or module == 'threadx':
        return 'ENTRY_POINT'

    # Archives
    if '_archive' in module:
        return 'ARCHIVE'

    # Code mort évident (CLI, etc.)
    if 'optimization.run' in module:
        return 'DEAD_CODE'  # CLI entry point (CLI supprimé)

    # Modules vides ou très petits
    if analysis['loc'] < 10:
        return 'DEAD_CODE'

    # Modules avec peu d'exports et peu de LOC
    total_exports = sum(len(v) for v in analysis['exports'].values())
    if analysis['loc'] < 50 and total_exports == 0:
        return 'DEAD_CODE'

    return 'TO_EXAMINE'


def main():
    """Analyse tous les modules isolés."""
    # Charge le rapport JSON
    report_path = PROJECT_ROOT / 'module_dependency_analysis.json'
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    isolated = []
    for category, modules in report['isolated_modules']['by_category'].items():
        for module in modules:
            # Trouve le fichier correspondant
            module_parts = module.replace('threadx.', '').split('.')
            possible_paths = [
                SRC_ROOT / '/'.join(module_parts) / '__init__.py',
                SRC_ROOT / ('/'.join(module_parts) + '.py')
            ]

            file_path = None
            for p in possible_paths:
                if p.exists():
                    file_path = p
                    break

            if file_path:
                isolated.append((module, file_path))

    print("=" * 80)
    print("🔍 ANALYSE DÉTAILLÉE DES MODULES ISOLÉS")
    print("=" * 80)

    results_by_category = {
        'ENTRY_POINT': [],
        'ARCHIVE': [],
        'DEAD_CODE': [],
        'TO_EXAMINE': []
    }

    for module, file_path in isolated:
        analysis = analyze_module(module, file_path)
        category = categorize_module(module, analysis)
        results_by_category[category].append((module, analysis))

    # Affichage par catégorie
    for category, items in results_by_category.items():
        if not items:
            continue

        emoji = {
            'ENTRY_POINT': '✅',
            'ARCHIVE': '📦',
            'DEAD_CODE': '❌',
            'TO_EXAMINE': '🔍'
        }[category]

        print(f"\n{emoji} {category} ({len(items)} modules)")
        print("-" * 80)

        for module, analysis in items:
            print(f"\n  Module: {module}")
            print(f"  Fichier: {analysis['file']}")
            print(f"  LOC: {analysis['loc']}")
            print(f"  Exports: {sum(len(v) for v in analysis['exports'].values())} " +
                  f"({len(analysis['exports']['classes'])} classes, " +
                  f"{len(analysis['exports']['functions'])} fonctions)")

            if analysis['exports']['classes']:
                print(f"    Classes: {', '.join(analysis['exports']['classes'][:5])}")
            if analysis['exports']['functions']:
                print(f"    Fonctions: {', '.join(analysis['exports']['functions'][:5])}")

            if category == 'TO_EXAMINE':
                print(f"  Docstring: {analysis['docstring']}")

    # Résumé
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ")
    print("=" * 80)

    total_dead = len(results_by_category['DEAD_CODE'])
    total_to_examine = len(results_by_category['TO_EXAMINE'])

    print(f"\n✅ Points d'entrée (à garder): {len(results_by_category['ENTRY_POINT'])}")
    print(f"📦 Archives (à garder): {len(results_by_category['ARCHIVE'])}")
    print(f"❌ Code mort confirmé (à supprimer): {total_dead}")
    print(f"🔍 À examiner manuellement: {total_to_examine}")

    if total_dead > 0:
        print(f"\n🗑️  FICHIERS À SUPPRIMER ({total_dead}):")
        for module, analysis in results_by_category['DEAD_CODE']:
            print(f"  - {analysis['file']}")

    # Sauvegarde rapport détaillé
    detailed_report = {
        category: [
            {**analysis, 'module': module}
            for module, analysis in items
        ]
        for category, items in results_by_category.items()
    }

    output = PROJECT_ROOT / 'isolated_modules_analysis.json'
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Rapport détaillé: {output}")


if __name__ == '__main__':
    main()
