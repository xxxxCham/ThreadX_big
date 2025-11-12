#!/usr/bin/env python3
"""
Analyse complète des dépendances entre modules ThreadX.

Ce script :
1. Liste tous les modules Python dans src/threadx/
2. Analyse tous les imports (internes threadx)
3. Identifie les modules isolés (jamais importés)
4. Génère un rapport détaillé avec graphe de dépendances

Usage:
    python tools/analyze_module_dependencies.py
"""

import ast
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "threadx"


class ImportAnalyzer(ast.NodeVisitor):
    """Analyse les imports d'un fichier Python."""

    def __init__(self, module_path: str):
        self.module_path = module_path
        self.imports: Set[str] = set()

    def visit_Import(self, node):
        """Visite import xxx"""
        for alias in node.names:
            if alias.name.startswith('threadx.'):
                self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visite from xxx import yyy"""
        if node.module and node.module.startswith('threadx.'):
            self.imports.add(node.module)
        elif node.level > 0:  # Relative import
            # Reconstruit le module absolu
            parts = self.module_path.replace('\\', '/').split('/')
            src_idx = parts.index('threadx')
            current_module_parts = parts[src_idx:-1]  # Sans le .py

            # Remonte selon le niveau
            for _ in range(node.level - 1):
                if current_module_parts:
                    current_module_parts.pop()

            # Ajoute le module importé
            if node.module:
                full_module = '.'.join(current_module_parts + [node.module])
            else:
                full_module = '.'.join(current_module_parts)

            self.imports.add(full_module)

        self.generic_visit(node)


def get_module_name(file_path: Path, src_root: Path) -> str:
    """Convertit un chemin de fichier en nom de module."""
    # src_root pointe déjà vers src/threadx/
    rel_path = file_path.relative_to(src_root)

    # Si c'est __init__.py, on prend le dossier parent
    if rel_path.stem == '__init__':
        parts = list(rel_path.parts[:-1])
    else:
        # Sinon on prend le fichier sans .py
        parts = list(rel_path.parts)
        if parts and parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]

    # Construit le nom du module
    if not parts:
        return 'threadx'
    return 'threadx.' + '.'.join(parts)


def analyze_file(file_path: Path, src_root: Path) -> Tuple[str, Set[str]]:
    """Analyse un fichier Python et retourne (nom_module, set_imports)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        module_name = get_module_name(file_path, src_root)
        analyzer = ImportAnalyzer(str(file_path))
        analyzer.visit(tree)

        return module_name, analyzer.imports
    except SyntaxError as e:
        print(f"⚠️  Syntax error in {file_path}: {e}")
        return get_module_name(file_path, src_root), set()
    except Exception as e:
        print(f"⚠️  Error analyzing {file_path}: {e}")
        return get_module_name(file_path, src_root), set()


def find_all_modules(src_root: Path) -> List[Path]:
    """Trouve tous les fichiers .py dans src/threadx/."""
    return list(src_root.rglob('*.py'))


def build_dependency_graph() -> Tuple[Dict[str, Set[str]], Dict[str, Path]]:
    """
    Construit le graphe de dépendances.

    Returns:
        (dependencies, module_paths)
        - dependencies: {module: set(modules_importés)}
        - module_paths: {module: chemin_fichier}
    """
    print(f"🔍 Analyse des modules dans {SRC_ROOT}")

    all_files = find_all_modules(SRC_ROOT)
    print(f"📁 {len(all_files)} fichiers Python trouvés")

    dependencies: Dict[str, Set[str]] = {}
    module_paths: Dict[str, Path] = {}

    for file_path in all_files:
        module_name, imports = analyze_file(file_path, SRC_ROOT)
        dependencies[module_name] = imports
        module_paths[module_name] = file_path

    return dependencies, module_paths


def get_top_level_modules(dependencies: Dict[str, Set[str]]) -> Set[str]:
    """Extrait les modules top-level (ex: threadx.backtest)."""
    top_level = set()
    for module in dependencies.keys():
        parts = module.split('.')
        if len(parts) >= 2:
            top_level.add('.'.join(parts[:2]))
    return top_level


def find_isolated_modules(dependencies: Dict[str, Set[str]]) -> Set[str]:
    """Trouve les modules jamais importés par d'autres."""
    all_modules = set(dependencies.keys())
    imported_modules = set()

    for imports in dependencies.values():
        imported_modules.update(imports)
        # Ajoute aussi les modules parents
        for imp in imports:
            parts = imp.split('.')
            for i in range(2, len(parts) + 1):
                imported_modules.add('.'.join(parts[:i]))

    isolated = all_modules - imported_modules

    # Filtre les __init__.py qui sont normaux
    isolated_filtered = {
        m for m in isolated
        if not m.endswith('.__init__')
    }

    return isolated_filtered


def categorize_by_top_level(modules: Set[str]) -> Dict[str, List[str]]:
    """Catégorise les modules par top-level."""
    categorized = defaultdict(list)
    for module in sorted(modules):
        parts = module.split('.')
        if len(parts) >= 2:
            top = parts[1]
            categorized[top].append(module)
        else:
            categorized['root'].append(module)
    return dict(categorized)


def count_importers(module: str, dependencies: Dict[str, Set[str]]) -> int:
    """Compte combien de modules importent ce module."""
    count = 0
    module_prefix = module + '.'
    for imports in dependencies.values():
        for imp in imports:
            if imp == module or imp.startswith(module_prefix):
                count += 1
                break
    return count


def generate_report(dependencies: Dict[str, Set[str]], module_paths: Dict[str, Path]):
    """Génère un rapport détaillé."""
    print("\n" + "=" * 80)
    print("📊 RAPPORT D'ANALYSE DES DÉPENDANCES")
    print("=" * 80)

    # Top-level modules
    top_level = get_top_level_modules(dependencies)
    print(f"\n🎯 Modules top-level détectés ({len(top_level)}):")
    for module in sorted(top_level):
        file_count = sum(1 for m in dependencies.keys() if m.startswith(module + '.') or m == module)
        importers = count_importers(module, dependencies)
        status = "✅ ACTIF" if importers > 0 else "⚠️  ISOLÉ"
        print(f"  {status} {module} ({file_count} fichiers, {importers} importations)")

    # Modules isolés
    isolated = find_isolated_modules(dependencies)
    print(f"\n🔍 Modules isolés ({len(isolated)}):")
    categorized = categorize_by_top_level(isolated)

    for top, modules in sorted(categorized.items()):
        print(f"\n  📦 {top}/ ({len(modules)} modules isolés):")
        for module in modules:
            path = module_paths.get(module, Path("unknown"))
            rel_path = path.relative_to(PROJECT_ROOT) if path != Path("unknown") else "unknown"
            print(f"    ❌ {module}")
            print(f"       {rel_path}")

    # Modules les plus importés
    print(f"\n📈 Top 10 modules les plus importés:")
    import_counts = {}
    for module in dependencies.keys():
        if not module.endswith('.__init__'):
            import_counts[module] = count_importers(module, dependencies)

    top_imported = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for module, count in top_imported:
        print(f"  ✅ {module}: {count} importations")

    # Sauvegarde JSON
    report = {
        'total_modules': len(dependencies),
        'top_level_modules': sorted(top_level),
        'isolated_modules': {
            'count': len(isolated),
            'by_category': {k: sorted(v) for k, v in categorized.items()}
        },
        'top_imported': [{'module': m, 'count': c} for m, c in top_imported],
        'dependencies': {k: sorted(v) for k, v in dependencies.items()}
    }

    output_file = PROJECT_ROOT / 'module_dependency_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Rapport JSON sauvegardé: {output_file}")

    return isolated, categorized


def main():
    """Point d'entrée principal."""
    print("🚀 Analyse des dépendances ThreadX")
    print(f"📂 Racine: {PROJECT_ROOT}")

    dependencies, module_paths = build_dependency_graph()
    isolated, categorized = generate_report(dependencies, module_paths)

    print("\n" + "=" * 80)
    print("✅ Analyse terminée !")
    print("=" * 80)

    if isolated:
        print(f"\n⚠️  {len(isolated)} modules isolés détectés - vérification manuelle nécessaire")
        return 1
    else:
        print("\n✅ Aucun module isolé - tous les modules sont utilisés !")
        return 0


if __name__ == '__main__':
    sys.exit(main())
