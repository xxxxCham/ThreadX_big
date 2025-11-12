"""
Analyse d'accès et de communication entre les fichiers ThreadX
Génère un rapport détaillé sur :
- Les imports et dépendances inter-modules
- Les fonctions/classes utilisées vs non utilisées
- Le graphe de communication entre fichiers
- Le code potentiellement mort
"""

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Ajout du path pour imports ThreadX
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class CodeAnalyzer:
    """Analyseur de code pour détecter les communications et l'utilisation"""

    def __init__(self, src_path: Path):
        self.src_path = src_path
        self.python_files: List[Path] = []
        self.imports_graph: Dict[str, Set[str]] = defaultdict(set)
        self.definitions: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"classes": [], "functions": []}
        )
        self.usages: Dict[str, Set[str]] = defaultdict(set)
        self.file_imports: Dict[str, List[Dict[str, str]]] = {}

    def scan_files(self) -> None:
        """Scanne tous les fichiers Python du projet"""
        self.python_files = list(self.src_path.rglob("*.py"))
        print(f"📁 Trouvé {len(self.python_files)} fichiers Python")

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyse un fichier Python"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Extraire les imports
            imports = self._extract_imports(tree, file_path)

            # Extraire les définitions
            definitions = self._extract_definitions(tree)

            # Extraire les usages
            usages = self._extract_usages(tree)

            return {
                "imports": imports,
                "definitions": definitions,
                "usages": usages,
                "loc": len(content.splitlines()),
            }

        except Exception as e:
            print(f"⚠️  Erreur lors de l'analyse de {file_path}: {e}")
            return {
                "imports": [],
                "definitions": {"classes": [], "functions": []},
                "usages": [],
                "loc": 0,
            }

    def _extract_imports(self, tree: ast.AST, file_path: Path) -> List[Dict[str, str]]:
        """Extrait tous les imports d'un fichier"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "name": alias.asname or alias.name,
                            "line": node.lineno,
                        }
                    )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )

        return imports

    def _extract_definitions(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extrait toutes les définitions de classes et fonctions"""
        definitions = {"classes": [], "functions": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                definitions["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                # Ignorer les méthodes privées (commençant par _)
                if not node.name.startswith("_"):
                    definitions["functions"].append(node.name)

        return definitions

    def _extract_usages(self, tree: ast.AST) -> List[str]:
        """Extrait tous les noms utilisés dans le code"""
        usages = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                usages.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Pour les appels comme module.function()
                if isinstance(node.value, ast.Name):
                    usages.add(f"{node.value.id}.{node.attr}")

        return list(usages)

    def build_dependency_graph(self) -> None:
        """Construit le graphe de dépendances entre modules"""
        for file_path in self.python_files:
            relative_path = file_path.relative_to(self.src_path)
            module_name = (
                str(relative_path)
                .replace("\\", ".")
                .replace("/", ".")
                .replace(".py", "")
            )

            analysis = self.analyze_file(file_path)

            # Stocker les imports
            self.file_imports[module_name] = analysis["imports"]

            # Stocker les définitions
            self.definitions[module_name] = analysis["definitions"]

            # Construire le graphe d'imports
            for imp in analysis["imports"]:
                imported_module = imp["module"]
                # Filtrer pour ne garder que les imports internes ThreadX
                if imported_module.startswith("threadx"):
                    self.imports_graph[module_name].add(imported_module)

            # Stocker les usages
            for usage in analysis["usages"]:
                self.usages[module_name].add(usage)

    def find_unused_definitions(self) -> Dict[str, Dict[str, List[str]]]:
        """Trouve les définitions potentiellement non utilisées"""
        unused = defaultdict(lambda: {"classes": [], "functions": []})

        # Pour chaque module
        for module_name, defs in self.definitions.items():
            # Vérifier chaque classe
            for class_name in defs["classes"]:
                used = False

                # Chercher dans tous les autres modules
                for other_module, usages_set in self.usages.items():
                    if other_module != module_name:
                        if class_name in usages_set:
                            used = True
                            break

                if not used:
                    unused[module_name]["classes"].append(class_name)

            # Vérifier chaque fonction
            for func_name in defs["functions"]:
                used = False

                for other_module, usages_set in self.usages.items():
                    if other_module != module_name:
                        if func_name in usages_set:
                            used = True
                            break

                if not used:
                    unused[module_name]["functions"].append(func_name)

        return dict(unused)

    def calculate_module_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Calcule les métriques pour chaque module"""
        metrics = {}

        for file_path in self.python_files:
            relative_path = file_path.relative_to(self.src_path)
            module_name = (
                str(relative_path)
                .replace("\\", ".")
                .replace("/", ".")
                .replace(".py", "")
            )

            analysis = self.analyze_file(file_path)

            num_imports = len(analysis["imports"])
            num_threadx_imports = sum(
                1 for imp in analysis["imports"] if "threadx" in imp["module"]
            )
            num_classes = len(analysis["definitions"]["classes"])
            num_functions = len(analysis["definitions"]["functions"])

            metrics[module_name] = {
                "loc": analysis["loc"],
                "imports_total": num_imports,
                "imports_threadx": num_threadx_imports,
                "classes": num_classes,
                "functions": num_functions,
                "definitions_total": num_classes + num_functions,
            }

        return metrics

    def generate_report(self, output_path: Path) -> None:
        """Génère un rapport complet d'analyse"""
        print("\n🔍 Génération du rapport d'analyse...")

        # Construire le graphe de dépendances
        self.build_dependency_graph()

        # Calculer les métriques
        metrics = self.calculate_module_metrics()

        # Trouver le code non utilisé
        unused = self.find_unused_definitions()

        # Créer le rapport
        report = {
            "summary": {
                "total_files": len(self.python_files),
                "total_modules": len(self.definitions),
                "total_loc": sum(m["loc"] for m in metrics.values()),
                "total_classes": sum(m["classes"] for m in metrics.values()),
                "total_functions": sum(m["functions"] for m in metrics.values()),
            },
            "dependency_graph": {
                module: list(deps) for module, deps in self.imports_graph.items()
            },
            "module_metrics": metrics,
            "potentially_unused": {
                module: data
                for module, data in unused.items()
                if data["classes"] or data["functions"]
            },
            "top_imported_modules": self._get_top_imported_modules(),
            "isolated_modules": self._find_isolated_modules(),
        }

        # Sauvegarder le rapport JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ Rapport sauvegardé : {output_path}")

        # Afficher un résumé
        self._print_summary(report)

    def _get_top_imported_modules(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Trouve les modules les plus importés"""
        import_counts = defaultdict(int)

        for deps in self.imports_graph.values():
            for dep in deps:
                import_counts[dep] += 1

        return sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def _find_isolated_modules(self) -> List[str]:
        """Trouve les modules qui n'importent rien et ne sont importés par rien"""
        isolated = []

        all_imported = set()
        for deps in self.imports_graph.values():
            all_imported.update(deps)

        for module_name in self.definitions.keys():
            # Pas d'imports sortants
            no_imports = len(self.imports_graph.get(module_name, [])) == 0
            # Pas importé par d'autres
            not_imported = module_name not in all_imported

            if no_imports and not_imported:
                isolated.append(module_name)

        return isolated

    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Affiche un résumé du rapport"""
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DE L'ANALYSE DE CODE")
        print("=" * 80)

        summary = report["summary"]
        print(f"\n📁 Fichiers analysés : {summary['total_files']}")
        print(f"📝 Lignes de code : {summary['total_loc']:,}")
        print(f"🏗️  Classes définies : {summary['total_classes']}")
        print(f"⚙️  Fonctions définies : {summary['total_functions']}")

        print(f"\n🔗 Top 10 des modules les plus importés :")
        for module, count in report["top_imported_modules"]:
            print(f"   {module}: {count} imports")

        print(
            f"\n⚠️  Modules potentiellement isolés : {len(report['isolated_modules'])}"
        )
        for module in report["isolated_modules"][:5]:
            print(f"   - {module}")

        unused_count = sum(
            len(data["classes"]) + len(data["functions"])
            for data in report["potentially_unused"].values()
        )
        print(f"\n🗑️  Définitions potentiellement non utilisées : {unused_count}")

        print("\n" + "=" * 80)
        print(f"📄 Rapport complet disponible dans le fichier JSON")
        print("=" * 80 + "\n")


def main():
    """Point d'entrée principal"""
    print("🚀 Démarrage de l'analyse de code ThreadX\n")

    src_path = PROJECT_ROOT / "src" / "threadx"
    output_path = PROJECT_ROOT / "code_analysis_report.json"

    if not src_path.exists():
        print(f"❌ Dossier source introuvable : {src_path}")
        sys.exit(1)

    analyzer = CodeAnalyzer(src_path)
    analyzer.scan_files()
    analyzer.generate_report(output_path)

    print("\n✨ Analyse terminée avec succès!")


if __name__ == "__main__":
    main()
