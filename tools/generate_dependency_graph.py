"""
Générateur de graphe de dépendances visuelles pour ThreadX
Crée un diagramme DOT/GraphViz des communications entre modules
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List


class DependencyGraphGenerator:
    """Génère un graphe visuel des dépendances"""

    def __init__(self, report_path: Path):
        with open(report_path, "r", encoding="utf-8") as f:
            self.report = json.load(f)

        self.dependency_graph = self.report["dependency_graph"]
        self.module_metrics = self.report["module_metrics"]
        self.isolated = set(self.report["isolated_modules"])

    def generate_dot(self, output_path: Path, include_isolated: bool = False) -> None:
        """Génère un fichier DOT pour GraphViz"""

        lines = [
            "digraph ThreadX {",
            "  rankdir=LR;",
            "  node [shape=box, style=rounded];",
            "  ",
            "  // Légende",
            "  subgraph cluster_legend {",
            '    label="Légende";',
            '    legend_core [label="Module Core\\n(>5 imports)", fillcolor=lightblue, style=filled];',
            '    legend_util [label="Module Utilitaire\\n(3-5 imports)", fillcolor=lightgreen, style=filled];',
            '    legend_normal [label="Module Normal\\n(<3 imports)", fillcolor=white];',
            '    legend_isolated [label="Module Isolé", fillcolor=lightgray, style=filled];',
            "  }",
            "  ",
        ]

        # Calculer le nombre d'imports entrants pour chaque module
        incoming_counts = defaultdict(int)
        for source, targets in self.dependency_graph.items():
            for target in targets:
                incoming_counts[target] += 1

        # Grouper les modules par catégorie
        core_modules = []
        util_modules = []
        normal_modules = []
        isolated_modules = []

        all_modules = set(self.dependency_graph.keys())
        for target in incoming_counts.keys():
            all_modules.add(target)

        for module in all_modules:
            count = incoming_counts.get(module, 0)

            if module in self.isolated:
                isolated_modules.append(module)
            elif count >= 5:
                core_modules.append(module)
            elif count >= 3:
                util_modules.append(module)
            else:
                normal_modules.append(module)

        # Ajouter les nœuds par catégorie
        if core_modules:
            lines.append("  // Modules Core (très importés)")
            for module in sorted(core_modules):
                label = self._format_label(module, incoming_counts.get(module, 0))
                lines.append(
                    f'  "{module}" [label="{label}", fillcolor=lightblue, style=filled];'
                )
            lines.append("  ")

        if util_modules:
            lines.append("  // Modules Utilitaires")
            for module in sorted(util_modules):
                label = self._format_label(module, incoming_counts.get(module, 0))
                lines.append(
                    f'  "{module}" [label="{label}", fillcolor=lightgreen, style=filled];'
                )
            lines.append("  ")

        if normal_modules:
            lines.append("  // Modules Normaux")
            for module in sorted(normal_modules):
                label = self._format_label(module, incoming_counts.get(module, 0))
                lines.append(f'  "{module}" [label="{label}"];')
            lines.append("  ")

        if include_isolated and isolated_modules:
            lines.append("  // Modules Isolés")
            for module in sorted(isolated_modules):
                label = self._format_label(module, 0)
                lines.append(
                    f'  "{module}" [label="{label}", fillcolor=lightgray, style=filled];'
                )
            lines.append("  ")

        # Ajouter les arêtes (dépendances)
        lines.append("  // Dépendances")
        for source, targets in sorted(self.dependency_graph.items()):
            if not include_isolated and source in self.isolated:
                continue

            for target in sorted(targets):
                if not include_isolated and target in self.isolated:
                    continue

                lines.append(f'  "{source}" -> "{target}";')

        lines.append("}")

        # Écrire le fichier
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Graphe DOT généré : {output_path}")
        print(f"   Modules core : {len(core_modules)}")
        print(f"   Modules utilitaires : {len(util_modules)}")
        print(f"   Modules normaux : {len(normal_modules)}")
        if include_isolated:
            print(f"   Modules isolés : {len(isolated_modules)}")

    def _format_label(self, module: str, import_count: int) -> str:
        """Formate le label d'un nœud"""
        short_name = module.split(".")[-1] if "." in module else module

        if import_count > 0:
            return f"{short_name}\\n({import_count} imports)"
        else:
            return short_name

    def generate_mermaid(self, output_path: Path, max_nodes: int = 30) -> None:
        """Génère un diagramme Mermaid (pour Markdown)"""

        # Calculer les imports entrants
        incoming_counts = defaultdict(int)
        for source, targets in self.dependency_graph.items():
            for target in targets:
                incoming_counts[target] += 1

        # Prendre uniquement les modules les plus importés
        top_modules = sorted(incoming_counts.items(), key=lambda x: x[1], reverse=True)[
            :max_nodes
        ]
        top_module_names = {name for name, _ in top_modules}

        lines = [
            "```mermaid",
            "graph LR",
            "  ",
            "  %% Modules principaux de ThreadX",
            "  ",
        ]

        # Ajouter les nœuds
        for module, count in top_modules:
            short_name = module.split(".")[-1]
            node_id = module.replace(".", "_")

            if count >= 5:
                lines.append(f"  {node_id}[{short_name}<br/>{count} imports]:::core")
            elif count >= 3:
                lines.append(f"  {node_id}[{short_name}<br/>{count} imports]:::util")
            else:
                lines.append(f"  {node_id}[{short_name}<br/>{count} imports]")

        lines.append("  ")

        # Ajouter les arêtes
        for source, targets in self.dependency_graph.items():
            if source not in top_module_names:
                continue

            source_id = source.replace(".", "_")
            for target in targets:
                if target in top_module_names:
                    target_id = target.replace(".", "_")
                    lines.append(f"  {source_id} --> {target_id}")

        lines.extend(
            [
                "  ",
                "  %% Styles",
                "  classDef core fill:#add8e6,stroke:#333,stroke-width:2px",
                "  classDef util fill:#90ee90,stroke:#333,stroke-width:1px",
                "```",
            ]
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Diagramme Mermaid généré : {output_path}")
        print(f"   Top {len(top_modules)} modules inclus")

    def generate_stats_report(self, output_path: Path) -> None:
        """Génère un rapport textuel des statistiques"""

        lines = [
            "# 📊 Statistiques de Dépendances ThreadX",
            "",
            "## Modules par Nombre d'Imports Entrants",
            "",
        ]

        # Calculer imports entrants
        incoming_counts = defaultdict(int)
        for source, targets in self.dependency_graph.items():
            for target in targets:
                incoming_counts[target] += 1

        sorted_modules = sorted(
            incoming_counts.items(), key=lambda x: x[1], reverse=True
        )

        lines.append("| Rang | Module | Imports Entrants | LOC |")
        lines.append("|------|--------|------------------|-----|")

        for i, (module, count) in enumerate(sorted_modules[:30], 1):
            loc = self.module_metrics.get(module, {}).get("loc", "N/A")
            lines.append(f"| {i} | `{module}` | {count} | {loc} |")

        lines.extend(
            [
                "",
                "## Modules par Nombre d'Imports Sortants",
                "",
                "| Rang | Module | Imports Sortants | LOC |",
                "|------|--------|------------------|-----|",
            ]
        )

        outgoing = [(m, len(deps)) for m, deps in self.dependency_graph.items()]
        sorted_outgoing = sorted(outgoing, key=lambda x: x[1], reverse=True)

        for i, (module, count) in enumerate(sorted_outgoing[:30], 1):
            loc = self.module_metrics.get(module, {}).get("loc", "N/A")
            lines.append(f"| {i} | `{module}` | {count} | {loc} |")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Rapport statistiques généré : {output_path}")


def main():
    """Point d'entrée"""
    print("🎨 Génération des visualisations de dépendances...\n")

    project_root = Path(__file__).parent.parent
    report_path = project_root / "code_analysis_report.json"

    if not report_path.exists():
        print(f"❌ Rapport d'analyse introuvable : {report_path}")
        print("   Lancez d'abord : python tools/code_analysis_access.py")
        return

    generator = DependencyGraphGenerator(report_path)

    # Générer différents formats
    generator.generate_dot(
        project_root / "dependency_graph.dot", include_isolated=False
    )

    generator.generate_dot(
        project_root / "dependency_graph_full.dot", include_isolated=True
    )

    generator.generate_mermaid(project_root / "dependency_graph.mermaid.md")

    generator.generate_stats_report(project_root / "dependency_stats.md")

    print("\n✨ Visualisations générées avec succès!")
    print("\n📝 Pour générer un PNG avec GraphViz :")
    print("   dot -Tpng dependency_graph.dot -o dependency_graph.png")


if __name__ == "__main__":
    main()
