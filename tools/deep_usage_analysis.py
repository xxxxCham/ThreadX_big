"""
Analyse d'utilisation approfondie - ThreadX
===========================================

Analyse chaque fonction/classe marquée comme "inutilisée" en vérifiant:
1. Imports directs
2. Usage via getattr/registry
3. Héritage/protocoles
4. Exports publics (__all__)
5. Usage dans tests

Génère un rapport détaillé par module.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Chemins
REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src" / "threadx"
ANALYSIS_REPORT = REPO_ROOT / "code_analysis_report.json"
OUTPUT_REPORT = REPO_ROOT / "docs" / "cleanup" / "deep_usage_analysis.md"


def load_analysis_data() -> Dict:
    """Charge le rapport d'analyse JSON."""
    with open(ANALYSIS_REPORT, "r", encoding="utf-8") as f:
        return json.load(f)


def grep_usage(function_name: str, search_roots: List[Path]) -> List[Tuple[Path, int]]:
    """
    Recherche récursive d'une fonction/classe dans les fichiers Python.

    Returns: Liste de (fichier, numéro_ligne)
    """
    matches = []
    for root in search_roots:
        for py_file in root.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        # Cherche usage direct (appel, import, héritage)
                        if re.search(rf"\b{re.escape(function_name)}\b", line):
                            matches.append((py_file, line_num))
            except Exception:
                continue
    return matches


def is_exported_in_all(module_path: Path, name: str) -> bool:
    """Vérifie si une fonction est dans __all__ du module."""
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Cherche __all__ = [...]
            all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
            if all_match:
                all_content = all_match.group(1)
                return f'"{name}"' in all_content or f"'{name}'" in all_content
    except Exception:
        pass
    return False


def is_protocol_method(module_path: Path, name: str) -> bool:
    """Vérifie si c'est une méthode de protocole/ABC."""
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Cherche @abstractmethod ou Protocol
            if "Protocol" in content or "@abstractmethod" in content:
                if f"def {name}" in content:
                    return True
    except Exception:
        pass
    return False


def categorize_unused_definition(
    module: str, def_type: str, name: str, analysis_data: Dict
) -> Dict[str, any]:
    """
    Catégorise une définition "inutilisée" avec vérification approfondie.

    Returns:
        {
            "name": str,
            "type": str,
            "module": str,
            "usage_count": int,
            "usages": List[str],
            "exported": bool,
            "protocol": bool,
            "verdict": "KEEP" | "VERIFY" | "DELETE",
            "reason": str
        }
    """
    module_path = (
        SRC_ROOT / module.replace(".", "/").replace("threadx/", "") / "__init__.py"
    )
    if not module_path.exists():
        # Essayer sans __init__.py
        module_path = SRC_ROOT / (
            module.replace(".", "/").replace("threadx/", "") + ".py"
        )

    # Recherche usages
    usages = grep_usage(name, [SRC_ROOT, REPO_ROOT / "tests"])
    usage_count = len(usages)

    # Vérifie exports publics
    exported = is_exported_in_all(module_path, name) if module_path.exists() else False

    # Vérifie protocoles
    protocol = is_protocol_method(module_path, name) if module_path.exists() else False

    # Catégorisation
    verdict = "DELETE"
    reason = "Aucun usage trouvé"

    if usage_count > 2:  # Définition + au moins 1 usage réel
        verdict = "KEEP"
        reason = f"Utilisé {usage_count - 1} fois"
    elif exported:
        verdict = "KEEP"
        reason = "API publique (__all__)"
    elif protocol:
        verdict = "KEEP"
        reason = "Méthode Protocol/ABC"
    elif def_type == "class":
        # Les classes peuvent être utilisées via __init_subclass__, registries
        verdict = "VERIFY"
        reason = "Classe - vérifier registries"
    elif usage_count == 2:
        # Définition + 1 usage potentiel
        verdict = "VERIFY"
        reason = "1 usage trouvé - vérifier contexte"

    return {
        "name": name,
        "type": def_type,
        "module": module,
        "usage_count": usage_count,
        "usages": [f"{u[0].relative_to(REPO_ROOT)}:{u[1]}" for u in usages[:5]],
        "exported": exported,
        "protocol": protocol,
        "verdict": verdict,
        "reason": reason,
    }


def analyze_by_module(analysis_data: Dict) -> Dict[str, List[Dict]]:
    """Analyse toutes les définitions inutilisées par module."""
    unused = analysis_data.get("potentially_unused", {})

    results_by_module = defaultdict(lambda: {"KEEP": [], "VERIFY": [], "DELETE": []})

    for module, definitions in unused.items():
        for def_type, names in definitions.items():
            for name in names:
                result = categorize_unused_definition(
                    module, def_type, name, analysis_data
                )
                results_by_module[module][result["verdict"]].append(result)

    return dict(results_by_module)


def generate_markdown_report(results: Dict[str, List[Dict]]) -> str:
    """Génère le rapport Markdown détaillé."""
    lines = [
        "# 🔬 ANALYSE D'UTILISATION APPROFONDIE - ThreadX",
        "",
        f"**Date**: 2025-11-08",
        f"**Modules analysés**: {len(results)}",
        "",
        "---",
        "",
        "## 📊 RÉSUMÉ EXÉCUTIF PAR MODULE",
        "",
    ]

    # Statistiques globales
    total_keep = sum(len(r["KEEP"]) for r in results.values())
    total_verify = sum(len(r["VERIFY"]) for r in results.values())
    total_delete = sum(len(r["DELETE"]) for r in results.values())

    lines.extend(
        [
            "| Verdict | Nombre | Pourcentage |",
            "|---------|--------|-------------|",
            f"| ✅ **KEEP** (conserver) | {total_keep} | {total_keep/(total_keep+total_verify+total_delete)*100:.1f}% |",
            f"| 🔍 **VERIFY** (vérifier) | {total_verify} | {total_verify/(total_keep+total_verify+total_delete)*100:.1f}% |",
            f"| ❌ **DELETE** (supprimer) | {total_delete} | {total_delete/(total_keep+total_verify+total_delete)*100:.1f}% |",
            "",
            "---",
            "",
        ]
    )

    # Détails par module
    for module in sorted(results.keys()):
        module_results = results[module]
        keep_count = len(module_results["KEEP"])
        verify_count = len(module_results["VERIFY"])
        delete_count = len(module_results["DELETE"])

        if keep_count + verify_count + delete_count == 0:
            continue

        lines.extend(
            [
                f"## 📦 `{module}`",
                "",
                f"**Statistiques** : ✅ {keep_count} | 🔍 {verify_count} | ❌ {delete_count}",
                "",
            ]
        )

        # KEEP
        if module_results["KEEP"]:
            lines.append("### ✅ À CONSERVER")
            lines.append("")
            for item in module_results["KEEP"]:
                lines.append(
                    f"- **{item['type']}** `{item['name']}` - *{item['reason']}*"
                )
                if item["usages"]:
                    lines.append(f"  - Usages: `{item['usages'][0]}`")
            lines.append("")

        # VERIFY
        if module_results["VERIFY"]:
            lines.append("### 🔍 À VÉRIFIER")
            lines.append("")
            for item in module_results["VERIFY"]:
                lines.append(
                    f"- **{item['type']}** `{item['name']}` - *{item['reason']}*"
                )
            lines.append("")

        # DELETE
        if module_results["DELETE"]:
            lines.append("### ❌ À SUPPRIMER")
            lines.append("")
            for item in module_results["DELETE"]:
                lines.append(
                    f"- **{item['type']}** `{item['name']}` - *{item['reason']}*"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    print("🔬 Analyse d'utilisation approfondie...")

    # Charge données
    analysis_data = load_analysis_data()

    # Analyse par module
    print("📊 Analyse par module...")
    results = analyze_by_module(analysis_data)

    # Génère rapport
    print("📝 Génération rapport Markdown...")
    markdown = generate_markdown_report(results)

    # Sauvegarde
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✅ Rapport généré : {OUTPUT_REPORT}")

    # Stats finales
    total_keep = sum(len(r["KEEP"]) for r in results.values())
    total_verify = sum(len(r["VERIFY"]) for r in results.values())
    total_delete = sum(len(r["DELETE"]) for r in results.values())

    print(f"\n📊 Résumé :")
    print(f"   ✅ À conserver : {total_keep}")
    print(f"   🔍 À vérifier : {total_verify}")
    print(f"   ❌ À supprimer : {total_delete}")


if __name__ == "__main__":
    main()
