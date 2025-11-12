#!/usr/bin/env python3
"""
Analyse le rapport code_analysis_report.json et identifie :
1. Les définitions vraiment inutilisées (priorité haute)
2. Les faux positifs (APIs publiques, __init__, etc.)
3. Les modules isolés réels
"""

import json
from pathlib import Path
from collections import defaultdict

REPORT_PATH = Path(__file__).parent.parent / "code_analysis_report.json"
OUTPUT_PATH = (
    Path(__file__).parent.parent / "docs" / "cleanup" / "unused_code_analysis.md"
)


def load_report():
    """Charge le rapport JSON"""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_false_positive(module_name, def_type, def_name):
    """Détermine si une définition est un faux positif"""

    # __init__.py exports (APIs publiques)
    if module_name.endswith("__init__"):
        return True, "API publique (__init__.py)"

    # Fonctions privées avec _ (intentionnellement non exportées)
    if def_name.startswith("_") and not def_name.startswith("__"):
        return True, "Fonction/classe privée (convention _name)"

    # Méthodes magiques Python
    if def_name.startswith("__") and def_name.endswith("__"):
        return True, "Méthode magique Python"

    # Points d'entrée CLI/scripts
    if def_name in ["main", "run", "execute", "app", "cli"]:
        return True, "Point d'entrée (main/run/app)"

    # Callbacks Streamlit (pattern render_*)
    if def_name.startswith("render_") or def_name.startswith("page_"):
        return True, "Callback UI (Streamlit/Dash)"

    # Classes de test
    if module_name.endswith("test_") or "testing" in module_name:
        return True, "Code de test"

    # Archive/legacy explicite
    if "_archive" in module_name or "_legacy" in module_name:
        return True, "Code archivé/legacy"

    # Dataclasses/NamedTuples (souvent utilisées via instantiation)
    if def_type == "class" and any(
        keyword in def_name
        for keyword in ["Config", "Settings", "Event", "Info", "Stats", "Result"]
    ):
        return True, "Dataclass/Config (usage via instantiation)"

    return False, None


def categorize_unused_code(report):
    """Catégorise le code inutilisé"""

    categories = {
        "SUPPRIMER": [],  # Code vraiment mort
        "FAUX_POSITIF": [],  # APIs, callbacks, etc.
        "A_VERIFIER": [],  # Incertain, nécessite inspection manuelle
    }

    for module, defs in report["potentially_unused"].items():
        for class_name in defs["classes"]:
            is_fp, reason = is_false_positive(module, "class", class_name)

            entry = {
                "module": module,
                "type": "class",
                "name": class_name,
                "reason": reason,
            }

            if is_fp:
                categories["FAUX_POSITIF"].append(entry)
            else:
                # Heuristique : classes sans __init__ probablement mortes
                categories["A_VERIFIER"].append(entry)

        for func_name in defs["functions"]:
            is_fp, reason = is_false_positive(module, "function", func_name)

            entry = {
                "module": module,
                "type": "function",
                "name": func_name,
                "reason": reason,
            }

            if is_fp:
                categories["FAUX_POSITIF"].append(entry)
            else:
                categories["SUPPRIMER"].append(entry)

    return categories


def analyze_isolated_modules(report):
    """Analyse les modules isolés (pas d'imports/importés)"""

    isolated = report["isolated_modules"]

    real_isolated = []
    false_positives = []

    for module in isolated:
        # __init__.py sont normalement isolés
        if module.endswith("__init__"):
            false_positives.append((module, "__init__.py (normal)"))
        # Points d'entrée
        elif any(name in module for name in ["main", "app", "cli", "__main__"]):
            false_positives.append((module, "Point d'entrée"))
        # Tests
        elif "test" in module or "testing" in module:
            false_positives.append((module, "Module de test"))
        # Archive
        elif "_archive" in module or "_legacy" in module:
            false_positives.append((module, "Archivé"))
        else:
            real_isolated.append(module)

    return real_isolated, false_positives


def generate_markdown_report(report, categories, isolated_analysis):
    """Génère un rapport Markdown détaillé"""

    real_isolated, isolated_fps = isolated_analysis

    md = f"""# 🗑️ ANALYSE DU CODE INUTILISÉ - ThreadX

**Date**: 2025-11-08
**Fichiers analysés**: {report['summary']['total_files']}
**LOC total**: {report['summary']['total_loc']:,}
**Définitions totales**: {report['summary']['total_classes'] + report['summary']['total_functions']}

---

## 📊 RÉSUMÉ EXÉCUTIF

| Catégorie | Nombre | Action |
|-----------|--------|--------|
| **À SUPPRIMER** (code mort confirmé) | {len(categories['SUPPRIMER'])} | ❌ Supprimer |
| **À VÉRIFIER** (incertain) | {len(categories['A_VERIFIER'])} | 🔍 Inspection manuelle |
| **FAUX POSITIFS** (APIs, callbacks) | {len(categories['FAUX_POSITIF'])} | ✅ Conserver |
| **Modules isolés réels** | {len(real_isolated)} | 🔍 Investiguer |
| **Modules isolés (faux positifs)** | {len(isolated_fps)} | ✅ Normal |

---

## ❌ CODE À SUPPRIMER ({len(categories['SUPPRIMER'])} définitions)

Ces définitions semblent réellement inutilisées et peuvent être supprimées en toute sécurité :

"""

    # Grouper par module
    by_module = defaultdict(list)
    for entry in categories["SUPPRIMER"]:
        by_module[entry["module"]].append(entry)

    for module in sorted(by_module.keys()):
        md += f"\n### `{module}`\n\n"
        for entry in by_module[module]:
            md += f"- [ ] **{entry['type']}** `{entry['name']}`\n"

    md += f"""

---

## 🔍 CODE À VÉRIFIER ({len(categories['A_VERIFIER'])} définitions)

Ces définitions nécessitent une inspection manuelle car elles pourraient être :
- Utilisées via getattr() ou registries
- Importées dynamiquement
- API publique documentée

"""

    by_module = defaultdict(list)
    for entry in categories["A_VERIFIER"]:
        by_module[entry["module"]].append(entry)

    # Limiter aux 50 premières
    count = 0
    for module in sorted(by_module.keys()):
        if count >= 50:
            md += f"\n... et {len(categories['A_VERIFIER']) - 50} autres définitions\n"
            break
        md += f"\n### `{module}`\n\n"
        for entry in by_module[module]:
            if count >= 50:
                break
            md += f"- [ ] **{entry['type']}** `{entry['name']}`\n"
            count += 1

    md += f"""

---

## ✅ FAUX POSITIFS ({len(categories['FAUX_POSITIF'])} définitions)

Ces définitions sont correctement utilisées, malgré l'analyse statique :

**Répartition par raison :**
"""

    # Compter par raison
    reason_counts = defaultdict(int)
    for entry in categories["FAUX_POSITIF"]:
        reason_counts[entry["reason"]] += 1

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        md += f"\n- **{reason}** : {count} définitions"

    md += f"""

---

## 🏝️ MODULES ISOLÉS RÉELS ({len(real_isolated)} modules)

Ces modules ne sont ni importés ni n'importent rien :

"""

    for module in sorted(real_isolated):
        md += f"- [ ] `{module}`\n"

    md += f"""

---

## 📈 TOP MODULES IMPORTÉS

Les modules les plus utilisés dans le projet :

"""

    for module, count in report["top_imported_modules"][:15]:
        md += f"- **{module}** : {count} imports\n"

    md += f"""

---

## 🎯 RECOMMANDATIONS D'ACTION

### Priorité 1 : Supprimer le code mort confirmé
1. Examiner la section "CODE À SUPPRIMER"
2. Vérifier une dernière fois avec grep : `grep -r "nom_fonction" src/`
3. Supprimer les définitions confirmées mortes
4. Relancer les tests : `pytest tests/`

### Priorité 2 : Investiguer les modules isolés
1. Examiner chaque module listé dans "MODULES ISOLÉS RÉELS"
2. Déterminer s'ils sont obsolètes ou simplement mal intégrés
3. Soit les supprimer, soit les connecter au reste du code

### Priorité 3 : Vérifier le code incertain
1. Inspection manuelle des 50 premiers éléments "À VÉRIFIER"
2. Chercher dans la documentation si c'est une API publique
3. Vérifier si utilisé via registries ou imports dynamiques

---

**Rapport généré automatiquement par `tools/analyze_unused_code.py`**
"""

    return md


def main():
    """Point d'entrée principal"""
    print("📊 Chargement du rapport d'analyse...")
    report = load_report()

    print("🔍 Catégorisation du code inutilisé...")
    categories = categorize_unused_code(report)

    print("🏝️ Analyse des modules isolés...")
    isolated_analysis = analyze_isolated_modules(report)

    print("📝 Génération du rapport Markdown...")
    markdown = generate_markdown_report(report, categories, isolated_analysis)

    # Sauvegarde
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n✅ Rapport généré : {OUTPUT_PATH}")
    print(f"\n📊 Résumé :")
    print(f"   ❌ À supprimer : {len(categories['SUPPRIMER'])}")
    print(f"   🔍 À vérifier : {len(categories['A_VERIFIER'])}")
    print(f"   ✅ Faux positifs : {len(categories['FAUX_POSITIF'])}")
    print(f"   🏝️ Modules isolés : {len(isolated_analysis[0])}")


if __name__ == "__main__":
    main()
