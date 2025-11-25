"""
audit_logging.py - Analyse du système de logging ThreadX
========================================================

Usage:
    python audit_logging.py
"""

import sys
from pathlib import Path
from collections import defaultdict

# Setup path
sys.path.insert(0, str(Path("src").absolute()))


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def audit_log_config():
    """Analyse la configuration du système de logging."""
    print_section("1. Configuration Logging Central")

    try:
        from threadx.utils.log import get_logger

        # Tester la création d'un logger
        test_logger = get_logger("test_audit")

        print(f"✅ Logger factory disponible : threadx.utils.log.get_logger()")
        print(f"   Niveau effectif : {test_logger.level}")
        print(f"   Handlers : {len(test_logger.handlers)} handler(s)")

        for i, handler in enumerate(test_logger.handlers):
            print(f"      Handler {i}: {type(handler).__name__}")
            print(f"         Niveau : {handler.level}")
            if hasattr(handler, 'formatter') and handler.formatter:
                print(f"         Format : {handler.formatter._fmt}")

        return True

    except Exception as e:
        print(f"❌ Erreur analyse config logging : {e}")
        return False


def audit_log_usage():
    """Scan tous les fichiers Python pour analyser l'usage du logging."""
    print_section("2. Usage du Logging par Module")

    src_dir = Path("src/threadx")

    # Stats globales
    stats = {
        'total_files': 0,
        'files_with_logging': 0,
        'by_level': defaultdict(int),
        'by_module': {},
    }

    for py_file in src_dir.rglob("*.py"):
        stats['total_files'] += 1

        try:
            content = py_file.read_text(encoding='utf-8')

            # Chercher get_logger()
            if 'get_logger' in content:
                stats['files_with_logging'] += 1

                # Compter les niveaux de log
                log_levels = {
                    'DEBUG': content.count('logger.debug('),
                    'INFO': content.count('logger.info('),
                    'WARNING': content.count('logger.warning('),
                    'ERROR': content.count('logger.error('),
                }

                total_logs = sum(log_levels.values())

                if total_logs > 0:
                    relative_path = py_file.relative_to(src_dir)
                    stats['by_module'][str(relative_path)] = log_levels

                    for level, count in log_levels.items():
                        stats['by_level'][level] += count

        except Exception:
            pass

    # Affichage stats globales
    print(f"📊 Statistiques Globales")
    print(f"   Fichiers Python : {stats['total_files']}")
    print(f"   Fichiers avec logging : {stats['files_with_logging']} ({stats['files_with_logging']/stats['total_files']*100:.1f}%)")
    print(f"\n📈 Répartition par Niveau")

    total_logs = sum(stats['by_level'].values())
    for level, count in sorted(stats['by_level'].items(), key=lambda x: -x[1]):
        pct = count / total_logs * 100 if total_logs > 0 else 0
        print(f"   {level:8s} : {count:4d} ({pct:5.1f}%)")

    # Top 10 modules avec le plus de logs
    print(f"\n🔝 Top 10 Modules (nombre de logs)")

    sorted_modules = sorted(
        stats['by_module'].items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )[:10]

    for module, levels in sorted_modules:
        total = sum(levels.values())
        print(f"   {module:50s} : {total:3d} logs")
        for level, count in levels.items():
            if count > 0:
                print(f"      {level:8s} : {count:3d}")

    return stats


def audit_llm_optimizer_logging():
    """Analyse spécifique du logging dans l'optimisation LLM."""
    print_section("3. Logging Optimisation LLM (détaillé)")

    llm_modules = [
        "llm/agents/analyst.py",
        "llm/agents/strategist.py",
        "llm/agents/base_agent.py",
        "llm/client.py",
        "llm/ollama_manager.py",
        "ui/page_llm_optimizer.py",
        "optimization/engine.py",
    ]

    src_dir = Path("src/threadx")

    for module_path in llm_modules:
        full_path = src_dir / module_path

        if not full_path.exists():
            print(f"⚠️  {module_path:50s} : Fichier non trouvé")
            continue

        try:
            content = full_path.read_text(encoding='utf-8')

            # Analyse détaillée
            has_logger = 'get_logger' in content
            log_counts = {
                'DEBUG': content.count('logger.debug('),
                'INFO': content.count('logger.info('),
                'WARNING': content.count('logger.warning('),
                'ERROR': content.count('logger.error('),
            }

            total = sum(log_counts.values())

            status = "✅" if has_logger else "❌"
            print(f"{status} {module_path:50s} : {total:3d} logs")

            if total > 0:
                details = ", ".join([f"{level}:{count}" for level, count in log_counts.items() if count > 0])
                print(f"      └─ {details}")

            # Chercher des patterns spécifiques
            patterns = {
                'timing': content.count('time.time()') + content.count('perf_counter'),
                'exceptions': content.count('except Exception'),
                'try/except sans log': content.count('except Exception:\n        pass'),
            }

            if any(patterns.values()):
                print(f"      📌 Patterns détectés :")
                for pattern, count in patterns.items():
                    if count > 0:
                        print(f"         • {pattern}: {count}")

        except Exception as e:
            print(f"❌ {module_path:50s} : Erreur lecture ({e})")


def audit_missing_logs():
    """Identifie les zones critiques sans logging suffisant."""
    print_section("4. Zones Critiques avec Logging Insuffisant")

    critical_checks = [
        {
            'name': 'ProcessPool Workers',
            'file': 'src/threadx/optimization/engine.py',
            'pattern': '_evaluate_combo_worker',
            'should_have': 'logger.debug',
        },
        {
            'name': 'LLM JSON Parsing',
            'file': 'src/threadx/llm/agents/analyst.py',
            'pattern': 'json.loads',
            'should_have': 'logger.warning',
        },
        {
            'name': 'GPU Memory Errors',
            'file': 'src/threadx/gpu/multi_gpu.py',
            'pattern': 'GPUMemoryError',
            'should_have': 'logger.error',
        },
    ]

    issues = []

    for check in critical_checks:
        file_path = Path(check['file'])

        if not file_path.exists():
            continue

        try:
            content = file_path.read_text(encoding='utf-8')

            has_pattern = check['pattern'] in content
            has_logging = check['should_have'] in content

            if has_pattern and not has_logging:
                issues.append(check['name'])
                print(f"⚠️  {check['name']:40s} : Pattern trouvé, logging manquant")
            elif has_pattern and has_logging:
                print(f"✅ {check['name']:40s} : Logging présent")
            else:
                print(f"ℹ️  {check['name']:40s} : Pattern non trouvé")

        except Exception:
            pass

    if not issues:
        print("\n✅ Aucune zone critique détectée sans logging")
    else:
        print(f"\n⚠️  {len(issues)} zone(s) critique(s) à améliorer")


def main():
    """Point d'entrée principal."""
    print("\n" + "🔍 " * 20)
    print("  AUDIT SYSTÈME DE LOGGING - ThreadX v2.0")
    print("🔍 " * 20)

    results = []

    # Exécuter tous les audits
    results.append(("Config Logging", audit_log_config()))

    stats = audit_log_usage()
    results.append(("Usage Global", stats['files_with_logging'] > 0))

    audit_llm_optimizer_logging()

    audit_missing_logs()

    # Résumé final
    print_section("📊 RÉSUMÉ AUDIT")

    for check_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status}  {check_name}")

    print("\n💡 Recommandations :")
    print("   1. Vérifier que tous les modules critiques utilisent get_logger()")
    print("   2. Ajouter logs DEBUG pour exceptions non-critiques")
    print("   3. Ajouter logs INFO pour étapes clés du workflow LLM")
    print("   4. Ajouter timing logs pour profiling (time.time())")
    print("   5. Standardiser format : [module] - [niveau] - [message]")


if __name__ == "__main__":
    main()
