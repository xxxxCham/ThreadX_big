"""
Script d'orchestration pour le Quant Research Lab.

Workflow complet:
1. Sweep baseline sur stratégie existante
2. Analyst analyse les résultats
3. Strategist propose modifications
4. CodeWriter génère code Python
5. Critic valide et décide promotion

V1: Exécution séquentielle simple, une génération à la fois.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ajouter src/ au PYTHONPATH si nécessaire
if str(Path(__file__).parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.codewriter import CodeWriter
from threadx.llm.agents.critic import Critic, ValidationCriteria
from threadx.llm.agents.strategist import Strategist

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evolution_loop.log"),
    ],
)

logger = logging.getLogger(__name__)


class EvolutionLoop:
    """
    Orchestrateur pour le processus d'évolution de stratégies.

    V1 Workflow (séquentiel):
    1. Load baseline sweep results
    2. Analyst.analyze_sweep_results()
    3. Strategist.propose_modifications()
    4. CodeWriter.run()
    5. Critic.run()
    6. Save generation report
    """

    def __init__(
        self,
        output_dir: str | Path = "results/ai_evolution",
        experimental_dir: str | Path = "src/threadx/strategy/experimental",
        debug: bool = False,
    ):
        """
        Initialise l'orchestrateur.

        Args:
            output_dir: Dossier pour sauvegarder les rapports de génération
            experimental_dir: Dossier des stratégies expérimentales
            debug: Active logs détaillés
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experimental_dir = Path(experimental_dir)

        # Initialiser agents
        self.analyst = Analyst(debug=debug)
        self.strategist = Strategist(debug=debug)
        self.codewriter = CodeWriter(
            output_dir=experimental_dir,
            debug=debug,
        )
        self.critic = Critic(
            criteria=ValidationCriteria(
                min_sharpe=0.5,
                max_drawdown_pct=-30.0,
                min_trades=10,
                min_win_rate_pct=35.0,
            ),
            experimental_dir=experimental_dir,
            debug=debug,
        )

        logger.info("🚀 EvolutionLoop initialisé")
        logger.info(f"   📁 Output: {self.output_dir}")
        logger.info(f"   📁 Experimental: {self.experimental_dir}")

    def run_generation(
        self,
        base_strategy: str,
        sweep_results: dict[str, Any],
        task: str = "improve_sharpe",
        generation_num: int = 1,
    ) -> dict[str, Any]:
        """
        Exécute une génération complète de stratégie.

        Args:
            base_strategy: Nom de la stratégie de base (ex: "Bollinger_Dual")
            sweep_results: Résultats du sweep (format OptimizationEngine)
            task: Objectif d'amélioration
            generation_num: Numéro de génération

        Returns:
            dict contenant tous les résultats de la génération
        """
        logger.info("=" * 80)
        logger.info(f"🧬 GÉNÉRATION #{generation_num}")
        logger.info(f"   Base: {base_strategy}")
        logger.info(f"   Task: {task}")
        logger.info("=" * 80)

        generation_report = {
            "generation_num": generation_num,
            "base_strategy": base_strategy,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "steps": {},
        }

        try:
            # Étape 1: Analyst
            logger.info("\n📊 ÉTAPE 1/5: Analyst - Analyse des résultats de sweep")
            analyst_start = datetime.now()

            analyst_result = self.analyst.analyze_sweep_results(
                sweep_results=sweep_results,
                focus="sharpe",  # ou "drawdown", "winrate" selon task
            )

            analyst_duration = (datetime.now() - analyst_start).total_seconds()

            generation_report["steps"]["analyst"] = {
                "status": analyst_result.get("status", "unknown"),
                "duration_sec": analyst_duration,
                "patterns_found": len(analyst_result.get("analysis", {}).get("patterns", [])),
                "recommendations_count": len(analyst_result.get("analysis", {}).get("recommendations", [])),
                "result": analyst_result,
            }

            logger.info(f"   ✅ Analyst terminé en {analyst_duration:.1f}s")
            logger.info(f"   Patterns: {generation_report['steps']['analyst']['patterns_found']}")

            # Étape 2: Strategist
            logger.info("\n🎯 ÉTAPE 2/5: Strategist - Proposition de modifications")
            strategist_start = datetime.now()

            strategist_result = self.strategist.propose_modifications(
                analysis=analyst_result,
                base_strategy=base_strategy,
                objective=task,
            )

            strategist_duration = (datetime.now() - strategist_start).total_seconds()

            generation_report["steps"]["strategist"] = {
                "status": strategist_result.get("status", "unknown"),
                "duration_sec": strategist_duration,
                "proposals_count": len(strategist_result.get("proposals", [])),
                "result": strategist_result,
            }

            logger.info(f"   ✅ Strategist terminé en {strategist_duration:.1f}s")
            logger.info(f"   Propositions: {generation_report['steps']['strategist']['proposals_count']}")

            # Étape 3: CodeWriter
            logger.info("\n🔨 ÉTAPE 3/5: CodeWriter - Génération de code")
            codewriter_start = datetime.now()

            # Extraire métriques baseline pour contexte
            failed_metrics = None
            if "baseline" in sweep_results:
                baseline = sweep_results["baseline"]
                failed_metrics = {
                    "current_sharpe": baseline.get("sharpe_ratio", 0.0),
                    "current_drawdown": baseline.get("max_drawdown", 0.0),
                    "current_trades": baseline.get("total_trades", 0),
                }

            codewriter_result = self.codewriter.run(
                task=task,
                base_strategy=base_strategy,
                analysis=analyst_result,
                proposals=strategist_result,
                failed_metrics=failed_metrics,
            )

            codewriter_duration = (datetime.now() - codewriter_start).total_seconds()

            generation_report["steps"]["codewriter"] = {
                "status": codewriter_result.get("status", "unknown"),
                "duration_sec": codewriter_duration,
                "filename": codewriter_result.get("filename", "N/A"),
                "class_name": codewriter_result.get("class_name", "N/A"),
                "filepath": codewriter_result.get("filepath", "N/A"),
                "result": codewriter_result,
            }

            logger.info(f"   ✅ CodeWriter terminé en {codewriter_duration:.1f}s")
            logger.info(f"   Fichier: {codewriter_result.get('filename', 'N/A')}")

            if codewriter_result.get("status") != "success":
                logger.error(f"   ❌ CodeWriter a échoué: {codewriter_result.get('error')}")
                generation_report["status"] = "failed_codewriter"
                return generation_report

            # Étape 4: Critic
            logger.info("\n🧪 ÉTAPE 4/5: Critic - Validation")
            critic_start = datetime.now()

            critic_result = self.critic.run(
                strategy_file=codewriter_result["filename"],
            )

            critic_duration = (datetime.now() - critic_start).total_seconds()

            generation_report["steps"]["critic"] = {
                "status": critic_result.get("status", "unknown"),
                "duration_sec": critic_duration,
                "recommendation": critic_result.get("recommendation", "N/A"),
                "promoted": critic_result.get("promoted", False),
                "result": critic_result,
            }

            logger.info(f"   ✅ Critic terminé en {critic_duration:.1f}s")
            logger.info(f"   Recommandation: {critic_result.get('recommendation')}")

            # Étape 5: Décision
            logger.info("\n🎯 ÉTAPE 5/5: Décision de promotion")

            if critic_result.get("status") == "approved":
                logger.info("   ✅ STRATÉGIE APPROUVÉE")
                generation_report["status"] = "approved"

                # V1: Pas de promotion automatique (manuel)
                # V2: Copier vers strategy/ + enregistrer dans registry
                logger.info("   ℹ️  Promotion manuelle requise (V1)")

            else:
                logger.info("   ❌ STRATÉGIE REJETÉE")
                generation_report["status"] = "rejected"
                generation_report["rejection_reasons"] = critic_result.get("errors", [])

            # Sauvegarder rapport
            self._save_generation_report(generation_report)

            return generation_report

        except Exception as e:
            logger.error(f"❌ Erreur durant génération: {e}", exc_info=True)
            generation_report["status"] = "error"
            generation_report["error"] = str(e)
            return generation_report

    def _save_generation_report(self, report: dict[str, Any]) -> Path:
        """
        Sauvegarde le rapport de génération en JSON.

        Args:
            report: Rapport de génération

        Returns:
            Path du fichier sauvegardé
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_strategy = report.get("base_strategy", "unknown")
        gen_num = report.get("generation_num", 0)

        filename = f"{timestamp}_{base_strategy}_gen{gen_num}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\n💾 Rapport sauvegardé: {filepath}")

        return filepath


def main():
    """Point d'entrée CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quant Research Lab - Evolution Loop V1"
    )
    parser.add_argument(
        "--base-strategy",
        required=True,
        help="Stratégie de base (ex: Bollinger_Dual)",
    )
    parser.add_argument(
        "--sweep-results",
        required=True,
        help="Path vers fichier JSON des résultats de sweep",
    )
    parser.add_argument(
        "--task",
        default="improve_sharpe",
        choices=["improve_sharpe", "reduce_drawdown", "increase_winrate"],
        help="Objectif d'amélioration",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=1,
        help="Numéro de génération",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active logs détaillés",
    )

    args = parser.parse_args()

    # Charger résultats de sweep
    sweep_path = Path(args.sweep_results)
    if not sweep_path.exists():
        logger.error(f"❌ Fichier sweep non trouvé: {sweep_path}")
        sys.exit(1)

    with open(sweep_path, encoding="utf-8") as f:
        sweep_results = json.load(f)

    # Initialiser loop
    loop = EvolutionLoop(debug=args.debug)

    # Exécuter génération
    result = loop.run_generation(
        base_strategy=args.base_strategy,
        sweep_results=sweep_results,
        task=args.task,
        generation_num=args.generation,
    )

    # Afficher résumé
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ GÉNÉRATION")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Base Strategy: {result['base_strategy']}")
    print(f"Task: {result['task']}")
    print(f"\nÉtapes:")
    for step_name, step_data in result.get("steps", {}).items():
        print(f"  {step_name}: {step_data.get('status')} ({step_data.get('duration_sec', 0):.1f}s)")

    if result["status"] == "approved":
        print(f"\n✅ Stratégie générée et approuvée!")
        print(f"   Fichier: {result['steps']['codewriter'].get('filename')}")
    elif result["status"] == "rejected":
        print(f"\n❌ Stratégie rejetée")
        print(f"   Raisons: {result.get('rejection_reasons', [])}")

    print("=" * 80)


if __name__ == "__main__":
    main()
