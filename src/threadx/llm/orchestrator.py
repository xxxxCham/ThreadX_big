"""
Optimization Orchestrator - Boucle d'optimisation autonome multi-agents.

Coordonne Analyst, Strategist, Critic pour améliorer stratégies trading de façon autonome.

Workflow (7 étapes):
    1. Backtest initial → RunResult
    2. Analyse Analyst → Diagnostic JSON
    3. Propositions Strategist → Liste configs candidates
    4. Validation Critic → Filtrage overfitting/risques
    5. Backtests parallèles → Scores propositions
    6. Sélection meilleure → Mise à jour params
    7. Mise à jour mémoire → Historique convergence

Boucle jusqu'à:
    - Convergence (N cycles sans amélioration)
    - Score cible atteint (ex: Sharpe > 2.0)
    - Max itérations

Author: ThreadX Framework
Version: 1.0 - Autonomous Multi-Agent Optimization
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from threadx.backtest.engine import BacktestEngine, RunResult
from threadx.indicators.bank import IndicatorBank
from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.critic import Critic
from threadx.llm.agents.strategist import Strategist
from threadx.llm.context_manager import ContextManager
from threadx.llm.memory import OptimizationMemory
from threadx.utils.log import get_logger


@dataclass
class OptimizationConfig:
    """Configuration pour orchestrateur autonome."""

    strategy_name: str
    """Nom stratégie à optimiser (ex: 'ma_crossover')."""

    initial_params: dict[str, Any]
    """Paramètres initiaux de la stratégie."""

    target_sharpe: float = 2.0
    """Score cible Sharpe ratio pour arrêt automatique."""

    max_iterations: int = 30
    """Nombre max d'itérations de la boucle."""

    convergence_threshold: int = 3
    """Nombre cycles stagnation avant arrêt."""

    proposals_per_iteration: int = 3
    """Nombre propositions générées par Strategist."""

    memory_size: int = 10
    """Historique itérations conservées."""

    export_dir: Path | None = None
    """Dossier export résultats (None = pas d'export)."""

    log_callback: Any = None
    """Callback(iteration, message, level) pour logs temps réel UI."""

    code_callback: Any = None
    """Callback(iteration, agent, code, description) pour code généré UI."""


@dataclass
class IterationResult:
    """Résultat d'une itération d'optimisation."""

    iteration: int
    params: dict[str, Any]
    score: float
    metrics: dict[str, Any]
    analysis: dict[str, Any]
    proposals: list[dict[str, Any]]
    validated_proposals: list[dict[str, Any]]
    execution_time: float
    converged: bool = False
    reason: str = ""


class OptimizationOrchestrator:
    """
    Orchestrateur autonome d'optimisation multi-agents.

    Features:
    - Coordination 3 agents LLM (Analyst, Strategist, Critic)
    - Boucle itérative avec convergence automatique
    - Backtests parallèles pour validation propositions
    - Mémoire historique (évite repropositions)
    - Export résultats + logs structurés

    Attributes:
        config: Configuration optimisation
        analyst: Agent analyse quantitative
        strategist: Agent génération propositions
        critic: Agent validation propositions
        memory: Historique itérations
        backtest_engine: Moteur backtest GPU
        sweep_runner: Runner backtests parallèles
        logger: Logger structuré
    """

    def __init__(
        self,
        config: OptimizationConfig,
        data: pd.DataFrame,
        analyst_model: str = "deepseek-r1:70b",
        strategist_model: str = "gpt-oss:20b",
        critic_model: str = "deepseek-r1:70b",
        gpu_id: int = 0,
        debug: bool = False,
        log_callback: Any = None,
        code_callback: Any = None,
    ):
        """
        Initialise l'orchestrateur autonome.

        Args:
            config: Configuration optimisation
            data: DataFrame OHLCV pour backtests
            analyst_model: Modèle LLM Analyst (défaut: deepseek-r1:70b)
            strategist_model: Modèle LLM Strategist (défaut: gpt-oss:20b)
            critic_model: Modèle LLM Critic (défaut: deepseek-r1:70b)
            gpu_id: ID GPU pour backtests (défaut: 0)
            debug: Active logs détaillés
            log_callback: Callback(iteration, message, level) logs temps réel
            code_callback: Callback(iteration, agent, code, desc) code généré
        """
        self.config = config
        self.data = data
        self.debug = debug

        # Callbacks UI temps réel (priorité args puis config)
        self.log_callback = log_callback or config.log_callback
        self.code_callback = code_callback or config.code_callback

        # Logger
        self.logger = get_logger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Context Manager - Inventaire données + registry stratégies
        self.context_manager = ContextManager()

        # Agents LLM
        self.analyst = Analyst(model=analyst_model, debug=debug)
        self.strategist = Strategist(model=strategist_model, debug=debug)
        self.critic = Critic(model=critic_model, debug=debug)

        # Mémoire optimisation
        self.memory = OptimizationMemory(max_size=config.memory_size)

        # Moteur backtest (API réelle: pas de gpu_id en __init__)
        self.backtest_engine = BacktestEngine()

        # IndicatorBank pour calcul indicateurs
        self.indicator_bank = IndicatorBank()

        # GPU ID pour backtests (stocké, utilisé dans run())
        self.gpu_id = gpu_id

        # Historique itérations
        self.iterations: list[IterationResult] = []

        # Stats globales
        self.start_time: float = 0.0
        self.total_backtests: int = 0

        self.logger.info(
            f"🤖 Orchestrator initialized: strategy={config.strategy_name}, "
            f"target_sharpe={config.target_sharpe}, max_iter={config.max_iterations}"
        )

    def run(self) -> dict[str, Any]:
        """
        Lance la boucle d'optimisation autonome.

        Returns:
            dict avec:
            - best_params: Meilleurs paramètres trouvés
            - best_score: Meilleur score Sharpe
            - iterations: Historique complet
            - total_backtests: Nombre backtests exécutés
            - execution_time: Temps total (secondes)
            - converged: Convergence atteinte
            - reason: Raison arrêt
        """
        self.start_time = time.time()
        self.logger.info("🚀 Starting autonomous optimization loop...")

        # Stats convergence
        best_score = float("-inf")
        stagnation_count = 0
        converged = False
        stop_reason = ""

        # Paramètres courants
        current_params = self.config.initial_params.copy()

        for iteration in range(self.config.max_iterations):
            iter_start = time.time()
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📍 ITERATION {iteration + 1}/{self.config.max_iterations}")
            self.logger.info(f"{'='*60}")

            # Log callback UI
            if self.log_callback:
                self.log_callback(
                    iteration + 1,
                    f"Starting iteration {iteration + 1}/{self.config.max_iterations}",
                    "INFO",
                )

            # === ÉTAPE 1: Backtest initial ===
            self.logger.info("⚙️  Step 1/7: Running initial backtest...")
            if self.log_callback:
                self.log_callback(iteration + 1, "Running initial backtest...", "INFO")

            result = self._run_backtest(current_params)
            self.total_backtests += 1

            current_score = result.metrics.get("sharpe_ratio", 0.0)
            self.logger.info(f"   → Sharpe: {current_score:.3f}")

            if self.log_callback:
                self.log_callback(
                    iteration + 1, f"Backtest complete: Sharpe={current_score:.3f}", "INFO"
                )

            # === ÉTAPE 2: Analyse Analyst ===
            self.logger.info("🕵️  Step 2/7: Analyst analyzing results...")
            if self.log_callback:
                self.log_callback(iteration + 1, "Analyst analyzing results...", "INFO")

            analysis = self._analyze_results(result, current_params)
            self.logger.info(
                f"   → Global score: {analysis.get('score_global', 0)}/10"
            )

            if self.log_callback:
                self.log_callback(
                    iteration + 1,
                    f"Analysis complete: Score {analysis.get('score_global', 0)}/10",
                    "INFO",
                )

            # === ÉTAPE 3: Génération propositions Strategist ===
            self.logger.info("💡 Step 3/7: Strategist generating proposals...")
            if self.log_callback:
                self.log_callback(iteration + 1, "Strategist proposing improvements...", "INFO")

            proposals = self._generate_proposals(
                current_params, analysis, self.config.proposals_per_iteration
            )
            self.logger.info(f"   → {len(proposals)} proposals generated")

            # Code callback (propositions = code paramètres)
            if self.code_callback and proposals:
                for i, prop in enumerate(proposals):
                    code_str = json.dumps(prop, indent=2)
                    self.code_callback(
                        iteration + 1,
                        "Strategist",
                        code_str,
                        f"Proposal {i+1}/{len(proposals)}",
                    )

            if self.log_callback:
                self.log_callback(
                    iteration + 1, f"{len(proposals)} proposals generated", "INFO"
                )

            # === ÉTAPE 4: Validation Critic ===
            self.logger.info("🔍 Step 4/7: Critic validating proposals...")
            validated = self._validate_proposals(proposals, analysis, current_params)
            self.logger.info(
                f"   → {len(validated)}/{len(proposals)} proposals validated"
            )

            # === ÉTAPE 5: Backtests parallèles ===
            if validated:
                self.logger.info("⚡ Step 5/7: Running parallel backtests...")
                proposal_scores = self._parallel_backtest(validated)
                self.total_backtests += len(validated)
            else:
                self.logger.warning("   ⚠️  No validated proposals, skipping backtests")
                proposal_scores = {}

            # === ÉTAPE 6: Sélection meilleure ===
            self.logger.info("🎯 Step 6/7: Selecting best configuration...")
            if proposal_scores:
                best_proposal = max(
                    proposal_scores.items(), key=lambda x: x[1]["sharpe_ratio"]
                )
                best_proposal_params = best_proposal[0]
                best_proposal_score = best_proposal[1]["sharpe_ratio"]

                # Amélioration ?
                if best_proposal_score > current_score:
                    self.logger.info(
                        f"   ✅ Improvement: {current_score:.3f} → {best_proposal_score:.3f}"
                    )
                    current_params = validated[best_proposal_params]
                    current_score = best_proposal_score
                    stagnation_count = 0
                else:
                    self.logger.info(f"   ⏸️  No improvement (best: {current_score:.3f})")
                    stagnation_count += 1
            else:
                self.logger.info("   ⏸️  No proposals to test")
                stagnation_count += 1

            # === ÉTAPE 7: Mise à jour mémoire ===
            self.logger.info("💾 Step 7/7: Updating memory...")
            self.memory.add(
                iteration_num=iteration + 1,
                params=current_params,
                score=current_score,
                metrics=result.metrics,
                analysis=analysis,
                proposals=proposals,
            )

            # Sauvegarde résultat itération
            iter_result = IterationResult(
                iteration=iteration + 1,
                params=current_params.copy(),
                score=current_score,
                metrics=result.metrics.copy(),
                analysis=analysis,
                proposals=proposals,
                validated_proposals=validated,
                execution_time=time.time() - iter_start,
            )
            self.iterations.append(iter_result)

            # Mise à jour meilleur score
            if current_score > best_score:
                best_score = current_score

            # === Vérification convergence ===
            # Condition 1: Score cible atteint
            if current_score >= self.config.target_sharpe:
                converged = True
                stop_reason = f"Target Sharpe {self.config.target_sharpe} reached"
                self.logger.info(f"🎉 {stop_reason}")
                break

            # Condition 2: Stagnation prolongée
            if stagnation_count >= self.config.convergence_threshold:
                converged = True
                stop_reason = (
                    f"Stagnation detected ({stagnation_count} cycles without improvement)"
                )
                self.logger.info(f"⏹️  {stop_reason}")
                break

            self.logger.info(
                f"📊 Iteration summary: score={current_score:.3f}, "
                f"stagnation={stagnation_count}/{self.config.convergence_threshold}"
            )

        # Si boucle terminée sans convergence
        if not converged:
            stop_reason = f"Max iterations ({self.config.max_iterations}) reached"

        total_time = time.time() - self.start_time

        # Export résultats si configuré
        if self.config.export_dir:
            self._export_results()

        # Résultat final
        final_result = {
            "best_params": current_params,
            "best_score": best_score,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "score": it.score,
                    "execution_time": it.execution_time,
                }
                for it in self.iterations
            ],
            "total_backtests": self.total_backtests,
            "execution_time": total_time,
            "converged": converged,
            "reason": stop_reason,
        }

        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏁 OPTIMIZATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Best Sharpe: {best_score:.3f}")
        self.logger.info(f"Total backtests: {self.total_backtests}")
        self.logger.info(f"Execution time: {total_time:.1f}s")
        self.logger.info(f"Converged: {converged} ({stop_reason})")
        self.logger.info(f"Best params: {current_params}")

        return final_result

    def _run_backtest(self, params: dict[str, Any]) -> RunResult:
        """
        Exécute un backtest avec paramètres donnés.

        Args:
            params: Paramètres stratégie

        Returns:
            RunResult avec métriques + trades
        """
        # Calculer indicateurs requis (Bollinger + ATR pour BacktestEngine)
        indicators = {
            "bollinger": self.indicator_bank.ensure(
                "bollinger",
                {"period": params.get("bb_period", 20), "std": params.get("bb_std", 2.0)},
                self.data,
                symbol="SYNTHETIC",
                timeframe="15m",
            ),
            "atr": self.indicator_bank.ensure(
                "atr",
                {"period": params.get("atr_period", 14)},
                self.data,
                symbol="SYNTHETIC",
                timeframe="15m",
            ),
        }

        # Exécuter backtest avec API réelle
        # BacktestEngine.run(df_1m, indicators, params, symbol, timeframe, use_gpu)
        result = self.backtest_engine.run(
            df_1m=self.data,
            indicators=indicators,
            params=params,
            symbol="SYNTHETIC",
            timeframe="15m",
            use_gpu=False,  # POC en CPU
        )

        return result

    def _analyze_results(self, result: RunResult, params: dict[str, Any]) -> dict[str, Any]:
        """
        Analyse résultats backtest via Analyst.

        Args:
            result: Résultat backtest
            params: Paramètres utilisés

        Returns:
            Analyse JSON structurée
        """
        # Convertir RunResult en JSON LLM avec adapters
        from threadx.llm.adapters import backtest_result_to_llm_json

        backtest_json = backtest_result_to_llm_json(result)

        # Appeler Analyst avec signature réelle: analyze_backtest(backtest_result, params)
        analysis = self.analyst.analyze_backtest(
            backtest_result=backtest_json, params=params
        )

        return analysis

    def _generate_proposals(
        self, current_params: dict[str, Any], analysis: dict[str, Any], n_proposals: int
    ) -> list[dict[str, Any]]:
        """
        Génère propositions via Strategist.

        Args:
            current_params: Params actuels
            analysis: Analyse Analyst
            n_proposals: Nombre propositions à générer

        Returns:
            Liste propositions JSON
        """
        # Définir param_specs (min/max pour chaque paramètre)
        param_specs = {
            "entry_z": {"min": 1.0, "max": 4.0, "type": "float"},
            "k_sl": {"min": 0.5, "max": 3.0, "type": "float"},
            "leverage": {"min": 1, "max": 10, "type": "int"},
            "bb_period": {"min": 10, "max": 50, "type": "int"},
            "bb_std": {"min": 1.5, "max": 3.0, "type": "float"},
            "atr_period": {"min": 7, "max": 30, "type": "int"},
        }

        # Appeler Strategist avec signature réelle: propose_modifications(analysis, current_params, param_specs, n_proposals)
        result = self.strategist.propose_modifications(
            analysis=analysis,
            current_params=current_params,
            param_specs=param_specs,
            n_proposals=n_proposals,
        )

        # Extraire liste propositions du résultat
        proposals = result.get("proposals", [])

        return proposals

    def _validate_proposals(
        self, proposals: list[dict[str, Any]], analysis: dict[str, Any], current_params: dict[str, Any]
    ) -> dict[int, dict[str, Any]]:
        """
        Valide propositions via Critic.

        Args:
            proposals: Liste propositions Strategist
            analysis: Analyse Analyst
            current_params: Paramètres actuels

        Returns:
            Dict {index: validation_result}
        """
        # Définir param_specs (min/max pour validation)
        param_specs = {
            "entry_z": {"min": 1.0, "max": 4.0},
            "k_sl": {"min": 0.5, "max": 3.0},
            "leverage": {"min": 1, "max": 10},
            "bb_period": {"min": 10, "max": 50},
            "bb_std": {"min": 1.5, "max": 3.0},
            "atr_period": {"min": 7, "max": 30},
        }

        # Appeler Critic avec signature réelle: validate_proposals(proposals, analysis, current_params, param_specs)
        validation_result = self.critic.validate_proposals(
            proposals=proposals,
            analysis=analysis,
            current_params=current_params,
            param_specs=param_specs,
        )

        # Extraire propositions validées
        validated = {}
        for prop in proposals:
            prop_id = prop.get("id")
            if prop_id in validation_result.get("propositions_validees", []):
                validated[prop_id] = prop.get("modifications", {})

        return validated

    def _parallel_backtest(
        self, validated_proposals: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, float]]:
        """
        Exécute backtests parallèles pour propositions validées.

        Args:
            validated_proposals: Dict {proposal_id: params}

        Returns:
            Dict {proposal_id: {"sharpe_ratio": float, ...}}
        """
        scores = {}

        # Backtests séquentiels pour l'instant (parallélisation future)
        for prop_id, params in validated_proposals.items():
            try:
                result = self._run_backtest(params)
                scores[prop_id] = result.metrics
            except Exception as e:
                self.logger.error(f"Backtest failed for proposal {prop_id}: {e}")
                scores[prop_id] = {"sharpe_ratio": float("-inf")}

        return scores

    def _export_results(self) -> None:
        """Exporte résultats optimisation dans fichiers JSON."""
        if self.config.export_dir is None:
            self.logger.warning("Export dir not configured, skipping export")
            return

        export_dir = Path(self.config.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export iterations
        iterations_file = export_dir / "iterations.json"
        iterations_data = [
            {
                "iteration": it.iteration,
                "params": it.params,
                "score": it.score,
                "metrics": it.metrics,
                "analysis": it.analysis,
                "execution_time": it.execution_time,
            }
            for it in self.iterations
        ]

        with open(iterations_file, "w") as f:
            json.dump(iterations_data, f, indent=2)

        # Export mémoire
        memory_file = export_dir / "memory.json"
        self.memory.export_to_file(memory_file)

        self.logger.info(f"📁 Results exported to {export_dir}")

    def get_convergence_plot_data(self) -> dict[str, list]:
        """
        Retourne données pour graphique convergence.

        Returns:
            dict avec iterations, scores, execution_times
        """
        return {
            "iterations": [it.iteration for it in self.iterations],
            "scores": [it.score for it in self.iterations],
            "execution_times": [it.execution_time for it in self.iterations],
        }

    def __repr__(self) -> str:
        """Représentation textuelle orchestrateur."""
        return (
            f"OptimizationOrchestrator("
            f"strategy={self.config.strategy_name}, "
            f"iterations={len(self.iterations)}/{self.config.max_iterations}, "
            f"backtests={self.total_backtests})"
        )
