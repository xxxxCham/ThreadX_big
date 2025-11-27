"""
ThreadX - LLM Iteration Engine
==============================

Moteur d'itération pour l'optimisation automatique de stratégies via LLM.

**Workflow complet**:
1. 🔄 Sweep GPU initial → Grid search sur stratégie
2. 🧠 Analyst → Analyse quantitative des résultats
3. 🎨 Strategist → Génération de propositions créatives
4. ✅ Validation → Test des propositions via backtest GPU
5. 🔁 Itération → Si amélioration, recommencer avec nouvelle baseline
6. 🌍 Cross-Testing → Validation sur tokens/timeframes multiples (optionnel)
7. 📊 Rapport final → Export JSON + Markdown

**Architecture**:
- Boucle d'amélioration jusqu'à convergence ou max_iterations
- Critères d'arrêt configurables (min_improvement, max_no_improvement)
- Support multi-token/timeframe pour robustesse
- Intégration avec RunReport pour historique

Usage:
    >>> from threadx.llm.iteration_engine import LLMIterationEngine
    >>>
    >>> engine = LLMIterationEngine(
    ...     analyst_model="deepseek-r1:32b",
    ...     strategist_model="deepseek-r1:14b",
    ...     max_iterations=5,
    ... )
    >>>
    >>> result = engine.run(
    ...     strategy_name="MA_Crossover",
    ...     data=df_market,
    ...     sweep_params=sweep_config,
    ...     baseline_params=baseline,
    ... )
    >>>
    >>> print(f"Amélioration: {result['improvement_pct']:.1f}%")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.strategist import Strategist
from threadx.llm.run_report import (
    LLMRunReport,
    RunIndex,
    create_report_from_run,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)


# ============================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================


@dataclass
class IterationConfig:
    """Configuration du moteur d'itération."""

    # Modèles LLM (vérifiés localement 26/11/2025)
    analyst_model: str = "deepseek-r1:32b"
    strategist_model: str = "deepseek-r1:8b"  # Plus rapide pour itérations

    # Limites d'itération
    max_iterations: int = 5
    max_no_improvement: int = 2  # Arrêt si N itérations sans amélioration

    # Seuils
    min_improvement_pct: float = 1.0  # Minimum 1% d'amélioration pour continuer
    min_sharpe_threshold: float = 0.0  # Ignorer configs avec Sharpe < seuil

    # Propositions
    n_proposals_per_iteration: int = 3
    top_n_analysis: int = 5

    # Cross-testing (optionnel)
    enable_cross_testing: bool = False
    cross_test_tokens: list[str] = field(default_factory=lambda: ["BTCUSDC", "ETHUSDC"])
    cross_test_timeframes: list[str] = field(default_factory=lambda: ["1h", "4h"])

    # Sauvegarde
    save_reports: bool = True
    reports_dir: str = "reports/llm_iterations"

    # Callbacks
    on_iteration_start: Callable[[int, dict], None] | None = None
    on_iteration_end: Callable[[int, dict], None] | None = None
    on_improvement: Callable[[float, float, dict], None] | None = None

    # Debug
    debug: bool = False


@dataclass
class IterationResult:
    """Résultat d'une itération."""

    iteration_num: int
    baseline_sharpe: float
    best_sharpe: float
    improvement_pct: float
    best_proposal_name: str | None
    best_params: dict[str, Any]
    all_proposals_tested: list[dict]
    analyst_patterns: list[str]
    analyst_recommendations: list[str]
    duration_seconds: float
    is_improvement: bool


@dataclass
class EngineResult:
    """Résultat final du moteur d'itération."""

    # Résumé
    strategy_name: str
    total_iterations: int
    total_duration_seconds: float

    # Progression
    initial_sharpe: float
    final_sharpe: float
    total_improvement_pct: float

    # Meilleure configuration
    best_params: dict[str, Any]
    best_proposal_name: str | None

    # Historique
    iterations: list[IterationResult]

    # Cross-testing (si activé)
    cross_test_results: dict[str, dict] | None

    # Convergence
    converged: bool
    convergence_reason: str

    # Rapport
    report: LLMRunReport | None
    report_path: str | None


# ============================================================
# MAIN ENGINE CLASS
# ============================================================


class LLMIterationEngine:
    """
    Moteur d'itération pour optimisation LLM de stratégies.

    Orchestre le cycle complet:
    Sweep → Analyst → Strategist → Tests → (Itération) → Cross-test → Rapport
    """

    def __init__(
        self,
        config: IterationConfig | None = None,
        backtest_fn: Callable | None = None,
        sweep_fn: Callable | None = None,
        **kwargs,
    ):
        """
        Initialise le moteur d'itération.

        Args:
            config: Configuration complète (ou utiliser kwargs)
            backtest_fn: Fonction de backtest personnalisée (optionnel)
                         Signature: (data, strategy_name, params) -> dict avec sharpe_ratio, etc.
            sweep_fn: Fonction de sweep personnalisée (optionnel)
                      Signature: (data, strategy_name, sweep_params) -> list[dict]
            **kwargs: Arguments pour IterationConfig si config non fourni
        """
        # Configuration
        if config:
            self.config = config
        else:
            self.config = IterationConfig(**kwargs)

        # Agents LLM
        self.analyst = Analyst(
            model=self.config.analyst_model,
            debug=self.config.debug,
        )
        self.strategist = Strategist(
            model=self.config.strategist_model,
            debug=self.config.debug,
        )

        # Fonctions de backtest/sweep (injectables pour tests)
        self._backtest_fn = backtest_fn or self._default_backtest
        self._sweep_fn = sweep_fn or self._default_sweep

        # État
        self._current_iteration = 0
        self._iterations_without_improvement = 0
        self._iteration_history: list[IterationResult] = []

        # Logger
        self.logger = get_logger(f"{__name__}.LLMIterationEngine")
        if self.config.debug:
            self.logger.setLevel(logging.DEBUG)

        self.logger.info(
            f"🚀 LLMIterationEngine initialisé "
            f"(analyst={self.config.analyst_model}, strategist={self.config.strategist_model})"
        )

    # ────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        sweep_params: dict[str, list],
        baseline_params: dict[str, Any],
        param_specs: dict[str, dict] | None = None,
        _symbol: str = "BTCUSDC",
        _timeframe: str = "1h",
    ) -> EngineResult:
        """
        Lance le cycle complet d'optimisation itérative.

        Args:
            strategy_name: Nom de la stratégie (ex: "MA_Crossover")
            data: DataFrame OHLCV pour backtests
            sweep_params: Paramètres du sweep {param: [values]}
            baseline_params: Paramètres initiaux (baseline)
            param_specs: Spécifications des paramètres (min/max/step) pour Strategist
            symbol: Symbole du marché
            timeframe: Timeframe des données

        Returns:
            EngineResult avec tous les détails de l'optimisation
        """
        start_time = time.time()

        self.logger.info("=" * 80)
        self.logger.info(f"🎯 DÉMARRAGE OPTIMISATION ITÉRATIVE: {strategy_name}")
        self.logger.info(f"   Max iterations: {self.config.max_iterations}")
        self.logger.info(f"   Baseline params: {baseline_params}")
        self.logger.info("=" * 80)

        # Initialisation
        self._current_iteration = 0
        self._iterations_without_improvement = 0
        self._iteration_history = []

        current_params = baseline_params.copy()
        current_sharpe = self._run_single_backtest(data, strategy_name, current_params)
        initial_sharpe = current_sharpe
        best_sharpe_global = current_sharpe
        best_params_global = current_params.copy()
        best_proposal_name = None

        self.logger.info(f"📊 Baseline Sharpe: {initial_sharpe:.4f}")

        # Variables pour sweep (exécuté une seule fois)
        sweep_results = None
        sweep_duration = 0.0

        # Boucle d'itération
        converged = False
        convergence_reason = ""

        while self._current_iteration < self.config.max_iterations:
            self._current_iteration += 1
            iteration_start = time.time()

            self.logger.info(f"\n{'─' * 60}")
            self.logger.info(
                f"🔄 ITÉRATION {self._current_iteration}/{self.config.max_iterations}"
            )
            self.logger.info(f"   Current Sharpe: {current_sharpe:.4f}")
            self.logger.info(f"{'─' * 60}")

            # Callback début itération
            if self.config.on_iteration_start:
                self.config.on_iteration_start(
                    self._current_iteration,
                    {
                        "current_sharpe": current_sharpe,
                        "current_params": current_params,
                    },
                )

            # ─────────────────────────────────────────────────────────
            # ÉTAPE 1: SWEEP (seulement à la première itération)
            # ─────────────────────────────────────────────────────────
            if sweep_results is None:
                self.logger.info("🔍 Étape 1: Exécution du Sweep GPU...")
                sweep_start = time.time()
                sweep_results = self._sweep_fn(data, strategy_name, sweep_params)
                sweep_duration = time.time() - sweep_start

                if not sweep_results:
                    self.logger.error("❌ Sweep n'a retourné aucun résultat")
                    convergence_reason = "sweep_failed"
                    break

                self.logger.info(
                    f"   ✅ {len(sweep_results)} configs testées en {sweep_duration:.1f}s"
                )

            # ─────────────────────────────────────────────────────────
            # ÉTAPE 2: ANALYSE (Analyst)
            # ─────────────────────────────────────────────────────────
            self.logger.info("🧠 Étape 2: Analyse Analyst...")
            analyst_start = time.time()

            analysis_result = self._run_analyst(sweep_results)
            analyst_duration = time.time() - analyst_start

            patterns = analysis_result.get("analysis", {}).get("patterns", [])
            recommendations = analysis_result.get("analysis", {}).get(
                "recommendations", []
            )

            self.logger.info(
                f"   ✅ {len(patterns)} patterns, {len(recommendations)} recommandations"
            )

            # ─────────────────────────────────────────────────────────
            # ÉTAPE 3: PROPOSITIONS (Strategist)
            # ─────────────────────────────────────────────────────────
            self.logger.info("🎨 Étape 3: Génération Strategist...")
            strategist_start = time.time()

            proposals_result = self._run_strategist(
                analysis_result,
                current_params,
                param_specs or {},
            )
            strategist_duration = time.time() - strategist_start

            proposals = proposals_result.get("proposals", [])
            self.logger.info(f"   ✅ {len(proposals)} propositions générées")

            if not proposals:
                self.logger.warning("⚠️ Aucune proposition valide générée")
                self._iterations_without_improvement += 1

                if (
                    self._iterations_without_improvement
                    >= self.config.max_no_improvement
                ):
                    convergence_reason = "no_proposals"
                    converged = True
                    break
                continue

            # ─────────────────────────────────────────────────────────
            # ÉTAPE 4: TESTS DES PROPOSITIONS
            # ─────────────────────────────────────────────────────────
            self.logger.info("✅ Étape 4: Test des propositions...")

            test_results = self._test_proposals(
                data, strategy_name, proposals, current_params
            )

            # Trouver la meilleure proposition
            best_proposal = None
            best_sharpe_iter = current_sharpe

            for res in test_results:
                sharpe = res.get("sharpe_ratio", 0)
                if sharpe > best_sharpe_iter:
                    best_sharpe_iter = sharpe
                    best_proposal = res

            # ─────────────────────────────────────────────────────────
            # ÉVALUATION DE L'AMÉLIORATION
            # ─────────────────────────────────────────────────────────
            improvement_pct = 0.0
            if current_sharpe != 0:
                improvement_pct = (
                    (best_sharpe_iter - current_sharpe) / abs(current_sharpe)
                ) * 100

            is_improvement = improvement_pct >= self.config.min_improvement_pct

            iteration_duration = time.time() - iteration_start

            # Créer résultat d'itération
            iter_result = IterationResult(
                iteration_num=self._current_iteration,
                baseline_sharpe=current_sharpe,
                best_sharpe=best_sharpe_iter,
                improvement_pct=improvement_pct,
                best_proposal_name=best_proposal.get("name") if best_proposal else None,
                best_params=(
                    best_proposal.get("params", {}) if best_proposal else current_params
                ),
                all_proposals_tested=test_results,
                analyst_patterns=patterns,
                analyst_recommendations=recommendations,
                duration_seconds=iteration_duration,
                is_improvement=is_improvement,
            )
            self._iteration_history.append(iter_result)

            # Log résultat
            if is_improvement and best_proposal is not None:
                self.logger.info(
                    f"🎉 AMÉLIORATION: {current_sharpe:.4f} → {best_sharpe_iter:.4f} ({improvement_pct:+.1f}%)"
                )

                # Callback amélioration
                if self.config.on_improvement:
                    self.config.on_improvement(
                        current_sharpe, best_sharpe_iter, best_proposal
                    )

                # Mettre à jour la baseline
                current_sharpe = best_sharpe_iter
                current_params = best_proposal.get("params", current_params)
                self._iterations_without_improvement = 0

                # Mettre à jour le meilleur global
                if best_sharpe_iter > best_sharpe_global:
                    best_sharpe_global = best_sharpe_iter
                    best_params_global = current_params.copy()
                    best_proposal_name = best_proposal.get("name")
            else:
                self.logger.info(
                    f"❌ Pas d'amélioration significative ({improvement_pct:+.1f}%)"
                )
                self._iterations_without_improvement += 1

            # Callback fin itération
            if self.config.on_iteration_end:
                self.config.on_iteration_end(
                    self._current_iteration, iter_result.__dict__
                )

            # Vérifier critères d'arrêt
            if self._iterations_without_improvement >= self.config.max_no_improvement:
                convergence_reason = "no_improvement"
                converged = True
                self.logger.info(
                    f"🛑 Arrêt: {self.config.max_no_improvement} itérations sans amélioration"
                )
                break

        # Fin de boucle
        if not converged:
            convergence_reason = "max_iterations"
            self.logger.info(
                f"🛑 Arrêt: max_iterations atteint ({self.config.max_iterations})"
            )

        # ─────────────────────────────────────────────────────────────
        # CROSS-TESTING (optionnel)
        # ─────────────────────────────────────────────────────────────
        cross_test_results = None
        if self.config.enable_cross_testing:
            self.logger.info("\n🌍 Cross-testing sur tokens/timeframes multiples...")
            cross_test_results = self._run_cross_testing(
                strategy_name, best_params_global
            )

        # ─────────────────────────────────────────────────────────────
        # GÉNÉRATION DU RAPPORT
        # ─────────────────────────────────────────────────────────────
        total_duration = time.time() - start_time
        total_improvement = (
            ((best_sharpe_global - initial_sharpe) / abs(initial_sharpe)) * 100
            if initial_sharpe != 0
            else 0
        )

        report = None
        report_path = None

        if self.config.save_reports and sweep_results:
            self.logger.info("📁 Génération du rapport...")

            report = create_report_from_run(
                strategy_name=strategy_name,
                sweep_results=sweep_results,
                sweep_params=sweep_params,
                sweep_duration=sweep_duration,
                analysis_result=analysis_result if analysis_result else None,
                analyst_model=self.config.analyst_model,
                analyst_duration=analyst_duration if "analyst_duration" in dir() else 0,
                proposals_result=proposals_result if proposals_result else None,
                baseline_params=baseline_params,
                baseline_sharpe=initial_sharpe,
                strategist_model=self.config.strategist_model,
                strategist_duration=(
                    strategist_duration if "strategist_duration" in dir() else 0
                ),
                test_results=test_results if "test_results" in dir() else [],
                config={
                    "max_iterations": self.config.max_iterations,
                    "total_iterations": self._current_iteration,
                    "convergence_reason": convergence_reason,
                },
            )

            # Sauvegarder
            index = RunIndex(reports_dir=self.config.reports_dir)
            report_path = str(
                index.save_report(
                    report,
                    tags=[
                        strategy_name,
                        f"iter_{self._current_iteration}",
                        convergence_reason,
                    ],
                )
            )

        # ─────────────────────────────────────────────────────────────
        # RÉSULTAT FINAL
        # ─────────────────────────────────────────────────────────────
        result = EngineResult(
            strategy_name=strategy_name,
            total_iterations=self._current_iteration,
            total_duration_seconds=total_duration,
            initial_sharpe=initial_sharpe,
            final_sharpe=best_sharpe_global,
            total_improvement_pct=total_improvement,
            best_params=best_params_global,
            best_proposal_name=best_proposal_name,
            iterations=self._iteration_history,
            cross_test_results=cross_test_results,
            converged=converged,
            convergence_reason=convergence_reason,
            report=report,
            report_path=report_path,
        )

        # Log final
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🏁 OPTIMISATION TERMINÉE")
        self.logger.info(f"   Iterations: {self._current_iteration}")
        self.logger.info(f"   Durée: {total_duration:.1f}s")
        self.logger.info(
            f"   Sharpe: {initial_sharpe:.4f} → {best_sharpe_global:.4f} ({total_improvement:+.1f}%)"
        )
        self.logger.info(f"   Convergence: {convergence_reason}")
        if report_path:
            self.logger.info(f"   Rapport: {report_path}")
        self.logger.info("=" * 80)

        return result

    def run_single_iteration(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        sweep_results: list[dict],
        current_params: dict[str, Any],
        param_specs: dict[str, dict] | None = None,
    ) -> IterationResult:
        """
        Exécute une seule itération (utile pour intégration Streamlit).

        Args:
            strategy_name: Nom de la stratégie
            data: DataFrame OHLCV
            sweep_results: Résultats du sweep (pré-calculés)
            current_params: Paramètres actuels (baseline pour cette itération)
            param_specs: Spécifications des paramètres

        Returns:
            IterationResult de cette itération
        """
        self._current_iteration += 1
        iteration_start = time.time()

        # Sharpe actuel
        current_sharpe = self._run_single_backtest(data, strategy_name, current_params)

        # Analyst
        analysis_result = self._run_analyst(sweep_results)
        patterns = analysis_result.get("analysis", {}).get("patterns", [])
        recommendations = analysis_result.get("analysis", {}).get("recommendations", [])

        # Strategist
        proposals_result = self._run_strategist(
            analysis_result,
            current_params,
            param_specs or {},
        )
        proposals = proposals_result.get("proposals", [])

        # Tests
        test_results = self._test_proposals(
            data, strategy_name, proposals, current_params
        )

        # Meilleure proposition
        best_proposal = None
        best_sharpe_iter = current_sharpe

        for res in test_results:
            sharpe = res.get("sharpe_ratio", 0)
            if sharpe > best_sharpe_iter:
                best_sharpe_iter = sharpe
                best_proposal = res

        # Calcul amélioration
        improvement_pct = 0.0
        if current_sharpe != 0:
            improvement_pct = (
                (best_sharpe_iter - current_sharpe) / abs(current_sharpe)
            ) * 100

        is_improvement = improvement_pct >= self.config.min_improvement_pct
        iteration_duration = time.time() - iteration_start

        return IterationResult(
            iteration_num=self._current_iteration,
            baseline_sharpe=current_sharpe,
            best_sharpe=best_sharpe_iter,
            improvement_pct=improvement_pct,
            best_proposal_name=best_proposal.get("name") if best_proposal else None,
            best_params=(
                best_proposal.get("params", {}) if best_proposal else current_params
            ),
            all_proposals_tested=test_results,
            analyst_patterns=patterns,
            analyst_recommendations=recommendations,
            duration_seconds=iteration_duration,
            is_improvement=is_improvement,
        )

    # ────────────────────────────────────────────────────────────────
    # INTERNAL METHODS
    # ────────────────────────────────────────────────────────────────

    def _run_analyst(self, sweep_results: list[dict]) -> dict:
        """Exécute l'analyse Analyst sur les résultats du sweep."""
        try:
            df_sweep = pd.DataFrame(sweep_results)
            return self.analyst.analyze_sweep_results(
                sweep_df=df_sweep,
                top_n=self.config.top_n_analysis,
            )
        except Exception as e:
            self.logger.error(f"Erreur Analyst: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis": {
                    "patterns": [],
                    "key_metrics": {},
                    "trade_offs": [],
                    "recommendations": [],
                },
            }

    def _run_strategist(
        self,
        analysis_result: dict,
        current_params: dict,
        param_specs: dict,
    ) -> dict:
        """Génère des propositions via Strategist."""
        try:
            return self.strategist.propose_modifications(
                analysis=analysis_result,
                current_params=current_params,
                param_specs=param_specs,
                n_proposals=self.config.n_proposals_per_iteration,
            )
        except Exception as e:
            self.logger.error(f"Erreur Strategist: {e}")
            return {
                "status": "error",
                "error": str(e),
                "proposals": [],
                "total_generated": 0,
                "total_valid": 0,
            }

    def _test_proposals(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        proposals: list[dict],
        baseline_params: dict,
    ) -> list[dict]:
        """Teste chaque proposition via backtest."""
        results = []

        for prop in proposals:
            # Fusionner baseline + proposition
            merged_params = {**baseline_params, **prop.get("params", {})}

            try:
                sharpe = self._run_single_backtest(data, strategy_name, merged_params)

                results.append(
                    {
                        "name": prop.get("name", "Unknown"),
                        "params": merged_params,
                        "sharpe_ratio": sharpe,
                        "success": True,
                    }
                )

                self.logger.debug(f"   ✓ {prop.get('name')}: Sharpe={sharpe:.4f}")

            except Exception as e:
                self.logger.warning(f"   ✗ {prop.get('name')}: {e}")
                results.append(
                    {
                        "name": prop.get("name", "Unknown"),
                        "params": merged_params,
                        "sharpe_ratio": None,
                        "error": str(e),
                        "success": False,
                    }
                )

        return results

    def _run_single_backtest(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        params: dict,
    ) -> float:
        """Exécute un backtest et retourne le Sharpe ratio."""
        result = self._backtest_fn(data, strategy_name, params)

        # Extraire Sharpe selon le format du résultat
        if isinstance(result, dict):
            return float(result.get("sharpe_ratio", result.get("sharpe", 0.0)))
        elif hasattr(result, "metrics"):
            return float(result.metrics.get("sharpe_ratio", 0.0))
        elif hasattr(result, "sharpe_ratio"):
            return float(result.sharpe_ratio)
        else:
            return 0.0

    def _run_cross_testing(
        self,
        _strategy_name: str,
        _params: dict,
    ) -> dict[str, dict]:
        """
        Cross-testing sur tokens/timeframes multiples.

        Note: Implémentation basique - à enrichir selon les besoins.
        """
        results = {}

        for token in self.config.cross_test_tokens:
            for tf in self.config.cross_test_timeframes:
                key = f"{token}_{tf}"
                self.logger.info(f"   Testing {key}...")

                # Note: Dans une vraie implémentation, charger les données
                # correspondantes et exécuter le backtest
                results[key] = {
                    "status": "not_implemented",
                    "note": "Charger les données via load_ohlcv() pour cross-test",
                }

        return results

    # ────────────────────────────────────────────────────────────────
    # DEFAULT IMPLEMENTATIONS (can be overridden)
    # ────────────────────────────────────────────────────────────────

    def _default_backtest(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        params: dict,
    ) -> dict:
        """
        Implémentation par défaut du backtest.

        Utilise run_backtest_gpu si disponible.
        """
        try:
            from threadx.ui.backtest_bridge import run_backtest_gpu

            result = run_backtest_gpu(
                df=data,
                strategy=strategy_name,
                params=params,
            )

            return {
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
                "total_return": result.metrics.get("total_return", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "win_rate": result.metrics.get("win_rate", 0.0),
            }

        except ImportError:
            self.logger.warning("run_backtest_gpu non disponible, utiliser fallback")
            # Fallback simple (mock)
            return {"sharpe_ratio": 0.0}

    def _default_sweep(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        sweep_params: dict,
    ) -> list[dict]:
        """
        Implémentation par défaut du sweep.

        Utilise SweepRunner si disponible.
        """
        try:
            from itertools import product

            from threadx.indicators.bank import IndicatorBank, IndicatorSettings
            from threadx.optimization.engine import SweepRunner
            from threadx.optimization.scenarios import ScenarioSpec

            # Créer IndicatorBank
            settings = IndicatorSettings(use_gpu=True)
            indicator_bank = IndicatorBank(settings)

            # Créer SweepRunner
            runner = SweepRunner(
                indicator_bank=indicator_bank,
                max_workers=30,
                use_multigpu=True,
            )

            # Convertir sweep_params au format ScenarioSpec
            scenario_params = {
                param: {"values": values} for param, values in sweep_params.items()
            }
            scenario = ScenarioSpec(type="grid", params=scenario_params)

            # Exécuter
            results_df = runner.run_grid(
                grid_spec=scenario,
                real_data=data,
                symbol="BTCUSDC",
                timeframe="1h",
                strategy_name=strategy_name,
                reuse_cache=True,
            )

            return list(results_df.to_dict("records"))

        except ImportError as e:
            self.logger.warning(f"SweepRunner non disponible: {e}")

            # Fallback: sweep séquentiel simple
            from itertools import product

            results = []
            param_names = list(sweep_params.keys())
            param_values = list(sweep_params.values())

            for combo in product(*param_values):
                params = dict(zip(param_names, combo))
                result = self._default_backtest(data, strategy_name, params)
                results.append(
                    {
                        "params": params,
                        **result,
                    }
                )

            return results


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "LLMIterationEngine",
    "IterationConfig",
    "IterationResult",
    "EngineResult",
]
