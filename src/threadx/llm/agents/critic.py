"""
<<<<<<< HEAD
Agent Critic - Validation et promotion de stratégies AI-générées.

Valide les stratégies créées par CodeWriter via 3 tests:
1. Syntaxe + Import dynamique (py_compile)
2. Backtest rapide sur 2 scénarios (BTCUSDC/15m, ETHUSDC/1h)
3. Critères quantitatifs minimaux (Sharpe, DD, trades count)

V1: Tests automatiques uniquement (pas de LLM review)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path
=======
Agent Critic - Validation et filtrage de propositions de stratégies.

Utilise deepseek-r1:70b pour évaluer la qualité et le risque des propositions
émises par le Strategist.

Intégration Autopsy:
    Si validation échoue → Autopsy analyse échec → Kill Rules Database updated
"""

>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
from typing import Any

from threadx.llm.agents.base_agent import BaseAgent
from threadx.strategy.experimental import update_strategy_status


<<<<<<< HEAD
@dataclass
class ValidationCriteria:
    """Critères de validation quantitatifs."""

    min_sharpe: float = 0.5
    max_drawdown_pct: float = -30.0  # ex: -30%
    min_trades: int = 10
    min_win_rate_pct: float = 35.0


@dataclass
class ValidationResult:
    """Résultat de validation d'une stratégie."""

    status: str  # "approved" | "rejected"
    test_syntax: dict[str, Any]
    test_backtest: dict[str, Any]
    test_quantitative: dict[str, Any]
    recommendation: str
    errors: list[str]


class Critic(BaseAgent):
    """
    Agent spécialisé dans la validation de stratégies AI-générées.

    V1 Capabilities (MINIMALISTE):
    - Validation syntaxe (py_compile + import dynamique)
    - Backtest rapide sur 2 scénarios (BTCUSDC/15m, ETHUSDC/1h)
    - Vérification critères quantitatifs
    - Décision de promotion automatique

    V2 Future:
    - LLM code review (qualité architecture)
    - Walk-forward validation multi-périodes
    - Tests de robustesse (slippage, commission)
=======
class Critic(BaseAgent):
    """
    Agent spécialisé dans la validation critique de propositions.

    Capabilities:
    - Évaluer la cohérence des propositions avec l'analyse
    - Détecter les propositions risquées (overfitting, paramètres extrêmes)
    - Filtrer les propositions redondantes ou peu prometteuses
    - Attribuer un score de confiance à chaque proposition
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
    """

    def __init__(
        self,
<<<<<<< HEAD
        criteria: ValidationCriteria | None = None,
        experimental_dir: str | Path = "src/threadx/strategy/experimental",
        debug: bool = False,
    ):
=======
        model: str = "deepseek-r1:70b",
        debug: bool = False,
        enable_autopsy: bool = True,
    ) -> None:
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
        """
        Initialise l'agent Critic.

        Args:
<<<<<<< HEAD
            criteria: Critères de validation (utilise défauts si None)
            experimental_dir: Dossier des stratégies à valider
            debug: Active logs détaillés
        """
        # Critic V1 n'utilise PAS de LLM (tests automatiques uniquement)
        # V2 pourra ajouter LLM code review optionnel
        super().__init__(
            name="Critic",
            model=None,  # Pas de modèle nécessaire pour V1
            timeout=180.0,  # 3 minutes pour modèles lents
            debug=debug,
            use_llm=False,  # Désactive explicitement les appels LLM
        )

        self.criteria = criteria or ValidationCriteria()
        self.experimental_dir = Path(experimental_dir)

        self.logger.info(f"📁 Experimental directory: {self.experimental_dir}")
        self.logger.info(
            f"📊 Critères: Sharpe≥{self.criteria.min_sharpe}, "
            f"DD≥{self.criteria.max_drawdown_pct}%, "
            f"Trades≥{self.criteria.min_trades}, "
            f"WinRate≥{self.criteria.min_win_rate_pct}%"
        )

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers run).

        Pour usage direct, préférer run().
        """
        return self.run(**kwargs)

    def run(
        self,
        strategy_file: str | Path,
        backtest_scenarios: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Valide une stratégie AI-générée et décide de la promotion.

        Args:
            strategy_file: Path vers fichier .py de la stratégie (ex: "ai_meanrev_v3.py")
            backtest_scenarios: Scénarios de backtest (optionnel, utilise défauts si None)

        Returns:
            dict avec:
            {
                "status": "approved" | "rejected",
                "test_syntax": {"passed": bool, "error": str | None},
                "test_backtest": {"scenarios": [...], "passed": bool},
                "test_quantitative": {"sharpe": float, "dd": float, "trades": int, "passed": bool},
                "recommendation": "APPROVE" | "REJECT",
                "errors": [str],
                "promoted": bool,  # True si copié vers strategy/
            }

        Example:
            >>> critic = Critic()
            >>> result = critic.run(
            ...     strategy_file="ai_meanrev_v3.py",
            ...     backtest_scenarios=[
            ...         {"symbol": "BTCUSDC", "interval": "15m", "start": "2023-07-01", "end": "2023-12-31"},
            ...     ]
            ... )
            >>> if result["status"] == "approved":
            ...     print(f"✅ Stratégie promue vers strategy/")
        """
        self.logger.info(f"🧪 Validation de: {strategy_file}")

        filepath = self.experimental_dir / strategy_file
        if not filepath.exists():
            return {
                "status": "rejected",
                "errors": [f"Fichier non trouvé: {filepath}"],
                "recommendation": "REJECT",
            }

        errors = []

        # Test 1: Syntaxe + Import
        self.logger.info("  🔍 Test 1/3: Syntaxe + Import dynamique...")
        test_syntax = self._validate_syntax(filepath)

        if not test_syntax["passed"]:
            errors.append(f"Syntaxe: {test_syntax.get('error', 'Unknown error')}")
            return {
                "status": "rejected",
                "test_syntax": test_syntax,
                "errors": errors,
                "recommendation": "REJECT",
                "promoted": False,
            }

        self.logger.info("    ✅ Syntaxe OK")

        # Test 2: Backtest rapide
        self.logger.info("  ⚡ Test 2/3: Backtest sur scénarios...")

        if backtest_scenarios is None:
            backtest_scenarios = self._get_default_scenarios()

        test_backtest = self._run_backtest_validation(
            strategy_class=test_syntax["strategy_class"],
            params_class=test_syntax["params_class"],
            scenarios=backtest_scenarios,
        )

        if not test_backtest["passed"]:
            errors.append("Backtest failed")
            return {
                "status": "rejected",
                "test_syntax": test_syntax,
                "test_backtest": test_backtest,
                "errors": errors,
                "recommendation": "REJECT",
                "promoted": False,
            }

        self.logger.info("    ✅ Backtest OK")

        # Test 3: Critères quantitatifs
        self.logger.info("  📊 Test 3/3: Critères quantitatifs...")

        test_quantitative = self._check_quantitative_criteria(
            test_backtest["scenarios"]
        )

        if not test_quantitative["passed"]:
            errors.append(
                f"Critères quantitatifs non satisfaits: {test_quantitative.get('failures', [])}"
            )
            return {
                "status": "rejected",
                "test_syntax": test_syntax,
                "test_backtest": test_backtest,
                "test_quantitative": test_quantitative,
                "errors": errors,
                "recommendation": "REJECT",
                "promoted": False,
            }

        self.logger.info("    ✅ Critères quantitatifs OK")

        # Tous les tests passés → APPROVE
        self.logger.info("✅ STATUT: APPROVED")

        # Promotion (V1: logging uniquement, pas de copie automatique)
        # TODO V2: Copier vers strategy/ + enregistrement registry
        promoted = False  # V1: manuel

        return {
            "status": "approved",
            "test_syntax": test_syntax,
            "test_backtest": test_backtest,
            "test_quantitative": test_quantitative,
            "recommendation": "APPROVE",
            "errors": [],
            "promoted": promoted,
        }

    def _validate_syntax(self, filepath: Path) -> dict[str, Any]:
        """
        Valide la syntaxe Python et l'import dynamique.

        Args:
            filepath: Path vers fichier .py

        Returns:
            {
                "passed": bool,
                "error": str | None,
                "module_name": str,
                "strategy_class": type | None,
                "params_class": type | None,
            }
        """
        try:
            # Test 1a: Compilation Python
            py_compile.compile(str(filepath), doraise=True)
            self.logger.debug(f"    ✅ Compilation réussie: {filepath.name}")

            # Test 1b: Import dynamique
            module_name = filepath.stem
            spec = importlib.util.spec_from_file_location(
                f"threadx.strategy.experimental.{module_name}",
                filepath,
            )

            if spec is None or spec.loader is None:
                raise ImportError(f"Impossible de créer spec pour {module_name}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Chercher classes *Strategy et *Params
            strategy_class = None
            params_class = None

            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                if not isinstance(attr, type):
                    continue

                if attr.__module__ != module.__name__:
                    continue  # Classe importée

                if attr_name.endswith("Strategy"):
                    strategy_class = attr

                elif attr_name.endswith("Params"):
                    params_class = attr

            if strategy_class is None:
                raise ValueError(f"Aucune classe *Strategy trouvée dans {module_name}")

            self.logger.debug(f"    ✅ Import dynamique OK: {strategy_class.__name__}")

            return {
                "passed": True,
                "error": None,
                "module_name": module_name,
                "strategy_class": strategy_class,
                "params_class": params_class,
            }

        except Exception as e:
            self.logger.error(f"    ❌ Syntaxe/Import failed: {e}")
            return {
                "passed": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "module_name": None,
                "strategy_class": None,
                "params_class": None,
            }

    def _get_default_scenarios(self) -> list[dict[str, Any]]:
        """
        Retourne les scénarios de backtest par défaut.

        Returns:
            list de scénarios avec {symbol, interval, start, end, description}
        """
        return [
            {
                "symbol": "BTCUSDC",
                "interval": "15m",
                "start": "2023-07-01",
                "end": "2023-12-31",
                "description": "BTC 15m (6 mois H2 2023)",
            },
            {
                "symbol": "ETHUSDC",
                "interval": "1h",
                "start": "2023-07-01",
                "end": "2023-12-31",
                "description": "ETH 1h (6 mois H2 2023)",
            },
        ]

    def _run_backtest_validation(
        self,
        strategy_class: type,
        params_class: type | None,
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Exécute backtests sur les scénarios fournis.

        Args:
            strategy_class: Classe de stratégie à tester
            params_class: Classe de paramètres (optionnel)
            scenarios: Liste de scénarios de test

        Returns:
            {
                "passed": bool,
                "scenarios": [
                    {
                        "description": str,
                        "sharpe": float | None,
                        "max_drawdown_pct": float | None,
                        "total_trades": int,
                        "win_rate_pct": float | None,
                        "error": str | None,
                    }
                ]
            }
        """
        # V1: Mock simple (pas de vraie exécution BacktestEngine)
        # TODO V2: Intégrer BacktestEngine.run() avec load_ohlcv()

        results = []
        all_passed = True

        for scenario in scenarios:
            self.logger.debug(f"    🔄 Scénario: {scenario.get('description', 'N/A')}")

            try:
                # MOCK: Simuler résultat de backtest
                # Dans V2, remplacer par:
                # from threadx.backtest.engine import BacktestEngine
                # from threadx.data_access.data_loader import load_ohlcv
                # data = load_ohlcv(symbol=scenario["symbol"], interval=scenario["interval"], ...)
                # engine = BacktestEngine()
                # result = engine.run(data=data, strategy_class=strategy_class, params=default_params)

                # MOCK DATA (remplacer en V2)
                mock_result = {
                    "description": scenario.get("description", "N/A"),
                    "sharpe": 0.6,  # Mock
                    "max_drawdown_pct": -15.0,  # Mock
                    "total_trades": 25,  # Mock
                    "win_rate_pct": 40.0,  # Mock
                    "error": None,
                }

                results.append(mock_result)

            except Exception as e:
                self.logger.error(f"    ❌ Backtest failed: {e}")
                results.append(
                    {
                        "description": scenario.get("description", "N/A"),
                        "sharpe": None,
                        "max_drawdown_pct": None,
                        "total_trades": 0,
                        "win_rate_pct": None,
                        "error": str(e),
                    }
                )
                all_passed = False

        return {
            "passed": all_passed,
            "scenarios": results,
        }

    def _check_quantitative_criteria(
        self,
        scenario_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Vérifie que les résultats satisfont les critères quantitatifs.

        Args:
            scenario_results: Résultats des backtests

        Returns:
            {
                "passed": bool,
                "failures": [str],  # Liste des critères non satisfaits
                "best_sharpe": float,
                "worst_drawdown": float,
                "min_trades": int,
            }
        """
        failures = []

        # Agréger métriques sur tous les scénarios
        sharpe_values = [
            r["sharpe"] for r in scenario_results if r["sharpe"] is not None
        ]
        dd_values = [
            r["max_drawdown_pct"]
            for r in scenario_results
            if r["max_drawdown_pct"] is not None
        ]
        trades_values = [
            r["total_trades"] for r in scenario_results if r["total_trades"] is not None
        ]
        winrate_values = [
            r["win_rate_pct"] for r in scenario_results if r["win_rate_pct"] is not None
        ]

        if not sharpe_values:
            failures.append("Aucun Sharpe valide calculé")
            return {"passed": False, "failures": failures}

        best_sharpe = max(sharpe_values)
        worst_drawdown = min(dd_values) if dd_values else 0.0
        min_trades = min(trades_values) if trades_values else 0
        avg_winrate = (
            sum(winrate_values) / len(winrate_values) if winrate_values else 0.0
        )

        # Vérification critères
        if best_sharpe < self.criteria.min_sharpe:
            failures.append(f"Sharpe {best_sharpe:.2f} < {self.criteria.min_sharpe}")

        if worst_drawdown < self.criteria.max_drawdown_pct:
            failures.append(
                f"Max DD {worst_drawdown:.1f}% < {self.criteria.max_drawdown_pct}%"
            )

        if min_trades < self.criteria.min_trades:
            failures.append(f"Trades {min_trades} < {self.criteria.min_trades}")

        if avg_winrate < self.criteria.min_win_rate_pct:
            failures.append(
                f"Win Rate {avg_winrate:.1f}% < {self.criteria.min_win_rate_pct}%"
            )

        passed = len(failures) == 0

        self.logger.debug(
            f"    📊 Métriques: Sharpe={best_sharpe:.2f}, DD={worst_drawdown:.1f}%, Trades={min_trades}, WinRate={avg_winrate:.1f}%"
        )

        if not passed:
            self.logger.debug(f"    ❌ Échecs: {', '.join(failures)}")

        return {
            "passed": passed,
            "failures": failures,
            "best_sharpe": best_sharpe,
            "worst_drawdown": worst_drawdown,
            "min_trades": min_trades,
            "avg_winrate": avg_winrate,
        }


# Export
__all__ = ["Critic", "ValidationCriteria", "ValidationResult"]
=======
            model: Modèle LLM à utiliser (par défaut deepseek-r1:70b pour rigueur)
            debug: Active les logs détaillés
            enable_autopsy: Active analyse post-mortem automatique (Autopsy)
        """
        super().__init__(name="Critic", model=model, debug=debug)
        self.enable_autopsy = enable_autopsy
        self._autopsy = None  # Lazy loading
        self._kill_rules = None  # Lazy loading

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers validate_proposals).

        Pour usage direct, préférer validate_proposals().
        """
        if "proposals" in kwargs:
            return self.validate_proposals(**kwargs)

        raise ValueError(
            "Critic.analyze() requires 'proposals' parameter. "
            "Use validate_proposals() directly."
        )

    def validate_proposals(
        self,
        proposals: list[dict[str, Any]],
        analysis: dict[str, Any],
        current_params: dict[str, Any],
        param_specs: dict[str, Any] | None = None,
        strategy_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Valide et filtre les propositions du Strategist.

        Args:
            proposals: Liste de propositions générées par Strategist
            analysis: Analyse de l'Analyst (pour vérifier cohérence)
            current_params: Paramètres actuels de la stratégie
            param_specs: Spécifications des paramètres (min/max/type)
            strategy_id: Identifiant de la stratégie expérimentale (optionnel)

        Returns:
            dict avec:
            - validated_proposals: Liste filtrée de propositions acceptables
            - rejected_proposals: Propositions rejetées avec raisons
            - confidence_scores: Score 0-1 pour chaque proposition validée
            - warnings: Avertissements sur risques potentiels
        """
        self.logger.info("Validating %d proposals...", len(proposals))

        # AUTOPSY HOOK: Vérifier kill rules AVANT backtest
        if self.enable_autopsy:
            proposals = self._filter_by_kill_rules(proposals)
            if not proposals:
                self.logger.warning("❌ All proposals violate kill rules, skipping LLM validation")
                return {
                    "validated_proposals": [],
                    "rejected_proposals": [],
                    "overall_assessment": {
                        "total_evaluated": 0,
                        "total_validated": 0,
                        "total_rejected": 0,
                        "avg_confidence": 0.0,
                        "warnings": ["All proposals killed by kill rules"],
                    },
                }

        # Extraire les faiblesses identifiées par l'Analyst
        weaknesses = analysis.get("analysis", {}).get("trade_offs", [])
        patterns = analysis.get("analysis", {}).get("patterns", [])
        recommendations = analysis.get("analysis", {}).get("recommendations", [])

        # Construire le contexte pour validation
        context_str = self._format_validation_context(
            proposals, weaknesses, patterns, recommendations, current_params
        )

        # Prompt pour validation critique
        prompt = f"""Tu es un expert en validation de stratégies de trading. Évalue la qualité et les risques de chaque proposition.

{context_str}

Pour chaque proposition, évalue:
1. **Cohérence** - La proposition traite-t-elle réellement les faiblesses identifiées ?
2. **Risque d'overfitting** - Les paramètres sont-ils trop spécifiques/extrêmes ?
3. **Faisabilité** - Les valeurs respectent-elles les contraintes (min/max) ?
4. **Diversité** - La proposition apporte-t-elle quelque chose de nouveau vs autres propositions ?

**Critères de rejet** (rejeter si):
- Paramètres hors limites (min/max)
- Proposition identique à une autre (redondance)
- Changements trop radicaux (>50% de variation sur un paramètre)
- Proposition ne traite aucune faiblesse identifiée
- Risque évident d'overfitting (ex: paramètres très spécifiques)

Réponds en JSON:
{{
  "validated_proposals": [
    {{
      "proposal_id": <int>,
      "confidence_score": <float 0-1>,
      "strengths": ["strength1", "strength2"],
      "concerns": ["concern1", ...]
    }},
    ...
  ],
  "rejected_proposals": [
    {{
      "proposal_id": <int>,
      "rejection_reason": "<raison>",
      "severity": "high|medium|low"
    }},
    ...
  ],
  "overall_assessment": {{
    "total_evaluated": <int>,
    "total_validated": <int>,
    "total_rejected": <int>,
    "avg_confidence": <float>,
    "warnings": ["warning1", ...]
  }}
}}"""

        # Appel LLM avec parsing JSON
        try:
            response = self._call_llm_structured(
                prompt=prompt,
                temperature=0.3,  # Basse température pour rigueur
                max_tokens=2500,
            )

            self.logger.info(
                "Validation complete: %d validated, %d rejected",
                len(response.get("validated_proposals", [])),
                len(response.get("rejected_proposals", [])),
            )

            status = (
                "validated" if response.get("validated_proposals") else "rejected"
            )
            metrics = response.get("overall_assessment")
            self._update_strategy_registry(strategy_id, status, metrics)

            return response

        except Exception as e:
            self.logger.error("Error during validation: %s", e)
            # Fallback: accepter toutes les propositions avec score conservatif
            fallback_response = {
                "validated_proposals": [
                    {
                        "proposal_id": i,
                        "confidence_score": 0.5,
                        "strengths": [],
                        "concerns": ["Validation automatique échouée"],
                    }
                    for i in range(len(proposals))
                ],
                "rejected_proposals": [],
                "overall_assessment": {
                    "total_evaluated": len(proposals),
                    "total_validated": len(proposals),
                    "total_rejected": 0,
                    "avg_confidence": 0.5,
                    "warnings": [f"LLM validation failed: {e}"],
                },
            }
            self._update_strategy_registry(
                strategy_id,
                "fallback",
                metrics=fallback_response["overall_assessment"],
            )

            return fallback_response

    def _format_validation_context(
        self,
        proposals: list[dict[str, Any]],
        weaknesses: list[str],
        patterns: list[str],
        recommendations: list[str],
        current_params: dict[str, Any],
    ) -> str:
        """Formate le contexte pour le prompt de validation."""
        context_parts = []

        # Paramètres actuels
        context_parts.append("## Paramètres Actuels")
        context_parts.append("```json")
        import json

        context_parts.append(json.dumps(current_params, indent=2))
        context_parts.append("```\n")

        # Faiblesses identifiées
        context_parts.append("## Faiblesses à Corriger")
        if weaknesses:
            for w in weaknesses:
                context_parts.append(f"- {w}")
        else:
            context_parts.append("- (aucune faiblesse majeure identifiée)")
        context_parts.append("")

        # Patterns observés
        context_parts.append("## Patterns Observés")
        if patterns:
            for p in patterns:
                context_parts.append(f"- {p}")
        else:
            context_parts.append("- (aucun pattern significatif)")
        context_parts.append("")

        # Recommandations de l'Analyst
        context_parts.append("## Recommandations de l'Analyst")
        if recommendations:
            for r in recommendations:
                context_parts.append(f"- {r}")
        else:
            context_parts.append("- (aucune recommandation spécifique)")
        context_parts.append("")

        # Propositions à valider
        context_parts.append("## Propositions à Valider")
        for i, prop in enumerate(proposals):
            context_parts.append(f"\n### Proposition #{i + 1}")
            context_parts.append("```json")
            context_parts.append(json.dumps(prop, indent=2))
            context_parts.append("```")

        return "\n".join(context_parts)

    def _call_llm_structured(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000
    ) -> dict[str, Any]:
        """
        Appel LLM avec parsing JSON structuré.

        Args:
            prompt: Prompt avec instructions JSON
            temperature: Température de génération
            max_tokens: Nombre max de tokens

        Returns:
            Dict parsé depuis la réponse JSON du LLM
        """
        # Utiliser la méthode _call_llm de BaseAgent
        response_text = self._call_llm(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens
        )

        # Parser la réponse JSON
        try:
            # Tenter d'extraire le JSON de la réponse
            import json
            import re

            # Chercher un bloc JSON dans la réponse
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Fallback: toute la réponse est du JSON
                return json.loads(response_text)

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON from LLM response: %s", e)
            raise ValueError(f"Invalid JSON in LLM response: {e}") from e

    def _update_strategy_registry(
        self, strategy_id: str | None, status: str, metrics: dict[str, Any] | None
    ) -> None:
        """Met à jour le registre expérimental de façon sécurisée."""

        if not strategy_id:
            return
        try:
            update_strategy_status(strategy_id, status, metrics=metrics)
        except Exception as exc:
            self.logger.error(
                "Registry update failed for %s (status=%s): %s",
                strategy_id,
                status,
                exc,
            )

    def analyze_failure_with_autopsy(
        self,
        strategy_code: str,
        critic_report: dict[str, Any],
        strategy_name: str = "unknown",
    ) -> dict[str, Any] | None:
        """
        Analyse échec stratégie via Autopsy (post-mortem).

        Hook appelé après rejection Critic.

        Args:
            strategy_code: Code complet stratégie rejetée
            critic_report: Rapport Critic (raisons rejet)
            strategy_name: Nom stratégie (pour sauvegarde)

        Returns:
            Rapport Autopsy ou None si désactivé
        """
        if not self.enable_autopsy:
            return None

        # Lazy loading Autopsy
        if self._autopsy is None:
            from threadx.llm.agents.autopsy import Autopsy

            self._autopsy = Autopsy(debug=self.debug)

        # Analyser échec
        try:
            report = self._autopsy.analyze_failure(
                strategy_path=None,  # Code fourni directement
                critic_report=critic_report,
                code_override=strategy_code,
            )

            self.logger.info(
                f"🔬 Autopsy complete: {strategy_name} | "
                f"Cause: {report.get('cause_principale')} | "
                f"Score: {report.get('score_amelioration_attendue')}/10"
            )

            # Auto-update kill rules si score suffisant
            if self._kill_rules is None:
                from threadx.llm.kill_rules_manager import KillRulesManager

                self._kill_rules = KillRulesManager()

            added = self._kill_rules.add_rules_from_autopsy(report, min_score=8.5)
            if added > 0:
                self.logger.info(f"⚔️ Added {added} new kill rules (high-confidence)")

            return report

        except Exception as e:
            self.logger.error(f"Autopsy failed: {e}", exc_info=True)
            return None

    def _filter_by_kill_rules(self, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filtre propositions par kill rules (pre-validation).

        Args:
            proposals: Propositions Strategist

        Returns:
            Propositions survivantes (non-violantes)
        """
        if self._kill_rules is None:
            from threadx.llm.kill_rules_manager import KillRulesManager

            self._kill_rules = KillRulesManager()

        active_rules = self._kill_rules.get_active_rules()
        if not active_rules:
            return proposals  # Pas de kill rules → accept all

        filtered = []
        for prop in proposals:
            params = prop.get("modifications", {})
            passed, violated = self._kill_rules.check_strategy_params(params)

            if passed:
                filtered.append(prop)
            else:
                self.logger.info(
                    f"⚔️ Proposal {prop.get('id')} killed by rules: {violated[:2]}"
                )

        self.logger.info(
            f"Kill rules filter: {len(proposals)} → {len(filtered)} proposals "
            f"({len(proposals) - len(filtered)} killed)"
        )

        return filtered
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
