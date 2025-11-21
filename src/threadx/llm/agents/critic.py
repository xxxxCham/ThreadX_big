"""
Agent Critic - Validation et filtrage de propositions de stratégies.

Utilise deepseek-r1:70b pour évaluer la qualité et le risque des propositions
émises par le Strategist.

Intégration Autopsy:
    Si validation échoue → Autopsy analyse échec → Kill Rules Database updated
"""

from typing import Any

from threadx.llm.agents.base_agent import BaseAgent


class Critic(BaseAgent):
    """
    Agent spécialisé dans la validation critique de propositions.

    Capabilities:
    - Évaluer la cohérence des propositions avec l'analyse
    - Détecter les propositions risquées (overfitting, paramètres extrêmes)
    - Filtrer les propositions redondantes ou peu prometteuses
    - Attribuer un score de confiance à chaque proposition
    """

    def __init__(
        self,
        model: str = "deepseek-r1:70b",
        debug: bool = False,
        enable_autopsy: bool = True,
    ) -> None:
        """
        Initialise l'agent Critic.

        Args:
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
    ) -> dict[str, Any]:
        """
        Valide et filtre les propositions du Strategist.

        Args:
            proposals: Liste de propositions générées par Strategist
            analysis: Analyse de l'Analyst (pour vérifier cohérence)
            current_params: Paramètres actuels de la stratégie
            param_specs: Spécifications des paramètres (min/max/type)

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

            return response

        except Exception as e:
            self.logger.error("Error during validation: %s", e)
            # Fallback: accepter toutes les propositions avec score conservatif
            return {
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
