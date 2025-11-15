"""
Agent Strategist - Génération créative de propositions de stratégies.

Utilise gpt-oss:20b pour proposer des modifications de paramètres basées
sur les analyses de l'Analyst.
"""

from typing import Any

from threadx.llm.agents.base_agent import BaseAgent


class Strategist(BaseAgent):
    """
    Agent spécialisé dans la génération de propositions créatives.

    Capabilities:
    - Proposer N modifications de paramètres basées sur une analyse
    - Valider que les propositions respectent les contraintes (min/max)
    - Formater les propositions pour exécution automatique
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        debug: bool = False,
    ) -> None:
        """
        Initialise l'agent Strategist.

        Args:
            model: Modèle LLM à utiliser (par défaut gpt-oss:20b pour créativité)
            debug: Active les logs détaillés
        """
        super().__init__(name="Strategist", model=model, debug=debug)

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers propose_modifications).

        Pour usage direct, préférer propose_modifications().
        """
        if "analysis" in kwargs:
            return self.propose_modifications(**kwargs)

        raise ValueError(
            "Strategist.analyze() requires 'analysis' parameter. "
            "Use propose_modifications() directly."
        )

    def propose_modifications(
        self,
        analysis: dict[str, Any],
        current_params: dict[str, Any],
        param_specs: dict[str, dict[str, Any]],
        n_proposals: int = 3,
    ) -> dict[str, Any]:
        """
        Propose N modifications de paramètres basées sur une analyse.

        Args:
            analysis: Résultat de Analyst.analyze_sweep_results()
            current_params: Paramètres de la config actuelle/baseline
            param_specs: Specs des paramètres (ex: {"short_period": {"min": 5, "max": 50, "type": "int"}})
            n_proposals: Nombre de propositions à générer

        Returns:
            dict avec:
            - proposals: Liste de N dicts de paramètres modifiés
            - rationale: Justifications pour chaque proposition
        """
        self.logger.info("Generating %d parameter proposals...", n_proposals)

        # Extraire insights clés de l'analyse
        patterns = analysis.get("analysis", {}).get("patterns", [])
        recommendations = analysis.get("analysis", {}).get("recommendations", [])
        trade_offs = analysis.get("analysis", {}).get("trade_offs", [])

        # Construire contexte pour LLM
        context_str = self._format_analysis_context(
            patterns, recommendations, trade_offs, current_params, param_specs
        )

        # Prompt pour génération créative avec consignes système
        system_instructions = """
🎯 OBJECTIFS PRIORITAIRES:
- Maximiser le Sharpe Ratio (risque/rendement optimal)
- Minimiser le drawdown maximum (protection du capital)
- Maintenir un win rate > 50% (cohérence stratégique)
- Optimiser le nombre de trades (ni trop, ni trop peu)

📊 APPROCHE DE PROPOSITION:
- Modifications incrémentielles (pas de changements brutaux)
- Exploiter les patterns identifiés dans les meilleures configs
- Tester des zones peu explorées (diversification)
- Valider la cohérence logique des propositions

⚠️ CONTRAINTES CRITIQUES:
- risk_per_trade: TOUJOURS dans [0.005, 0.02]
- max_hold_bars: Adapter selon volatilité (range typique 20-150)
- Stop Loss / Take Profit: Ratio minimum 1:1.5 (asymétrie favorable)
- Respecter STRICTEMENT les plages min/max des paramètres

💡 PRINCIPES:
- Privilégier robustesse > performance brute (éviter overfitting)
- Documenter clairement le raisonnement (transparence)
- 3 approches: Conservative (stabilité), Aggressive (rendement), Exploratoire (découverte)
- Chaque proposition doit être testable immédiatement
"""

        prompt = f"""{system_instructions}

Tu es un expert en optimisation de stratégies de trading. Génère {n_proposals} propositions de modifications de paramètres.

{context_str}

Génère {n_proposals} propositions **différentes et créatives**:
1. Une approche conservative (petites modifications, réduire risque)
2. Une approche aggressive (exploiter patterns identifiés, maximiser Sharpe)
3. Une approche exploratoire (tester zones peu explorées)

**Contraintes strictes**:
- Respecter les min/max de chaque paramètre
- Propositions doivent être testables (valeurs concrètes)
- Justifier chaque modification

Réponds en JSON:
{{
  "proposals": [
    {{
      "name": "Conservative",
      "params": {{"short_period": 12, "long_period": 35, ...}},
      "rationale": "Réduit drawdown observé en augmentant long_period..."
    }},
    ...
  ]
}}
"""

        # Appel LLM structuré
        result = self._call_llm_structured(
            prompt=prompt,
            expected_schema={
                "proposals": list,
            },
            temperature=0.8,  # Haute température pour créativité
            max_tokens=2500,
        )

        # Valider et filtrer propositions
        validated_proposals = self._validate_and_filter_proposals(
            result.get("proposals", []), param_specs
        )

        self.logger.info("Generated %d valid proposals", len(validated_proposals))

        return {
            "proposals": validated_proposals,
            "total_generated": len(result.get("proposals", [])),
            "total_valid": len(validated_proposals),
        }

    def validate_constraints(
        self, proposals: list[dict[str, Any]], param_specs: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Valide que les propositions respectent les contraintes min/max.

        Args:
            proposals: Liste de propositions (dicts avec 'params' key)
            param_specs: Specs des paramètres avec min/max

        Returns:
            Liste filtrée des propositions valides uniquement
        """
        valid = []

        for prop in proposals:
            params = prop.get("params", {})
            is_valid = True

            for param_name, value in params.items():
                if param_name not in param_specs:
                    self.logger.warning(
                        "Unknown parameter '%s' in proposal, skipping", param_name
                    )
                    is_valid = False
                    break

                spec = param_specs[param_name]
                min_val = spec.get("min")
                max_val = spec.get("max")

                if min_val is not None and value < min_val:
                    self.logger.warning(
                        "Parameter %s = %s below min %s, skipping proposal",
                        param_name,
                        value,
                        min_val,
                    )
                    is_valid = False
                    break

                if max_val is not None and value > max_val:
                    self.logger.warning(
                        "Parameter %s = %s above max %s, skipping proposal",
                        param_name,
                        value,
                        max_val,
                    )
                    is_valid = False
                    break

            if is_valid:
                valid.append(prop)

        return valid

    def format_proposals(self, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Formate les propositions pour exécution automatique (ScenarioSpec compatible).

        Args:
            proposals: Liste de propositions brutes du LLM

        Returns:
            Liste de propositions formatées avec clés standardisées
        """
        formatted = []

        for i, prop in enumerate(proposals):
            formatted.append(
                {
                    "proposal_id": i + 1,
                    "name": prop.get("name", f"Proposal_{i+1}"),
                    "params": prop.get("params", {}),
                    "rationale": prop.get("rationale", ""),
                }
            )

        return formatted

    # --- Méthodes privées ---

    def _format_analysis_context(
        self,
        patterns: list[str],
        recommendations: list[str],
        trade_offs: list[str],
        current_params: dict[str, Any],
        param_specs: dict[str, dict[str, Any]],
    ) -> str:
        """Formate le contexte d'analyse pour le LLM."""
        lines = ["**Analyse des résultats:**"]

        if patterns:
            lines.append("\nPatterns identifiés:")
            for p in patterns:
                lines.append(f"  - {p}")

        if recommendations:
            lines.append("\nRecommandations:")
            for r in recommendations:
                lines.append(f"  - {r}")

        if trade_offs:
            lines.append("\nTrade-offs observés:")
            for t in trade_offs:
                lines.append(f"  - {t}")

        lines.append("\n**Paramètres actuels (baseline):**")
        for k, v in current_params.items():
            lines.append(f"  {k}: {v}")

        lines.append("\n**Contraintes des paramètres:**")
        for param, spec in param_specs.items():
            min_val = spec.get("min", "N/A")
            max_val = spec.get("max", "N/A")
            param_type = spec.get("type", "unknown")
            lines.append(f"  {param}: type={param_type}, min={min_val}, max={max_val}")

        return "\n".join(lines)

    def _validate_and_filter_proposals(
        self, proposals: list[dict[str, Any]], param_specs: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Valide et filtre les propositions (wrapper de validate_constraints + format).

        Args:
            proposals: Propositions brutes du LLM
            param_specs: Specs des paramètres

        Returns:
            Propositions validées et formatées
        """
        # Valider contraintes
        valid_proposals = self.validate_constraints(proposals, param_specs)

        # Formater pour exécution
        formatted_proposals = self.format_proposals(valid_proposals)

        return formatted_proposals
