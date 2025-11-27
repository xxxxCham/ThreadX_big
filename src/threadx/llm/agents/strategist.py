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
        model: str = "deepseek-r1:32b",
        debug: bool = False,
    ) -> None:
        """
        Initialise l'agent Strategist.

        Args:
            model: Modèle LLM à utiliser (par défaut deepseek-r1:32b pour cohérence avec Analyst)
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
- **MA_Crossover/EMA_Cross: slow_period DOIT TOUJOURS être > fast_period** (BLOQUANT si non respecté)
- **Bollinger_Breakout: bb_std_dev DOIT être > 0** (BLOQUANT si non respecté)
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

        # Appel LLM structuré avec retry et fallback
        try:
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

            # Si aucune proposition valide, générer fallback
            if len(validated_proposals) == 0:
                self.logger.warning(
                    "⚠️ No valid LLM proposals after validation, "
                    "generating fallback proposals..."
                )
                validated_proposals = self._generate_fallback_proposals(
                    current_params, param_specs, n_proposals
                )
                source = "fallback_invalid"
            else:
                source = "llm"

            self.logger.info("Generated %d valid proposals (source: %s)", len(validated_proposals), source)

            return {
                "proposals": validated_proposals,
                "total_generated": len(result.get("proposals", [])),
                "total_valid": len(validated_proposals),
                "source": source,
            }

        except Exception as e:
            # Fallback complet si LLM échoue
            self.logger.error(
                "❌ LLM proposal generation failed: %s, using fallback heuristics",
                str(e)
            )
            fallback_proposals = self._generate_fallback_proposals(
                current_params, param_specs, n_proposals
            )

            return {
                "proposals": fallback_proposals,
                "total_generated": 0,
                "total_valid": len(fallback_proposals),
                "source": "fallback_error",
                "error": str(e),
            }

    def validate_constraints(
        self, proposals: list[dict[str, Any]], param_specs: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Valide que les propositions respectent les contraintes min/max ET les contraintes stratégiques.

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
            rejection_reason = None

            # Validation 1: Contraintes min/max des paramètres
            for param_name, value in params.items():
                if param_name not in param_specs:
                    self.logger.warning(
                        "Unknown parameter '%s' in proposal, skipping", param_name
                    )
                    is_valid = False
                    rejection_reason = f"Unknown parameter '{param_name}'"
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
                    rejection_reason = f"{param_name}={value} < min({min_val})"
                    break

                if max_val is not None and value > max_val:
                    self.logger.warning(
                        "Parameter %s = %s above max %s, skipping proposal",
                        param_name,
                        value,
                        max_val,
                    )
                    is_valid = False
                    rejection_reason = f"{param_name}={value} > max({max_val})"
                    break

            # Validation 2: Contraintes stratégiques spécifiques
            if is_valid:
                # MA_Crossover/EMA_Cross : slow_period DOIT être > fast_period
                if "fast_period" in params and "slow_period" in params:
                    fast = params["fast_period"]
                    slow = params["slow_period"]
                    if slow <= fast:
                        self.logger.warning(
                            "❌ MA_Crossover constraint violated: slow_period (%s) <= fast_period (%s), skipping",
                            slow,
                            fast,
                        )
                        is_valid = False
                        rejection_reason = f"slow_period ({slow}) <= fast_period ({fast})"

                # Bollinger_Breakout : bb_std_dev DOIT être > 0
                if "bb_std_dev" in params:
                    bb_std = params["bb_std_dev"]
                    if bb_std <= 0:
                        self.logger.warning(
                            "❌ Bollinger constraint violated: bb_std_dev (%s) <= 0, skipping",
                            bb_std,
                        )
                        is_valid = False
                        rejection_reason = f"bb_std_dev ({bb_std}) <= 0"

                # Take Profit DOIT être >= Stop Loss * 1.5 (ratio asymétrique)
                if "stop_loss_pct" in params and "take_profit_pct" in params:
                    sl = params["stop_loss_pct"]
                    tp = params["take_profit_pct"]
                    if tp < sl * 1.5:
                        self.logger.warning(
                            "⚠️ TP/SL ratio low: %s/%s = %.2fx (recommended >1.5x), keeping but flagged",
                            tp,
                            sl,
                            tp / sl if sl > 0 else 0,
                        )
                        # Non-bloquant mais signalé (on garde la proposition)

            if is_valid:
                valid.append(prop)
            else:
                self.logger.info("Rejected proposal '%s': %s", prop.get("name", "Unknown"), rejection_reason)

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

    def _generate_fallback_proposals(
        self,
        current_params: dict[str, Any],
        param_specs: dict[str, dict[str, Any]],
        n_proposals: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Génère propositions par défaut via heuristiques Python (fallback si LLM échoue).

        Stratégies:
        1. Conservative: Augmentation +10% paramètres "lents" (slow, long, period)
        2. Aggressive: Réduction SL, augmentation TP, périodes rapides, multiplicateurs
        3. Baseline: Configuration actuelle inchangée

        Règles heuristiques étendues:
        - Règle 1: Slow/long/period → +10% (stabilité)
        - Règle 2: Stop loss → -10% (moins de stops)
        - Règle 3: Take profit → +10% (gains plus élevés)
        - Règle 4: Thresholds/z-scores → -10% (plus permissif)
        - Règle 5: Multiplicateurs (ATR, volatilité) → +10% (plus de marge)
        - Règle 6: Périodes rapides (fast, short) → -10% (plus réactif)

        Args:
            current_params: Paramètres baseline
            param_specs: Specs avec min/max
            n_proposals: Nombre de propositions (max 3)

        Returns:
            Liste de propositions formatées (dict avec name, params, rationale)
        """
        proposals = []

        # PROPOSITION 1: Conservative (+10% params lents, -10% thresholds)
        conservative = current_params.copy()
        conservative_changes = []

        for param, value in conservative.items():
            if param not in param_specs:
                continue

            # Règle 1: Augmenter paramètres "lents" de 10%
            if any(keyword in param.lower() for keyword in ["slow", "long", "period"]):
                if not any(keyword in param.lower() for keyword in ["fast", "short"]):
                    new_val = value * 1.1
                    max_val = param_specs[param].get("max")
                    if max_val is not None:
                        new_val = min(new_val, max_val)

                    # Arrondir selon type
                    if param_specs[param].get("type") == "int":
                        new_val = int(new_val)

                    if new_val != value:
                        conservative[param] = new_val
                        conservative_changes.append(f"{param}: {value} → {new_val}")
            
            # Règle 4: Réduire thresholds/z-scores de 10% (plus permissif)
            elif any(keyword in param.lower() for keyword in ["threshold", "_z", "trigger", "zscore"]):
                new_val = value * 0.9
                min_val = param_specs[param].get("min")
                if min_val is not None:
                    new_val = max(new_val, min_val)

                if param_specs[param].get("type") == "int":
                    new_val = int(new_val)

                if new_val != value:
                    conservative[param] = new_val
                    conservative_changes.append(f"{param}: {value} → {new_val}")

        proposals.append({
            "name": "Conservative (fallback)",
            "params": conservative,
            "rationale": (
                f"Règles heuristiques: +10% params lents (stabilité), -10% thresholds (permissivité). "
                f"Modifications: {', '.join(conservative_changes) if conservative_changes else 'baseline inchangée'}"
            ),
        })

        # PROPOSITION 2: Aggressive (SL -10%, TP +10%, périodes rapides -10%, multiplicateurs +10%)
        aggressive = current_params.copy()
        aggressive_changes = []

        for param, value in aggressive.items():
            if param not in param_specs:
                continue

            # Règle 2: Réduire stop loss de 10%
            if "stop" in param.lower() or "sl" in param.lower():
                new_val = value * 0.9
                min_val = param_specs[param].get("min")
                if min_val is not None:
                    new_val = max(new_val, min_val)

                if param_specs[param].get("type") == "int":
                    new_val = int(new_val)

                if new_val != value:
                    aggressive[param] = new_val
                    aggressive_changes.append(f"{param}: {value} → {new_val}")

            # Règle 3: Augmenter take profit de 10%
            elif "take" in param.lower() or "tp" in param.lower():
                new_val = value * 1.1
                max_val = param_specs[param].get("max")
                if max_val is not None:
                    new_val = min(new_val, max_val)

                if param_specs[param].get("type") == "int":
                    new_val = int(new_val)

                if new_val != value:
                    aggressive[param] = new_val
                    aggressive_changes.append(f"{param}: {value} → {new_val}")
            
            # Règle 5: Augmenter multiplicateurs de 10% (ATR, volatilité)
            elif any(keyword in param.lower() for keyword in ["multiplier", "mult", "factor", "atr"]):
                if not any(keyword in param.lower() for keyword in ["stop", "sl"]):  # Éviter double application
                    new_val = value * 1.1
                    max_val = param_specs[param].get("max")
                    if max_val is not None:
                        new_val = min(new_val, max_val)

                    if param_specs[param].get("type") == "int":
                        new_val = int(new_val)

                    if new_val != value:
                        aggressive[param] = new_val
                        aggressive_changes.append(f"{param}: {value} → {new_val}")
            
            # Règle 6: Réduire périodes rapides de 10% (plus réactif)
            elif any(keyword in param.lower() for keyword in ["fast", "short"]):
                if "period" in param.lower() or "ma" in param.lower() or "ema" in param.lower():
                    new_val = value * 0.9
                    min_val = param_specs[param].get("min")
                    if min_val is not None:
                        new_val = max(new_val, min_val)

                    if param_specs[param].get("type") == "int":
                        new_val = int(new_val)

                    if new_val != value:
                        aggressive[param] = new_val
                        aggressive_changes.append(f"{param}: {value} → {new_val}")

        proposals.append({
            "name": "Aggressive (fallback)",
            "params": aggressive,
            "rationale": (
                f"Règles heuristiques: SL -10%, TP +10%, multiplicateurs +10%, périodes rapides -10%. "
                f"Modifications: {', '.join(aggressive_changes) if aggressive_changes else 'baseline inchangée'}"
            ),
        })

        # PROPOSITION 3: Baseline (inchangée)
        if n_proposals >= 3:
            proposals.append({
                "name": "Baseline (unchanged)",
                "params": current_params.copy(),
                "rationale": "Configuration actuelle (baseline) sans modification.",
            })

        # Validation finale : corriger automatiquement les contraintes stratégiques
        validated_proposals = []
        for prop in proposals[:n_proposals]:
            fixed_params = self._fix_strategic_constraints(prop["params"])
            validated_proposals.append({
                "name": prop["name"],
                "params": fixed_params,
                "rationale": prop["rationale"],
            })

        return validated_proposals

    def _fix_strategic_constraints(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Corrige automatiquement les contraintes stratégiques dans une proposition.

        Règles:
        1. MA_Crossover/EMA_Cross: Si slow_period <= fast_period, swap ou ajustement intelligent
        2. Bollinger: Si bb_std_dev <= 0, fixer à 2.0 (valeur standard)
        3. TP/SL: Si TP < SL * 1.5, ajuster TP à SL * 2.0

        Args:
            params: Paramètres bruts (potentiellement invalides)

        Returns:
            Paramètres corrigés (garantis valides pour contraintes stratégiques)
        """
        fixed = params.copy()

        # Règle 1: MA_Crossover → slow_period DOIT être > fast_period
        if "fast_period" in fixed and "slow_period" in fixed:
            fast = fixed["fast_period"]
            slow = fixed["slow_period"]

            if slow <= fast:
                # Stratégie de correction: swap si possible, sinon ajustement
                if fast < 100:  # Seulement si fast est raisonnable comme slow
                    self.logger.info(
                        "Auto-fix MA_Crossover: swapping fast_period (%s) ↔ slow_period (%s)",
                        fast,
                        slow,
                    )
                    fixed["fast_period"] = slow
                    fixed["slow_period"] = fast
                else:
                    # fast trop grand pour devenir slow, ajuster différemment
                    self.logger.info(
                        "Auto-fix MA_Crossover: setting slow_period = fast_period (%s) + 10",
                        fast,
                    )
                    fixed["slow_period"] = fast + 10

        # Règle 2: Bollinger → bb_std_dev DOIT être > 0
        if "bb_std_dev" in fixed and fixed["bb_std_dev"] <= 0:
            self.logger.info(
                "Auto-fix Bollinger: bb_std_dev (%s) → 2.0 (standard value)",
                fixed["bb_std_dev"],
            )
            fixed["bb_std_dev"] = 2.0

        # Règle 3: TP/SL → Ratio minimum 1.5:1 (recommandé 2:1)
        if "stop_loss_pct" in fixed and "take_profit_pct" in fixed:
            sl = fixed["stop_loss_pct"]
            tp = fixed["take_profit_pct"]

            if sl > 0 and tp < sl * 1.5:
                new_tp = sl * 2.0  # Ratio 2:1 (plus sûr que 1.5:1)
                self.logger.info(
                    "Auto-fix TP/SL ratio: take_profit_pct (%.2f) → %.2f (ratio 2:1 with SL %.2f)",
                    tp,
                    new_tp,
                    sl,
                )
                fixed["take_profit_pct"] = new_tp

        return fixed

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
