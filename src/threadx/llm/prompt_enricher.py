"""
Prompt Enricher - Enrichissement Prompts Agents avec Contexte Global
====================================================================

Injecte contexte structuré dans prompts agents LLM :
1. Inventaire données disponibles (tokens, dates, qualité)
2. Registry stratégies existantes (versions, performances)
3. Contraintes validation (éviter données invalides)
4. Historique optimisation (mémoire itérations)

Architecture:
    PromptEnricher.enrich(agent_type, base_prompt, context) → prompt_enrichi
    → Agents reçoivent contexte complet avant génération
    → Peuvent prendre décisions éclairées (choix token, params, etc)

Author: ThreadX Framework
Version: 1.0 - Context-Aware Prompting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from threadx.llm.context_manager import ContextManager
    from threadx.llm.memory import OptimizationMemory


class PromptEnricher:
    """Enrichisseur prompts avec contexte global."""

    @staticmethod
    def enrich_analyst_prompt(
        base_prompt: str,
        context_manager: ContextManager,
        strategy_name: str,
        backtest_result: dict[str, Any],
        memory: OptimizationMemory | None = None,
    ) -> str:
        """
        Enrichit prompt Analyst avec contexte.

        Args:
            base_prompt: Prompt base Analyst
            context_manager: Manager contexte global
            strategy_name: Nom stratégie analysée
            backtest_result: Résultat backtest à analyser
            memory: Mémoire optimisation (historique)

        Returns:
            Prompt enrichi
        """
        # Contexte global
        ctx = context_manager.get_full_context(strategy_name)

        # Construire section contexte
        context_section = f"""
# CONTEXTE GLOBAL DISPONIBLE

## Données Disponibles
{_format_data_inventory(ctx['data_inventory'])}

## Stratégie Analysée
{_format_strategy_context(ctx['strategy_registry'])}

## Historique Optimisation
{_format_memory_context(memory) if memory else 'Pas d\'historique (première itération)'}

## Contraintes Validation
- Éviter tokens avec qualité <80% (gaps massifs)
- Vérifier disponibilité token sur période complète
- Considérer tokens alternatifs si données insuffisantes
- Tenir compte historique 2024-01-01 → aujourd'hui

---
"""

        # Ajouter résultat backtest
        result_section = f"""
# RÉSULTAT BACKTEST À ANALYSER

```json
{_format_backtest_result(backtest_result)}
```

---
"""

        # Composer prompt final
        return context_section + result_section + base_prompt

    @staticmethod
    def enrich_strategist_prompt(
        base_prompt: str,
        context_manager: ContextManager,
        strategy_name: str,
        current_params: dict[str, Any],
        analyst_diagnosis: dict[str, Any],
        memory: OptimizationMemory | None = None,
        autopsy_feedback: str | None = None,
    ) -> str:
        """
        Enrichit prompt Strategist avec contexte + autopsy feedback.

        Args:
            base_prompt: Prompt base Strategist
            context_manager: Manager contexte global
            strategy_name: Nom stratégie
            current_params: Paramètres actuels
            analyst_diagnosis: Diagnostic Analyst
            memory: Mémoire optimisation
            autopsy_feedback: Feedback échecs Autopsy (patterns + kill rules)

        Returns:
            Prompt enrichi
        """
        ctx = context_manager.get_full_context(strategy_name)

        # AUTOPSY FEEDBACK (si disponible)
        autopsy_section = ""
        if autopsy_feedback:
            autopsy_section = f"""
# ⚔️ FEEDBACK AUTOPSY - ÉCHECS PRÉCÉDENTS

{autopsy_feedback}

**Tu DOIS éviter ces patterns à tout prix.**

---
"""

        context_section = f"""
# CONTEXTE GLOBAL DISPONIBLE

## Données Disponibles
{_format_data_inventory(ctx['data_inventory'])}

## Stratégie à Optimiser
{_format_strategy_context(ctx['strategy_registry'])}

### Paramètres Actuels
```json
{_format_params(current_params)}
```

## Historique Optimisation
{_format_memory_context(memory) if memory else 'Première itération'}

## Diagnostic Analyst
```json
{_format_analyst_diagnosis(analyst_diagnosis)}
```

## Contraintes Génération Propositions
1. **Respect disponibilité données**: Vérifier token disponible sur période
2. **Évolution stratégie**: Créer nouvelles versions, pas écraser existantes
3. **Diversité**: Proposer approches variées (conservatrice, agressive, innovante)
4. **Persistence**: Propositions seront sauvegardées dans registry avec versioning
5. **Validation**: Propositions passeront validation Critic (overfitting, risques)

---
"""

        result_section = f"""
# DIAGNOSTIC À AMÉLIORER

{analyst_diagnosis.get('summary', 'Pas de résumé')}

**Score Analyst**: {analyst_diagnosis.get('score', 0)}/10

---
"""

        return autopsy_section + context_section + result_section + base_prompt

    @staticmethod
    def enrich_critic_prompt(
        base_prompt: str,
        context_manager: ContextManager,
        strategy_name: str,
        proposals: list[dict[str, Any]],
        memory: OptimizationMemory | None = None,
    ) -> str:
        """
        Enrichit prompt Critic avec contexte.

        Args:
            base_prompt: Prompt base Critic
            context_manager: Manager contexte global
            strategy_name: Nom stratégie
            proposals: Propositions à valider
            memory: Mémoire optimisation

        Returns:
            Prompt enrichi
        """
        ctx = context_manager.get_full_context(strategy_name)

        context_section = f"""
# CONTEXTE GLOBAL DISPONIBLE

## Données Disponibles
{_format_data_inventory(ctx['data_inventory'])}

## Stratégie Concernée
{_format_strategy_context(ctx['strategy_registry'])}

## Historique Optimisation
{_format_memory_context(memory) if memory else 'Première itération'}

## Contraintes Validation Critic
1. **Overfitting**: Détecter params trop spécifiques (ex: stop_loss=18.7324%)
2. **Risques données**: Vérifier token disponible + qualité suffisante
3. **Cohérence stratégie**: Params compatibles avec logique stratégie
4. **Robustesse**: Penaliser configurations fragiles (gaps, volatilité extrême)
5. **Diversité**: Favoriser approches complémentaires, pas redondantes

---
"""

        proposals_section = f"""
# PROPOSITIONS À VALIDER

{_format_proposals(proposals)}

---
"""

        return context_section + proposals_section + base_prompt


# =============================================================================
# HELPER FORMATTERS
# =============================================================================


def _format_data_inventory(inventory: dict[str, Any]) -> str:
    """Formate inventaire données pour prompt."""
    lines = [
        f"**Période globale**: {inventory['global_period']['start']} → {inventory['global_period']['end']}",
        f"**Tokens disponibles**: {inventory['total_tokens']}",
        "",
        "### Tokens Principaux",
    ]

    # Top 5 tokens haute qualité
    tokens = inventory.get("tokens", {})
    sorted_tokens = sorted(
        tokens.items(),
        key=lambda x: float(x[1]["quality_score"].rstrip("%")) / 100,
        reverse=True,
    )

    for symbol, info in sorted_tokens[:5]:
        lines.append(
            f"- **{symbol}**: {info['available_since']} → {info['available_until']}, "
            f"Quality {info['quality_score']}, "
            f"Timeframes: {', '.join(info['timeframes'])}"
        )

    # Recommandations
    recs = inventory.get("recommendations", [])
    if recs:
        lines.append("")
        lines.append("### Recommandations")
        for rec in recs:
            lines.append(f"- {rec}")

    return "\n".join(lines)


def _format_strategy_context(registry: dict[str, Any]) -> str:
    """Formate contexte stratégie pour prompt."""
    if "error" in registry:
        return f"⚠️ {registry['error']}"

    if "strategy_name" in registry:
        # Contexte stratégie spécifique
        lines = [
            f"**Nom**: {registry['strategy_name']}",
            f"**Versions totales**: {registry['total_versions']}",
            "",
            "### Version Actuelle (Latest)",
        ]

        latest = registry["latest_version"]
        lines.append(f"- **Version**: {latest['version']}")
        lines.append(f"- **Créée**: {latest['created_at']} par {latest['created_by']}")
        if latest.get("performance"):
            lines.append(f"- **Performance**: Sharpe {latest['performance'].get('sharpe_ratio', 'N/A')}")
        if latest.get("tier_s_score"):
            lines.append(f"- **Tier S**: {latest['tier_s_score']}/100")

        # Meilleure version
        best = registry.get("best_version")
        if best and best["version"] != latest["version"]:
            lines.append("")
            lines.append("### Meilleure Version (Best Sharpe)")
            lines.append(f"- **Version**: {best['version']}")
            lines.append(f"- **Sharpe**: {best['performance'].get('sharpe_ratio', 'N/A')}")
            lines.append(f"- **Tier S**: {best.get('tier_s_score', 'N/A')}/100")

        return "\n".join(lines)
    else:
        # Contexte global stratégies
        lines = [f"**Stratégies totales**: {registry['total_strategies']}", ""]

        strategies = registry.get("strategies", {})
        for name, info in list(strategies.items())[:3]:
            lines.append(
                f"- **{name}**: {info['total_versions']} versions, "
                f"Best Sharpe {info['best_sharpe']:.2f}, "
                f"Best Tier S {info['best_tier_s']:.0f}/100"
            )

        return "\n".join(lines)


def _format_memory_context(memory: OptimizationMemory | None) -> str:
    """Formate historique mémoire pour prompt."""
    if not memory or len(memory.iterations) == 0:
        return "Pas d'historique (première itération)"

    lines = [f"**Itérations**: {len(memory.iterations)}", ""]

    # Dernières 3 itérations
    recent = memory.iterations[-3:]
    for i, iter_data in enumerate(recent, start=len(memory.iterations) - 2):
        lines.append(f"### Itération {i}")
        lines.append(f"- Sharpe: {iter_data.get('sharpe_ratio', 'N/A')}")
        lines.append(f"- Analyst Score: {iter_data.get('analyst_score', 'N/A')}/10")
        lines.append(f"- Propositions: {iter_data.get('proposals_count', 0)}")

    # Convergence
    if memory.has_converged():
        lines.append("")
        lines.append("⚠️ **Convergence détectée** (pas d'amélioration)")

    return "\n".join(lines)


def _format_backtest_result(result: dict[str, Any]) -> str:
    """Formate résultat backtest pour prompt."""
    import json

    return json.dumps(result, indent=2)


def _format_params(params: dict[str, Any]) -> str:
    """Formate paramètres pour prompt."""
    import json

    return json.dumps(params, indent=2)


def _format_analyst_diagnosis(diagnosis: dict[str, Any]) -> str:
    """Formate diagnostic Analyst pour prompt."""
    import json

    return json.dumps(diagnosis, indent=2)


def _format_proposals(proposals: list[dict[str, Any]]) -> str:
    """Formate propositions pour prompt."""
    lines = []

    for i, prop in enumerate(proposals, start=1):
        lines.append(f"### Proposition {i}")
        lines.append(f"```json")
        import json

        lines.append(json.dumps(prop, indent=2))
        lines.append(f"```")
        lines.append("")

    return "\n".join(lines)
