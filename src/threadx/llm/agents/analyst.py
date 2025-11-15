"""
Agent Analyst - Analyse quantitative de résultats de backtests.

Utilise deepseek-r1:70b pour analyser des résultats de sweep/backtests
et identifier des patterns significatifs.
"""

from typing import Any

import pandas as pd

from threadx.llm.agents.base_agent import BaseAgent


class Analyst(BaseAgent):
    """
    Agent spécialisé dans l'analyse quantitative de résultats de backtests.

    Capabilities:
    - Analyser les résultats d'un sweep (top N configurations)
    - Analyser un backtest individuel en profondeur
    - Identifier des patterns communs dans des configurations performantes
    """

    def __init__(
        self,
        model: str = "deepseek-r1:70b",
        debug: bool = False,
    ) -> None:
        """
        Initialise l'agent Analyst.

        Args:
            model: Modèle LLM à utiliser (par défaut deepseek-r1:70b pour analyse)
            debug: Active les logs détaillés
        """
        super().__init__(name="Analyst", model=model, debug=debug)

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers analyze_sweep_results).

        Pour usage direct, préférer analyze_sweep_results() ou analyze_backtest().
        """
        if "sweep_df" in kwargs or (args and isinstance(args[0], pd.DataFrame)):
            sweep_df = kwargs.get("sweep_df", args[0] if args else None)
            top_n = kwargs.get("top_n", 5)
            return self.analyze_sweep_results(sweep_df, top_n)

        raise ValueError(
            "Analyst.analyze() requires 'sweep_df' (DataFrame) parameter. "
            "Use analyze_sweep_results() or analyze_backtest() directly."
        )

    def analyze_sweep_results(
        self, sweep_df: pd.DataFrame, top_n: int = 5
    ) -> dict[str, Any]:
        """
        Analyse les résultats d'un sweep pour identifier les meilleures configs.

        Args:
            sweep_df: DataFrame avec colonnes [strategy, param1, param2, ..., sharpe_ratio, etc.]
            top_n: Nombre de top configurations à analyser en détail

        Returns:
            dict avec:
            - top_configs: Liste des N meilleures configs avec métriques
            - analysis: Analyse qualitative LLM (patterns, recommandations)
            - patterns: Patterns identifiés (ex: "short_period < 15 dans 4/5 top configs")
        """
        self.logger.info("Analyzing sweep results (top %d configs)...", top_n)

        # Trier par Sharpe ratio (ou autre métrique)
        if "sharpe_ratio" not in sweep_df.columns:
            raise ValueError("sweep_df must contain 'sharpe_ratio' column")

        top_df = sweep_df.nlargest(top_n, "sharpe_ratio")

        # Préparer données pour le LLM
        configs_str = self._format_sweep_results(top_df)

        # Prompt pour analyse quantitative avec consignes système
        system_instructions = """
🎯 OBJECTIFS PRIORITAIRES:
- Maximiser le Sharpe Ratio (risque/rendement optimal)
- Minimiser le drawdown maximum (protection du capital)
- Maintenir un win rate > 50% (cohérence stratégique)
- Optimiser le nombre de trades (éviter over/under-trading)

📊 APPROCHE D'ANALYSE:
- Identifier les patterns reproductibles dans les meilleures configurations
- Détecter les corrélations entre paramètres (interactions non-linéaires)
- Privilégier la robustesse à la performance brute (éviter overfitting)
- Analyser les trade-offs (ex: rendement vs stabilité)

⚠️ CONTRAINTES CRITIQUES:
- risk_per_trade: Rester dans [0.005, 0.02] (gestion risque stricte)
- max_hold_bars: Adapter selon volatilité observée
- Stop Loss / Take Profit: Ratio minimum 1:1.5 (asymétrie favorable)
- Respecter TOUJOURS les plages min/max des paramètres

💡 PRINCIPES:
- Préférer solutions simples et explicables
- Documenter clairement le raisonnement
- Signaler les anomalies ou incohérences dans les données
"""
        
        prompt = f"""{system_instructions}

Analyse les {top_n} meilleures configurations de backtest ci-dessous.

Résultats du sweep (triés par Sharpe ratio):
{configs_str}

Identifie:
1. **Patterns communs** dans les paramètres performants (ex: "short_period souvent < 15")
2. **Métriques clés** (Sharpe moyen, drawdown max, win rate)
3. **Trade-offs observés** (ex: "Sharpe élevé mais drawdown important")
4. **Recommandations** pour prochaines optimisations (plages de paramètres prometteuses)

Réponds en JSON avec:
{{
  "patterns": ["pattern1", "pattern2", ...],
  "key_metrics": {{"avg_sharpe": X, "max_drawdown_avg": Y, ...}},
  "trade_offs": ["trade-off1", ...],
  "recommendations": ["rec1", "rec2", ...]
}}
"""

        # Appel LLM structuré
        analysis_result = self._call_llm_structured(
            prompt=prompt,
            expected_schema={
                "patterns": list,
                "key_metrics": dict,
                "trade_offs": list,
                "recommendations": list,
            },
            temperature=0.3,  # Basse température pour analyse factuelle
            max_tokens=2000,
        )

        # Identifier patterns quantitatifs (complément à l'analyse LLM)
        quantitative_patterns = self._identify_quantitative_patterns(top_df)

        return {
            "top_configs": top_df.to_dict("records"),
            "analysis": analysis_result,
            "quantitative_patterns": quantitative_patterns,
        }

    def analyze_backtest(
        self, backtest_result: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analyse en profondeur un backtest individuel.

        Args:
            backtest_result: Résultat backtest (sharpe, drawdown, total_return, trades, etc.)
            params: Paramètres utilisés (strategy, short_period, long_period, etc.)

        Returns:
            dict avec:
            - assessment: Évaluation qualitative (strong/medium/weak)
            - strengths: Points forts identifiés
            - weaknesses: Points faibles identifiés
            - suggestions: Modifications suggérées des paramètres
        """
        self.logger.info("Analyzing individual backtest...")

        # Formater résultats pour LLM
        results_str = self._format_backtest_result(backtest_result, params)

        prompt = f"""Analyse en détail ce backtest de stratégie trading:

{results_str}

Évalue:
1. **Performance globale** (strong/medium/weak) basée sur Sharpe, drawdown, win rate
2. **Points forts** (métriques excellentes, comportement robuste)
3. **Points faibles** (risques, incohérences, métriques faibles)
4. **Suggestions** de modifications (ex: "Augmenter long_period pour réduire drawdown")

Réponds en JSON:
{{
  "assessment": "strong/medium/weak",
  "strengths": ["strength1", ...],
  "weaknesses": ["weakness1", ...],
  "suggestions": ["suggestion1", ...]
}}
"""

        analysis = self._call_llm_structured(
            prompt=prompt,
            expected_schema={
                "assessment": str,
                "strengths": list,
                "weaknesses": list,
                "suggestions": list,
            },
            temperature=0.4,
            max_tokens=1500,
        )

        return analysis

    def identify_patterns(self, configs_list: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Identifie des patterns communs dans une liste de configurations.

        Args:
            configs_list: Liste de configs (dicts avec params + métriques)

        Returns:
            dict avec:
            - common_params: Paramètres avec valeurs fréquentes (ex: {"short_period": [10, 12]})
            - correlations: Corrélations observées (ex: "short_period < 15 → Sharpe > 1.5")
        """
        self.logger.info("Identifying patterns in %d configurations...", len(configs_list))

        # Convertir en DataFrame pour analyse
        df = pd.DataFrame(configs_list)

        # Analyser distributions
        param_cols = [c for c in df.columns if c not in ["sharpe_ratio", "total_return", "max_drawdown"]]
        distributions = {}
        for col in param_cols:
            if df[col].dtype in ["int64", "float64"]:
                distributions[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

        # Demander au LLM d'interpréter
        configs_str = df.to_string(max_rows=20)
        prompt = f"""Analyse ces configurations de trading pour identifier des patterns:

{configs_str}

Distributions des paramètres:
{distributions}

Identifie:
1. **Paramètres critiques** (ceux qui varient le plus entre configs performantes)
2. **Corrélations** (ex: "Quand short_period < 15, Sharpe > 1.5 dans 80% des cas")
3. **Plages optimales** (ex: "long_period entre 30-50 semble optimal")

Réponds en JSON:
{{
  "critical_params": ["param1", "param2"],
  "correlations": ["correlation1", ...],
  "optimal_ranges": {{"param1": "10-15", "param2": "30-50", ...}}
}}
"""

        patterns = self._call_llm_structured(
            prompt=prompt,
            expected_schema={
                "critical_params": list,
                "correlations": list,
                "optimal_ranges": dict,
            },
            temperature=0.3,
            max_tokens=1500,
        )

        return {
            "patterns": patterns,
            "distributions": distributions,
        }

    # --- Méthodes privées de formatage ---

    def _format_sweep_results(self, df: pd.DataFrame) -> str:
        """Formate un DataFrame de sweep pour le LLM (texte tabulaire lisible)."""
        # Colonnes clés à afficher
        key_cols = ["strategy", "sharpe_ratio", "total_return", "max_drawdown"]
        param_cols = [c for c in df.columns if c not in key_cols and not c.startswith("_")]

        display_cols = key_cols + param_cols[:5]  # Limiter à 5 params pour lisibilité
        return str(df[display_cols].to_string(index=False))

    def _format_backtest_result(self, result: dict[str, Any], params: dict[str, Any]) -> str:
        """Formate un résultat backtest + params en texte lisible."""
        lines = ["**Paramètres:**"]
        for k, v in params.items():
            lines.append(f"  {k}: {v}")

        lines.append("\n**Résultats:**")
        for k, v in result.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def _identify_quantitative_patterns(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Identifie des patterns quantitatifs simples (complément LLM).

        Ex: "short_period < 15 dans 4/5 top configs"
        """
        patterns = {}

        # Paramètres numériques uniquement
        param_cols = [c for c in df.columns if c not in ["sharpe_ratio", "total_return", "max_drawdown"]]
        numeric_cols = [c for c in param_cols if df[c].dtype in ["int64", "float64"]]

        for col in numeric_cols:
            patterns[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "range": (float(df[col].min()), float(df[col].max())),
            }

        return patterns
