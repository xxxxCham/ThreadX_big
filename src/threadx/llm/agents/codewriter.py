"""
Agent CodeWriter - Génération de code pour stratégies de trading.

Génère du code Python de stratégies ThreadX basé sur analyses LLM
et résultats de sweep. V1: Modifications de stratégies existantes uniquement.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from threadx.llm.agents.base_agent import BaseAgent


class CodeWriter(BaseAgent):
    """
    Agent spécialisé dans la génération de code Python pour stratégies.

    V1 Capabilities (MINIMALISTE):
    - Modifier stratégies existantes (BollingerDual, MACrossover, etc.)
    - Générer code Python valide conforme au Protocol Strategy
    - Sauvegarder dans strategy/experimental/
    - Validation syntaxe basique

    V2 Future:
    - Créer nouvelles stratégies from scratch
    - Extraction automatique de schémas params/indicators
    - Tests unitaires auto-générés
    """

    # Prompt système avec contraintes strictes ThreadX
    SYSTEM_PROMPT = """Tu es un expert en stratégies de trading quantitatives Python.

🎯 TÂCHE: Modifier une stratégie ThreadX existante

📋 ARCHITECTURE THREADX (OBLIGATOIRE):

1. **Imports requis**:
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from threadx.strategy.model import RunStats, Trade
```

2. **Classe Params** (Paramètres de stratégie):
```python
@dataclass
class MyStrategyParams:
    # Indicateurs
    bb_period: int = 20
    bb_std: float = 2.0

    # Risk Management
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    risk_per_trade: float = 0.015  # 1.5%

    # Position Management
    max_hold_bars: int = 100
```

3. **Classe Strategy** (Logique trading):
```python
class MyStrategy:
    '''Docstring expliquant la stratégie.'''

    def run(
        self,
        data: pd.DataFrame,
        indicators: dict[str, Any],
        params: dict[str, Any] | MyStrategyParams,
    ) -> RunStats:
        '''
        Exécute le backtest.

        Args:
            data: DataFrame OHLCV [open, high, low, close, volume]
            indicators: dict depuis IndicatorBank {"bollinger": {...}, "atr": Series, ...}
            params: Paramètres (dict ou MyStrategyParams)

        Returns:
            RunStats avec trades, equity, métriques
        '''
        # Conversion params si dict
        if isinstance(params, dict):
            params = MyStrategyParams(**params)

        # Extraction indicateurs (depuis IndicatorBank!)
        bb = indicators.get("bollinger", {})
        bb_upper = bb.get("upper")
        bb_lower = bb.get("lower")
        bb_middle = bb.get("middle")

        # LOGIQUE DE TRADING ICI
        trades = []
        position = None

        for i in range(1, len(data)):
            close = data["close"].iloc[i]

            # Entry logic
            if position is None and close < bb_lower.iloc[i]:
                position = {
                    "entry_idx": i,
                    "entry_price": close,
                    "entry_time": data.index[i],
                    "stop_loss": close * (1 - params.stop_loss_pct / 100),
                }

            # Exit logic
            elif position is not None:
                if close >= bb_middle.iloc[i] or close <= position["stop_loss"]:
                    trade = Trade(
                        entry_ts=position["entry_time"],
                        exit_ts=data.index[i],
                        price_entry=position["entry_price"],
                        price_exit=close,
                        side="LONG",
                        qty=1.0,
                        pnl_realized=close - position["entry_price"],
                    )
                    trades.append(trade)
                    position = None

        # Retourner RunStats
        from threadx.strategy.model import RunStatsDict
        return RunStatsDict(
            trades=[t.__dict__ for t in trades],
            total_trades=len(trades),
        )
```

⚠️ CONTRAINTES CRITIQUES:

1. **Utiliser UNIQUEMENT IndicatorBank**:
   - NE PAS calculer indicateurs manuellement
   - Extraire depuis `indicators` dict fourni en paramètre
   - Indicateurs disponibles: bollinger, atr, sma, ema, rsi

2. **NE PAS modifier**:
   - data_access (lecture données)
   - backtest/engine.py
   - indicators/bank.py
   - Aucun autre fichier du framework

3. **Gestion risque OBLIGATOIRE**:
   - Stop-loss sur CHAQUE position
   - Position sizing (risk_per_trade)
   - Max hold bars (éviter positions zombies)

4. **Code déterministe**:
   - Pas de random.random() sans seed
   - Comportement reproductible

5. **Bibliothèques autorisées**:
   - NumPy, Pandas: OUI ✅
   - TA-Lib, autres: NON ❌

📤 FORMAT DE SORTIE: JSON strict

```json
{
    "status": "success",
    "filename": "ai_strategy_v1.py",
    "class_name": "AIStrategyV1",
    "code": "# Code Python complet ici...",
    "explanation": "Cette modification améliore...",
    "changes_summary": [
        "Ajout filtre volume sur entrées",
        "Stop-loss dynamique basé sur ATR",
        "Réduction max_hold_bars de 150 à 100"
    ]
}
```

Si ERREUR:
```json
{
    "status": "error",
    "error": "Raison de l'échec..."
}
```

💡 PRINCIPES V1 (MINIMALISTE):

- **Modifier stratégies existantes** (BollingerDual, MACrossover, etc.)
- Garder même structure de paramètres (compatibilité Optimization Engine)
- Modifications incrémentielles (pas de refonte complète)
- Commentaires expliquant POURQUOI, pas QUOI
"""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        output_dir: str | Path = "src/threadx/strategy/experimental",
        debug: bool = False,
    ):
        """
        Initialise l'agent CodeWriter.

        Args:
            model: Modèle LLM (défaut: deepseek-r1:32b pour cohérence)
            output_dir: Dossier de sortie pour stratégies générées
            debug: Active logs détaillés
        """
        super().__init__(name="CodeWriter", model=model, timeout=120.0, debug=debug)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"📁 Output directory: {self.output_dir}")

    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Point d'entrée générique (délègue vers run).

        Pour usage direct, préférer run().
        """
        return self.run(**kwargs)

    def run(
        self,
        task: str,
        base_strategy: str,
        analysis: dict[str, Any],
        proposals: dict[str, Any],
        failed_metrics: dict[str, Any] | None = None,
        ideas: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Génère du code de stratégie modifiée basé sur analyse LLM.

        Args:
            task: Type de modification ("improve_sharpe", "reduce_drawdown", etc.)
            base_strategy: Nom de la stratégie à modifier ("Bollinger_Dual", "MA_Crossover", etc.)
            analysis: Résultat de Analyst.analyze_sweep_results()
            proposals: Résultat de Strategist.propose_modifications()
            failed_metrics: Métriques actuelles insatisfaisantes (optionnel)
            ideas: Idées additionnelles pour guider génération (optionnel)

        Returns:
            dict avec:
            {
                "status": "success" | "error",
                "filename": "ai_strategy_v1.py",
                "class_name": "AIStrategyV1",
                "code": "...",  # Code Python complet
                "explanation": "...",
                "changes_summary": [str],
                "error": str (si status=error)
            }

        Example:
            >>> codewriter = CodeWriter()
            >>> result = codewriter.run(
            ...     task="improve_sharpe",
            ...     base_strategy="Bollinger_Dual",
            ...     analysis=analyst_result,
            ...     proposals=strategist_result,
            ...     failed_metrics={"current_sharpe": 0.5, "target_sharpe": 0.8}
            ... )
            >>> if result["status"] == "success":
            ...     print(f"Code généré: {result['filename']}")
        """
        self.logger.info(
            f"🔨 Génération code: task={task}, base={base_strategy}"
        )

        # Construire contexte pour LLM
        context = self._build_context(
            task, base_strategy, analysis, proposals, failed_metrics, ideas
        )

        # Prompt utilisateur
        prompt = f"""{context}

**TÂCHE**: {self._task_description(task)}

**STRATÉGIE DE BASE**: {base_strategy}

**OBJECTIF**: Générer une version améliorée de cette stratégie en Python.

**CONTRAINTES**:
- Garder la même structure de paramètres (compatibilité Optimization Engine)
- Utiliser UNIQUEMENT IndicatorBank pour indicateurs
- Ajouter gestion risque stricte (stop-loss, position sizing)
- Code déterministe et reproductible

**OUTPUT**: JSON avec "status", "filename", "class_name", "code", "explanation", "changes_summary"
"""

        try:
            # Appel LLM structuré
            result = self._call_llm_structured(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                expected_schema={
                    "status": str,
                    "filename": str,
                    "class_name": str,
                    "code": str,
                },
                temperature=0.7,  # Créativité modérée
                max_tokens=4000,  # Code peut être long
            )

            # Validation résultat
            if result.get("status") != "success":
                return result

            # Extraction et nettoyage du code
            code = self._extract_python_code(result.get("code", ""))

            # Sauvegarde du code
            filepath = self._save_code(
                filename=result["filename"],
                code=code,
                metadata={
                    "task": task,
                    "base_strategy": base_strategy,
                    "class_name": result["class_name"],
                    "explanation": result.get("explanation", ""),
                }
            )

            # Retourner résultat enrichi
            return {
                **result,
                "code": code,
                "filepath": str(filepath),
            }

        except Exception as e:
            self.logger.error(f"❌ Code generation failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)}"
            }

    def _build_context(
        self,
        task: str,
        base_strategy: str,
        analysis: dict[str, Any],
        proposals: dict[str, Any],
        failed_metrics: dict[str, Any] | None,
        ideas: list[str] | None,
    ) -> str:
        """Construit le contexte pour le prompt LLM."""
        lines = []

        # Métriques actuelles
        if failed_metrics:
            lines.append("📊 **MÉTRIQUES ACTUELLES** (insatisfaisantes):")
            for key, val in failed_metrics.items():
                lines.append(f"  - {key}: {val}")
            lines.append("")

        # Patterns de l'Analyst
        patterns = analysis.get("analysis", {}).get("patterns", [])
        if patterns:
            lines.append("🔍 **PATTERNS IDENTIFIÉS** (Analyst):")
            for pattern in patterns[:5]:  # Limiter à top 5
                lines.append(f"  - {pattern}")
            lines.append("")

        # Recommandations de l'Analyst
        recommendations = analysis.get("analysis", {}).get("recommendations", [])
        if recommendations:
            lines.append("💡 **RECOMMANDATIONS** (Analyst):")
            for rec in recommendations[:5]:
                lines.append(f"  - {rec}")
            lines.append("")

        # Propositions du Strategist
        proposals_list = proposals.get("proposals", [])
        if proposals_list:
            lines.append("🎯 **PROPOSITIONS DE PARAMÈTRES** (Strategist):")
            for i, prop in enumerate(proposals_list[:3], 1):
                name = prop.get("name", f"Proposition {i}")
                rationale = prop.get("rationale", "N/A")
                params = prop.get("params", {})
                lines.append(f"  {i}. **{name}**:")
                lines.append(f"     Rationale: {rationale}")
                lines.append(f"     Params: {params}")
            lines.append("")

        # Idées additionnelles
        if ideas:
            lines.append("💭 **IDÉES ADDITIONNELLES**:")
            for idea in ideas:
                lines.append(f"  - {idea}")
            lines.append("")

        return "\n".join(lines)

    def _task_description(self, task: str) -> str:
        """Retourne la description détaillée d'une tâche."""
        descriptions = {
            "improve_sharpe": "Améliorer le ratio de Sharpe en optimisant le risk/reward",
            "reduce_drawdown": "Réduire le drawdown maximum via gestion de risque stricte",
            "increase_winrate": "Augmenter le win rate en améliorant la qualité des signaux",
            "optimize_trades": "Optimiser le nombre de trades (ni trop, ni trop peu)",
            "modify_strategy": "Modifier la stratégie selon les recommandations d'analyse",
        }

        return descriptions.get(task, task)

    def _extract_python_code(self, code_str: str) -> str:
        """
        Extrait le code Python depuis la réponse LLM.

        Gère les cas:
        - Code brut Python
        - Code dans bloc ```python ... ```
        - Code avec commentaires avant/après

        Args:
            code_str: String contenant le code (potentiellement avec markdown)

        Returns:
            Code Python nettoyé
        """
        # Chercher bloc ```python ... ```
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, code_str, re.DOTALL)

        if match:
            code = match.group(1)
            self.logger.debug("✅ Code extrait depuis bloc ```python")
            return code.strip()

        # Sinon, supposer que c'est du code brut
        self.logger.debug("⚠️  Pas de bloc markdown, utilise code brut")
        return code_str.strip()

    def _save_code(
        self,
        filename: str,
        code: str,
        metadata: dict[str, Any],
    ) -> Path:
        """
        Sauvegarde le code généré dans experimental/.

        Args:
            filename: Nom du fichier (ex: "ai_strategy_v1.py")
            code: Code Python complet
            metadata: Métadonnées (task, base_strategy, etc.)

        Returns:
            Path du fichier sauvegardé
        """
        # Assurer extension .py
        if not filename.endswith(".py"):
            filename += ".py"

        filepath = self.output_dir / filename

        # Header avec métadonnées
        header = f'''"""
AI-Generated Strategy: {metadata.get("class_name", "Unknown")}
{'=' * 60}

**Generated**: {self._get_timestamp()}
**Base Strategy**: {metadata.get("base_strategy", "N/A")}
**Task**: {metadata.get("task", "N/A")}

**Explanation**:
{metadata.get("explanation", "N/A")}

⚠️ ATTENTION:
- Stratégie générée automatiquement par LLM
- Doit passer validation Critic avant utilisation
- NE PAS utiliser en production sans revue humaine
"""

'''

        # Écriture du fichier
        full_code = header + code

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_code)

        self.logger.info(f"✅ Code sauvegardé: {filepath}")

        return filepath

    def _get_timestamp(self) -> str:
        """Retourne timestamp ISO pour métadonnées."""
        from datetime import datetime
        return datetime.now().isoformat()


# Export
__all__ = ["CodeWriter"]
