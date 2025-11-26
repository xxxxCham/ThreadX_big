"""
Autopsy Agent - Analyse Post-Mortem Stratégies Échouées
=======================================================

Mission unique: Recevoir résultats stratégie rejetée par Critic,
identifier cause précise échec, proposer correctifs concrets + kill rules permanentes.

Transforme chaque échec en apprentissage permanent du système.

Workflow:
    1. Reçoit: code stratégie + rapport Critic complet
    2. Analyse: cause principale (drawdown, overfitting, frais, etc)
    3. Diagnostique: symptômes précis + poids cause
    4. Prescrit: correctifs concrets + kill rules
    5. Score: amélioration attendue (0-10)

Output JSON:
    {
        "cause_principale": "drawdown_range | trop_de_trades | ...",
        "poids_cause": 0.87,
        "symptomes_cles": ["drawdown -42% juillet-août", ...],
        "correctifs_concrets": ["ajouter filtre ADX > 23", ...],
        "kill_rules_proposees": ["rejeter si avg_trade_duration > 30h", ...],
        "score_amelioration_attendue": 9.1
    }

Integration:
    - Lancé automatiquement après échec Critic
    - Kill rules auto-ajoutées si score ≥ 8.5
    - Feedback injecté dans prompt Strategist

Author: ThreadX Framework
Version: 1.0 - Auto-Learning System
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from threadx.llm.agents.base_agent import BaseAgent
from threadx.utils.log import get_logger

logger = get_logger(__name__)


class Autopsy(BaseAgent):
    """
    Agent spécialisé analyse post-mortem stratégies rejetées.
    
    Produit rapport ultra-précis + kill rules permanentes.
    Chaque échec devient apprentissage permanent.
    """

    PROMPT_TEMPLATE = """Tu es Autopsy, médecin légiste des stratégies de trading.

Voici le code complet de la stratégie qui vient d'être rejetée :

```python
{code}
```

Voici le rapport complet du Critic (métriques sur plusieurs tokens/timeframes) :

```json
{critic_report}
```

**TA MISSION :**

1. Identifier LA cause principale de l'échec (une seule, la plus impactante)
2. Lister 3-5 symptômes précis observables dans les métriques
3. Proposer 3-5 correctifs concrets et immédiats (code-level, pas théoriques)
4. Proposer 2-4 kill rules permanentes (règles dures qui rejetteront automatiquement les futures stratégies similaires SANS les tester)
5. Donner un score d'amélioration attendue (0-10) si on applique tes correctifs

**FORMAT DE RÉPONSE (EXCLUSIVEMENT JSON) :**

```json
{{
  "strategy_name": "{strategy_name}",
  "cause_principale": "drawdown_range | trop_de_trades | overfitting_token | mauvais_rr | fragilite_range | frais_tueurs | sur_optimisation_params | manque_filtres_tendance | volatilite_excessive | win_rate_faible | autre",
  "poids_cause": 0.XX,
  "symptomes_cles": [
    "symptôme précis 1 avec chiffres",
    "symptôme précis 2 avec chiffres",
    "symptôme précis 3"
  ],
  "correctifs_concrets": [
    "correctif code-level 1 (ex: ajouter filtre ADX > 23)",
    "correctif code-level 2 (ex: réduire max_hold_bars à 120)",
    "correctif code-level 3"
  ],
  "kill_rules_proposees": [
    "rejeter si average_trade_duration > 30h",
    "rejeter si win_rate sur SOL < 60%",
    "rejeter si profit_factor < 2.1"
  ],
  "score_amelioration_attendue": X.X,
  "explication_breve": "1-2 phrases expliquant pourquoi cette cause est LA principale"
}}
```

**CONTRAINTES CRITIQUES :**

- Cause principale = UNE SEULE (la plus impactante, pas une liste)
- Symptômes = CHIFFRES PRÉCIS des métriques (pas de vague "mauvais sharpe", mais "sharpe 0.87 sur ETH vs 2.1 sur BTC")
- Correctifs = CODE-LEVEL (pas "améliorer robustesse", mais "ajouter condition volume > SMA(20)")
- Kill rules = MESURABLES (pas "éviter overfitting", mais "rejeter si win_rate_variance > 15%")
- Score = RÉALISTE (si cause mineure, score <5; si cause majeure facilement fixable, score 8-9)

**SOIS IMPITOYABLE, PRÉCIS, ET FOCALISÉ SUR L'ALPHA RÉEL.**

Pas de bla-bla. Pas de pitié. Chirurgie de précision uniquement.
"""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        temperature: float = 0.0,
        debug: bool = True,
        reports_dir: Path = Path("./autopsy_reports"),
    ):
        """
        Initialise Autopsy agent.

        Args:
            model: Modèle LLM (deepseek-r1:32b recommandé pour précision)
            temperature: 0.0 pour déterminisme maximal
            debug: Active logs détaillés
            reports_dir: Dossier sauvegarde rapports
        """
        super().__init__(model=model, temperature=temperature, debug=debug)
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"🔬 Autopsy Agent initialized: model={model}, reports_dir={reports_dir}"
        )

    def analyze_failure(
        self,
        strategy_path: Path,
        critic_report: dict[str, Any],
        code_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyse complète stratégie rejetée.

        Args:
            strategy_path: Chemin fichier stratégie
            critic_report: Rapport complet Critic (métriques multi-token/timeframe)
            code_override: Code stratégie (si None, lit depuis strategy_path)

        Returns:
            Rapport JSON structuré avec cause, symptômes, correctifs, kill rules
        """
        # Lire code stratégie
        if code_override:
            code = code_override
        else:
            if not strategy_path.exists():
                logger.error(f"Strategy file not found: {strategy_path}")
                return {"error": f"File not found: {strategy_path}"}
            code = strategy_path.read_text()

        strategy_name = strategy_path.stem

        # Construire prompt enrichi
        prompt = self.PROMPT_TEMPLATE.format(
            code=code,
            critic_report=json.dumps(critic_report, indent=2, default=str),
            strategy_name=strategy_name,
        )

        if self.debug:
            logger.debug(f"Autopsy prompt length: {len(prompt)} chars")

        # Appel LLM
        try:
            raw_response = self.client.complete(
                prompt, max_tokens=2000, temperature=self.temperature
            )

            if self.debug:
                logger.debug(f"Autopsy raw response:\n{raw_response[:500]}...")

            # Parse JSON tolérant
            report = self.client._parse_json_tolerant(raw_response)

            # Enrichissement métadonnées
            report["raw_critic"] = critic_report
            report["code_snapshot"] = code[:2000]  # Debug (premiers 2000 chars)
            report["timestamp"] = datetime.now().isoformat()
            report["model_used"] = self.model

            # Validation structure
            self._validate_report(report)

            # Logs
            logger.info(
                f"🔬 AUTOPSY → {strategy_name} | "
                f"Cause: {report.get('cause_principale', 'N/A')} | "
                f"Score amélioration: {report.get('score_amelioration_attendue', 0)}/10"
            )

            # Sauvegarde automatique
            self._save_report(strategy_name, report)

            return report

        except Exception as e:
            logger.error(f"Autopsy failed on {strategy_name}: {e}", exc_info=True)
            return {
                "error": str(e),
                "strategy_name": strategy_name,
                "timestamp": datetime.now().isoformat(),
            }

    def _validate_report(self, report: dict[str, Any]) -> None:
        """
        Valide structure rapport Autopsy.

        Args:
            report: Rapport à valider

        Raises:
            ValueError: Si structure invalide
        """
        required_fields = [
            "cause_principale",
            "poids_cause",
            "symptomes_cles",
            "correctifs_concrets",
            "kill_rules_proposees",
            "score_amelioration_attendue",
        ]

        missing = [f for f in required_fields if f not in report]
        if missing:
            logger.warning(f"Autopsy report missing fields: {missing}")

        # Validation types
        if "poids_cause" in report:
            try:
                poids = float(report["poids_cause"])
                if not 0 <= poids <= 1:
                    logger.warning(f"poids_cause hors range [0,1]: {poids}")
            except (ValueError, TypeError):
                logger.warning(f"poids_cause not a number: {report['poids_cause']}")

        if "score_amelioration_attendue" in report:
            try:
                score = float(report["score_amelioration_attendue"])
                if not 0 <= score <= 10:
                    logger.warning(f"score_amelioration hors range [0,10]: {score}")
            except (ValueError, TypeError):
                logger.warning(
                    f"score_amelioration not a number: {report['score_amelioration_attendue']}"
                )

    def _save_report(self, strategy_name: str, report: dict[str, Any]) -> None:
        """
        Sauvegarde rapport sur disque.

        Args:
            strategy_name: Nom stratégie
            report: Rapport à sauvegarder
        """
        report_path = self.reports_dir / f"{strategy_name}.json"

        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"💾 Autopsy report saved: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save autopsy report: {e}")

    def get_all_reports(self) -> list[dict[str, Any]]:
        """
        Retourne tous rapports Autopsy existants.

        Returns:
            Liste rapports triés par timestamp (plus récents en premier)
        """
        reports = []

        for report_file in self.reports_dir.glob("*.json"):
            try:
                with open(report_file) as f:
                    report = json.load(f)
                    reports.append(report)
            except Exception as e:
                logger.warning(f"Failed to load report {report_file}: {e}")

        # Trier par timestamp (plus récents en premier)
        reports.sort(
            key=lambda r: r.get("timestamp", ""), reverse=True
        )

        return reports

    def get_failure_patterns_summary(self) -> dict[str, Any]:
        """
        Génère résumé patterns d'échec (heatmap data).

        Returns:
            Dict avec stats par cause:
            {
                "trop_de_trades": {
                    "count": 87,
                    "last_seen": "2024-11-21T05:30:00",
                    "avg_sharpe_victims": 0.94,
                    "avg_improvement_score": 8.7
                },
                ...
            }
        """
        reports = self.get_all_reports()

        # Aggregation par cause
        patterns = {}

        for report in reports:
            cause = report.get("cause_principale", "autre")

            if cause not in patterns:
                patterns[cause] = {
                    "count": 0,
                    "last_seen": None,
                    "sharpe_victims": [],
                    "improvement_scores": [],
                }

            patterns[cause]["count"] += 1

            # Timestamp
            timestamp = report.get("timestamp")
            if timestamp and (
                patterns[cause]["last_seen"] is None
                or timestamp > patterns[cause]["last_seen"]
            ):
                patterns[cause]["last_seen"] = timestamp

            # Sharpe moyen (extraction depuis critic_report)
            critic = report.get("raw_critic", {})
            if isinstance(critic, dict):
                # Chercher sharpe_ratio dans métriques
                for key, value in critic.items():
                    if "sharpe" in key.lower() and isinstance(value, (int, float)):
                        patterns[cause]["sharpe_victims"].append(value)
                        break

            # Score amélioration
            score = report.get("score_amelioration_attendue")
            if score is not None:
                try:
                    patterns[cause]["improvement_scores"].append(float(score))
                except (ValueError, TypeError):
                    pass

        # Calculer moyennes
        summary = {}
        for cause, data in patterns.items():
            summary[cause] = {
                "count": data["count"],
                "last_seen": data["last_seen"],
                "avg_sharpe_victims": (
                    sum(data["sharpe_victims"]) / len(data["sharpe_victims"])
                    if data["sharpe_victims"]
                    else None
                ),
                "avg_improvement_score": (
                    sum(data["improvement_scores"]) / len(data["improvement_scores"])
                    if data["improvement_scores"]
                    else None
                ),
            }

        # Trier par count (plus fréquents en premier)
        summary = dict(
            sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True)
        )

        return summary

    def generate_strategist_feedback(self, top_n: int = 5) -> str:
        """
        Génère feedback structuré pour prompt Strategist.

        Args:
            top_n: Nombre top causes à inclure

        Returns:
            Texte markdown formaté pour injection prompt
        """
        reports = self.get_all_reports()
        patterns = self.get_failure_patterns_summary()

        total_failures = len(reports)

        # Top N causes
        top_causes = list(patterns.items())[:top_n]

        feedback = f"""## ⚠️ APPRENTISSAGE ÉCHECS PASSÉS

**Total échecs analysés** : {total_failures}

### Top {top_n} Causes d'Échec (à éviter absolument)

"""

        for i, (cause, data) in enumerate(top_causes, start=1):
            last_seen = data["last_seen"]
            if last_seen:
                try:
                    dt = datetime.fromisoformat(last_seen)
                    delay = datetime.now() - dt
                    if delay.days > 0:
                        delay_str = f"il y a {delay.days}j"
                    elif delay.seconds > 3600:
                        delay_str = f"il y a {delay.seconds // 3600}h"
                    else:
                        delay_str = f"il y a {delay.seconds // 60}min"
                except Exception:
                    delay_str = "récemment"
            else:
                delay_str = "N/A"

            avg_sharpe = data["avg_sharpe_victims"]
            sharpe_str = f"{avg_sharpe:.2f}" if avg_sharpe else "N/A"

            feedback += f"{i}. **{cause}** – {data['count']} occurrences (dernière fois {delay_str}, Sharpe moyen victimes: {sharpe_str})\n"

        feedback += "\n"
        return feedback
