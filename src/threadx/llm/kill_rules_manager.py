"""
Kill Rules Manager - Gestion Règles Dures Rejet Automatique
==========================================================

Base données kill rules permanentes, auto-updatée par Autopsy.
Reject stratégies AVANT backtests lourds → gain temps ×10.

Workflow:
    1. Autopsy génère kill rules après échec (si score ≥ 8.5)
    2. KillRulesManager valide + ajoute règles à base
    3. Critic applique kill rules AVANT backtests
    4. Strategist reçoit kill rules dans prompt (feedback)

Kill Rule Format:
    "rejeter si average_trade_duration > 30h"
    "rejeter si win_rate sur SOL < 60%"
    "rejeter si profit_factor < 2.1"

Persistence:
    - kill_rules.json (JSON array, auto-save)
    - Déduplication automatique
    - Timestamp + metadata par règle

Author: ThreadX Framework
Version: 1.0 - Auto-Learning System
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from threadx.utils.log import get_logger

logger = get_logger(__name__)


class KillRulesManager:
    """
    Manager règles dures rejet automatique.

    Auto-updaté par Autopsy après échecs.
    Utilisé par Critic pour filtrage pré-backtest.
    """

    def __init__(self, rules_path: Path = Path("./kill_rules.json")):
        """
        Initialise manager.

        Args:
            rules_path: Chemin fichier JSON kill rules
        """
        self.rules_path = rules_path
        self.rules: list[dict[str, Any]] = []

        # Charger règles existantes
        self.load()

        logger.info(
            f"⚔️ Kill Rules Manager initialized: {len(self.rules)} rules loaded"
        )

    def load(self) -> None:
        """Charge règles depuis disque."""
        if not self.rules_path.exists():
            logger.info(f"No existing kill rules file, starting fresh")
            self.rules = []
            return

        try:
            with open(self.rules_path) as f:
                data = json.load(f)

            # Rétrocompatibilité: anciennes versions = array simple strings
            if isinstance(data, list):
                if data and isinstance(data[0], str):
                    # Convertir ancien format
                    self.rules = [
                        {
                            "rule": rule,
                            "added_at": None,
                            "source": "legacy",
                            "active": True,
                        }
                        for rule in data
                    ]
                else:
                    self.rules = data
            else:
                logger.warning(f"Invalid kill rules format: {type(data)}")
                self.rules = []

            logger.info(f"✅ Loaded {len(self.rules)} kill rules from {self.rules_path}")

        except Exception as e:
            logger.error(f"Failed to load kill rules: {e}")
            self.rules = []

    def save(self) -> None:
        """Sauvegarde règles sur disque."""
        try:
            self.rules_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.rules_path, "w") as f:
                json.dump(self.rules, f, indent=2, default=str)

            logger.info(f"💾 Saved {len(self.rules)} kill rules to {self.rules_path}")

        except Exception as e:
            logger.error(f"Failed to save kill rules: {e}")

    def add_rule(
        self,
        rule_text: str,
        source: str = "autopsy",
        improvement_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Ajoute nouvelle kill rule.

        Args:
            rule_text: Texte règle (ex: "rejeter si avg_trade_duration > 30h")
            source: Source règle (autopsy, manual, strategist)
            improvement_score: Score amélioration Autopsy (si applicable)
            metadata: Métadonnées additionnelles

        Returns:
            True si ajoutée, False si dupliquée
        """
        # Normaliser règle
        rule_normalized = self._normalize_rule(rule_text)

        # Vérifier doublons
        for existing in self.rules:
            if self._normalize_rule(existing["rule"]) == rule_normalized:
                logger.info(f"⚠️ Kill rule already exists (skipped): {rule_text}")
                return False

        # Créer entrée
        rule_entry = {
            "rule": rule_text,
            "added_at": datetime.now().isoformat(),
            "source": source,
            "improvement_score": improvement_score,
            "active": True,
            "metadata": metadata or {},
        }

        self.rules.append(rule_entry)
        self.save()

        logger.info(f"✅ Kill rule added: {rule_text} (source: {source})")
        return True

    def add_rules_from_autopsy(
        self, autopsy_report: dict[str, Any], min_score: float = 8.5
    ) -> int:
        """
        Ajoute kill rules depuis rapport Autopsy (si score suffisant).

        Args:
            autopsy_report: Rapport Autopsy complet
            min_score: Score minimum amélioration pour activer règles

        Returns:
            Nombre règles ajoutées
        """
        score = autopsy_report.get("score_amelioration_attendue", 0)

        try:
            score = float(score)
        except (ValueError, TypeError):
            logger.warning(f"Invalid improvement score: {score}")
            return 0

        if score < min_score:
            logger.info(
                f"⚠️ Autopsy score {score:.1f} < {min_score} threshold, "
                f"kill rules not activated"
            )
            return 0

        # Extraire kill rules proposées
        proposed_rules = autopsy_report.get("kill_rules_proposees", [])

        if not proposed_rules:
            logger.warning("No kill rules proposed in autopsy report")
            return 0

        # Ajouter chaque règle
        added = 0
        for rule in proposed_rules:
            if self.add_rule(
                rule_text=rule,
                source="autopsy",
                improvement_score=score,
                metadata={
                    "strategy_name": autopsy_report.get("strategy_name"),
                    "cause_principale": autopsy_report.get("cause_principale"),
                    "timestamp": autopsy_report.get("timestamp"),
                },
            ):
                added += 1

        logger.info(
            f"⚔️ Added {added}/{len(proposed_rules)} kill rules from autopsy "
            f"(score {score:.1f})"
        )

        return added

    def check_strategy_params(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Vérifie si paramètres stratégie violent kill rules.

        Args:
            params: Paramètres stratégie à valider

        Returns:
            (passed: bool, violated_rules: list[str])
        """
        violated = []

        for rule_entry in self.rules:
            if not rule_entry.get("active", True):
                continue

            rule_text = rule_entry["rule"]

            # Évaluation simple pattern matching
            # Format: "rejeter si PARAM OPERATOR VALUE"
            if self._evaluate_rule(rule_text, params):
                violated.append(rule_text)

        passed = len(violated) == 0

        if not passed:
            logger.warning(
                f"❌ Strategy params violate {len(violated)} kill rules: {violated[:3]}"
            )

        return passed, violated

    def check_backtest_result(
        self, result: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Vérifie si résultat backtest viole kill rules.

        Args:
            result: Résultat backtest (métriques)

        Returns:
            (passed: bool, violated_rules: list[str])
        """
        violated = []

        for rule_entry in self.rules:
            if not rule_entry.get("active", True):
                continue

            rule_text = rule_entry["rule"]

            if self._evaluate_rule(rule_text, result):
                violated.append(rule_text)

        passed = len(violated) == 0

        if not passed:
            logger.warning(
                f"❌ Backtest result violates {len(violated)} kill rules: {violated[:3]}"
            )

        return passed, violated

    def _normalize_rule(self, rule_text: str) -> str:
        """
        Normalise texte règle pour comparaison.

        Args:
            rule_text: Règle brute

        Returns:
            Règle normalisée (lowercase, espaces trimés)
        """
        return re.sub(r"\s+", " ", rule_text.lower().strip())

    def _evaluate_rule(self, rule_text: str, data: dict[str, Any]) -> bool:
        """
        Évalue règle contre données (params ou résultat backtest).

        Args:
            rule_text: Règle à évaluer
            data: Données (params stratégie ou métriques backtest)

        Returns:
            True si règle violée (stratégie doit être rejetée)
        """
        # Pattern matching simple
        # Format: "rejeter si PARAM OPERATOR VALUE"
        # Ex: "rejeter si average_trade_duration > 30h"
        # Ex: "rejeter si win_rate sur SOL < 60%"

        rule_lower = rule_text.lower()

        # Extraction pattern
        # Regex: rejeter si <param> <op> <value>
        match = re.search(
            r"rejeter\s+si\s+(\w+(?:_\w+)*)\s*([<>=!]+)\s*([\d.]+)",
            rule_lower,
        )

        if not match:
            # Pattern alternatif: "rejeter si win_rate sur TOKEN < VALUE"
            match = re.search(
                r"rejeter\s+si\s+(\w+(?:_\w+)*)\s+sur\s+(\w+)\s*([<>=!]+)\s*([\d.]+)",
                rule_lower,
            )

            if match:
                param, token, operator, value_str = match.groups()
                # Chercher param_TOKEN dans data
                key = f"{param}_{token.upper()}"
                if key not in data:
                    return False  # Pas de donnée, pas de violation

                try:
                    param_value = float(data[key])
                    threshold = float(value_str)
                    return self._compare(param_value, operator, threshold)
                except (ValueError, TypeError):
                    return False

            # Pas de pattern reconnu
            return False

        param, operator, value_str = match.groups()

        # Chercher param dans data (insensible casse)
        param_value = None
        for key, val in data.items():
            if key.lower() == param.lower():
                param_value = val
                break

        if param_value is None:
            return False  # Pas de donnée, pas de violation

        # Parser value (gérer unités: h, %, etc)
        try:
            # Retirer unités
            value_clean = re.sub(r"[hH%]", "", value_str)
            threshold = float(value_clean)

            # Convertir param_value en float
            if isinstance(param_value, str):
                param_value = float(re.sub(r"[hH%]", "", param_value))
            else:
                param_value = float(param_value)

            # Comparer
            return self._compare(param_value, operator, threshold)

        except (ValueError, TypeError):
            return False

    def _compare(self, value: float, operator: str, threshold: float) -> bool:
        """
        Compare valeur avec seuil selon opérateur.

        Args:
            value: Valeur à tester
            operator: Opérateur (>, <, >=, <=, ==, !=)
            threshold: Seuil

        Returns:
            True si condition satisfaite (violation)
        """
        operator = operator.strip()

        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator in ("==", "="):
            return abs(value - threshold) < 1e-6
        elif operator == "!=":
            return abs(value - threshold) >= 1e-6
        else:
            return False

    def get_active_rules(self) -> list[str]:
        """
        Retourne liste règles actives (textes seulement).

        Returns:
            Liste textes règles actives
        """
        return [
            rule["rule"] for rule in self.rules if rule.get("active", True)
        ]

    def get_rules_summary(self) -> dict[str, Any]:
        """
        Résumé statistiques kill rules.

        Returns:
            Dict stats (total, actives, par source, top rules)
        """
        total = len(self.rules)
        active = sum(1 for r in self.rules if r.get("active", True))

        # Par source
        by_source = {}
        for rule in self.rules:
            source = rule.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1

        # Top rules (par score amélioration)
        top_rules = sorted(
            [r for r in self.rules if r.get("improvement_score")],
            key=lambda r: r.get("improvement_score", 0),
            reverse=True,
        )[:5]

        return {
            "total_rules": total,
            "active_rules": active,
            "by_source": by_source,
            "top_rules": [
                {
                    "rule": r["rule"],
                    "score": r.get("improvement_score"),
                    "added": r.get("added_at"),
                }
                for r in top_rules
            ],
        }

    def deactivate_rule(self, rule_text: str) -> bool:
        """
        Désactive règle (sans supprimer).

        Args:
            rule_text: Texte règle à désactiver

        Returns:
            True si trouvée et désactivée
        """
        rule_normalized = self._normalize_rule(rule_text)

        for rule_entry in self.rules:
            if self._normalize_rule(rule_entry["rule"]) == rule_normalized:
                rule_entry["active"] = False
                self.save()
                logger.info(f"⚠️ Kill rule deactivated: {rule_text}")
                return True

        logger.warning(f"Kill rule not found: {rule_text}")
        return False

    def generate_prompt_section(self) -> str:
        """
        Génère section Markdown pour prompt Strategist/CodeWriter.

        Returns:
            Texte formaté kill rules actives
        """
        active_rules = self.get_active_rules()

        if not active_rules:
            return "**Kill Rules Actives** : Aucune (toutes stratégies acceptées)\n"

        section = f"## ⚔️ KILL RULES ACTIVES ({len(active_rules)})\n\n"
        section += "**Toute proposition violant une kill rule sera rejetée automatiquement avec score 0/10.**\n\n"
        section += "**Règles permanentes à respecter absolument** :\n\n"

        for i, rule in enumerate(active_rules, start=1):
            section += f"{i}. {rule}\n"

        section += "\n**Tu DOIS éviter ces patterns à tout prix.**\n"

        return section
