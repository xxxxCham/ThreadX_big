"""
ThreadX - LLM Run Report & Index System
========================================

Système de génération de rapports et d'indexation pour les runs Multi-LLM.

Features:
- Génération de rapports JSON structurés
- Index centralisé pour recherche/réutilisation
- Export HTML optionnel
- Catégorisation: Sweep, Analysis, Proposals, Tests

Usage:
    >>> from threadx.llm.run_report import LLMRunReport, RunIndex
    >>>
    >>> # Créer un rapport
    >>> report = LLMRunReport(
    ...     strategy_name="MA_Crossover",
    ...     sweep_results=sweep_df,
    ...     analysis=analyst_result,
    ...     proposals=strategist_result,
    ...     test_results=test_results,
    ... )
    >>>
    >>> # Sauvegarder et indexer
    >>> index = RunIndex()
    >>> report_path = index.save_report(report)
    >>>
    >>> # Rechercher des runs
    >>> runs = index.search(strategy="MA_Crossover", min_sharpe=0.5)
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Répertoire par défaut pour les rapports
DEFAULT_REPORTS_DIR = Path("reports/llm_runs")


# ============================================================
# REPORT DATA STRUCTURES
# ============================================================


@dataclass
class SweepSection:
    """Section des résultats du sweep."""

    total_configs: int
    duration_seconds: float
    top_configs: list[dict[str, Any]]  # Top 10-20 configs
    params_tested: dict[str, list[Any]]  # Paramètres et leurs valeurs
    metrics_summary: dict[str, float]  # avg_sharpe, max_sharpe, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_sweep_results(
        cls,
        results: list[dict],
        sweep_params: dict,
        duration: float,
        top_n: int = 20
    ) -> "SweepSection":
        """Crée une SweepSection depuis les résultats bruts du sweep."""
        df = pd.DataFrame(results)

        # Extraire top configs
        if "sharpe_ratio" in df.columns:
            top_df = df.nlargest(top_n, "sharpe_ratio")
        elif "sharpe" in df.columns:
            top_df = df.nlargest(top_n, "sharpe")
        else:
            top_df = df.head(top_n)

        top_configs = top_df.to_dict("records")

        # Calculer métriques summary
        metrics_summary = {}
        for col in ["sharpe_ratio", "sharpe", "total_return", "pnl_pct", "max_drawdown", "win_rate"]:
            if col in df.columns:
                metrics_summary[f"avg_{col}"] = float(df[col].mean())
                metrics_summary[f"max_{col}"] = float(df[col].max())
                metrics_summary[f"min_{col}"] = float(df[col].min())
                metrics_summary[f"std_{col}"] = float(df[col].std())

        return cls(
            total_configs=len(results),
            duration_seconds=duration,
            top_configs=top_configs,
            params_tested=sweep_params,
            metrics_summary=metrics_summary,
        )


@dataclass
class AnalysisSection:
    """Section de l'analyse LLM (Analyst)."""

    model_used: str
    duration_seconds: float
    patterns: list[str]
    key_metrics: dict[str, float]
    trade_offs: list[str]
    recommendations: list[str]
    raw_response: Optional[str] = None  # Réponse brute pour debug

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_analyst_result(
        cls,
        result: dict,
        model: str,
        duration: float
    ) -> "AnalysisSection":
        """Crée une AnalysisSection depuis le résultat de l'Analyst."""
        analysis = result.get("analysis", {})

        return cls(
            model_used=model,
            duration_seconds=duration,
            patterns=analysis.get("patterns", []),
            key_metrics=analysis.get("key_metrics", {}),
            trade_offs=analysis.get("trade_offs", []),
            recommendations=analysis.get("recommendations", []),
            raw_response=result.get("_raw_response"),
        )


@dataclass
class ProposalItem:
    """Une proposition individuelle du Strategist."""

    name: str
    rationale: str
    params: dict[str, Any]
    changes_from_baseline: dict[str, dict[str, Any]]  # {param: {old, new}}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProposalsSection:
    """Section des propositions LLM (Strategist)."""

    model_used: str
    duration_seconds: float
    baseline_params: dict[str, Any]
    baseline_sharpe: float
    proposals: list[ProposalItem]
    total_generated: int
    total_valid: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["proposals"] = [p.to_dict() for p in self.proposals]
        return result

    @classmethod
    def from_strategist_result(
        cls,
        result: dict,
        baseline_params: dict,
        baseline_sharpe: float,
        model: str,
        duration: float
    ) -> "ProposalsSection":
        """Crée une ProposalsSection depuis le résultat du Strategist."""
        proposals = []

        for prop in result.get("proposals", []):
            # Calculer les changements par rapport à baseline
            changes = {}
            for param, new_val in prop.get("params", {}).items():
                old_val = baseline_params.get(param)
                if old_val != new_val:
                    changes[param] = {"old": old_val, "new": new_val}

            proposals.append(ProposalItem(
                name=prop.get("name", "Unknown"),
                rationale=prop.get("rationale", ""),
                params=prop.get("params", {}),
                changes_from_baseline=changes,
            ))

        return cls(
            model_used=model,
            duration_seconds=duration,
            baseline_params=baseline_params,
            baseline_sharpe=baseline_sharpe,
            proposals=proposals,
            total_generated=result.get("total_generated", len(proposals)),
            total_valid=result.get("total_valid", len(proposals)),
        )


@dataclass
class TestResultItem:
    """Résultat de test d'une proposition."""

    name: str
    params: dict[str, Any]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_trades: int
    loss_trades: int
    vs_baseline_sharpe: float  # Différence avec baseline
    is_improvement: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestsSection:
    """Section des résultats de tests des propositions."""

    total_tested: int
    successful_tests: int
    failed_tests: int
    results: list[TestResultItem]
    best_proposal: Optional[str]  # Nom de la meilleure proposition
    best_sharpe: float
    baseline_sharpe: float
    improvement_found: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["results"] = [r.to_dict() for r in self.results]
        return result

    @classmethod
    def from_test_results(
        cls,
        test_results: list[dict],
        baseline_sharpe: float
    ) -> "TestsSection":
        """Crée une TestsSection depuis les résultats de tests."""
        results = []
        best_proposal = None
        best_sharpe = baseline_sharpe

        for res in test_results:
            sharpe = res.get("sharpe_ratio", 0.0)
            trades = res.get("trades", [])

            # Calculer profit/loss trades
            profit_trades = sum(1 for t in trades if t.get("pnl", t.get("pnl_realized", 0)) > 0)
            total_trades = len(trades)
            loss_trades = total_trades - profit_trades

            item = TestResultItem(
                name=res.get("name", "Unknown"),
                params=res.get("params", {}),
                sharpe_ratio=sharpe,
                total_return=res.get("total_return", 0.0),
                max_drawdown=res.get("max_drawdown", 0.0),
                win_rate=res.get("win_rate", 0.0),
                total_trades=total_trades,
                profit_trades=profit_trades,
                loss_trades=loss_trades,
                vs_baseline_sharpe=sharpe - baseline_sharpe,
                is_improvement=sharpe > baseline_sharpe,
            )
            results.append(item)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_proposal = res.get("name")

        return cls(
            total_tested=len(test_results),
            successful_tests=len([r for r in test_results if r.get("sharpe_ratio") is not None]),
            failed_tests=len([r for r in test_results if r.get("sharpe_ratio") is None]),
            results=results,
            best_proposal=best_proposal,
            best_sharpe=best_sharpe,
            baseline_sharpe=baseline_sharpe,
            improvement_found=best_sharpe > baseline_sharpe,
        )


@dataclass
class LLMRunReport:
    """
    Rapport complet d'un run Multi-LLM.

    Structure:
    - metadata: Infos générales (date, stratégie, durée totale)
    - sweep: Résultats du sweep GPU
    - analysis: Analyse de l'Analyst
    - proposals: Propositions du Strategist
    - tests: Résultats des tests
    - conclusion: Résumé et meilleure config
    """

    # Métadonnées
    run_id: str
    timestamp: str
    strategy_name: str
    total_duration_seconds: float

    # Sections
    sweep: SweepSection
    analysis: Optional[AnalysisSection] = None
    proposals: Optional[ProposalsSection] = None
    tests: Optional[TestsSection] = None

    # Configuration utilisée
    config: dict[str, Any] = field(default_factory=dict)

    # Conclusion
    best_config: Optional[dict[str, Any]] = None
    best_sharpe: float = 0.0
    summary: str = ""

    def __post_init__(self):
        """Génère le run_id si non fourni."""
        if not self.run_id:
            # Générer un ID unique basé sur timestamp + stratégie
            hash_input = f"{self.timestamp}_{self.strategy_name}"
            self.run_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        """Convertit le rapport en dictionnaire sérialisable."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "strategy_name": self.strategy_name,
            "total_duration_seconds": self.total_duration_seconds,
            "sweep": self.sweep.to_dict(),
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "proposals": self.proposals.to_dict() if self.proposals else None,
            "tests": self.tests.to_dict() if self.tests else None,
            "config": self.config,
            "best_config": self.best_config,
            "best_sharpe": self.best_sharpe,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Sérialise le rapport en JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "LLMRunReport":
        """Reconstruit un rapport depuis un dictionnaire."""
        # Reconstruire les sections
        sweep = SweepSection(**data["sweep"])

        analysis = None
        if data.get("analysis"):
            analysis = AnalysisSection(**data["analysis"])

        proposals = None
        if data.get("proposals"):
            props_data = data["proposals"]
            props_data["proposals"] = [
                ProposalItem(**p) for p in props_data.get("proposals", [])
            ]
            proposals = ProposalsSection(**props_data)

        tests = None
        if data.get("tests"):
            tests_data = data["tests"]
            tests_data["results"] = [
                TestResultItem(**r) for r in tests_data.get("results", [])
            ]
            tests = TestsSection(**tests_data)

        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            strategy_name=data["strategy_name"],
            total_duration_seconds=data["total_duration_seconds"],
            sweep=sweep,
            analysis=analysis,
            proposals=proposals,
            tests=tests,
            config=data.get("config", {}),
            best_config=data.get("best_config"),
            best_sharpe=data.get("best_sharpe", 0.0),
            summary=data.get("summary", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LLMRunReport":
        """Reconstruit un rapport depuis JSON."""
        return cls.from_dict(json.loads(json_str))

    def generate_summary(self) -> str:
        """Génère un résumé textuel du run."""
        lines = [
            f"# Rapport Multi-LLM: {self.strategy_name}",
            f"",
            f"**Run ID:** {self.run_id}",
            f"**Date:** {self.timestamp}",
            f"**Durée totale:** {self.total_duration_seconds:.1f}s",
            f"",
            f"## Sweep GPU",
            f"- Configurations testées: {self.sweep.total_configs}",
            f"- Durée: {self.sweep.duration_seconds:.1f}s",
        ]

        if self.sweep.metrics_summary:
            avg_sharpe = self.sweep.metrics_summary.get("avg_sharpe_ratio",
                         self.sweep.metrics_summary.get("avg_sharpe", 0))
            max_sharpe = self.sweep.metrics_summary.get("max_sharpe_ratio",
                         self.sweep.metrics_summary.get("max_sharpe", 0))
            lines.append(f"- Sharpe moyen: {avg_sharpe:.3f}")
            lines.append(f"- Meilleur Sharpe: {max_sharpe:.3f}")

        if self.analysis:
            lines.extend([
                f"",
                f"## Analyse (Analyst)",
                f"- Modèle: {self.analysis.model_used}",
                f"- Patterns identifiés: {len(self.analysis.patterns)}",
                f"- Recommandations: {len(self.analysis.recommendations)}",
            ])

        if self.proposals:
            lines.extend([
                f"",
                f"## Propositions (Strategist)",
                f"- Modèle: {self.proposals.model_used}",
                f"- Propositions valides: {self.proposals.total_valid}/{self.proposals.total_generated}",
                f"- Baseline Sharpe: {self.proposals.baseline_sharpe:.3f}",
            ])

        if self.tests:
            lines.extend([
                f"",
                f"## Tests des Propositions",
                f"- Propositions testées: {self.tests.total_tested}",
                f"- Amélioration trouvée: {'✅ Oui' if self.tests.improvement_found else '❌ Non'}",
            ])
            if self.tests.best_proposal:
                lines.append(f"- Meilleure proposition: {self.tests.best_proposal}")
                lines.append(f"- Meilleur Sharpe: {self.tests.best_sharpe:.3f}")

        lines.extend([
            f"",
            f"## Conclusion",
            f"**Meilleur Sharpe final:** {self.best_sharpe:.3f}",
        ])

        self.summary = "\n".join(lines)
        return self.summary


# ============================================================
# RUN INDEX SYSTEM
# ============================================================


@dataclass
class IndexEntry:
    """Entrée dans l'index des runs."""

    run_id: str
    timestamp: str
    strategy_name: str
    best_sharpe: float
    total_configs: int
    improvement_found: bool
    report_path: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_report(cls, report: LLMRunReport, report_path: str) -> "IndexEntry":
        """Crée une entrée d'index depuis un rapport."""
        return cls(
            run_id=report.run_id,
            timestamp=report.timestamp,
            strategy_name=report.strategy_name,
            best_sharpe=report.best_sharpe,
            total_configs=report.sweep.total_configs,
            improvement_found=report.tests.improvement_found if report.tests else False,
            report_path=report_path,
            tags=[],
        )


class RunIndex:
    """
    Gestionnaire d'index pour les runs Multi-LLM.

    Permet de:
    - Sauvegarder des rapports avec indexation automatique
    - Rechercher des runs par critères
    - Charger des rapports existants

    L'index est stocké dans un fichier JSON centralisé.
    """

    def __init__(self, reports_dir: Path | str = DEFAULT_REPORTS_DIR):
        """
        Initialise le gestionnaire d'index.

        Args:
            reports_dir: Répertoire racine pour les rapports
        """
        self.reports_dir = Path(reports_dir)
        self.index_path = self.reports_dir / "index.json"
        self._index: dict[str, IndexEntry] = {}

        # Créer le répertoire si nécessaire
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Charger l'index existant
        self._load_index()

    def _load_index(self):
        """Charge l'index depuis le fichier JSON."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._index = {
                    k: IndexEntry(**v) for k, v in data.get("entries", {}).items()
                }
                logger.info(f"Index chargé: {len(self._index)} runs")
            except Exception as e:
                logger.warning(f"Erreur chargement index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self):
        """Sauvegarde l'index dans le fichier JSON."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "total_runs": len(self._index),
            "entries": {k: v.to_dict() for k, v in self._index.items()},
        }

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Index sauvegardé: {len(self._index)} runs")

    def save_report(
        self,
        report: LLMRunReport,
        tags: list[str] | None = None
    ) -> Path:
        """
        Sauvegarde un rapport et l'ajoute à l'index.

        Args:
            report: Rapport à sauvegarder
            tags: Tags optionnels pour faciliter la recherche

        Returns:
            Path: Chemin du fichier rapport sauvegardé
        """
        # Générer le nom du répertoire
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dir_name = f"{date_str}_{report.strategy_name}_{report.run_id}"
        report_dir = self.reports_dir / dir_name
        report_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le rapport JSON
        report_path = report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())

        # Sauvegarder le résumé Markdown
        summary_path = report_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report.generate_summary())

        # Créer l'entrée d'index
        entry = IndexEntry.from_report(report, str(report_path.relative_to(self.reports_dir)))
        if tags:
            entry.tags = tags

        # Ajouter à l'index
        self._index[report.run_id] = entry
        self._save_index()

        logger.info(f"Rapport sauvegardé: {report_path}")
        return report_path

    def load_report(self, run_id: str) -> LLMRunReport | None:
        """
        Charge un rapport depuis son run_id.

        Args:
            run_id: Identifiant du run

        Returns:
            LLMRunReport ou None si non trouvé
        """
        entry = self._index.get(run_id)
        if not entry:
            logger.warning(f"Run non trouvé: {run_id}")
            return None

        report_path = self.reports_dir / entry.report_path
        if not report_path.exists():
            logger.warning(f"Fichier rapport non trouvé: {report_path}")
            return None

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return LLMRunReport.from_json(f.read())
        except Exception as e:
            logger.error(f"Erreur chargement rapport {run_id}: {e}")
            return None

    def search(
        self,
        strategy: str | None = None,
        min_sharpe: float | None = None,
        max_sharpe: float | None = None,
        improvement_only: bool = False,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[IndexEntry]:
        """
        Recherche des runs selon critères.

        Args:
            strategy: Filtrer par nom de stratégie
            min_sharpe: Sharpe minimum
            max_sharpe: Sharpe maximum
            improvement_only: Seulement les runs avec amélioration
            tags: Filtrer par tags (OR)
            limit: Nombre max de résultats

        Returns:
            Liste d'IndexEntry correspondant aux critères
        """
        results = []

        for entry in self._index.values():
            # Filtres
            if strategy and entry.strategy_name != strategy:
                continue
            if min_sharpe is not None and entry.best_sharpe < min_sharpe:
                continue
            if max_sharpe is not None and entry.best_sharpe > max_sharpe:
                continue
            if improvement_only and not entry.improvement_found:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue

            results.append(entry)

        # Trier par date décroissante
        results.sort(key=lambda x: x.timestamp, reverse=True)

        return results[:limit]

    def list_all(self, limit: int = 100) -> list[IndexEntry]:
        """Liste tous les runs (triés par date décroissante)."""
        return self.search(limit=limit)

    def list_strategies(self) -> list[str]:
        """Liste toutes les stratégies indexées."""
        return list(set(e.strategy_name for e in self._index.values()))

    def get_stats(self) -> dict:
        """Retourne des statistiques sur l'index."""
        if not self._index:
            return {"total_runs": 0}

        entries = list(self._index.values())
        sharpes = [e.best_sharpe for e in entries]

        return {
            "total_runs": len(entries),
            "strategies": self.list_strategies(),
            "avg_sharpe": sum(sharpes) / len(sharpes),
            "max_sharpe": max(sharpes),
            "min_sharpe": min(sharpes),
            "improvements_found": sum(1 for e in entries if e.improvement_found),
            "total_configs_tested": sum(e.total_configs for e in entries),
        }

    def delete_run(self, run_id: str, delete_files: bool = True) -> bool:
        """
        Supprime un run de l'index.

        Args:
            run_id: Identifiant du run
            delete_files: Si True, supprime aussi les fichiers

        Returns:
            True si supprimé, False si non trouvé
        """
        entry = self._index.get(run_id)
        if not entry:
            return False

        if delete_files:
            report_path = self.reports_dir / entry.report_path
            report_dir = report_path.parent
            if report_dir.exists():
                import shutil
                shutil.rmtree(report_dir)
                logger.info(f"Fichiers supprimés: {report_dir}")

        del self._index[run_id]
        self._save_index()

        logger.info(f"Run supprimé de l'index: {run_id}")
        return True


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def create_report_from_run(
    strategy_name: str,
    sweep_results: list[dict],
    sweep_params: dict,
    sweep_duration: float,
    analysis_result: dict | None = None,
    analyst_model: str = "",
    analyst_duration: float = 0.0,
    proposals_result: dict | None = None,
    baseline_params: dict | None = None,
    baseline_sharpe: float = 0.0,
    strategist_model: str = "",
    strategist_duration: float = 0.0,
    test_results: list[dict] | None = None,
    config: dict | None = None,
) -> LLMRunReport:
    """
    Fonction helper pour créer un rapport depuis les résultats d'un run.

    Args:
        strategy_name: Nom de la stratégie optimisée
        sweep_results: Résultats du sweep GPU
        sweep_params: Paramètres testés dans le sweep
        sweep_duration: Durée du sweep en secondes
        analysis_result: Résultat de l'Analyst (optionnel)
        analyst_model: Modèle utilisé par l'Analyst
        analyst_duration: Durée de l'analyse
        proposals_result: Résultat du Strategist (optionnel)
        baseline_params: Paramètres baseline
        baseline_sharpe: Sharpe de la baseline
        strategist_model: Modèle utilisé par le Strategist
        strategist_duration: Durée de la génération
        test_results: Résultats des tests (optionnel)
        config: Configuration du run

    Returns:
        LLMRunReport complet
    """
    timestamp = datetime.now().isoformat()

    # Créer les sections
    sweep = SweepSection.from_sweep_results(
        sweep_results, sweep_params, sweep_duration
    )

    analysis = None
    if analysis_result:
        analysis = AnalysisSection.from_analyst_result(
            analysis_result, analyst_model, analyst_duration
        )

    proposals = None
    if proposals_result and baseline_params is not None:
        proposals = ProposalsSection.from_strategist_result(
            proposals_result, baseline_params, baseline_sharpe,
            strategist_model, strategist_duration
        )

    tests = None
    if test_results:
        tests = TestsSection.from_test_results(test_results, baseline_sharpe)

    # Calculer durée totale
    total_duration = sweep_duration + analyst_duration + strategist_duration

    # Déterminer best config et sharpe
    best_sharpe = baseline_sharpe
    best_config = baseline_params

    if tests and tests.improvement_found and tests.best_proposal:
        best_sharpe = tests.best_sharpe
        # Trouver les params de la meilleure proposition
        for res in tests.results:
            if res.name == tests.best_proposal:
                best_config = res.params
                break

    # Créer le rapport
    report = LLMRunReport(
        run_id="",  # Sera généré automatiquement
        timestamp=timestamp,
        strategy_name=strategy_name,
        total_duration_seconds=total_duration,
        sweep=sweep,
        analysis=analysis,
        proposals=proposals,
        tests=tests,
        config=config or {},
        best_config=best_config,
        best_sharpe=best_sharpe,
    )

    # Générer le résumé
    report.generate_summary()

    return report


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "LLMRunReport",
    "SweepSection",
    "AnalysisSection",
    "ProposalsSection",
    "ProposalItem",
    "TestsSection",
    "TestResultItem",
    "IndexEntry",
    "RunIndex",
    "create_report_from_run",
    "DEFAULT_REPORTS_DIR",
]
