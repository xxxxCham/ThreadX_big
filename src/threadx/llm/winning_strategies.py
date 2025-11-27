"""
Winning Strategies Manager
==========================

Module pour la gestion et le stockage automatique des stratégies gagnantes.

Une stratégie est considérée "gagnante" si elle respecte les critères suivants :
- Sharpe Ratio > seuil minimum (configurable)
- Performance positive en cross-testing
- Robustesse sur plusieurs tokens/timeframes
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Dossier par défaut pour les stratégies gagnantes
DEFAULT_WINNING_DIR = (
    Path(__file__).parent.parent.parent.parent / "reports" / "winning_strategies"
)


@dataclass
class WinningCriteria:
    """Critères pour qualifier une stratégie comme 'gagnante'."""

    min_sharpe: float = 0.5
    min_avg_cross_sharpe: float = 0.2  # Sharpe moyen sur cross-testing
    min_positive_cross_ratio: float = 0.6  # % de tokens avec Sharpe > 0
    min_trades: int = 5
    max_drawdown_pct: float = -20.0  # Max DD en % (négatif)
    min_profit_factor: float = 1.2


@dataclass
class CrossTestResult:
    """Résultat d'un test sur un token/timeframe."""

    token: str
    timeframe: str
    sharpe_ratio: Optional[float]
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    n_trades: Optional[int] = None
    status: str = "✅"


@dataclass
class WinningStrategy:
    """Structure d'une stratégie gagnante sauvegardée."""

    # Identifiants
    id: str
    timestamp: str
    strategy_name: str

    # Paramètres de la stratégie
    params: Dict[str, Any]

    # Métriques principales (sur le token/timeframe principal)
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float

    # Configuration du test principal
    primary_token: str
    primary_timeframe: str

    # Résultats cross-testing
    cross_test_results: List[Dict[str, Any]] = field(default_factory=list)
    avg_cross_sharpe: float = 0.0
    positive_cross_ratio: float = 0.0

    # Métadonnées
    models_used: Dict[str, str] = field(default_factory=dict)
    run_id: Optional[str] = None  # ID du run LLM d'origine
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Sérialise en JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WinningStrategy":
        """Crée une instance depuis un dictionnaire."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "WinningStrategy":
        """Crée une instance depuis une chaîne JSON."""
        return cls.from_dict(json.loads(json_str))


class WinningStrategiesManager:
    """Gestionnaire des stratégies gagnantes."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        criteria: Optional[WinningCriteria] = None,
    ):
        """
        Initialise le gestionnaire.

        Args:
            storage_dir: Dossier de stockage (défaut: reports/winning_strategies/)
            criteria: Critères de qualification (défaut: critères standards)
        """
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_WINNING_DIR
        self.criteria = criteria or WinningCriteria()
        self.index_file = self.storage_dir / "index.json"

        # Créer le dossier si nécessaire
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Charger ou initialiser l'index
        self._load_index()

    def _load_index(self) -> None:
        """Charge l'index des stratégies gagnantes."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.warning(f"[WinningManager] Erreur chargement index: {e}")
                self.index = self._create_empty_index()
        else:
            self.index = self._create_empty_index()

    def _create_empty_index(self) -> Dict[str, Any]:
        """Crée un index vide."""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_strategies": 0,
            "criteria": asdict(self.criteria),
            "entries": {},
        }

    def _save_index(self) -> None:
        """Sauvegarde l'index."""
        self.index["updated_at"] = datetime.now().isoformat()
        self.index["total_strategies"] = len(self.index["entries"])

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, default=str)

    def is_winning(
        self,
        sharpe_ratio: float,
        cross_results: Optional[List[Dict[str, Any]]] = None,
        total_trades: int = 0,
        max_drawdown: float = 0.0,
        profit_factor: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Vérifie si une stratégie est gagnante selon les critères.

        Returns:
            (is_winning, reason) - True/False et raison de la décision
        """
        # Critère 1: Sharpe minimum
        if sharpe_ratio < self.criteria.min_sharpe:
            return (
                False,
                f"Sharpe ({sharpe_ratio:.3f}) < minimum ({self.criteria.min_sharpe})",
            )

        # Critère 2: Nombre de trades minimum
        if total_trades < self.criteria.min_trades:
            return (
                False,
                f"Trades ({total_trades}) < minimum ({self.criteria.min_trades})",
            )

        # Critère 3: Drawdown maximum
        if max_drawdown < self.criteria.max_drawdown_pct:
            return (
                False,
                f"Drawdown ({max_drawdown:.1f}%) < limite ({self.criteria.max_drawdown_pct}%)",
            )

        # Critère 4: Profit factor minimum
        if profit_factor > 0 and profit_factor < self.criteria.min_profit_factor:
            return (
                False,
                f"Profit Factor ({profit_factor:.2f}) < minimum ({self.criteria.min_profit_factor})",
            )

        # Critère 5: Cross-testing (si disponible)
        if cross_results and len(cross_results) > 1:
            valid_sharpes = [
                r.get("sharpe_ratio", 0)
                for r in cross_results
                if r.get("sharpe_ratio") is not None
            ]

            if valid_sharpes:
                avg_sharpe = sum(valid_sharpes) / len(valid_sharpes)
                positive_ratio = sum(1 for s in valid_sharpes if s > 0) / len(
                    valid_sharpes
                )

                if avg_sharpe < self.criteria.min_avg_cross_sharpe:
                    return (
                        False,
                        f"Sharpe moyen cross ({avg_sharpe:.3f}) < minimum ({self.criteria.min_avg_cross_sharpe})",
                    )

                if positive_ratio < self.criteria.min_positive_cross_ratio:
                    return (
                        False,
                        f"Ratio positif cross ({positive_ratio:.0%}) < minimum ({self.criteria.min_positive_cross_ratio:.0%})",
                    )

        return True, "✅ Tous les critères sont satisfaits"

    def save_strategy(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        primary_token: str,
        primary_timeframe: str,
        cross_results: Optional[List[Dict[str, Any]]] = None,
        models_used: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        notes: str = "",
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Sauvegarde une stratégie gagnante.

        Args:
            strategy_name: Nom de la stratégie
            params: Paramètres optimisés
            metrics: Métriques de performance
            primary_token: Token principal utilisé
            primary_timeframe: Timeframe principal
            cross_results: Résultats du cross-testing
            models_used: Modèles LLM utilisés
            run_id: ID du run d'origine
            notes: Notes additionnelles
            tags: Tags pour classification
            force: Force la sauvegarde même si non gagnante

        Returns:
            Path du fichier sauvegardé ou None si non sauvegardé
        """
        cross_results = cross_results or []
        models_used = models_used or {}
        tags = tags or []

        # Extraire métriques
        sharpe = metrics.get("sharpe_ratio", 0)
        total_return = metrics.get("total_return", metrics.get("total_pnl_pct", 0))
        max_dd = metrics.get("max_drawdown", metrics.get("max_drawdown_pct", 0))
        win_rate = metrics.get("win_rate", metrics.get("win_rate_pct", 0))
        total_trades = metrics.get("total_trades", 0)
        profit_factor = metrics.get("profit_factor", 0)

        # Vérifier si gagnante
        is_winner, reason = self.is_winning(
            sharpe_ratio=sharpe,
            cross_results=cross_results,
            total_trades=total_trades,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
        )

        if not is_winner and not force:
            logger.info(f"[WinningManager] Stratégie non gagnante: {reason}")
            return None

        # Calculer métriques cross
        avg_cross_sharpe = 0.0
        positive_cross_ratio = 0.0

        if cross_results:
            valid_sharpes = [
                r.get("sharpe_ratio", 0)
                for r in cross_results
                if r.get("sharpe_ratio") is not None
            ]
            if valid_sharpes:
                avg_cross_sharpe = sum(valid_sharpes) / len(valid_sharpes)
                positive_cross_ratio = sum(1 for s in valid_sharpes if s > 0) / len(
                    valid_sharpes
                )

        # Générer ID unique
        timestamp = datetime.now()
        strategy_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{strategy_name[:10]}_{sharpe:.2f}".replace(
            ".", "p"
        )

        # Créer l'objet stratégie
        winning = WinningStrategy(
            id=strategy_id,
            timestamp=timestamp.isoformat(),
            strategy_name=strategy_name,
            params=params,
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_factor=profit_factor,
            primary_token=primary_token,
            primary_timeframe=primary_timeframe,
            cross_test_results=cross_results,
            avg_cross_sharpe=avg_cross_sharpe,
            positive_cross_ratio=positive_cross_ratio,
            models_used=models_used,
            run_id=run_id,
            notes=notes if is_winner else f"[FORCE SAVE] {reason}",
            tags=tags + [strategy_name, f"sharpe_{sharpe:.2f}", primary_token],
        )

        # Sauvegarder le fichier JSON
        filename = f"{strategy_id}.json"
        filepath = self.storage_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(winning.to_json())

        # Mettre à jour l'index
        self.index["entries"][strategy_id] = {
            "id": strategy_id,
            "timestamp": winning.timestamp,
            "strategy_name": strategy_name,
            "sharpe_ratio": sharpe,
            "avg_cross_sharpe": avg_cross_sharpe,
            "params": params,
            "primary_token": primary_token,
            "primary_timeframe": primary_timeframe,
            "file": filename,
            "tags": winning.tags,
        }

        self._save_index()

        logger.info(
            f"[WinningManager] ✅ Stratégie gagnante sauvegardée: {strategy_id} "
            f"(Sharpe={sharpe:.3f}, AvgCross={avg_cross_sharpe:.3f})"
        )

        return filepath

    def list_strategies(
        self,
        strategy_name: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        token: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Liste les stratégies gagnantes avec filtres optionnels.

        Returns:
            Liste des stratégies triées par Sharpe décroissant
        """
        entries = list(self.index["entries"].values())

        # Filtres
        if strategy_name:
            entries = [e for e in entries if e["strategy_name"] == strategy_name]

        if min_sharpe is not None:
            entries = [e for e in entries if e["sharpe_ratio"] >= min_sharpe]

        if token:
            entries = [e for e in entries if e.get("primary_token") == token]

        # Tri par Sharpe décroissant
        entries.sort(key=lambda x: x.get("sharpe_ratio", 0), reverse=True)

        return entries[:limit]

    def get_strategy(self, strategy_id: str) -> Optional[WinningStrategy]:
        """Charge une stratégie complète par son ID."""
        if strategy_id not in self.index["entries"]:
            return None

        filename = self.index["entries"][strategy_id].get("file")
        filepath = self.storage_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return WinningStrategy.from_json(f.read())

    def get_best_params_for_strategy(
        self, strategy_name: str
    ) -> Optional[Dict[str, Any]]:
        """Retourne les meilleurs paramètres pour une stratégie donnée."""
        strategies = self.list_strategies(strategy_name=strategy_name, limit=1)
        if strategies:
            return strategies[0].get("params")
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les stratégies gagnantes."""
        entries = list(self.index["entries"].values())

        if not entries:
            return {"total": 0}

        sharpes = [e["sharpe_ratio"] for e in entries]
        strategies = set(e["strategy_name"] for e in entries)
        tokens = set(e.get("primary_token", "N/A") for e in entries)

        return {
            "total": len(entries),
            "strategies": list(strategies),
            "tokens": list(tokens),
            "sharpe_avg": sum(sharpes) / len(sharpes),
            "sharpe_max": max(sharpes),
            "sharpe_min": min(sharpes),
            "last_updated": self.index.get("updated_at"),
        }


# Instance globale pour accès facile
_manager_instance: Optional[WinningStrategiesManager] = None


def get_winning_manager() -> WinningStrategiesManager:
    """Retourne l'instance globale du gestionnaire."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = WinningStrategiesManager()
    return _manager_instance


def save_winning_strategy(
    strategy_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    primary_token: str = "BTCUSDC",
    primary_timeframe: str = "1h",
    cross_results: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Optional[Path]:
    """
    Fonction utilitaire pour sauvegarder une stratégie gagnante.

    Exemple:
        save_winning_strategy(
            strategy_name="MA_Crossover",
            params={"fast_period": 5, "slow_period": 20, ...},
            metrics={"sharpe_ratio": 1.5, "max_drawdown": -10, ...},
            primary_token="BTCUSDC",
            primary_timeframe="1h",
            cross_results=[{"token": "ETHUSDC", "sharpe_ratio": 1.2, ...}]
        )
    """
    manager = get_winning_manager()
    return manager.save_strategy(
        strategy_name=strategy_name,
        params=params,
        metrics=metrics,
        primary_token=primary_token,
        primary_timeframe=primary_timeframe,
        cross_results=cross_results,
        **kwargs,
    )
