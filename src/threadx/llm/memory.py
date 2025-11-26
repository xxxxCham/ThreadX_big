"""
Optimization Memory - Gestion de l'historique des itérations d'optimisation.

Permet de conserver le contexte des dernières itérations pour éviter de re-tester
les mêmes configurations et orienter les propositions futures.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any


class OptimizationMemory:
    """
    Gère la mémoire contextuelle des itérations d'optimisation.

    Features:
    - Stockage des N dernières itérations (paramètres, scores, analyses)
    - Détection des configurations déjà testées (évite redondance)
    - Export/import pour persistance entre sessions
    - Statistiques sur convergence (amélioration moyenne, stagnation)

    Attributes:
        max_size: Nombre maximum d'itérations conservées en mémoire
        iterations: Historique des itérations (FIFO)
    """

    def __init__(self, max_size: int = 10):
        """
        Initialise la mémoire d'optimisation.

        Args:
            max_size: Nombre max d'itérations à conserver (default: 10)
        """
        self.max_size = max_size
        self.iterations: deque[dict[str, Any]] = deque(maxlen=max_size)
        self.logger = logging.getLogger(__name__)

    def add(
        self,
        iteration_num: int,
        params: dict[str, Any],
        score: float,
        metrics: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
        proposals: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Ajoute une itération à l'historique.

        Args:
            iteration_num: Numéro de l'itération
            params: Paramètres de la stratégie testée
            score: Score de qualité obtenu (ex: Sharpe ratio)
            metrics: Métriques complètes du backtest (optionnel)
            analysis: Analyse LLM de l'itération (optionnel)
            proposals: Propositions générées durant l'itération (optionnel)
        """
        iteration_data = {
            "iteration": iteration_num,
            "params": params.copy(),
            "score": score,
            "metrics": metrics or {},
            "analysis": analysis or {},
            "proposals": proposals or [],
        }

        self.iterations.append(iteration_data)
        self.logger.debug(
            "Added iteration %d to memory (score=%.3f)", iteration_num, score
        )

    def get_recent(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Récupère les N dernières itérations.

        Args:
            n: Nombre d'itérations à récupérer

        Returns:
            Liste des N dernières itérations (ou moins si historique court)
        """
        return list(self.iterations)[-n:]

    def has_tested(self, params: dict[str, Any], tolerance: float = 0.01) -> bool:
        """
        Vérifie si une configuration de paramètres a déjà été testée.

        Args:
            params: Paramètres à vérifier
            tolerance: Tolérance relative pour comparaison (0.01 = 1%)

        Returns:
            True si une config similaire existe déjà dans l'historique
        """
        for iteration in self.iterations:
            if self._params_similar(iteration["params"], params, tolerance):
                return True
        return False

    def _params_similar(
        self, params1: dict[str, Any], params2: dict[str, Any], tolerance: float
    ) -> bool:
        """
        Compare deux configurations de paramètres avec tolérance.

        Args:
            params1: Première config
            params2: Seconde config
            tolerance: Tolérance relative (ex: 0.01 pour 1%)

        Returns:
            True si les configs sont similaires à tolerance près
        """
        # Vérifier que les clés correspondent
        if set(params1.keys()) != set(params2.keys()):
            return False

        # Comparer chaque paramètre
        for key in params1.keys():
            val1 = params1[key]
            val2 = params2[key]

            # Si valeurs numériques, comparer avec tolérance
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 == 0:
                    # Éviter division par zéro
                    if abs(val2) > tolerance:
                        return False
                else:
                    relative_diff = abs(val1 - val2) / abs(val1)
                    if relative_diff > tolerance:
                        return False
            else:
                # Pour autres types, comparaison stricte
                if val1 != val2:
                    return False

        return True

    def get_best_iteration(self) -> dict[str, Any] | None:
        """
        Retourne l'itération avec le meilleur score.

        Returns:
            Dict de l'itération avec score max, ou None si historique vide
        """
        if not self.iterations:
            return None

        return max(self.iterations, key=lambda x: x["score"])

    def get_convergence_stats(self) -> dict[str, Any]:
        """
        Calcule des statistiques sur la convergence de l'optimisation.

        Returns:
            dict avec:
            - total_iterations: Nombre d'itérations effectuées
            - best_score: Meilleur score atteint
            - avg_improvement: Amélioration moyenne par itération
            - stagnation_count: Nombre d'itérations consécutives sans amélioration
        """
        if not self.iterations:
            return {
                "total_iterations": 0,
                "best_score": 0.0,
                "avg_improvement": 0.0,
                "stagnation_count": 0,
            }

        scores = [it["score"] for it in self.iterations]
        best_score = max(scores)

        # Calculer amélioration moyenne
        improvements = []
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i - 1]
            improvements.append(improvement)

        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        # Compter stagnation à la fin
        stagnation_count = 0
        for i in range(len(scores) - 1, 0, -1):
            if scores[i] <= scores[i - 1]:
                stagnation_count += 1
            else:
                break

        return {
            "total_iterations": len(self.iterations),
            "best_score": best_score,
            "avg_improvement": avg_improvement,
            "stagnation_count": stagnation_count,
            "score_trend": scores,
        }

    def export_to_file(self, filepath: Path | str) -> None:
        """
        Exporte l'historique dans un fichier JSON.

        Args:
            filepath: Chemin du fichier de sortie
        """
        filepath = Path(filepath)
        data = {
            "max_size": self.max_size,
            "iterations": list(self.iterations),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info("Memory exported to %s", filepath)

    def import_from_file(self, filepath: Path | str) -> None:
        """
        Importe un historique depuis un fichier JSON.

        Args:
            filepath: Chemin du fichier d'entrée
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Memory file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.max_size = data.get("max_size", self.max_size)
        self.iterations = deque(data.get("iterations", []), maxlen=self.max_size)

        self.logger.info(
            "Memory imported from %s (%d iterations)", filepath, len(self.iterations)
        )

    def clear(self) -> None:
        """Vide complètement l'historique."""
        self.iterations.clear()
        self.logger.info("Memory cleared")

    def __len__(self) -> int:
        """Retourne le nombre d'itérations en mémoire."""
        return len(self.iterations)

    def __repr__(self) -> str:
        """Représentation textuelle de la mémoire."""
        stats = self.get_convergence_stats()
        return (
            f"OptimizationMemory("
            f"iterations={stats['total_iterations']}, "
            f"best_score={stats['best_score']:.3f}, "
            f"stagnation={stats['stagnation_count']})"
        )
