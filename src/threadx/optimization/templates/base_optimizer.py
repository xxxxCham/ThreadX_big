"""
ThreadX Optimization Templates - Template Method Pattern
=========================================================

Classes de base pour optimisations avec Template Method Pattern.
Centralise la logique commune (prepare, iterate, finalize) et
laisse les sous-classes implémenter les détails spécifiques.

Usage:
    from threadx.optimization.templates import BaseOptimizer, OptimizationResult

    class GridOptimizer(BaseOptimizer):
        def run_iteration(self, params: Dict[str, Any]) -> float:
            # Implémentation spécifique grid search
            return score

Author: ThreadX Framework - Phase 2 Step 3.3 DRY Refactoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import time

from threadx.utils.common_imports import pd, np, create_logger

logger = create_logger(__name__)


@dataclass
class OptimizationResult:
    """
    Résultat unifié d'une optimization.

    Attributes:
        best_params: Meilleurs paramètres trouvés
        best_score: Meilleur score (métrique objectif)
        all_results: DataFrame de tous les essais
        iterations: Nombre total d'itérations
        duration_sec: Durée totale d'exécution
        convergence_history: Historique des meilleurs scores
        metadata: Métadonnées additionnelles (optimizer_type, etc.)
    """
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    iterations: int
    duration_sec: float
    convergence_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """
    Classe abstraite de base pour tous les optimizers.

    Implémente le Template Method Pattern:
    - prepare_data(): Préparation initiale
    - run_iteration(): Itération unique (à implémenter)
    - finalize(): Finalisation et cleanup
    - optimize(): Orchestration complète (template method)

    Fournit aussi:
    - Gestion centralisée des logs
    - Gestion centralisée des exceptions
    - Tracking de convergence
    - Validation des résultats
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        verbose: bool = True,
        early_stopping: Optional[int] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize base optimizer.

        Args:
            objective_fn: Fonction objectif à optimiser
                         Prend params dict, retourne score float
            maximize: True pour maximiser, False pour minimiser
            verbose: Afficher logs de progression
            early_stopping: Nombre d'itérations sans amélioration avant arrêt
            tolerance: Tolérance pour considérer une amélioration
        """
        self.objective_fn = objective_fn
        self.maximize = maximize
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.tolerance = tolerance

        # État interne
        self.logger = create_logger(f"{__name__}.{self.__class__.__name__}")
        self.results: List[Dict[str, Any]] = []
        self.convergence_history: List[float] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.iterations_count = 0
        self.iterations_without_improvement = 0
        self.start_time: Optional[float] = None

        self.logger.info(
            f"🔧 {self.__class__.__name__} initialized "
            f"({'maximize' if maximize else 'minimize'})"
        )

    def prepare_data(self) -> None:
        """
        Préparation avant optimization (hook).

        Peut être overridden par sous-classes pour:
        - Validation des paramètres
        - Initialisation de structures de données
        - Setup de caches
        """
        self.logger.debug("Preparing optimization data...")
        self.start_time = time.time()
        self.results = []
        self.convergence_history = []
        self.best_params = None
        self.best_score = None
        self.iterations_count = 0
        self.iterations_without_improvement = 0

    @abstractmethod
    def run_iteration(self, iteration: int) -> Tuple[Dict[str, Any], float]:
        """
        Exécute une itération d'optimization (à implémenter).

        Args:
            iteration: Numéro de l'itération (0-based)

        Returns:
            (params, score): Paramètres testés et score obtenu

        Raises:
            NotImplementedError: Si non implémenté par sous-classe
        """
        raise NotImplementedError("run_iteration() must be implemented by subclass")

    def finalize(self) -> OptimizationResult:
        """
        Finalisation après optimization (hook).

        Peut être overridden pour:
        - Cleanup de ressources
        - Post-processing des résultats
        - Génération de rapports

        Returns:
            OptimizationResult avec tous les résultats
        """
        duration = time.time() - self.start_time if self.start_time else 0

        # Créer DataFrame résultats
        results_df = pd.DataFrame(self.results)

        # Trouver meilleur si pas déjà fait
        if self.best_params is None and len(results_df) > 0:
            if self.maximize:
                best_idx = results_df['score'].idxmax()
            else:
                best_idx = results_df['score'].idxmin()

            best_row = results_df.loc[best_idx]
            self.best_params = best_row.drop('score').to_dict()
            self.best_score = best_row['score']

        self.logger.info(
            f"✅ Optimization complete: "
            f"{self.iterations_count} iterations in {duration:.2f}s"
        )
        self.logger.info(f"   Best: {self.best_params} → score={self.best_score:.4f}")

        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score or 0.0,
            all_results=results_df,
            iterations=self.iterations_count,
            duration_sec=duration,
            convergence_history=self.convergence_history,
            metadata={
                'optimizer_type': self.__class__.__name__,
                'maximize': self.maximize,
                'early_stopped': self.iterations_without_improvement >= (self.early_stopping or float('inf'))
            }
        )

    def optimize(self, max_iterations: int) -> OptimizationResult:
        """
        Template method principal pour optimization complète.

        Orchestration:
        1. prepare_data() - Préparation
        2. Boucle: run_iteration() avec gestion erreurs/convergence
        3. finalize() - Finalisation

        Args:
            max_iterations: Nombre maximum d'itérations

        Returns:
            OptimizationResult complet
        """
        self.logger.info(f"🚀 Starting optimization: max {max_iterations} iterations")

        # 1. Préparation
        try:
            self.prepare_data()
        except Exception as e:
            self.logger.error(f"Preparation failed: {e}", exc_info=True)
            raise

        # 2. Boucle d'optimization
        for iteration in range(max_iterations):
            try:
                # Exécuter itération (implémenté par sous-classe)
                params, score = self.run_iteration(iteration)

                # Enregistrer résultat
                self.results.append({**params, 'score': score})
                self.iterations_count += 1

                # Mettre à jour meilleur
                is_improvement = self._update_best(params, score)

                # Tracking convergence
                self.convergence_history.append(self.best_score)

                # Logs
                if self.verbose and (iteration + 1) % 10 == 0:
                    self.logger.info(
                        f"Progress: {iteration + 1}/{max_iterations} | "
                        f"Best: {self.best_score:.4f}"
                    )

                # Early stopping
                if self.early_stopping and self.iterations_without_improvement >= self.early_stopping:
                    self.logger.info(
                        f"⏹️  Early stopping after {self.early_stopping} "
                        f"iterations without improvement"
                    )
                    break

            except KeyboardInterrupt:
                self.logger.warning("⚠️  Optimization interrupted by user")
                break

            except Exception as e:
                self.logger.error(
                    f"Iteration {iteration} failed: {e}",
                    exc_info=self.verbose
                )
                # Continue malgré erreur d'une itération
                continue

        # 3. Finalisation
        try:
            return self.finalize()
        except Exception as e:
            self.logger.error(f"Finalization failed: {e}", exc_info=True)
            raise

    def _update_best(self, params: Dict[str, Any], score: float) -> bool:
        """
        Met à jour le meilleur résultat si amélioration.

        Args:
            params: Paramètres testés
            score: Score obtenu

        Returns:
            True si amélioration, False sinon
        """
        if self.best_score is None:
            self.best_params = params
            self.best_score = score
            self.iterations_without_improvement = 0
            return True

        # Vérifier amélioration
        if self.maximize:
            is_better = score > self.best_score + self.tolerance
        else:
            is_better = score < self.best_score - self.tolerance

        if is_better:
            self.best_params = params
            self.best_score = score
            self.iterations_without_improvement = 0

            if self.verbose:
                self.logger.debug(
                    f"✨ New best: {params} → {score:.4f} "
                    f"(iteration {self.iterations_count})"
                )

            return True
        else:
            self.iterations_without_improvement += 1
            return False

    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé de l'optimization.

        Returns:
            Dict avec statistiques principales
        """
        if not self.results:
            return {}

        scores = [r['score'] for r in self.results]

        return {
            'optimizer': self.__class__.__name__,
            'iterations': self.iterations_count,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'convergence_rate': len(self.convergence_history) / max(self.iterations_count, 1)
        }


# Exports
__all__ = [
    'BaseOptimizer',
    'OptimizationResult',
]
