"""
ThreadX Grid Search Optimizer
==============================

Grid search implementation using BaseOptimizer template.

Author: ThreadX Framework - Phase 2 Step 3.3
"""

from typing import Any, Dict, List, Tuple, Callable
from itertools import product

from threadx.utils.common_imports import create_logger
from .base_optimizer import BaseOptimizer

logger = create_logger(__name__)


class GridOptimizer(BaseOptimizer):
    """
    Grid search optimizer - teste toutes les combinaisons.

    Hérite de BaseOptimizer et implémente run_iteration()
    pour parcourir systématiquement une grille de paramètres.

    Usage:
        optimizer = GridOptimizer(
            param_grid={'period': [10, 20, 30], 'std': [1.5, 2.0]},
            objective_fn=my_backtest_fn,
            maximize=True
        )
        result = optimizer.optimize(max_iterations=6)  # 3*2 combinations
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        objective_fn: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        verbose: bool = True,
        parallel: bool = False,
        n_jobs: int = -1
    ):
        """
        Initialize grid search optimizer.

        Args:
            param_grid: Dict de paramètres avec listes de valeurs
                       Ex: {'period': [10, 20, 30], 'std': [1.5, 2.0]}
            objective_fn: Fonction à optimiser
            maximize: True pour maximiser
            verbose: Afficher progression
            parallel: Exécution parallèle (future feature)
            n_jobs: Nombre de workers (future feature)
        """
        super().__init__(
            objective_fn=objective_fn,
            maximize=maximize,
            verbose=verbose
        )

        self.param_grid = param_grid
        self.parallel = parallel
        self.n_jobs = n_jobs

        # Générer toutes les combinaisons
        self._generate_combinations()

        self.logger.info(
            f"Grid search initialized: {len(self.combinations)} combinations"
        )

    def _generate_combinations(self) -> None:
        """Génère toutes les combinaisons de paramètres."""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        # Produit cartésien
        combinations_tuples = list(product(*param_values))

        # Convertir en liste de dicts
        self.combinations = [
            dict(zip(param_names, combo))
            for combo in combinations_tuples
        ]

        self.logger.debug(
            f"Generated {len(self.combinations)} parameter combinations"
        )

    def prepare_data(self) -> None:
        """Override: Re-génère combinations si besoin."""
        super().prepare_data()

        # Validation de la grille
        if not self.param_grid:
            raise ValueError("param_grid cannot be empty")

        for param, values in self.param_grid.items():
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                raise ValueError(
                    f"Parameter '{param}' must have at least one value"
                )

    def run_iteration(self, iteration: int) -> Tuple[Dict[str, Any], float]:
        """
        Exécute une itération de grid search.

        Args:
            iteration: Index de l'itération (correspond à index dans combinations)

        Returns:
            (params, score): Paramètres testés et score
        """
        # Vérifier qu'on a encore des combinaisons
        if iteration >= len(self.combinations):
            raise StopIteration("All combinations tested")

        # Récupérer params pour cette itération
        params = self.combinations[iteration]

        # Évaluer objective function
        try:
            score = self.objective_fn(params)

            if self.verbose and (iteration + 1) % 5 == 0:
                self.logger.debug(
                    f"Iteration {iteration + 1}/{len(self.combinations)}: "
                    f"{params} → {score:.4f}"
                )

            return params, score

        except Exception as e:
            self.logger.warning(
                f"Objective function failed for {params}: {e}"
            )
            # Retourner score worst possible
            score = float('-inf') if self.maximize else float('inf')
            return params, score

    def optimize(self, max_iterations: int = None) -> 'OptimizationResult':
        """
        Override: Grid search teste TOUTES les combinaisons.

        Args:
            max_iterations: Ignoré (utilise len(combinations))

        Returns:
            OptimizationResult
        """
        # Préparer d'abord pour avoir combinations
        if not self.combinations:
            self.prepare_data()

        # Pour grid search, max_iterations = nombre de combinaisons
        actual_max = len(self.combinations)

        if max_iterations is not None and max_iterations != actual_max:
            self.logger.warning(
                f"max_iterations={max_iterations} ignored for grid search, "
                f"using {actual_max} (all combinations)"
            )

        return super().optimize(max_iterations=actual_max)

    def get_param_importance(self) -> Dict[str, float]:
        """
        Calcule l'importance de chaque paramètre (variance du score).

        Returns:
            Dict {param_name: importance_score}
        """
        if not self.results:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.results)

        importance = {}
        for param in self.param_grid.keys():
            # Variance du score groupé par valeur du paramètre
            grouped = df.groupby(param)['score'].var()
            importance[param] = grouped.mean()

        return importance


# Convenience function
def grid_search(
    param_grid: Dict[str, List[Any]],
    objective_fn: Callable[[Dict[str, Any]], float],
    maximize: bool = True,
    verbose: bool = True
) -> 'OptimizationResult':
    """
    Fonction helper pour grid search rapide.

    Args:
        param_grid: Grille de paramètres
        objective_fn: Fonction objectif
        maximize: Maximiser ou minimiser
        verbose: Logs verbose

    Returns:
        OptimizationResult

    Example:
        >>> result = grid_search(
        ...     param_grid={'period': [10, 20], 'std': [2.0, 2.5]},
        ...     objective_fn=my_backtest,
        ...     maximize=True
        ... )
        >>> print(result.best_params)
    """
    optimizer = GridOptimizer(
        param_grid=param_grid,
        objective_fn=objective_fn,
        maximize=maximize,
        verbose=verbose
    )
    return optimizer.optimize()
