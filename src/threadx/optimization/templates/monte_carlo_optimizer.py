"""
ThreadX Monte Carlo Optimizer
==============================

Monte Carlo (random search) implementation using BaseOptimizer template.

Author: ThreadX Framework - Phase 2 Step 3.3
"""

from typing import Any, Dict, Tuple, Callable
import numpy as np

from threadx.utils.common_imports import create_logger
from .base_optimizer import BaseOptimizer

logger = create_logger(__name__)


class MonteCarloOptimizer(BaseOptimizer):
    """
    Monte Carlo optimizer - tire des paramètres aléatoires.

    Hérite de BaseOptimizer et implémente run_iteration()
    pour tirer aléatoirement des paramètres dans des ranges.

    Usage:
        optimizer = MonteCarloOptimizer(
            param_ranges={'period': (10, 50), 'std': (1.0, 3.0)},
            objective_fn=my_backtest_fn,
            n_trials=100,
            maximize=True,
            seed=42
        )
        result = optimizer.optimize(max_iterations=100)
    """

    def __init__(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        maximize: bool = True,
        verbose: bool = True,
        seed: int = 42,
        early_stopping: int = None
    ):
        """
        Initialize Monte Carlo optimizer.

        Args:
            param_ranges: Dict de paramètres avec ranges (min, max)
                         Ex: {'period': (10, 50), 'std': (1.0, 3.0)}
            objective_fn: Fonction à optimiser
            n_trials: Nombre d'essais aléatoires
            maximize: True pour maximiser
            verbose: Afficher progression
            seed: Seed pour reproductibilité
            early_stopping: Arrêt si pas d'amélioration après N essais
        """
        super().__init__(
            objective_fn=objective_fn,
            maximize=maximize,
            verbose=verbose,
            early_stopping=early_stopping
        )

        self.param_ranges = param_ranges
        self.n_trials = n_trials
        self.seed = seed

        # Random state pour reproductibilité
        self.rng = np.random.RandomState(seed)

        self.logger.info(
            f"Monte Carlo initialized: {n_trials} trials "
            f"(seed={seed})"
        )

    def prepare_data(self) -> None:
        """Override: Validation des ranges."""
        super().prepare_data()

        # Validation
        if not self.param_ranges:
            raise ValueError("param_ranges cannot be empty")

        for param, (min_val, max_val) in self.param_ranges.items():
            if min_val >= max_val:
                raise ValueError(
                    f"Invalid range for '{param}': "
                    f"min ({min_val}) >= max ({max_val})"
                )

        # Réinitialiser RNG avec seed
        self.rng = np.random.RandomState(self.seed)

    def run_iteration(self, iteration: int) -> Tuple[Dict[str, Any], float]:
        """
        Exécute une itération Monte Carlo.

        Args:
            iteration: Numéro de l'itération

        Returns:
            (params, score): Paramètres tirés et score
        """
        # Tirer paramètres aléatoires
        params = self._sample_params()

        # Évaluer objective function
        try:
            score = self.objective_fn(params)

            if self.verbose and (iteration + 1) % 20 == 0:
                self.logger.debug(
                    f"Trial {iteration + 1}/{self.n_trials}: "
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

    def _sample_params(self) -> Dict[str, Any]:
        """
        Tire aléatoirement des paramètres dans les ranges.

        Returns:
            Dict de paramètres tirés
        """
        params = {}

        for name, (min_val, max_val) in self.param_ranges.items():
            # Détecter type de paramètre (int vs float)
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Paramètre entier
                params[name] = self.rng.randint(min_val, max_val + 1)
            else:
                # Paramètre float
                params[name] = self.rng.uniform(min_val, max_val)

        return params

    def optimize(self, max_iterations: int = None) -> OptimizationResult:
        """
        Lance l'optimisation Monte Carlo.

        Args:
            max_iterations: Nombre max d'itérations (défaut: n_trials)

        Returns:
            Résultat de l'optimisation
        """
        # Valider d'abord les ranges
        if not hasattr(self, '_prepared'):
            self.prepare_data()
            self._prepared = True

        if max_iterations is None:
            max_iterations = self.n_trials

        return super().optimize(max_iterations)

    def get_param_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Analyse la distribution des paramètres testés.

        Returns:
            Dict {param_name: {'mean': ..., 'std': ..., ...}}
        """
        if not self.results:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.results)

        distributions = {}
        for param in self.param_ranges.keys():
            values = df[param]
            distributions[param] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median()
            }

        return distributions

    def get_best_region(self, percentile: float = 90) -> Dict[str, Tuple[float, float]]:
        """
        Identifie la région des meilleurs paramètres.

        Args:
            percentile: Percentile pour définir "meilleurs" (90 = top 10%)

        Returns:
            Dict {param_name: (min, max)} de la région optimale
        """
        if not self.results:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.results)

        # Filtrer top percentile
        if self.maximize:
            threshold = df['score'].quantile(percentile / 100)
            top_df = df[df['score'] >= threshold]
        else:
            threshold = df['score'].quantile(1 - percentile / 100)
            top_df = df[df['score'] <= threshold]

        # Calculer ranges pour chaque param
        best_region = {}
        for param in self.param_ranges.keys():
            values = top_df[param]
            best_region[param] = (values.min(), values.max())

        return best_region


# Convenience function
def monte_carlo_search(
    param_ranges: Dict[str, Tuple[float, float]],
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 100,
    maximize: bool = True,
    seed: int = 42,
    verbose: bool = True
) -> 'OptimizationResult':
    """
    Fonction helper pour Monte Carlo search rapide.

    Args:
        param_ranges: Ranges de paramètres
        objective_fn: Fonction objectif
        n_trials: Nombre d'essais
        maximize: Maximiser ou minimiser
        seed: Seed aléatoire
        verbose: Logs verbose

    Returns:
        OptimizationResult

    Example:
        >>> result = monte_carlo_search(
        ...     param_ranges={'period': (10, 50), 'std': (1.0, 3.0)},
        ...     objective_fn=my_backtest,
        ...     n_trials=100,
        ...     maximize=True
        ... )
        >>> print(result.best_params)
    """
    optimizer = MonteCarloOptimizer(
        param_ranges=param_ranges,
        objective_fn=objective_fn,
        n_trials=n_trials,
        maximize=maximize,
        seed=seed,
        verbose=verbose
    )
    return optimizer.optimize()
