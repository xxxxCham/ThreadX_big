"""
Tests for Optimization Templates (Step 3.3)
============================================

Tests pour valider le refactoring DRY des optimizers
avec Template Method Pattern.

Author: ThreadX Framework - Phase 2 Step 3.3
"""

import pytest
import numpy as np
from typing import Dict, Any

from threadx.optimization.templates import (
    BaseOptimizer,
    GridOptimizer,
    MonteCarloOptimizer,
    OptimizationResult,
    grid_search,
    monte_carlo_search,
)


# ===== Fixtures =====


@pytest.fixture
def simple_objective():
    """Fonction objectif simple: minimise (x-5)² + (y-3)²"""

    def objective(params: Dict[str, Any]) -> float:
        x = params["x"]
        y = params["y"]
        return -((x - 5) ** 2 + (y - 3) ** 2)  # Négatif pour maximization

    return objective


@pytest.fixture
def quadratic_objective():
    """Fonction objectif quadratique: -(period/20 - 1)² - (std/2 - 1)²"""

    def objective(params: Dict[str, Any]) -> float:
        period = params["period"]
        std = params["std"]
        return -(((period / 20) - 1) ** 2 + ((std / 2) - 1) ** 2)

    return objective


@pytest.fixture
def failing_objective():
    """Fonction qui échoue pour certains paramètres"""

    def objective(params: Dict[str, Any]) -> float:
        x = params["x"]
        if x < 0:
            raise ValueError("x must be >= 0")
        return -(x**2)

    return objective


# ===== Tests BaseOptimizer =====


class ConcreteOptimizer(BaseOptimizer):
    """Optimizer concret pour tester BaseOptimizer"""

    def __init__(self, *args, values=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = values or [(1, 2), (3, 4), (5, 6)]
        self.current = 0

    def run_iteration(self, iteration: int):
        if iteration >= len(self.values):
            raise StopIteration()
        x, y = self.values[iteration]
        params = {"x": x, "y": y}
        score = self.objective_fn(params)
        return params, score


def test_base_optimizer_abstract():
    """BaseOptimizer ne peut pas être instancié directement"""
    with pytest.raises(TypeError):
        BaseOptimizer(objective_fn=lambda p: 0)


def test_base_optimizer_template_method(simple_objective):
    """Template method optimize() fonctionne correctement"""
    optimizer = ConcreteOptimizer(
        objective_fn=simple_objective, maximize=True, verbose=False
    )

    result = optimizer.optimize(max_iterations=3)

    assert isinstance(result, OptimizationResult)
    assert result.iterations == 3
    assert len(result.all_results) == 3
    assert result.best_params is not None
    assert result.best_score is not None
    assert result.duration_sec >= 0


def test_base_optimizer_convergence_tracking(simple_objective):
    """Convergence history est correctement trackée"""
    optimizer = ConcreteOptimizer(
        objective_fn=simple_objective, maximize=True, verbose=False
    )

    result = optimizer.optimize(max_iterations=3)

    assert len(result.convergence_history) == 3
    assert all(isinstance(s, (int, float)) for s in result.convergence_history)


def test_base_optimizer_early_stopping(simple_objective):
    """Early stopping fonctionne"""
    optimizer = ConcreteOptimizer(
        objective_fn=simple_objective,
        maximize=True,
        verbose=False,
        early_stopping=2,  # Stop après 2 itérations sans amélioration
    )

    # Créer valeurs qui ne s'améliorent pas
    optimizer.values = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    result = optimizer.optimize(max_iterations=5)

    # Devrait s'arrêter avant 5 itérations
    assert result.iterations <= 3  # 1 best + 2 sans amélioration


def test_base_optimizer_error_handling(failing_objective):
    """Erreurs dans objective_fn sont gérées"""
    optimizer = ConcreteOptimizer(
        objective_fn=failing_objective, maximize=True, verbose=False
    )

    # Inclure valeurs qui vont échouer
    optimizer.values = [(-1, 0), (1, 0), (2, 0)]

    # Ne devrait pas crash
    result = optimizer.optimize(max_iterations=3)

    # Au moins une itération devrait réussir
    assert result.iterations == 3


# ===== Tests GridOptimizer =====


def test_grid_optimizer_basic(quadratic_objective):
    """Grid search trouve l'optimum"""
    optimizer = GridOptimizer(
        param_grid={"period": [10, 20, 30], "std": [1.0, 2.0, 3.0]},
        objective_fn=quadratic_objective,
        maximize=True,
        verbose=False,
    )

    result = optimizer.optimize()

    # 3 * 3 = 9 combinaisons
    assert result.iterations == 9
    assert len(result.all_results) == 9

    # Optimum devrait être proche de period=20, std=2.0
    assert result.best_params["period"] == 20
    assert result.best_params["std"] == 2.0


def test_grid_optimizer_single_param():
    """Grid search avec un seul paramètre"""

    def objective(params):
        return -((params["x"] - 5) ** 2)

    optimizer = GridOptimizer(
        param_grid={"x": [1, 3, 5, 7, 9]},
        objective_fn=objective,
        maximize=True,
        verbose=False,
    )

    result = optimizer.optimize()

    assert result.iterations == 5
    assert result.best_params["x"] == 5  # Optimum global


def test_grid_optimizer_param_importance(quadratic_objective):
    """get_param_importance() fonctionne"""
    optimizer = GridOptimizer(
        param_grid={"period": [10, 20, 30], "std": [1.0, 2.0, 3.0]},
        objective_fn=quadratic_objective,
        maximize=True,
        verbose=False,
    )

    optimizer.optimize()
    importance = optimizer.get_param_importance()

    assert "period" in importance
    assert "std" in importance
    assert all(isinstance(v, (int, float)) for v in importance.values())


def test_grid_search_helper(quadratic_objective):
    """Fonction helper grid_search() fonctionne"""
    result = grid_search(
        param_grid={"period": [15, 20, 25], "std": [1.5, 2.0, 2.5]},
        objective_fn=quadratic_objective,
        maximize=True,
        verbose=False,
    )

    assert isinstance(result, OptimizationResult)
    assert result.iterations == 9


def test_grid_optimizer_empty_grid():
    """Validation: grille vide échoue"""
    with pytest.raises(ValueError, match="cannot be empty"):
        GridOptimizer(param_grid={}, objective_fn=lambda p: 0)


def test_grid_optimizer_invalid_param():
    """Validation: param sans valeurs échoue"""
    optimizer = GridOptimizer(
        param_grid={"x": []}, objective_fn=lambda p: 0, verbose=False
    )

    with pytest.raises(ValueError, match="must have at least one value"):
        optimizer.prepare_data()


# ===== Tests MonteCarloOptimizer =====


def test_monte_carlo_basic(simple_objective):
    """Monte Carlo trouve une bonne solution"""
    optimizer = MonteCarloOptimizer(
        param_ranges={"x": (0, 10), "y": (0, 10)},
        objective_fn=simple_objective,
        n_trials=50,
        maximize=True,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    assert result.iterations == 50
    assert len(result.all_results) == 50

    # Optimum est (5, 3), on devrait être proche
    assert abs(result.best_params["x"] - 5) < 2
    assert abs(result.best_params["y"] - 3) < 2


def test_monte_carlo_integer_params():
    """Monte Carlo avec paramètres entiers"""

    def objective(params):
        return -((params["n"] - 10) ** 2)

    optimizer = MonteCarloOptimizer(
        param_ranges={"n": (1, 20)},
        objective_fn=objective,
        n_trials=30,
        maximize=True,
        seed=42,
        verbose=False,
    )

    result = optimizer.optimize()

    # n devrait être entier
    assert isinstance(result.best_params["n"], (int, np.integer))


def test_monte_carlo_reproducibility():
    """Seed assure reproductibilité"""

    def objective(params):
        return params["x"] + params["y"]

    # Deux runs avec même seed
    result1 = monte_carlo_search(
        param_ranges={"x": (0, 10), "y": (0, 10)},
        objective_fn=objective,
        n_trials=20,
        seed=123,
        verbose=False,
    )

    result2 = monte_carlo_search(
        param_ranges={"x": (0, 10), "y": (0, 10)},
        objective_fn=objective,
        n_trials=20,
        seed=123,
        verbose=False,
    )

    # Résultats identiques
    assert result1.best_params == result2.best_params
    assert result1.best_score == result2.best_score


def test_monte_carlo_param_distributions(simple_objective):
    """get_param_distributions() fonctionne"""
    optimizer = MonteCarloOptimizer(
        param_ranges={"x": (0, 10), "y": (0, 10)},
        objective_fn=simple_objective,
        n_trials=50,
        seed=42,
        verbose=False,
    )

    optimizer.optimize()
    distributions = optimizer.get_param_distributions()

    assert "x" in distributions
    assert "y" in distributions
    assert "mean" in distributions["x"]
    assert "std" in distributions["x"]


def test_monte_carlo_best_region(simple_objective):
    """get_best_region() identifie zone optimale"""
    optimizer = MonteCarloOptimizer(
        param_ranges={"x": (0, 10), "y": (0, 10)},
        objective_fn=simple_objective,
        n_trials=100,
        maximize=True,
        seed=42,
        verbose=False,
    )

    optimizer.optimize()
    best_region = optimizer.get_best_region(percentile=90)

    assert "x" in best_region
    assert "y" in best_region

    # Région devrait être autour de (5, 3)
    x_min, x_max = best_region["x"]
    y_min, y_max = best_region["y"]

    assert 3 <= x_min <= 7
    assert 3 <= x_max <= 7
    assert 1 <= y_min <= 5
    assert 1 <= y_max <= 5


def test_monte_carlo_empty_ranges():
    """Validation: ranges vides échouent"""
    with pytest.raises(ValueError, match="cannot be empty"):
        MonteCarloOptimizer(param_ranges={}, objective_fn=lambda p: 0)


def test_monte_carlo_invalid_range():
    """Validation: range invalide échoue"""
    optimizer = MonteCarloOptimizer(
        param_ranges={"x": (10, 5)},  # min > max
        objective_fn=lambda p: 0,
        n_trials=10,
        verbose=False,
    )

    with pytest.raises(ValueError, match="Invalid range"):
        optimizer.prepare_data()


# ===== Tests OptimizationResult =====


def test_optimization_result_dataclass():
    """OptimizationResult est bien un dataclass"""
    import pandas as pd

    result = OptimizationResult(
        best_params={"x": 5},
        best_score=10.0,
        all_results=pd.DataFrame([{"x": 5, "score": 10.0}]),
        iterations=1,
        duration_sec=0.5,
    )

    assert result.best_params == {"x": 5}
    assert result.best_score == 10.0
    assert result.iterations == 1
    assert result.duration_sec == 0.5
    assert len(result.convergence_history) == 0  # default
    assert len(result.metadata) == 0  # default


# ===== Tests d'intégration =====


def test_grid_vs_monte_carlo_consistency():
    """Grid et Monte Carlo devraient converger vers même optimum"""

    def objective(params):
        return -((params["x"] - 3) ** 2)

    # Grid search
    grid_result = grid_search(
        param_grid={"x": list(range(0, 10))},
        objective_fn=objective,
        maximize=True,
        verbose=False,
    )

    # Monte Carlo
    mc_result = monte_carlo_search(
        param_ranges={"x": (0, 9)},
        objective_fn=objective,
        n_trials=100,
        seed=42,
        maximize=True,
        verbose=False,
    )

    # Les deux devraient trouver x=3
    assert grid_result.best_params["x"] == 3
    assert abs(mc_result.best_params["x"] - 3) <= 1  # Monte Carlo moins précis


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
