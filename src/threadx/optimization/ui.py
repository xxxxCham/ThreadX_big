"""
ThreadX Optimization UI Helpers
===============================

Backwards-compatibility shims that expose a small façade around
:class:`~threadx.optimization.engine.SweepRunner`.  Some legacy scripts still
import ``ParametricOptimizationUI`` from ``threadx.optimization.ui``; the class
below keeps those imports working while reusing the unified optimisation engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .engine import SweepRunner


@dataclass
class ParametricOptimizationUI:
    """Friendly wrapper that delegates to :class:`SweepRunner`."""

    runner: SweepRunner = field(default_factory=SweepRunner)

    def run_grid(self, grid_spec: dict[str, Any], *, reuse_cache: bool = True) -> pd.DataFrame:
        """Execute a grid sweep and return the resulting :class:`DataFrame`."""
        return self.runner.run_grid(grid_spec, reuse_cache=reuse_cache)

    def run_monte_carlo(
        self,
        mc_spec: dict[str, Any],
        *,
        reuse_cache: bool = True,
    ) -> pd.DataFrame:
        """Execute a Monte-Carlo optimisation and return the results."""
        return self.runner.run_monte_carlo(mc_spec, reuse_cache=reuse_cache)


def create_optimization_ui(runner: SweepRunner | None = None) -> ParametricOptimizationUI:
    """Factory helper used by older tooling."""
    return ParametricOptimizationUI(runner=runner or SweepRunner())


# Historical entry-point kept for compatibility -------------------------------------------------

def init_ui(*args: Any, **kwargs: Any) -> ParametricOptimizationUI:
    """Return a ready-to-use optimisation helper instance."""
    return create_optimization_ui(*args, **kwargs)
