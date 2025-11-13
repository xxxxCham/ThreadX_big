"""
ThreadX - Multi-Sweep Parallèle
=================================

Lancer plusieurs sweeps simultanément pour saturer toutes les ressources.

Architecture:
- N sweeps en parallèle (ProcessPoolExecutor)
- Chaque sweep utilise son propre SweepRunner avec workers réduits
- Total workers = N × workers_per_sweep ≈ CPU threads disponibles
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from threadx.indicators.bank import IndicatorBank
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec

logger = logging.getLogger(__name__)


@dataclass
class MultiSweepConfig:
    """Configuration pour multi-sweep parallèle."""

    n_parallel_sweeps: int = 4  # Nombre de sweeps simultanés
    workers_per_sweep: int | None = None  # Workers par sweep (None = auto)
    use_multigpu: bool = True


def _run_single_sweep(
    sweep_id: int,
    grid_spec: ScenarioSpec,
    real_data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    workers: int | None,
    use_multigpu: bool,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Fonction worker pour exécuter un sweep dans un process séparé.

    Cette fonction DOIT être au top-level (pas nested) pour être picklable.
    """
    # Créer IndicatorBank + SweepRunner dans le process worker
    from threadx.indicators.bank import IndicatorBank, IndicatorSettings

    settings = IndicatorSettings(use_gpu=True)
    bank = IndicatorBank(settings)

    runner = SweepRunner(
        indicator_bank=bank, max_workers=workers, use_multigpu=use_multigpu
    )

    logger.info(
        f"🔄 Sweep {sweep_id} démarré - {workers} workers, Multi-GPU: {use_multigpu}"
    )

    results = runner.run_grid(
        grid_spec=grid_spec,
        real_data=real_data,
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
    )

    logger.info(f"✅ Sweep {sweep_id} terminé - {len(results)} résultats")

    return sweep_id, results


class MultiSweepRunner:
    """Runner pour exécuter plusieurs sweeps en parallèle."""

    def __init__(self, config: MultiSweepConfig | None = None):
        self.config = config or MultiSweepConfig()

        # Calcul workers par sweep
        if self.config.workers_per_sweep is None:
            # Auto: diviser total workers par n_parallel_sweeps
            import os

            total_workers = os.cpu_count() or 16
            self.workers_per_sweep = max(8, total_workers // self.config.n_parallel_sweeps)
        else:
            self.workers_per_sweep = self.config.workers_per_sweep

        logger.info(
            f"🚀 MultiSweepRunner: {self.config.n_parallel_sweeps} sweeps × {self.workers_per_sweep} workers = {self.config.n_parallel_sweeps * self.workers_per_sweep} total workers"
        )

    def run_parallel_sweeps(
        self,
        grid_specs: list[ScenarioSpec],
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy_name: str = "Bollinger_Breakout",
    ) -> dict[int, list[dict[str, Any]]]:
        """
        Exécute plusieurs sweeps en parallèle.

        Args:
            grid_specs: Liste des grilles à exécuter (1 par sweep)
            real_data: Données OHLCV
            symbol: Symbole (BTCUSDC, etc.)
            timeframe: Timeframe (15m, etc.)
            strategy_name: Nom stratégie

        Returns:
            Dict {sweep_id: results}
        """
        if len(grid_specs) != self.config.n_parallel_sweeps:
            raise ValueError(
                f"grid_specs length ({len(grid_specs)}) != n_parallel_sweeps ({self.config.n_parallel_sweeps})"
            )

        logger.info(
            f"\n{'='*80}\n"
            f"🚀 MULTI-SWEEP PARALLÈLE\n"
            f"{'='*80}\n"
            f"  Sweeps simultanés: {self.config.n_parallel_sweeps}\n"
            f"  Workers par sweep: {self.workers_per_sweep}\n"
            f"  Total workers: {self.config.n_parallel_sweeps * self.workers_per_sweep}\n"
            f"  Multi-GPU: {self.config.use_multigpu}\n"
            f"{'='*80}"
        )

        results_by_sweep = {}

        # Lancer sweeps en parallèle via ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.config.n_parallel_sweeps) as executor:
            # Soumettre tous les sweeps
            futures = {}
            for sweep_id, grid_spec in enumerate(grid_specs):
                future = executor.submit(
                    _run_single_sweep,
                    sweep_id,
                    grid_spec,
                    real_data,
                    symbol,
                    timeframe,
                    strategy_name,
                    self.workers_per_sweep,
                    self.config.use_multigpu,
                )
                futures[future] = sweep_id

            # Collecter résultats au fur et à mesure
            for future in as_completed(futures):
                sweep_id = futures[future]
                try:
                    returned_id, results = future.result()
                    results_by_sweep[returned_id] = results
                    logger.info(
                        f"📊 Sweep {returned_id} collecté - {len(results)} résultats"
                    )
                except Exception as e:
                    logger.error(f"❌ Erreur sweep {sweep_id}: {e}")
                    import traceback

                    traceback.print_exc()
                    results_by_sweep[sweep_id] = []

        logger.info(
            f"\n{'='*80}\n"
            f"✅ MULTI-SWEEP TERMINÉ\n"
            f"  Total résultats: {sum(len(r) for r in results_by_sweep.values())}\n"
            f"{'='*80}"
        )

        return results_by_sweep
