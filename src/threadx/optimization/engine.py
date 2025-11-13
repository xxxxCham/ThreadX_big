"""
ThreadX Parametric Optimization Engine
======================================

Moteur d'optimisation paramétrique unifié qui utilise les composants existants :
- IndicatorBank (Phase 3) pour les calculs d'indicateurs avec cache GPU
- BacktestEngine (Phase 5) pour l'exécution des stratégies
- PerformanceCalculator (Phase 6) pour les métriques
- Cache intelligent avec TTL et checksums

Toutes les fonctions de calcul sont centralisées autour d'IndicatorBank.

Author: ThreadX Framework
Version: Phase 10 - Unified Compute Engine
"""

import hashlib
import itertools
import json
import time
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeout,
)
from typing import Any

import numpy as np
import pandas as pd

from threadx.indicators.bank import IndicatorBank
from threadx.utils.log import get_logger

from .pruning import pareto_soft_prune
from .scenarios import ScenarioSpec, generate_monte_carlo, generate_param_grid

# Multi-GPU support
try:
    from threadx.gpu.multi_gpu import get_default_manager

    MULTIGPU_AVAILABLE = True
except ImportError:
    MULTIGPU_AVAILABLE = False

# Monitoring système pour workers dynamiques
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = get_logger(__name__)

# Global stop flag to allow UI-triggered cancellation without direct runner reference
_GLOBAL_STOP_FLAG = False

# Optional process-global payloads to reduce per-task serialization overhead
_G_INDICATORS = None
_G_REAL_DATA = None
_G_SYMBOL = None
_G_TIMEFRAME = None
_G_STRATEGY_NAME = None


def _init_process_globals(
    computed_indicators: dict,
    real_data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
) -> None:
    global _G_INDICATORS, _G_REAL_DATA, _G_SYMBOL, _G_TIMEFRAME, _G_STRATEGY_NAME
    _G_INDICATORS = computed_indicators
    _G_REAL_DATA = real_data
    _G_SYMBOL = symbol
    _G_TIMEFRAME = timeframe
    _G_STRATEGY_NAME = strategy_name


def set_global_stop(stop: bool = True) -> None:
    """Définir le flag global pour arrêter l'exécution."""
    global _GLOBAL_STOP_FLAG
    _GLOBAL_STOP_FLAG = stop
    if stop:
        logger.warning("⏹️ ARRÊT GLOBAL DEMANDÉ")


# ✅ Fonction worker standalone (picklable pour ProcessPoolExecutor)
def _evaluate_combo_worker(
    combo: dict,
    computed_indicators: dict | None,
    real_data: pd.DataFrame | None,
    symbol: str | None,
    timeframe: str | None,
    strategy_name: str = "Bollinger_Breakout",
) -> dict:
    """
    Worker function pour évaluation combo (picklable pour ProcessPoolExecutor).

    Chaque process crée sa propre instance de stratégie + IndicatorBank.
    """
    try:
        # Import local pour éviter overhead dans process principal
        from threadx.indicators.bank import IndicatorBank, IndicatorSettings
        from threadx.strategy import BBAtrStrategy, BollingerDualStrategy

        # Créer IndicatorBank dans ce process worker
        settings = IndicatorSettings(use_gpu=True)
        indicator_bank = IndicatorBank(settings)

        # Mapping stratégie → classe
        strategy_classes = {
            "Bollinger_Breakout": BBAtrStrategy,
            "Bollinger_Dual": BollingerDualStrategy,
        }

        # Fallback to process globals if args are None (ProcessPool with initializer)
        ci = computed_indicators if computed_indicators is not None else _G_INDICATORS
        rd = real_data if real_data is not None else _G_REAL_DATA
        sym = symbol if symbol is not None else _G_SYMBOL
        tf = timeframe if timeframe is not None else _G_TIMEFRAME
        strat_name = combo.get("strategy", (strategy_name or _G_STRATEGY_NAME))
        strategy_class = strategy_classes.get(strat_name, BBAtrStrategy)

        # Créer stratégie avec IndicatorBank local
        strategy = strategy_class(symbol=sym, timeframe=tf, indicator_bank=indicator_bank)

        # Mapping paramètres
        strategy_params = {}
        for key, value in combo.items():
            if key == "bb_window":
                strategy_params["bb_period"] = value
            elif key == "bb_num_std":
                strategy_params["bb_std"] = value
            elif key == "atr_window":
                strategy_params["atr_period"] = value
            elif key == "atr_multiplier":
                strategy_params["atr_multiplier"] = value
            else:
                strategy_params[key] = value

        # Paramètres par défaut requis
        strategy_params.setdefault("risk_per_trade", 0.02)
        strategy_params.setdefault("leverage", 1)
        strategy_params.setdefault("max_hold_bars", 100)
        strategy_params.setdefault("spacing_bars", 5)
        strategy_params.setdefault("min_pnl_pct", 0.02)  # ✅ FIX: doit être positif
        strategy_params.setdefault("entry_z", 2.0)
        strategy_params.setdefault("trailing_stop", False)

        # Backtest
        equity, stats = strategy.backtest(
            df=rd,
            params=strategy_params,
            precomputed_indicators=ci,
        )

        # Résultat
        result = {"params": combo, "stats": stats.__dict__ if hasattr(stats, "__dict__") else {}}

        return result

    except Exception as e:
        logger.error(f"Erreur évaluation combo {combo}: {e}")
        import traceback
        traceback.print_exc()
        return {"params": combo, "stats": {}, "error": str(e)}


def is_global_stop_requested() -> bool:
    """Vérifier si un arrêt global a été demandé."""
    global _GLOBAL_STOP_FLAG
    return bool(_GLOBAL_STOP_FLAG)


def request_global_stop() -> None:
    """Request cancellation for any running optimization."""
    set_global_stop(True)


def clear_global_stop() -> None:
    """Clear the global stop flag."""
    global _GLOBAL_STOP_FLAG
    _GLOBAL_STOP_FLAG = False


class SweepRunner:
    """
    Runner de sweeps paramétriques unifié avec batch processing et early stopping.

    Utilise IndicatorBank pour la mutualisation des calculs d'indicateurs,
    device-agnostic computing via xp, et hooks de performance par stage.
    """

    def __init__(
        self,
        indicator_bank: IndicatorBank | None = None,
        max_workers: int | None = None,
        use_multigpu: bool = True,
        use_processes: bool = False,  # ✅ DEFAULT False: ThreadPool (GPU release GIL = OK)
    ):
        """
        Initialise le runner de sweeps avec support Multi-GPU.

        Args:
            indicator_bank: Instance IndicatorBank pour cache partagé
            max_workers: Nombre de workers (None = auto-détection dynamique)
            use_multigpu: Activer distribution Multi-GPU si disponible
            use_processes: True = ProcessPoolExecutor (vrai parallélisme),
                          False = ThreadPoolExecutor (limité par GIL)
        """
        self.indicator_bank = indicator_bank or IndicatorBank()
        self.logger = get_logger(__name__)
        self.use_processes = use_processes  # ✅ Stocker choix

        # Multi-GPU Manager
        self.use_multigpu = use_multigpu and MULTIGPU_AVAILABLE
        self.gpu_manager = None

        if self.use_multigpu:
            try:
                self.gpu_manager = get_default_manager()
                self.logger.info("✅ Multi-GPU activé")
            except Exception as e:
                self.logger.warning(f"⚠️ Multi-GPU non disponible: {e}")
                self.use_multigpu = False

        # Workers dynamiques
        if max_workers is None:
            self.max_workers = self._calculate_optimal_workers()
            self.logger.info(f"Workers calculés automatiquement: {self.max_workers}")
        else:
            self.max_workers = max_workers
            self.logger.info(f"Workers configurés manuellement: {self.max_workers}")

        # Log mode parallélisme
        mode = "ProcessPool (vrai parallélisme)" if use_processes else "ThreadPool (GIL limité)"
        self.logger.info(f"🔧 Mode parallélisme: {mode}")

        # État d'exécution
        self.is_running = False
        self.should_pause = False
        self.current_scenario = 0
        self.total_scenarios = 0

        # Hooks de performance
        self.stage_timings = {}
        self.start_time = None
        self._last_worker_adjustment = 0

        self.logger.info("🚀 SweepRunner initialisé avec IndicatorBank centralisé")

    def _calculate_optimal_workers(self) -> int:
        """
        Calcule dynamiquement le nombre optimal de workers.

        Basé sur:
        - Nombre de GPUs disponibles
        - VRAM disponible
        - RAM système disponible
        """
        # Base: nombre de CPU cores (physiques)
        if PSUTIL_AVAILABLE:
            base_workers = psutil.cpu_count(logical=False) or 4
        else:
            base_workers = 4

        if self.gpu_manager and self.use_multigpu:
            # Mode Multi-GPU: saturer les GPUs disponibles
            gpu_devices = [
                d for d in self.gpu_manager.available_devices if d.device_id != -1
            ]

            if len(gpu_devices) >= 2:
                # 2 GPUs: 60 workers par GPU = 120 total
                # RTX 5080 (16GB) + RTX 2060 (8GB) peuvent gérer 120 workers (GPU release GIL)
                optimal = len(gpu_devices) * 60
                self.logger.info(f"🚀 Multi-GPU: {len(gpu_devices)} GPUs → {optimal} workers")
            elif len(gpu_devices) == 1:
                # 1 GPU: 20 workers (saturer GPU unique)
                optimal = 20
            else:
                # Pas de GPU: utiliser CPU massivement
                optimal = base_workers * 3
        else:
            # Mode CPU-only: saturer CPU
            optimal = min(base_workers * 4, 32)

        # Vérifier RAM disponible (60 GB total → autoriser jusqu'à 40 workers)
        if PSUTIL_AVAILABLE:
            ram_gb = psutil.virtual_memory().available / (1024**3)

            if ram_gb < 16:
                # RAM limitée: réduire workers
                optimal = min(optimal, 8)
            elif ram_gb < 32:
                # RAM moyenne: limiter à 16
                optimal = min(optimal, 16)
            elif ram_gb >= 40:
                # RAM abondante (60 GB): autoriser 50+ workers
                optimal = min(optimal * 1.5, 50)
                self.logger.info(f"💾 RAM abondante ({ram_gb:.1f} GB) → max {optimal} workers")

        return max(optimal, 2)  # Minimum 2 workers

    def _adjust_workers_dynamically(
        self, current_batch: int, total_batches: int
    ) -> int:
        """
        Ajuste le nombre de workers en temps réel selon:
        - Utilisation GPU actuelle
        - Utilisation RAM
        - Performance observée
        """
        # Ajustement toutes les 5 batchs minimum
        if current_batch - self._last_worker_adjustment < 5:
            return self.max_workers

        if not PSUTIL_AVAILABLE:
            return self.max_workers

        # Monitoring ressources
        ram_used_pct = psutil.virtual_memory().percent

        current_workers = self.max_workers

        if self.gpu_manager and self.use_multigpu:
            try:
                # Récupérer stats GPU
                gpu_stats = self.gpu_manager.get_device_stats()

                # Moyenne utilisation GPU
                gpu_usage_values = [
                    stats.get("memory_used_pct", 0)
                    for stats in gpu_stats.values()
                    if stats.get("device_id", -1) != -1
                ]

                if gpu_usage_values:
                    gpu_usage_avg = np.mean(gpu_usage_values)

                    # Ajustement adaptatif
                    if gpu_usage_avg < 50 and ram_used_pct < 70:
                        # GPU sous-utilisé et RAM OK: augmenter workers
                        new_workers = min(current_workers + 2, 16)
                        self.logger.info(
                            f"↑ Augmentation workers: {current_workers} → {new_workers} (GPU: {gpu_usage_avg:.0f}%, RAM: {ram_used_pct:.0f}%)"
                        )
                        self._last_worker_adjustment = current_batch
                        return new_workers

                    elif gpu_usage_avg > 85 or ram_used_pct > 85:
                        # Saturation: réduire workers
                        new_workers = max(current_workers - 2, 2)
                        self.logger.warning(
                            f"↓ Réduction workers: {current_workers} → {new_workers} (GPU: {gpu_usage_avg:.0f}%, RAM: {ram_used_pct:.0f}%)"
                        )
                        self._last_worker_adjustment = current_batch
                        return new_workers
            except Exception as e:
                self.logger.debug(f"Erreur ajustement workers: {e}")

        return current_workers

    def run_grid(
        self,
        grid_spec: ScenarioSpec,
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy_name: str = "Bollinger_Breakout",
        *,
        reuse_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Exécute un sweep de grille paramétrique avec vraies données.

        Args:
            grid_spec: Spécification de la grille
            real_data: DataFrame OHLCV avec vraies données de marché
            symbol: Symbole tradé (ex: "BTC", "ETH")
            timeframe: Timeframe des données (ex: "1h", "15m")
            strategy_name: Nom de la stratégie à utiliser
            reuse_cache: Réutilise le cache IndicatorBank

        Returns:
            DataFrame des résultats classés

        Raises:
            ValueError: Si real_data est None ou vide
        """
        if real_data is None or real_data.empty:
            raise ValueError("Données OHLCV requises pour run_grid()")

        self.logger.info(
            f"Début sweep grille: {grid_spec} avec {len(real_data)} barres"
        )

        # Génération des combinaisons
        with self._time_stage("scenario_generation"):
            # Extraire les params du ScenarioSpec (dict ou ScenarioSpec)
            params = (
                grid_spec["params"] if isinstance(grid_spec, dict) else grid_spec.params
            )
            combinations = generate_param_grid(params)

        self.total_scenarios = len(combinations)
        self.logger.info(f"Grille générée: {self.total_scenarios} combinaisons")

        # Exécution (bounded feeder pour meilleures perfs mémoire/latence)
        results_df = self._execute_combinations_bounded(
            combinations,
            real_data,
            symbol,
            timeframe,
            strategy_name,
            reuse_cache=reuse_cache,
        )

        return results_df

    def run_monte_carlo(
        self,
        mc_spec: ScenarioSpec,
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy_name: str = "Bollinger_Breakout",
        *,
        reuse_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Exécute un sweep Monte Carlo avec vraies données.

        Args:
            mc_spec: Spécification Monte Carlo
            real_data: DataFrame OHLCV avec vraies données de marché
            symbol: Symbole tradé (ex: "BTC", "ETH")
            timeframe: Timeframe des données (ex: "1h", "15m")
            strategy_name: Nom de la stratégie à utiliser
            reuse_cache: Réutilise le cache IndicatorBank

        Returns:
            DataFrame des résultats avec pruning Pareto

        Raises:
            ValueError: Si real_data est None ou vide
        """
        if real_data is None or real_data.empty:
            raise ValueError("Données OHLCV requises pour run_monte_carlo()")

        self.logger.info(
            f"Début sweep Monte Carlo: {mc_spec} avec {len(real_data)} barres"
        )

        # Génération des scénarios
        with self._time_stage("scenario_generation"):
            # Extraire les params et autres attributs
            params = mc_spec["params"] if isinstance(mc_spec, dict) else mc_spec.params
            n_scenarios = (
                mc_spec.get("n_scenarios", 100)
                if isinstance(mc_spec, dict)
                else mc_spec.n_scenarios
            )
            seed = (
                mc_spec.get("seed", 42) if isinstance(mc_spec, dict) else mc_spec.seed
            )
            scenarios = generate_monte_carlo(params, n_scenarios, seed)

        self.total_scenarios = len(scenarios)
        self.logger.info(f"Monte Carlo généré: {self.total_scenarios} scénarios")

        # Exécution avec pruning adaptatif
        results_df = self._execute_combinations_with_pruning(
            scenarios,
            real_data,
            symbol,
            timeframe,
            strategy_name,
            reuse_cache=reuse_cache,
        )

        return results_df

    def _execute_combinations(
        self,
        combinations: list[dict],
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy_name: str = "Bollinger_Breakout",
        *,
        reuse_cache: bool = True,
    ) -> pd.DataFrame:
        """Exécute les combinaisons en mode batch avec vraies données."""
        self.is_running = True
        self.start_time = time.time()

        results = []

        try:
            # Extraction des indicateurs uniques pour batch processing
            with self._time_stage("indicator_extraction"):
                unique_indicators = self._extract_unique_indicators(combinations)

            # Calcul batch des indicateurs via IndicatorBank
            with self._time_stage("batch_indicators"):
                computed_indicators = self._compute_batch_indicators(
                    unique_indicators,
                    real_data,
                    symbol,
                    timeframe,
                    reuse_cache=reuse_cache,
                )

            # Évaluation PARALLÈLE des stratégies avec ThreadPoolExecutor
            with self._time_stage("strategy_evaluation"):
                completed_count = [0]  # Mutable counter pour tracking progress

                # ✅ Choisir ProcessPool ou ThreadPool selon config
                ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
                with ExecutorClass(max_workers=self.max_workers) as executor:
                    futures = {}
                    batch_size = (
                        1000  # Soumettre par batch pour éviter une queue géante
                    )
                    stop_requested = False

                    # Soumettre les futures par BATCH, en vérifiant le stop entre chaque
                    self.logger.info(
                        f"Début soumission {len(combinations)} combos par batch de {batch_size}"
                    )
                    for batch_idx in range(0, len(combinations), batch_size):
                        # Vérifier le stop AVANT de soumettre chaque batch
                        if self.should_pause or is_global_stop_requested():
                            stop_requested = True
                            self.logger.warning(
                                f"⏹️ Arrêt détecté avant soumission batch {batch_idx // batch_size}"
                            )
                            break

                        # Soumettre le batch
                        batch_end = min(batch_idx + batch_size, len(combinations))
                        for i in range(batch_idx, batch_end):
                            combo = combinations[i]

                            # ✅ Utiliser fonction standalone si ProcessPool, sinon méthode self
                            if self.use_processes:
                                worker_func = _evaluate_combo_worker
                            else:
                                worker_func = self._evaluate_single_combination

                            future = executor.submit(
                                worker_func,
                                combo,
                                computed_indicators,
                                real_data,
                                symbol,
                                timeframe,
                                strategy_name,
                            )
                            futures[future] = i

                        self.logger.debug(
                            f"Batch {batch_idx // batch_size}: {batch_end - batch_idx} futures soumises (total: {len(futures)})"
                        )

                    # Collecter les résultats au fur et à mesure
                    try:
                        for future in as_completed(futures):
                            # Vérifier le stop régulièrement
                            if self.should_pause or is_global_stop_requested():
                                if not stop_requested:
                                    self.logger.warning(
                                        f"⏹️ Arrêt demandé après {completed_count[0]} combos terminés"
                                    )
                                    stop_requested = True

                                # Annuler les futures restantes en queue
                                cancelled_count = 0
                                for f in futures:
                                    if f.cancel():
                                        cancelled_count += 1

                                if cancelled_count > 0:
                                    self.logger.warning(
                                        f"⏹️ {cancelled_count} futures annulées en queue"
                                    )
                                break

                            try:
                                result = future.result()
                                results.append(result)
                                completed_count[0] += 1

                                # Mise à jour du compteur de progression (thread-safe via GIL)
                                self.current_scenario = completed_count[0]

                                # Log de progression tous les 100 combos
                                if completed_count[0] % 100 == 0:
                                    self._log_progress()

                            except Exception as e:
                                self.logger.error(f"Erreur exécution combo: {e}")
                                # Continuer avec les autres combos
                                completed_count[0] += 1
                                results.append(
                                    {
                                        "error": str(e),
                                        "pnl": 0.0,
                                        "pnl_pct": 0.0,
                                        "sharpe": 0.0,
                                        "max_drawdown": 0.0,
                                        "win_rate": 0.0,
                                        "total_trades": 0,
                                    }
                                )
                    finally:
                        # Cleanup: cancel any remaining futures
                        remaining = sum(1 for f in futures if not f.done())
                        if remaining > 0:
                            for f in futures:
                                f.cancel()
                            self.logger.info(
                                f"Cleanup: {remaining} futures restantes annulées"
                            )

            # Construction du DataFrame final
            with self._time_stage("results_compilation"):
                results_df = pd.DataFrame(results)

        finally:
            self.is_running = False
            clear_global_stop()
            self._log_final_stats()

        return results_df

    def _execute_combinations_bounded(
        self,
        combinations: list[dict],
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy_name: str = "Bollinger_Breakout",
        *,
        reuse_cache: bool = True,
    ) -> pd.DataFrame:
        """Exécute les combinaisons avec un flux de tâches borné (feed/drain).

        Objectif: limiter la file de futures et la sérialisation, améliorer réactivité.
        """
        self.is_running = True
        self.start_time = time.time()

        results: list[dict] = []

        try:
            with self._time_stage("indicator_extraction"):
                unique_indicators = self._extract_unique_indicators(combinations)

            with self._time_stage("batch_indicators"):
                computed_indicators = self._compute_batch_indicators(
                    unique_indicators,
                    real_data,
                    symbol,
                    timeframe,
                    reuse_cache=reuse_cache,
                )

            with self._time_stage("strategy_evaluation"):
                completed_count = [0]
                ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

                inflight_limit = max(int(self.max_workers) * 4, 64)
                exec_kwargs = {}
                if self.use_processes:
                    exec_kwargs["initializer"] = _init_process_globals
                    exec_kwargs["initargs"] = (
                        computed_indicators,
                        real_data,
                        symbol,
                        timeframe,
                        strategy_name,
                    )

                with ExecutorClass(max_workers=self.max_workers, **exec_kwargs) as executor:
                    futures = {}
                    stop_requested = False

                    def submit_one(idx: int) -> None:
                        combo = combinations[idx]
                        if self.use_processes:
                            fut = executor.submit(
                                _evaluate_combo_worker, combo, None, None, None, None, strategy_name
                            )
                        else:
                            fut = executor.submit(
                                self._evaluate_single_combination,
                                combo,
                                computed_indicators,
                                real_data,
                                symbol,
                                timeframe,
                                strategy_name,
                            )
                        futures[fut] = idx

                    next_index = 0
                    prefill = min(inflight_limit, len(combinations))
                    for i in range(prefill):
                        submit_one(i)
                    next_index = prefill

                    while futures:
                        if self.should_pause or is_global_stop_requested():
                            stop_requested = True

                        try:
                            for fut in as_completed(list(futures.keys()), timeout=0.5):
                                futures.pop(fut, None)
                                try:
                                    result = fut.result()
                                except Exception as e:
                                    self.logger.error(f"Erreur exécution combo: {e}")
                                    completed_count[0] += 1
                                    results.append(
                                        {
                                            "error": str(e),
                                            "pnl": 0.0,
                                            "pnl_pct": 0.0,
                                            "sharpe": 0.0,
                                            "max_drawdown": 0.0,
                                            "win_rate": 0.0,
                                            "total_trades": 0,
                                        }
                                    )
                                else:
                                    results.append(result)
                                    completed_count[0] += 1
                                    self.current_scenario = completed_count[0]
                                    if completed_count[0] % 100 == 0:
                                        self._log_progress()
                        except FuturesTimeout:
                            pass

                        while (
                            not stop_requested
                            and next_index < len(combinations)
                            and len(futures) < inflight_limit
                        ):
                            submit_one(next_index)
                            next_index += 1

                    if stop_requested:
                        self.logger.warning(
                            f"⏹️ Arrêt demandé après {completed_count[0]} combos terminés"
                        )

            with self._time_stage("results_compilation"):
                results_df = pd.DataFrame(results)

        finally:
            self.is_running = False
            clear_global_stop()
            self._log_final_stats()

        return results_df

    def _execute_combinations_with_pruning(
        self,
        combinations: list[dict],
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy_name: str = "Bollinger_Breakout",
        *,
        reuse_cache: bool = True,
    ) -> pd.DataFrame:
        """Exécute avec pruning Pareto adaptatif et vraies données."""
        self.is_running = True
        self.start_time = time.time()

        results = []
        pruning_metadata = None

        try:
            # Extraction et calcul batch des indicateurs
            with self._time_stage("indicator_extraction"):
                unique_indicators = self._extract_unique_indicators(combinations)

            with self._time_stage("batch_indicators"):
                computed_indicators = self._compute_batch_indicators(
                    unique_indicators,
                    real_data,
                    symbol,
                    timeframe,
                    reuse_cache=reuse_cache,
                )

            # Évaluation avec pruning progressif
            with self._time_stage("strategy_evaluation_pruning"):
                batch_size = 50  # Taille de batch pour pruning

                for batch_start in range(0, len(combinations), batch_size):
                    if self.should_pause or is_global_stop_requested():
                        break

                    batch_end = min(batch_start + batch_size, len(combinations))
                    batch_combos = combinations[batch_start:batch_end]

                    # Évaluation du batch
                    batch_results = []
                    for combo in batch_combos:
                        result = self._evaluate_single_combination(
                            combo,
                            computed_indicators,
                            real_data,
                            symbol,
                            timeframe,
                            strategy_name,
                        )
                        batch_results.append(result)

                    results.extend(batch_results)
                    self.current_scenario = len(results)

                    # Pruning périodique
                    if len(results) >= 200:  # Seuil minimum pour pruning
                        current_df = pd.DataFrame(results)
                        pruned_df, pruning_metadata = pareto_soft_prune(
                            current_df, patience=200, quantile=0.85
                        )

                        if pruning_metadata["early_stop_triggered"]:
                            self.logger.info(
                                "Early stopping déclenché par pruning Pareto"
                            )
                            break

                    self._log_progress()

            # Construction finale
            with self._time_stage("results_compilation"):
                results_df = pd.DataFrame(results)

                # Pruning final si pas encore fait
                if len(results_df) > 100:
                    results_df, final_pruning = pareto_soft_prune(results_df)
                    if not pruning_metadata:
                        pruning_metadata = final_pruning

        finally:
            self.is_running = False
            clear_global_stop()
            self._log_final_stats(pruning_metadata)

        return results_df

    def _extract_unique_indicators(
        self, combinations: list[dict]
    ) -> dict[str, list[dict]]:
        """Extrait les indicateurs uniques pour batch processing."""
        indicators_by_type = {}

        for combo in combinations:
            # Parsing des paramètres par type d'indicateur
            for param_name, param_value in combo.items():
                if param_name.startswith("bb_"):
                    # Bollinger Bands
                    if "bollinger" not in indicators_by_type:
                        indicators_by_type["bollinger"] = []

                    # ✅ MAPPING: bb_window→period, bb_num_std→std
                    bb_params = {}
                    for name, value in combo.items():
                        if name == "bb_window":
                            bb_params["period"] = value
                        elif name == "bb_num_std":
                            bb_params["std"] = value

                    if bb_params and bb_params not in indicators_by_type["bollinger"]:
                        indicators_by_type["bollinger"].append(bb_params)

                elif param_name.startswith("atr_"):
                    # ATR
                    if "atr" not in indicators_by_type:
                        indicators_by_type["atr"] = []

                    # ✅ MAPPING: atr_window→period, atr_method→method (défaut: ema)
                    atr_params = {}
                    for name, value in combo.items():
                        if name == "atr_window":
                            atr_params["period"] = value
                        elif name == "atr_method":
                            atr_params["method"] = value
                        # atr_multiplier est pour la STRATÉGIE, pas l'indicateur

                    # ATR par défaut utilise EMA
                    if "method" not in atr_params:
                        atr_params["method"] = "ema"

                    if atr_params and atr_params not in indicators_by_type["atr"]:
                        indicators_by_type["atr"].append(atr_params)

        return indicators_by_type

    def _normalize_indicator_key(self, indicator_type: str, params: dict) -> str:
        """
        Génère clé normalisée IDENTIQUE à celle utilisée par _ensure_indicators.

        CRITICAL: Les clés doivent matcher exactement pour éviter recalcul !

        Args:
            indicator_type: "bollinger" ou "atr"
            params: Paramètres de l'indicateur

        Returns:
            Clé JSON normalisée (même format que strategy/bb_atr.py:509)
        """
        import json

        if indicator_type == "bollinger":
            # Format: {"period": XX, "std": X.X}
            normalized = {
                "period": params.get("period", 20),
                "std": params.get("std", 2.0)
            }
        elif indicator_type == "atr":
            # Format: {"period": XX}
            normalized = {
                "period": params.get("period", 14)
            }
        else:
            # Autres indicateurs: format générique
            normalized = params

        return json.dumps(normalized, sort_keys=True)

    def _compute_batch_indicators(
        self,
        unique_indicators: dict[str, list[dict]],
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        *,
        reuse_cache: bool = True,
    ) -> dict[str, dict]:
        """Calcule les indicateurs en batch avec VRAIES données."""
        computed = {}

        # ✅ Validation données réelles
        if real_data is None or real_data.empty:
            raise ValueError("Données OHLCV requises pour le sweep")

        close_data = real_data["close"]
        self.logger.info(f"Calcul batch indicateurs avec {len(real_data)} barres")

        for indicator_type, params_list in unique_indicators.items():
            computed[indicator_type] = {}

            if reuse_cache:
                # ✅ Utilisation du batch_ensure avec vraies données
                batch_results = self.indicator_bank.batch_ensure(
                    indicator_type,
                    params_list,
                    close_data,
                    symbol=symbol,
                    timeframe=timeframe,
                )

                # ✅ NORMALISATION: Utiliser MÊME format de clés que _ensure_indicators
                # Ceci évite KeyError → Fallback recalcul → Overhead 16x !!
                for params in params_list:
                    # Générer clé normalisée (IDENTIQUE à strategy/bb_atr.py:509)
                    normalized_key = self._normalize_indicator_key(indicator_type, params)

                    # Récupérer depuis batch_results (mapping interne IndicatorBank)
                    internal_key = self._params_to_key(params)

                    if internal_key in batch_results:
                        computed[indicator_type][normalized_key] = batch_results[internal_key]
                        self.logger.debug(f"✅ Mapped {indicator_type} {params} → key {normalized_key}")
                    else:
                        self.logger.warning(f"⚠️ Missing batch result for {indicator_type} {params}")

            else:
                # Calcul direct sans cache
                for params in params_list:
                    normalized_key = self._normalize_indicator_key(indicator_type, params)
                    result = self.indicator_bank.ensure(
                        indicator_type,
                        params,
                        close_data,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    computed[indicator_type][normalized_key] = result

        return computed

    def _evaluate_single_combination(
        self,
        combo: dict,
        computed_indicators: dict,
        real_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy_name: str = "Bollinger_Breakout",
    ) -> dict:
        """
        Évaluation avec VRAI backtest de stratégie.

        Utilise les vraies stratégies implementées dans threadx.strategy
        au lieu de simuler les résultats.
        """
        try:
            # ✅ Import des stratégies disponibles
            from threadx.strategy import BBAtrStrategy, BollingerDualStrategy

            # Mapping stratégie → classe
            strategy_classes = {
                "Bollinger_Breakout": BBAtrStrategy,
                "Bollinger_Dual": BollingerDualStrategy,
            }

            # Déterminer quelle stratégie utiliser
            # Priorité: param "strategy" dans combo, sinon stratégie par défaut
            strat_name = combo.get("strategy", strategy_name)

            strategy_class = strategy_classes.get(strat_name, BBAtrStrategy)

            # 🚀 OPTIMISATION CRITIQUE: Réutiliser instance existante si disponible
            # Évite de recréer GPU Manager, Bollinger, ATR, IndicatorBank pour chaque combo
            if not hasattr(self, "_cached_strategy_instances"):
                self._cached_strategy_instances = {}

            cache_key = (strat_name, symbol, timeframe)
            if cache_key not in self._cached_strategy_instances:
                # ✅ P0.2: INJECTER SINGLETON IndicatorBank (élimine recréation GPU Manager 16x)
                self._cached_strategy_instances[cache_key] = strategy_class(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_bank=self.indicator_bank  # ← Singleton partagé !
                )

            strategy = self._cached_strategy_instances[cache_key]

            # ✅ MAPPING: Transformer paramètres sweep → paramètres stratégie
            strategy_params = {}
            for key, value in combo.items():
                if key == "bb_window":
                    strategy_params["bb_period"] = value
                elif key == "bb_num_std":
                    strategy_params["bb_std"] = value
                elif key == "atr_window":
                    strategy_params["atr_period"] = value
                elif key == "atr_multiplier":
                    strategy_params["atr_multiplier"] = value
                else:
                    # Autres paramètres passent tels quels
                    strategy_params[key] = value

            # Paramètres par défaut requis
            if "entry_z" not in strategy_params:
                strategy_params["entry_z"] = 1.0  # Valeur par défaut

            # ✅ VRAI backtest avec vraies données + indicateurs pré-calculés
            equity_curve, run_stats = strategy.backtest(
                df=real_data,
                params=strategy_params,  # Utiliser les paramètres mappés
                initial_capital=10000.0,
                fee_bps=4.5,
                slippage_bps=0.0,
                precomputed_indicators=computed_indicators,  # 🚀 OPTIMISATION: Réutiliser batch indicators
            )

            # Retourner métriques réelles
            result = combo.copy()
            result.update(
                {
                    "pnl": run_stats.total_pnl,
                    "pnl_pct": run_stats.total_pnl_pct,
                    "sharpe": (
                        run_stats.sharpe_ratio
                        if hasattr(run_stats, "sharpe_ratio")
                        else 0.0
                    ),
                    "max_drawdown": (
                        run_stats.max_drawdown
                        if hasattr(run_stats, "max_drawdown")
                        else 0.0
                    ),
                    "win_rate": (
                        run_stats.win_rate if hasattr(run_stats, "win_rate") else 0.0
                    ),
                    "total_trades": run_stats.total_trades,
                }
            )

            return result

        except Exception as e:
            # Fallback en cas d'erreur: retourner résultats neutres
            self.logger.error(f"Erreur évaluation combo {combo}: {e}")
            result = combo.copy()
            result.update(
                {
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "error": str(e),
                }
            )
            return result

    def _params_to_key(self, params: dict) -> str:
        """Convertit des paramètres en clé de cache."""
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    def _time_stage(self, stage_name: str):
        """Context manager pour mesurer le temps par stage."""

        class StageTimer:
            def __init__(self, runner, name):
                self.runner = runner
                self.name = name
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                if self.name not in self.runner.stage_timings:
                    self.runner.stage_timings[self.name] = []
                self.runner.stage_timings[self.name].append(duration)

        return StageTimer(self, stage_name)

    def _log_progress(self):
        """Log du progrès d'exécution."""
        if self.total_scenarios > 0:
            progress = self.current_scenario / self.total_scenarios
            elapsed = time.time() - self.start_time if self.start_time else 0
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0

            self.logger.info(
                f"Progrès: {self.current_scenario}/{self.total_scenarios} "
                f"({progress:.1%}) - ETA: {eta:.1f}s"
            )

    def _log_final_stats(self, pruning_metadata: dict | None = None):
        """Log des statistiques finales."""
        total_time = time.time() - self.start_time if self.start_time else 0

        self.logger.info(
            f"Sweep terminé: {self.current_scenario} scénarios " f"en {total_time:.1f}s"
        )

        # Statistiques par stage
        for stage, timings in self.stage_timings.items():
            avg_time = np.mean(timings)
            total_stage_time = np.sum(timings)
            self.logger.info(
                f"  {stage}: {total_stage_time:.2f}s "
                f"(avg: {avg_time:.3f}s, {len(timings)} appels)"
            )

        # Métadonnées de pruning
        if pruning_metadata:
            self.logger.info(
                f"Pruning Pareto: {pruning_metadata['pruned_count']} configurations éliminées "
                f"({pruning_metadata['pruning_rate']:.1%})"
            )


class UnifiedOptimizationEngine:
    """
    Moteur d'optimisation unifié utilisant IndicatorBank comme seul moteur de calcul.

    Centralise tous les calculs via IndicatorBank pour éviter la duplication de code
    et garantir la cohérence entre UI, optimisation et backtesting.
    """

    def __init__(
        self, indicator_bank: IndicatorBank | None = None, max_workers: int = 4
    ):
        """
        Initialise le moteur d'optimisation unifié.

        Args:
            indicator_bank: Instance IndicatorBank existante (recommandé)
            max_workers: Nombre de workers pour le parallélisme
        """
        # Utilise l'IndicatorBank existant ou en crée un nouveau
        self.indicator_bank = indicator_bank or IndicatorBank()
        self.max_workers = max_workers
        self.logger = get_logger(__name__)

        # État d'exécution
        self.is_running = False
        self.should_pause = False
        self.progress_callback: Callable | None = None

        # Métriques
        self.total_combos = 0
        self.completed_combos = 0
        self.start_time: float | None = None

        self.logger.info(
            "🚀 UnifiedOptimizationEngine initialisé avec IndicatorBank centralisé"
        )

    def run_parameter_sweep(self, config: dict, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute un sweep de paramètres en utilisant uniquement IndicatorBank.

        Args:
            config: Configuration de sweep
            data: Données OHLCV source

        Returns:
            DataFrame des résultats classés
        """
        self.is_running = True
        self.start_time = time.time()

        try:
            # 1. Expansion de la grille de paramètres
            combinations = self._expand_parameter_grid(config.get("grid", {}))
            self.total_combos = len(combinations)
            self.completed_combos = 0

            self.logger.info(f"Démarrage sweep: {self.total_combos} combinaisons")

            # 2. Exécution parallèle via IndicatorBank
            results = []

            # ✅ Choisir ProcessPool ou ThreadPool selon config
            ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            with ExecutorClass(max_workers=self.max_workers) as executor:
                futures = {}
                batch_size = 1000
                stop_requested = False

                # Soumission des tâches par BATCH
                self.logger.info(
                    f"Soumission {len(combinations)} combos par batch de {batch_size}"
                )
                for batch_idx in range(0, len(combinations), batch_size):
                    if self.should_pause or is_global_stop_requested():
                        stop_requested = True
                        self.logger.warning(
                            f"⏹️ Arrêt avant batch {batch_idx // batch_size}"
                        )
                        break

                    batch_end = min(batch_idx + batch_size, len(combinations))
                    for i in range(batch_idx, batch_end):
                        combo = combinations[i]
                        future = executor.submit(
                            self._execute_single_combination, data, combo, config
                        )
                        futures[future] = i

                    self.logger.debug(
                        f"Batch: {batch_end - batch_idx} soumises (total: {len(futures)})"
                    )

                # Collecte des résultats
                try:
                    for future in as_completed(futures):
                        if self.should_pause or is_global_stop_requested():
                            if not stop_requested:
                                self.logger.warning(
                                    f"⏹️ Arrêt après {self.completed_combos} combos"
                                )
                                stop_requested = True

                            # Annuler les futures en queue
                            cancelled = sum(1 for f in futures if f.cancel())
                            if cancelled > 0:
                                self.logger.warning(f"⏹️ {cancelled} futures annulées")
                            break

                        try:
                            result = future.result()
                            results.append(result)
                            self.completed_combos += 1
                            self._update_progress()

                        except Exception as e:
                            self.logger.error(f"Erreur dans une combinaison: {e}")
                            self.completed_combos += 1
                            self._update_progress()
                finally:
                    # Cleanup
                    remaining = sum(1 for f in futures if not f.done())
                    if remaining > 0:
                        for f in futures:
                            f.cancel()
                        self.logger.info(f"Cleanup: {remaining} futures annulées")

            # 3. Classement des résultats
            if results:
                df = pd.DataFrame(results)
                df = self._score_and_rank_results(df, config.get("scoring", {}))

                # Statistiques finales
                duration = time.time() - self.start_time
                rate = self.completed_combos / duration if duration > 0 else 0

                self.logger.info(
                    f"Sweep terminé: {self.completed_combos}/{self.total_combos}"
                )
                self.logger.info(
                    f"Cache hits IndicatorBank: {self.indicator_bank.stats.get('cache_hits', 0)}"
                )
                self.logger.info(f"Vitesse: {rate:.1f} combinaisons/sec")

                return df
            else:
                self.logger.warning("Aucun résultat obtenu")
                return pd.DataFrame()

        finally:
            self.is_running = False
            clear_global_stop()

    def _expand_parameter_grid(self, grid_config: dict) -> list[dict]:
        """Expanse la configuration de grille en combinaisons."""
        all_combinations = []

        for indicator_type, params_config in grid_config.items():
            if indicator_type not in ["bollinger", "atr"]:
                self.logger.warning(f"Type indicateur non supporté: {indicator_type}")
                continue

            # Génération des valeurs pour chaque paramètre
            param_values = {}
            for param_name, param_def in params_config.items():
                if isinstance(param_def, dict) and "start" in param_def:
                    # Plage avec start/stop/step
                    values = self._generate_range(
                        param_def["start"], param_def["stop"], param_def["step"]
                    )
                elif isinstance(param_def, list):
                    # Liste de valeurs discrètes
                    values = param_def
                else:
                    # Valeur unique
                    values = [param_def]

                param_values[param_name] = values

            # Produit cartésien pour cet indicateur
            param_names = list(param_values.keys())
            for combo in itertools.product(*param_values.values()):
                combination = {
                    "indicator_type": indicator_type,
                    "params": dict(zip(param_names, combo)),
                }
                all_combinations.append(combination)

        # Déduplication
        unique_combinations = []
        seen = set()

        for combo in all_combinations:
            key = self._make_combination_key(combo)
            if key not in seen:
                seen.add(key)
                unique_combinations.append(combo)

        self.logger.info(
            f"Grille expansée: {len(all_combinations)} → {len(unique_combinations)} uniques"
        )
        return unique_combinations

    def _generate_range(self, start: float, stop: float, step: float) -> list[float]:
        """Génère une plage de valeurs avec gestion des flottants."""
        values = []
        current = start
        while current <= stop + 1e-10:  # Tolérance pour les erreurs de flottants
            values.append(round(current, 6))
            current += step
        return values

    def _make_combination_key(self, combo: dict) -> str:
        """Crée une clé unique pour une combinaison."""
        key_data = {
            "type": combo["indicator_type"],
            "params": sorted(combo["params"].items()),
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()

    def _execute_single_combination(
        self, data: pd.DataFrame, combo: dict, config: dict
    ) -> dict:
        """
        Exécute une combinaison de paramètres via IndicatorBank.

        Toute la logique de calcul passe par IndicatorBank pour centralisation.
        """
        start_time = time.time()

        try:
            # 1. Calcul de l'indicateur via IndicatorBank
            indicator_result = self.indicator_bank.ensure(
                indicator_type=combo["indicator_type"],
                params=combo["params"],
                data=data,
                symbol=config.get("dataset", {}).get("symbol", ""),
                timeframe=config.get("dataset", {}).get("timeframe", ""),
            )

            # 2. Génération de signaux basiques (mock pour démonstration)
            signals = self._generate_signals_from_indicator(
                data, indicator_result, combo
            )

            # 3. Calcul des métriques de performance
            metrics = self._calculate_performance_metrics(data, signals)

            # 4. Construction du résultat
            result = {
                "indicator_type": combo["indicator_type"],
                **combo["params"],
                "pnl": metrics["total_pnl"],
                "sharpe": metrics["sharpe_ratio"],
                "max_drawdown": abs(metrics["max_drawdown"]),
                "profit_factor": metrics["profit_factor"],
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "duration_sec": time.time() - start_time,
            }

            return result

        except Exception as e:
            self.logger.error(f"Erreur combinaison {combo}: {e}")
            # Résultat par défaut pour éviter de faire planter le sweep
            return {
                "indicator_type": combo["indicator_type"],
                **combo["params"],
                "pnl": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 1.0,
                "profit_factor": 1.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "duration_sec": time.time() - start_time,
                "error": str(e),
            }

    def _generate_signals_from_indicator(
        self, data: pd.DataFrame, indicator_result: Any, combo: dict
    ) -> pd.Series:
        """Génère des signaux de trading à partir des résultats d'indicateurs."""
        # Stratégie simple pour démonstration
        # En production, ceci serait dans un module strategy séparé

        if combo["indicator_type"] == "bollinger":
            # Signaux Bollinger: achat sur bande basse, vente sur bande haute
            if isinstance(indicator_result, tuple) and len(indicator_result) >= 3:
                upper, middle, lower = indicator_result[:3]
                price = data["close"]

                # Génération de signaux basiques
                signals = pd.Series(0, index=data.index)
                signals[price <= lower] = 1  # Long signal
                signals[price >= upper] = -1  # Short signal

                return signals

        elif combo["indicator_type"] == "atr":
            # Signaux ATR: volatility breakout
            if isinstance(indicator_result, np.ndarray):
                atr_values = pd.Series(indicator_result, index=data.index)
                price = data["close"]

                # Signaux basés sur les breakouts ATR
                signals = pd.Series(0, index=data.index)
                price_change = price.pct_change()
                atr_threshold = atr_values / price  # ATR en pourcentage

                signals[price_change > atr_threshold] = 1  # Long breakout
                signals[price_change < -atr_threshold] = -1  # Short breakout

                return signals

        # Signaux par défaut (random pour test)
        np.random.seed(42)
        return pd.Series(
            np.random.choice([0, 1, -1], size=len(data), p=[0.8, 0.1, 0.1]),
            index=data.index,
        )

    def _calculate_performance_metrics(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> dict:
        """Calcule les métriques de performance à partir des signaux."""
        # Calcul simple des returns basé sur les signaux
        price_returns = data["close"].pct_change()
        strategy_returns = (
            signals.shift(1) * price_returns
        )  # Décalage pour éviter look-ahead

        # Nettoyage
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) == 0:
            return self._empty_metrics()

        # Métriques de base
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (
            cumulative_returns.iloc[-1] - 1.0 if len(cumulative_returns) > 0 else 0.0
        )
        total_return = float(total_return)

        if len(strategy_returns) > 1:
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe = (
                (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
                if strategy_returns.std() > 0
                else 0
            )
        else:
            volatility = 0
            sharpe = 0

        # Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # Trades (approximation basée sur les changements de signal)
        signal_changes = signals.diff().abs()
        total_trades = signal_changes[signal_changes > 0].count()

        # Win rate (approximation)
        positive_returns = strategy_returns[strategy_returns > 0]
        win_rate = (
            len(positive_returns) / len(strategy_returns)
            if len(strategy_returns) > 0
            else 0
        )

        # Profit factor
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        profit_factor = (
            abs(wins.sum() / losses.sum())
            if len(losses) > 0 and losses.sum() != 0
            else 1.0
        )

        return {
            "total_pnl": total_return * 10000,  # En dollars sur base 10k
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "total_trades": int(total_trades),
            "win_rate": win_rate,
            "volatility": volatility,
        }

    def _empty_metrics(self) -> dict:
        """Métriques par défaut quand aucune donnée."""
        return {
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 1.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "volatility": 0.0,
        }

    def _score_and_rank_results(
        self, df: pd.DataFrame, scoring_config: dict
    ) -> pd.DataFrame:
        """Score et classe les résultats selon les critères."""
        if df.empty:
            return df

        # Tri par défaut: PnL descendant, puis Sharpe descendant, puis MaxDD ascendant
        primary = scoring_config.get("primary", "pnl")
        secondary = scoring_config.get("secondary", ["sharpe", "-max_drawdown"])

        sort_columns = [primary] + secondary
        ascending_flags = []

        for col in sort_columns:
            if col.startswith("-"):
                ascending_flags.append(True)  # Tri ascendant (pour MaxDD)
                col = col[1:]
            else:
                ascending_flags.append(False)  # Tri descendant (pour PnL, Sharpe)

        # Nettoyage des noms de colonnes
        clean_columns = [col.lstrip("-") for col in sort_columns]
        available_columns = [col for col in clean_columns if col in df.columns]

        if available_columns:
            corresponding_flags = ascending_flags[: len(available_columns)]
            df = df.sort_values(by=available_columns, ascending=corresponding_flags)

        # Ajout du rang
        df = df.reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        # Top-K si spécifié
        top_k = scoring_config.get("top_k")
        if top_k and len(df) > top_k:
            df = df.head(top_k)

        return df

    def _update_progress(self):
        """Met à jour la progression."""
        if self.progress_callback and self.total_combos > 0:
            progress = self.completed_combos / self.total_combos
            eta = None

            if self.start_time and self.completed_combos > 0:
                elapsed = time.time() - self.start_time
                rate = self.completed_combos / elapsed
                remaining = self.total_combos - self.completed_combos
                eta = remaining / rate if rate > 0 else None

            self.progress_callback(
                progress, self.completed_combos, self.total_combos, eta
            )

    # Méthodes de contrôle
    def pause(self):
        """Met en pause l'optimisation."""
        self.should_pause = True

    def resume(self):
        """Reprend l'optimisation."""
        self.should_pause = False

    def stop(self):
        """Arrête l'optimisation."""
        self.should_pause = True
        self.is_running = False

    # Accès aux statistiques du moteur principal
    def get_indicator_bank_stats(self) -> dict:
        """Retourne les statistiques de l'IndicatorBank."""
        return self.indicator_bank.stats.copy()

    def cleanup_cache(self) -> int:
        """Nettoie le cache de l'IndicatorBank."""
        return self.indicator_bank.cache_manager.cleanup_expired()


def create_unified_engine(
    indicator_bank: IndicatorBank | None = None,
) -> UnifiedOptimizationEngine:
    """
    Factory function pour créer un moteur d'optimisation unifié.

    Args:
        indicator_bank: Instance IndicatorBank existante (recommandé pour partage de cache)

    Returns:
        UnifiedOptimizationEngine configuré
    """
    return UnifiedOptimizationEngine(indicator_bank=indicator_bank)


# Configuration d'exemple pour les sweeps
DEFAULT_SWEEP_CONFIG = {
    "dataset": {
        "symbol": "BTCUSDC",
        "timeframe": "1h",
        "start": "2024-01-01",
        "end": "2024-12-31",
    },
    "grid": {
        "bollinger": {
            "period": [10, 15, 20, 25, 30],
            "std": {"start": 1.5, "stop": 3.0, "step": 0.1},
        },
        "atr": {
            "period": {"start": 10, "stop": 30, "step": 2},
            "method": ["ema", "sma"],
        },
    },
    "scoring": {
        "primary": "pnl",
        "secondary": ["sharpe", "-max_drawdown"],
        "top_k": 50,
    },
}
