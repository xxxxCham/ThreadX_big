"""
Parallel Sweep Manager for ThreadX
==================================

Orchestrates parametric sweep execution with adaptive parallelization.

Goals
- Probe different parallel configs (threads vs processes, worker counts).
- Pick the most efficient setup for the full sweep.
- Reuse IndicatorBank cache to avoid recomputation.
- Optionally report basic CPU/GPU/RAM usage during probes.

This module can be used as a standalone script or imported from other
components (e.g., the Streamlit UI) to determine an optimal configuration
before launching a large sweep.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    import psutil  # type: ignore

    HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional
    HAS_PSUTIL = False

from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import generate_param_grid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:  # GPU monitoring (optional)
    from threadx.gpu.multi_gpu import get_default_manager  # type: ignore

    def _gpu_stats_snapshot() -> dict[str, Any]:
        mgr = get_default_manager()
        try:
            return mgr.get_device_stats() or {}
        except Exception:
            return {}

    HAS_GPU_MON = True
except Exception:  # pragma: no cover - optional
    HAS_GPU_MON = False


# ------------------------------ Data models ------------------------------ #


@dataclass
class ProbeConfig:
    use_processes: bool
    max_workers: int


@dataclass
class ProbeResult:
    config: ProbeConfig
    n_cases: int
    elapsed: float
    throughput: float  # cases/sec
    cpu_pct_avg: float | None
    ram_used_gb: float | None
    gpu_util_avg: float | None


# ------------------------------ Utilities ------------------------------- #


def _safe_cpu_percent(interval: float = 0.0) -> float | None:
    if not HAS_PSUTIL:
        return None
    try:
        return float(psutil.cpu_percent(interval=interval))
    except Exception:
        return None


def _safe_ram_gb_used() -> float | None:
    if not HAS_PSUTIL:
        return None
    try:
        v = psutil.virtual_memory()
        return float((v.total - v.available) / (1024**3))
    except Exception:
        return None


def _gpu_utilization_avg(stats: dict[str, Any]) -> float | None:
    if not stats:
        return None
    vals: list[float] = []
    for s in stats.values():
        # Prefer memory_used_pct; adjust here if a utilization percent exists.
        util = s.get("memory_used_pct")
        if isinstance(util, (int, float)):
            vals.append(float(util))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _split_evenly(items: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    if parts <= 1:
        return [items]
    n = len(items)
    chunk = math.ceil(n / parts)
    return [items[i : i + chunk] for i in range(0, n, chunk)]


# --------------------------- Core probe routine -------------------------- #


def _run_probe(
    *,
    combos: list[dict[str, Any]],
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    use_processes: bool,
    max_workers: int,
    reuse_cache: bool,
    indicator_bank: IndicatorBank,
) -> ProbeResult:
    probe_cases = min(len(combos), max(8, min(64, len(combos))))
    sample = combos[:probe_cases]

    cpu_start = _safe_cpu_percent(0.0)
    ram_start = _safe_ram_gb_used()
    _gpu_start = _gpu_stats_snapshot() if HAS_GPU_MON else {}  # Reserved for future use

    runner = SweepRunner(
        indicator_bank=indicator_bank,
        max_workers=max_workers,
        use_multigpu=True,
        use_processes=use_processes,
    )

    t0 = time.time()

    # Use the engine's internal bounded executor for minimal queue overhead
    try:
        results_df = runner._execute_combinations_bounded(  # type: ignore[attr-defined]
            sample,
            data,
            symbol,
            timeframe,
            strategy_name,
            reuse_cache=reuse_cache,
        )
    except AttributeError:
        # Fallback to legacy implementation if bounded feeder is missing
        results_df = runner._execute_combinations(  # type: ignore[attr-defined]
            sample,
            data,
            symbol,
            timeframe,
            strategy_name,
            reuse_cache=reuse_cache,
        )

    elapsed = time.time() - t0
    n_cases = len(results_df) if isinstance(results_df, pd.DataFrame) else len(sample)
    throughput = float(n_cases / elapsed) if elapsed > 0 else 0.0

    # Lightweight snapshots after probe
    cpu_end = _safe_cpu_percent(0.0)
    ram_end = _safe_ram_gb_used()
    gpu_end = _gpu_stats_snapshot() if HAS_GPU_MON else {}

    cpu_avg = None
    if cpu_start is not None and cpu_end is not None:
        cpu_avg = (cpu_start + cpu_end) / 2.0
    ram_used_gb = None
    if ram_start is not None and ram_end is not None:
        ram_used_gb = max(0.0, ram_end - ram_start)
    gpu_avg = _gpu_utilization_avg(gpu_end) if HAS_GPU_MON else None

    return ProbeResult(
        config=ProbeConfig(use_processes=use_processes, max_workers=max_workers),
        n_cases=n_cases,
        elapsed=elapsed,
        throughput=throughput,
        cpu_pct_avg=cpu_avg,
        ram_used_gb=ram_used_gb,
        gpu_util_avg=gpu_avg,
    )


def _heuristic_candidates(
    *,
    cpu_cores: int,
    gpu_count: int,
    baseline: int | None = None,
) -> list[int]:
    """Generate a small candidate set for worker counts.

    Keep this modest to probe quickly (3–5 candidates typically).
    """
    candidates: set[int] = set()
    if baseline:
        candidates.add(max(1, baseline))
    # CPU-oriented
    candidates.update({max(1, cpu_cores // 2), max(1, cpu_cores), max(1, cpu_cores * 2)})
    # GPU-oriented (if any GPU, try a higher parallelism)
    if gpu_count > 0:
        candidates.update({8, 12, 16, 24})
    return sorted(x for x in candidates if x > 0)


def _detect_hardware() -> tuple[int, int, float]:
    cpu_cores = 4
    if HAS_PSUTIL:
        try:
            cpu_cores = psutil.cpu_count(logical=False) or (psutil.cpu_count() or 4)
        except Exception:
            cpu_cores = 4
    gpu_count = 0
    if HAS_GPU_MON:
        try:
            stats = _gpu_stats_snapshot()
            gpu_count = len([s for s in stats.values() if s.get("device_id", -1) != -1])
        except Exception:
            gpu_count = 0
    ram_gb = 0.0
    if HAS_PSUTIL:
        try:
            ram_gb = float(psutil.virtual_memory().available / (1024**3))
        except Exception:
            ram_gb = 0.0
    return cpu_cores, gpu_count, ram_gb


def _pick_best_config(probes: Iterable[ProbeResult]) -> ProbeResult | None:
    best: ProbeResult | None = None
    best_score = -1.0
    for p in probes:
        # Score favors throughput, penalizes worker count very lightly to avoid overkill
        penalty = math.log(max(1, p.config.max_workers)) * (0.05 if not p.config.use_processes else 0.08)
        score = p.throughput - penalty
        if score > best_score:
            best = p
            best_score = score
    return best


# ------------------------------ Public API ------------------------------- #


def probe_parallel_configs(
    *,
    params_grid: dict[str, Any],
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    reuse_cache: bool = True,
    try_processes: bool = True,
    extra_worker_candidates: list[int] | None = None,
    indicator_bank: IndicatorBank | None = None,
    log_level: int = logging.INFO,
) -> dict[str, Any]:
    """Probe several parallel configurations and return a report with the best.

    Does NOT execute the full sweep. Intended for UI usage to keep the
    Streamlit progress bar during the actual run.
    """
    logger.setLevel(log_level)

    combos = generate_param_grid(params_grid)
    if not combos:
        raise ValueError("Empty parameter grid: no combinations generated")

    bank = indicator_bank or IndicatorBank(IndicatorSettings())

    cpu_cores, gpu_count, ram_gb = _detect_hardware()
    logger.info(f"Hardware: CPU cores={cpu_cores}, GPUs={gpu_count}, free RAM~{ram_gb:.1f} GB")

    cand_workers = _heuristic_candidates(cpu_cores=cpu_cores, gpu_count=gpu_count)
    if extra_worker_candidates:
        cand_workers = sorted(set(cand_workers).union(set(extra_worker_candidates)))

    probe_matrix: list[ProbeResult] = []

    for w in cand_workers:
        try:
            r = _run_probe(
                combos=combos,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                use_processes=False,
                max_workers=w,
                reuse_cache=reuse_cache,
                indicator_bank=bank,
            )
            logger.info(f"Probe ThreadPool(w={w}): {r.throughput:.1f} cases/s in {r.elapsed:.2f}s")
            probe_matrix.append(r)
        except Exception as e:
            logger.warning(f"Probe ThreadPool(w={w}) failed: {e}")

    if try_processes:
        for w in {max(1, cpu_cores // 2), cpu_cores}:
            try:
                r = _run_probe(
                    combos=combos,
                    data=data,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=strategy_name,
                    use_processes=True,
                    max_workers=w,
                    reuse_cache=reuse_cache,
                    indicator_bank=bank,
                )
                logger.info(f"Probe ProcessPool(w={w}): {r.throughput:.1f} cases/s in {r.elapsed:.2f}s")
                probe_matrix.append(r)
            except Exception as e:
                logger.warning(f"Probe ProcessPool(w={w}) failed: {e}")

    if not probe_matrix:
        # Fallback conservative default instead of raising to keep UI responsive
        cpu_cores, gpu_count, _ = _detect_hardware()
        fallback_workers = max(2, cpu_cores or 4)
        return {
            "probes": [],
            "chosen": {
                "use_processes": False,
                "max_workers": fallback_workers,
                "probe_throughput_cps": 0.0,
            },
            "total_combos": len(combos),
            "warning": "Probe failed; using fallback ThreadPool configuration",
        }

    best = _pick_best_config(probe_matrix)
    assert best is not None

    report: dict[str, Any] = {
        "probes": [
            {
                "use_processes": pr.config.use_processes,
                "max_workers": pr.config.max_workers,
                "probe_cases": pr.n_cases,
                "elapsed_sec": pr.elapsed,
                "throughput_cps": pr.throughput,
                "cpu_pct_avg": pr.cpu_pct_avg,
                "ram_used_gb": pr.ram_used_gb,
                "gpu_util_avg": pr.gpu_util_avg,
            }
            for pr in probe_matrix
        ],
        "chosen": {
            "use_processes": best.config.use_processes,
            "max_workers": best.config.max_workers,
            "probe_throughput_cps": best.throughput,
        },
        "total_combos": len(combos),
    }

    return report


def optimize_sweep_execution(
    *,
    params_grid: dict[str, Any],
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    reuse_cache: bool = True,
    try_processes: bool = True,
    extra_worker_candidates: list[int] | None = None,
    log_level: int = logging.INFO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Determine an efficient parallel configuration and run the full sweep.

    Returns
    -------
    (results_df, report)
      - results_df: pandas DataFrame with sweep results
      - report: dict with probe details and chosen configuration
    """
    logger.setLevel(log_level)

    # Generate combinations once
    combos = generate_param_grid(params_grid)
    if not combos:
        raise ValueError("Empty parameter grid: no combinations generated")

    # Shared IndicatorBank to maximize cache reuse across probes and final run
    indicator_bank = IndicatorBank(IndicatorSettings())

    cpu_cores, gpu_count, ram_gb = _detect_hardware()
    logger.info(f"Hardware: CPU cores={cpu_cores}, GPUs={gpu_count}, free RAM≈{ram_gb:.1f} GB")

    # Candidate worker counts
    cand_workers = _heuristic_candidates(cpu_cores=cpu_cores, gpu_count=gpu_count)
    if extra_worker_candidates:
        cand_workers = sorted(set(cand_workers).union(set(extra_worker_candidates)))

    probe_matrix: list[ProbeResult] = []

    # Probe ThreadPool variants
    for w in cand_workers:
        try:
            r = _run_probe(
                combos=combos,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                use_processes=False,
                max_workers=w,
                reuse_cache=reuse_cache,
                indicator_bank=indicator_bank,
            )
            logger.info(f"Probe ThreadPool(w={w}): {r.throughput:.1f} cases/s in {r.elapsed:.2f}s")
            probe_matrix.append(r)
        except Exception as e:
            logger.warning(f"Probe ThreadPool(w={w}) failed: {e}")

    # Probe ProcessPool variants (optional)
    if try_processes:
        for w in {max(1, cpu_cores // 2), cpu_cores}:
            try:
                r = _run_probe(
                    combos=combos,
                    data=data,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=strategy_name,
                    use_processes=True,
                    max_workers=w,
                    reuse_cache=reuse_cache,
                    indicator_bank=indicator_bank,
                )
                logger.info(f"Probe ProcessPool(w={w}): {r.throughput:.1f} cases/s in {r.elapsed:.2f}s")
                probe_matrix.append(r)
            except Exception as e:
                logger.warning(f"Probe ProcessPool(w={w}) failed: {e}")

    if not probe_matrix:
        raise RuntimeError("No probe could be completed. Aborting sweep.")

    best = _pick_best_config(probe_matrix)
    assert best is not None
    logger.info(
        f"Chosen config: {'ProcessPool' if best.config.use_processes else 'ThreadPool'} "
        f"with {best.config.max_workers} workers (throughput≈{best.throughput:.1f} cases/s on probe)"
    )

    # Final run with chosen configuration
    final_runner = SweepRunner(
        indicator_bank=indicator_bank,
        max_workers=best.config.max_workers,
        use_multigpu=True,
        use_processes=best.config.use_processes,
    )

    t0 = time.time()
    try:
        results_df = final_runner._execute_combinations_bounded(  # type: ignore[attr-defined]
            combos,
            data,
            symbol,
            timeframe,
            strategy_name,
            reuse_cache=reuse_cache,
        )
    except AttributeError:
        results_df = final_runner._execute_combinations(  # type: ignore[attr-defined]
            combos,
            data,
            symbol,
            timeframe,
            strategy_name,
            reuse_cache=reuse_cache,
        )
    total_elapsed = time.time() - t0
    throughput = len(results_df) / total_elapsed if total_elapsed > 0 else 0.0

    report: dict[str, Any] = {
        "probes": [
            {
                "use_processes": pr.config.use_processes,
                "max_workers": pr.config.max_workers,
                "probe_cases": pr.n_cases,
                "elapsed_sec": pr.elapsed,
                "throughput_cps": pr.throughput,
                "cpu_pct_avg": pr.cpu_pct_avg,
                "ram_used_gb": pr.ram_used_gb,
                "gpu_util_avg": pr.gpu_util_avg,
            }
            for pr in probe_matrix
        ],
        "chosen": {
            "use_processes": best.config.use_processes,
            "max_workers": best.config.max_workers,
        },
        "final": {
            "total_cases": len(results_df),
            "elapsed_sec": total_elapsed,
            "throughput_cps": throughput,
        },
    }

    return results_df, report


# --------------------------------- CLI ---------------------------------- #


def _demo_cli() -> None:  # pragma: no cover - convenience entrypoint
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="ThreadX Parallel Sweep Manager")
    parser.add_argument("--symbol", default="BTC", help="Symbol, e.g., BTC")
    parser.add_argument("--timeframe", default="1h", help="Timeframe, e.g., 1h")
    parser.add_argument("--strategy", default="Bollinger_Breakout", help="Strategy name")
    parser.add_argument("--bars", type=int, default=2000, help="Number of synthetic bars")
    parser.add_argument("--log", default="INFO", help="Log level: DEBUG/INFO/WARN")
    args = parser.parse_args()

    # Synthetic OHLCV for quick runs
    idx = pd.date_range("2022-01-01", periods=args.bars, freq="H", tz="UTC")
    price = pd.Series(np.cumsum(np.random.randn(args.bars)) + 100, index=idx)
    df = pd.DataFrame(
        {
            "open": price + np.random.randn(args.bars) * 0.1,
            "high": price + np.abs(np.random.randn(args.bars)) * 0.2,
            "low": price - np.abs(np.random.randn(args.bars)) * 0.2,
            "close": price,
            "volume": np.random.rand(args.bars) * 1000,
        }
    )

    grid = {
        "strategy": {"value": args.strategy},
        "bb_window": {"values": [10, 20, 30, 40]},
        "bb_num_std": {"values": [1.5, 2.0]},
        "atr_window": {"values": [14, 20]},
        "atr_multiplier": {"values": [1.0, 1.5]},
    }

    level = getattr(logging, args.log.upper(), logging.INFO)
    results, rep = optimize_sweep_execution(
        params_grid=grid,
        data=df,
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_name=args.strategy,
        try_processes=True,
        log_level=level,
    )
    print("Chosen:", rep["chosen"])  # Short summary
    print("Final throughput:", f"{rep['final']['throughput_cps']:.2f} cases/s")
    print("Rows:", len(results))


if __name__ == "__main__":  # pragma: no cover
    _demo_cli()
