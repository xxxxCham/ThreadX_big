"""
ThreadX Sweep Engine - Phase 7
==============================

Parametric sweep engine with parallel execution, checkpointing, and append-only
Parquet storage. Integrates with Engine (Phase 5) and Performance (Phase 6).

Features:
- Multi-threaded parameter grid execution
- Deterministic results (seed=42)
- Checkpoint/resume capability
- Append-only Parquet storage with file locks
- GPU/CPU compatibility through engine delegation
- Windows 11 compatible

Author: ThreadX Framework
Version: Phase 7 - Sweep & Logging
"""

import json
import uuid
import hashlib
import time
import tempfile

# Platform-specific imports for file locking
if os.name == "nt":
    import msvcrt

    fcntl = None
else:
    import fcntl

    msvcrt = None
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from threadx.utils.log import get_logger, setup_logging_once

# Initialize logging
setup_logging_once()
logger = get_logger(__name__)

# Global file locks for thread safety
_file_locks: Dict[str, Lock] = {}
_locks_lock = Lock()


def _get_file_lock(file_path: Path) -> Lock:
    """Get or create thread-local lock for file operations."""
    with _locks_lock:
        str_path = str(file_path.absolute())
        if str_path not in _file_locks:
            _file_locks[str_path] = Lock()
        return _file_locks[str_path]


def _safe_json_dumps(obj: Any) -> str:
    """
    JSON serialize with sorted keys for deterministic output.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    str
        JSON string with sorted keys.
    """
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def make_run_id(seed: int, extra: Dict[str, Any]) -> str:
    """
    Create stable run identifier from seed and metadata.

    Parameters
    ----------
    seed : int
        Random seed for deterministic generation.
    extra : dict
        Additional metadata (symbol, timeframe, etc.).

    Returns
    -------
    str
        Stable UUID string based on input hash.

    Examples
    --------
    >>> run_id = make_run_id(42, {"symbol": "BTCUSDC", "timeframe": "15m"})
    >>> print(len(run_id))
    36
    """
    # Create deterministic hash from seed + extra
    content = f"{seed}:{_safe_json_dumps(extra)}"
    hash_bytes = hashlib.sha256(content.encode("utf-8")).digest()

    # Convert to UUID format for readability
    run_uuid = uuid.UUID(bytes=hash_bytes[:16])
    return str(run_uuid)


def validate_param_grid(
    param_grid: List[Union[Dict[str, Any], Any]],
) -> List[Dict[str, Any]]:
    """
    Validate and normalize parameter grid.

    Parameters
    ----------
    param_grid : list
        List of parameter dictionaries or dataclass instances.

    Returns
    -------
    list[dict]
        Normalized parameter dictionaries.

    Raises
    ------
    ValueError
        If param_grid is invalid or contains unsupported types.

    Examples
    --------
    >>> @dataclass
    ... class Params:
    ...     bb_period: int = 20
    ...     bb_std: float = 2.0
    >>>
    >>> grid = [Params(bb_period=14), {"bb_period": 21, "bb_std": 2.5}]
    >>> normalized = validate_param_grid(grid)
    >>> len(normalized)
    2
    """
    if not param_grid:
        raise ValueError("Parameter grid cannot be empty")

    if not isinstance(param_grid, list):
        raise ValueError("Parameter grid must be a list")

    normalized = []

    for i, params in enumerate(param_grid):
        try:
            if is_dataclass(params):
                # Convert dataclass to dict
                param_dict = asdict(params)
            elif isinstance(params, dict):
                param_dict = dict(params)  # Copy to avoid mutations
            else:
                raise ValueError(
                    f"Item {i}: Expected dict or dataclass, got {type(params)}"
                )

            # Validate required fields (basic validation)
            if not param_dict:
                raise ValueError(f"Item {i}: Parameter dictionary is empty")

            # Ensure all values are JSON serializable
            try:
                _safe_json_dumps(param_dict)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Item {i}: Parameters not JSON serializable: {e}")

            normalized.append(param_dict)

        except Exception as e:
            logger.error(f"Failed to validate parameter {i}: {e}")
            raise ValueError(f"Invalid parameter at index {i}: {e}")

    logger.debug(f"Validated {len(normalized)} parameter combinations")
    return normalized


def _execute_single_task(task_args: tuple) -> Dict[str, Any]:
    """
    Execute single backtest task.

    Parameters
    ----------
    task_args : tuple
        Arguments: (df, params, engine_func, symbol, timeframe,
                   initial_capital, fee_bps, slip_bps, use_gpu, task_id)

    Returns
    -------
    dict
        Task results with performance metrics.
    """
    (
        df,
        params,
        engine_func,
        symbol,
        timeframe,
        initial_capital,
        fee_bps,
        slip_bps,
        use_gpu,
        task_id,
    ) = task_args

    start_time = time.time()
    result = {
        "task_id": task_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "params_json": _safe_json_dumps(params),
        "success": False,
        "error": None,
        "duration_sec": 0.0,
        # Default performance metrics
        "final_equity": initial_capital,
        "pnl": 0.0,
        "total_return": 0.0,
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "expectancy": 0.0,
        "total_trades": 0,
        "win_trades": 0,
        "loss_trades": 0,
        "duration_days": 0.0,
        "annual_volatility": 0.0,
    }

    try:
        # Execute backtest through engine
        engine_result = engine_func(
            df=df,
            params=params,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            use_gpu=use_gpu,
            symbol=symbol,
            timeframe=timeframe,
        )

        # Extract returns and trades from engine result
        if isinstance(engine_result, dict):
            returns = engine_result.get("returns", pd.Series(dtype=float))
            trades = engine_result.get("trades", pd.DataFrame())
        elif isinstance(engine_result, tuple) and len(engine_result) >= 2:
            returns, trades = engine_result[:2]
        else:
            raise ValueError(f"Unexpected engine result type: {type(engine_result)}")

        # Calculate performance metrics using Phase 6
        try:
            from threadx.backtest.performance import summarize

            performance_metrics = summarize(
                trades=trades,
                returns=returns,
                initial_capital=initial_capital,
                risk_free=0.0,
                periods_per_year=365,
            )

            # Update result with performance metrics
            result.update(performance_metrics)
            result["success"] = True

        except ImportError:
            logger.warning("Performance module not available, using basic metrics")
            # Basic fallback metrics
            if not returns.empty:
                result["final_equity"] = initial_capital * (1 + returns.sum())
                result["pnl"] = result["final_equity"] - initial_capital
                result["total_return"] = (
                    result["final_equity"] / initial_capital - 1
                ) * 100

            if not trades.empty and "pnl" in trades.columns:
                result["total_trades"] = len(trades)
                result["win_trades"] = (trades["pnl"] > 0).sum()
                result["loss_trades"] = (trades["pnl"] < 0).sum()
                result["win_rate"] = (
                    result["win_trades"] / result["total_trades"]
                    if result["total_trades"] > 0
                    else 0.0
                )

            result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
        logger.warning(f"Task {task_id} failed: {e}")

    finally:
        result["duration_sec"] = time.time() - start_time

    return result


def run_grid(
    df: pd.DataFrame,
    param_grid: List[Union[Dict[str, Any], Any]],
    *,
    engine_func: Callable,
    symbol: str,
    timeframe: str,
    initial_capital: float = 10_000.0,
    fee_bps: float = 1.0,
    slip_bps: float = 0.0,
    max_workers: int = 8,
    seed: int = 42,
    use_gpu: bool = True,
    checkpoint_path: Optional[Path] = None,
    chunk_size: int = 50,
) -> pd.DataFrame:
    """
    Execute parameter grid sweep with parallel processing and checkpointing.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with datetime index.
    param_grid : list
        Parameter combinations to test (dicts or dataclasses).
    engine_func : callable
        Backtest engine function from Phase 5.
        Must return (returns, trades) or dict with these keys.
    symbol : str
        Trading symbol identifier.
    timeframe : str
        Data timeframe (e.g., "15m", "1h").
    initial_capital : float, default 10_000.0
        Starting capital for backtests.
    fee_bps : float, default 1.0
        Trading fees in basis points.
    slip_bps : float, default 0.0
        Slippage in basis points.
    max_workers : int, default 8
        Maximum parallel workers.
    seed : int, default 42
        Random seed for deterministic results.
    use_gpu : bool, default True
        Enable GPU acceleration (delegated to engine).
    checkpoint_path : Path, optional
        Path for checkpoint/resume functionality.
    chunk_size : int, default 50
        Batch size for processing and checkpointing.

    Returns
    -------
    pd.DataFrame
        Results with performance metrics and metadata.

    Raises
    ------
    ValueError
        If inputs are invalid.
    IOError
        If file operations fail.

    Examples
    --------
    >>> # Basic usage with mock engine
    >>> def mock_engine(df, params, **kwargs):
    ...     returns = pd.Series([0.001, -0.002, 0.003])
    ...     trades = pd.DataFrame({'pnl': [100, -50, 150]})
    ...     return returns, trades
    >>>
    >>> param_grid = [
    ...     {"bb_period": 20, "bb_std": 2.0},
    ...     {"bb_period": 14, "bb_std": 1.8}
    ... ]
    >>>
    >>> results = run_grid(
    ...     df=ohlcv_data,
    ...     param_grid=param_grid,
    ...     engine_func=mock_engine,
    ...     symbol="BTCUSDC",
    ...     timeframe="15m"
    ... )
    >>> print(f"Completed {len(results)} backtests")

    >>> # With GPU and checkpointing
    >>> results = run_grid(
    ...     df=ohlcv_data,
    ...     param_grid=large_param_grid,
    ...     engine_func=advanced_engine,
    ...     symbol="ETHUSD",
    ...     timeframe="1h",
    ...     use_gpu=True,
    ...     checkpoint_path=Path("sweep_checkpoint.parquet"),
    ...     max_workers=16,
    ...     chunk_size=100
    ... )
    """
    start_time = time.time()

    # Set random seed for deterministic results
    np.random.seed(seed)

    # Validate inputs
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    normalized_params = validate_param_grid(param_grid)
    n_tasks = len(normalized_params)

    # Limit workers for Windows stability
    max_workers = min(max_workers, os.cpu_count() or 8, 16)

    logger.info(f"Starting parameter sweep: {n_tasks} tasks, {max_workers} workers")
    logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}, Seed: {seed}")
    logger.info(f"GPU enabled: {use_gpu}, Chunk size: {chunk_size}")

    # Check for existing checkpoint
    completed_tasks = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_df = pd.read_parquet(checkpoint_path)
            completed_tasks = checkpoint_df["task_id"].tolist()
            logger.info(
                f"Resuming from checkpoint: {len(completed_tasks)} tasks completed"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # Prepare tasks (skip completed ones)
    all_tasks = []
    for i, params in enumerate(normalized_params):
        task_id = f"{symbol}_{timeframe}_{seed}_{i:06d}"

        if task_id not in completed_tasks:
            task_args = (
                df,
                params,
                engine_func,
                symbol,
                timeframe,
                initial_capital,
                fee_bps,
                slip_bps,
                use_gpu,
                task_id,
            )
            all_tasks.append(task_args)

    remaining_tasks = len(all_tasks)
    logger.info(
        f"Tasks to execute: {remaining_tasks} (skipped {n_tasks - remaining_tasks} completed)"
    )

    if remaining_tasks == 0:
        logger.info("All tasks already completed, loading checkpoint")
        return pd.read_parquet(checkpoint_path)

    # Execute tasks in chunks
    all_results = []
    completed_count = len(completed_tasks)

    # Load existing results if available
    if completed_tasks and checkpoint_path and checkpoint_path.exists():
        try:
            existing_df = pd.read_parquet(checkpoint_path)
            all_results.extend(existing_df.to_dict("records"))
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")

    # Process remaining tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks in chunks
        for chunk_start in range(0, remaining_tasks, chunk_size):
            chunk_end = min(chunk_start + chunk_size, remaining_tasks)
            chunk_tasks = all_tasks[chunk_start:chunk_end]

            logger.info(
                f"Processing chunk {chunk_start//chunk_size + 1}: "
                f"tasks {chunk_start+1}-{chunk_end} of {remaining_tasks}"
            )

            # Submit chunk
            future_to_args = {}
            for task_args in chunk_tasks:
                future = executor.submit(_execute_single_task, task_args)
                future_to_args[future] = task_args

            # Collect results
            chunk_results = []
            for future in as_completed(future_to_args):
                try:
                    result = future.result(timeout=300)  # 5 min timeout per task
                    chunk_results.append(result)
                    completed_count += 1

                    if completed_count % 10 == 0:
                        logger.info(f"Completed {completed_count}/{n_tasks} tasks")

                except Exception as e:
                    task_args = future_to_args[future]
                    task_id = task_args[-1]
                    logger.error(f"Task {task_id} failed with timeout/error: {e}")

                    # Create failed result
                    failed_result = {
                        "task_id": task_id,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "params_json": _safe_json_dumps(task_args[1]),
                        "success": False,
                        "error": str(e),
                        "duration_sec": 300.0,  # Timeout duration
                        "final_equity": initial_capital,
                        "pnl": 0.0,
                        "total_return": 0.0,
                        "sharpe": 0.0,
                        "total_trades": 0,
                    }
                    chunk_results.append(failed_result)
                    completed_count += 1

            # Add chunk results
            all_results.extend(chunk_results)

            # Save checkpoint
            if checkpoint_path:
                try:
                    checkpoint_df = pd.DataFrame(all_results)
                    # Atomic write
                    temp_path = checkpoint_path.with_suffix(".tmp")
                    checkpoint_df.to_parquet(temp_path, index=False)
                    temp_path.replace(checkpoint_path)

                    logger.debug(f"Checkpoint saved: {len(all_results)} results")
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")

    # Final results
    results_df = pd.DataFrame(all_results)
    total_time = time.time() - start_time

    # Add run metadata
    results_df["run_id"] = make_run_id(
        seed, {"symbol": symbol, "timeframe": timeframe, "n_tasks": n_tasks}
    )
    results_df["timestamp"] = pd.Timestamp.now(tz="UTC")
    results_df["tasks_per_min"] = n_tasks / (total_time / 60) if total_time > 0 else 0.0

    # Calculate success rate
    success_count = results_df["success"].sum()
    success_rate = success_count / len(results_df) * 100

    logger.info(
        f"Sweep completed: {success_count}/{n_tasks} successful ({success_rate:.1f}%)"
    )
    logger.info(
        f"Total time: {total_time:.1f}s, Rate: {results_df['tasks_per_min'].iloc[0]:.1f} tasks/min"
    )

    return results_df


def _acquire_file_lock(file_handle, blocking: bool = True):
    """Cross-platform file locking."""
    if os.name == "nt":  # Windows
        while True:
            try:
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except IOError:
                if not blocking:
                    raise
                time.sleep(0.01)
    else:  # Unix-like
        flag = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(file_handle.fileno(), flag)


def _release_file_lock(file_handle):
    """Release cross-platform file lock."""
    if os.name == "nt":  # Windows
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # Unix-like
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def append_run_history(
    results: pd.DataFrame, history_path: Path, *, lock_path: Optional[Path] = None
) -> Path:
    """
    Append results to run history with file locking.

    Parameters
    ----------
    results : pd.DataFrame
        Results to append with required columns.
    history_path : Path
        Path to history Parquet file.
    lock_path : Path, optional
        Custom lock file path. Uses history_path.lock if None.

    Returns
    -------
    Path
        Path to updated history file.

    Raises
    ------
    IOError
        If file operations fail.

    Examples
    --------
    >>> results_df = pd.DataFrame({
    ...     'run_id': ['run_001'],
    ...     'symbol': ['BTCUSDC'],
    ...     'sharpe': [1.5],
    ...     'success': [True]
    ... })
    >>> history_path = append_run_history(results_df, Path("runs.parquet"))
    >>> print(f"Updated history: {history_path}")
    """
    if results.empty:
        logger.warning("Empty results DataFrame, skipping append")
        return history_path

    # Prepare lock file
    if lock_path is None:
        lock_path = history_path.with_suffix(".lock")

    # Thread-local lock
    thread_lock = _get_file_lock(history_path)

    with thread_lock:
        # Create directory if needed
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # File-level lock
        with open(lock_path, "w") as lock_file:
            try:
                _acquire_file_lock(lock_file, blocking=True)

                # Read existing data
                existing_df = None
                if history_path.exists():
                    try:
                        existing_df = pd.read_parquet(history_path)
                        logger.debug(
                            f"Loaded existing history: {len(existing_df)} records"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read existing history: {e}")

                # Combine data
                if existing_df is not None and not existing_df.empty:
                    # Check for duplicates by run_id + task_id if available
                    if (
                        "task_id" in results.columns
                        and "task_id" in existing_df.columns
                    ):
                        # Remove duplicates from existing data
                        task_ids = set(results["task_id"])
                        existing_df = existing_df[
                            ~existing_df["task_id"].isin(task_ids)
                        ]

                    combined_df = pd.concat([existing_df, results], ignore_index=True)
                else:
                    combined_df = results.copy()

                # Atomic write
                temp_path = history_path.with_suffix(".tmp")
                combined_df.to_parquet(temp_path, index=False)
                temp_path.replace(history_path)

                logger.info(
                    f"Appended {len(results)} records to history "
                    f"(total: {len(combined_df)} records)"
                )

            finally:
                _release_file_lock(lock_file)

    return history_path


def update_best_by_run(
    history_path: Path, best_path: Path, *, sort_by: str = "sharpe"
) -> Path:
    """
    Update best results by run from history.

    Parameters
    ----------
    history_path : Path
        Path to run history Parquet.
    best_path : Path
        Path to best results Parquet.
    sort_by : str, default "sharpe"
        Metric to sort by for "best" selection.

    Returns
    -------
    Path
        Path to updated best results file.

    Examples
    --------
    >>> best_path = update_best_by_run(
    ...     Path("runs.parquet"),
    ...     Path("best_by_run.parquet"),
    ...     sort_by="sharpe"
    ... )
    >>> best_df = pd.read_parquet(best_path)
    >>> print(f"Best results: {len(best_df)} runs")
    """
    if not history_path.exists():
        logger.warning(f"History file not found: {history_path}")
        return best_path

    try:
        # Load history
        history_df = pd.read_parquet(history_path)

        if history_df.empty:
            logger.warning("Empty history file")
            return best_path

        # Filter successful runs only
        successful_df = history_df[history_df["success"] == True].copy()

        if successful_df.empty:
            logger.warning("No successful runs in history")
            return best_path

        # Group by run_id and find best result
        best_results = []

        for run_id, group in successful_df.groupby("run_id"):
            # Sort by metric (descending for most metrics, ascending for drawdown)
            ascending = sort_by in ["max_drawdown", "error", "duration_sec"]
            best_row = group.sort_values(
                [sort_by, "task_id"], ascending=[ascending, True]
            ).iloc[0]

            # Create best result record
            best_record = {
                "run_id": run_id,
                "best_task_id": best_row["task_id"],
                "best_params_json": best_row["params_json"],
                "best_sharpe": best_row.get("sharpe", 0.0),
                "best_sortino": best_row.get("sortino", 0.0),
                "best_cagr": best_row.get("cagr", 0.0),
                "best_win_rate": best_row.get("win_rate", 0.0),
                "best_profit_factor": best_row.get("profit_factor", 0.0),
                "best_max_drawdown": best_row.get("max_drawdown", 0.0),
                "best_total_trades": best_row.get("total_trades", 0),
                "best_final_equity": best_row.get("final_equity", 0.0),
                "sort_metric": sort_by,
                "sort_value": best_row.get(sort_by, 0.0),
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "symbol": best_row.get("symbol", ""),
                "timeframe": best_row.get("timeframe", ""),
            }
            best_results.append(best_record)

        if not best_results:
            logger.warning("No best results generated")
            return best_path

        # Create DataFrame
        best_df = pd.DataFrame(best_results)

        # Sort by metric value (stable sort)
        ascending = sort_by in ["max_drawdown", "error", "duration_sec"]
        best_df = best_df.sort_values(
            ["sort_value", "run_id"], ascending=[ascending, True]
        )

        # Atomic write with lock
        thread_lock = _get_file_lock(best_path)

        with thread_lock:
            best_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write
            temp_path = best_path.with_suffix(".tmp")
            best_df.to_parquet(temp_path, index=False)
            temp_path.replace(best_path)

        logger.info(f"Updated best results: {len(best_df)} runs, sorted by {sort_by}")

        return best_path

    except Exception as e:
        logger.error(f"Failed to update best results: {e}")
        raise


def load_run_history(history_path: Path) -> pd.DataFrame:
    """
    Load run history from Parquet file.

    Parameters
    ----------
    history_path : Path
        Path to history Parquet file.

    Returns
    -------
    pd.DataFrame
        History DataFrame, empty if file doesn't exist.

    Examples
    --------
    >>> history_df = load_run_history(Path("runs.parquet"))
    >>> print(f"Loaded {len(history_df)} historical runs")
    """
    if not history_path.exists():
        logger.debug(f"History file not found: {history_path}")
        return pd.DataFrame()

    try:
        history_df = pd.read_parquet(history_path)
        logger.debug(f"Loaded history: {len(history_df)} records")
        return history_df

    except Exception as e:
        logger.error(f"Failed to load history from {history_path}: {e}")
        return pd.DataFrame()


# Integration examples for docstrings
def _example_engine_func(df, params, **kwargs):
    """Example engine function for testing."""
    # Mock returns and trades
    n_periods = len(df)
    returns = pd.Series(np.random.randn(n_periods) * 0.01, index=df.index)

    trades_data = {
        "entry_time": df.index[::20][:5],  # Sample entries
        "exit_time": df.index[10::20][:5],  # Sample exits
        "pnl": np.random.randn(5) * 100,
        "side": ["LONG"] * 5,
    }
    trades = pd.DataFrame(trades_data)

    return returns, trades
