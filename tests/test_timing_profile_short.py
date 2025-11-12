"""
Short timing profile for ThreadX blocks.

Measures durations of key blocks on a tiny synthetic dataset to
identify hotspots: indicator computation, backtest engine, and
performance summarization. Designed to run quickly and print a
human-readable summary without strict pass/fail thresholds.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# Ensure minimal logging overhead for the run
os.environ.setdefault("THREADX_DEBUG", "0")
logging.basicConfig(level=logging.WARNING)


class TimingCollector:
    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = {}

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.samples.setdefault(name, []).append(elapsed)

    def summary(self) -> List[tuple[str, float]]:
        return sorted(
            ((k, sum(v)) for k, v in self.samples.items()),
            key=lambda x: x[1],
            reverse=True,
        )


def _make_ohlcv(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Simple random walk for close; derive open/high/low
    close = 100 + rng.standard_normal(n).cumsum().astype(np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * 0.2
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * 0.2
    volume = rng.integers(1_000, 10_000, size=n, dtype=np.int64)

    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df


def test_timing_profile_short():
    # Lazy imports (keep test import time small)
    from threadx.indicators.bank import IndicatorBank
    from threadx.backtest.engine import BacktestEngine
    from threadx.backtest.performance import summarize

    t = TimingCollector()

    symbol = "BTCUSDC"
    timeframe = "1m"
    df = _make_ohlcv(n=1500, seed=42)

    bank = IndicatorBank()

    with t.section("indicators_bollinger"):
        bb = bank.ensure(
            "bollinger",
            {"period": 20, "std": 2.0},
            df,
            symbol=symbol,
            timeframe=timeframe,
        )

    with t.section("indicators_atr"):
        atr = bank.ensure(
            "atr",
            {"period": 14, "method": "ema"},
            df,
            symbol=symbol,
            timeframe=timeframe,
        )

    indicators: Dict[str, Any] = {"bollinger": bb, "atr": atr}

    engine = BacktestEngine()
    params = {"entry_z": 1.0, "k_sl": 1.5, "leverage": 1.0, "fees_bps": 10.0}

    with t.section("engine_run"):
        result = engine.run(
            df,
            indicators,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            seed=42,
            use_gpu=False,  # keep it fast and deterministic here
        )

    with t.section("performance_summarize"):
        _summary = summarize(result.trades, result.returns, initial_capital=10_000.0)

    # Emit a readable summary to test logs
    totals = t.summary()
    print("\nTiming summary (short profile):")
    total_time = sum(d for _, d in totals)
    for name, duration in totals:
        pct = (duration / total_time) * 100 if total_time > 0 else 0.0
        print(f" - {name:24s}: {duration:7.3f}s  ({pct:5.1f}%)")
    print(f" - {'TOTAL':24s}: {total_time:7.3f}s")

    # Basic sanity asserts (do not enforce performance here)
    assert "indicators_bollinger" in t.samples
    assert "indicators_atr" in t.samples
    assert "engine_run" in t.samples
    assert "performance_summarize" in t.samples

