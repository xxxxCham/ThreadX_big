"""
Profile a short backtest and report hotspots.

Outputs:
- Section timings (indicators, engine, performance)
- cProfile top functions (threadx only), sorted by cumulative time
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import time
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@contextmanager
def section_timings(sections: Dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sections[name] = sections.get(name, 0.0) + (time.perf_counter() - t0)


def make_ohlcv(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + rng.standard_normal(n).cumsum().astype(np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * 0.2
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * 0.2
    volume = rng.integers(1_000, 10_000, size=n, dtype=np.int64)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def run_short_backtest() -> Dict[str, Any]:
    # Keep runtime logs quiet
    os.environ.setdefault("THREADX_DEBUG", "0")

    from threadx.indicators.bank import IndicatorBank
    from threadx.backtest.engine import BacktestEngine
    from threadx.backtest.performance import summarize

    timings: Dict[str, float] = {}
    df = make_ohlcv(n=5000, seed=42)

    with section_timings(timings, "indicators"):
        bank = IndicatorBank()
        bb = bank.ensure("bollinger", {"period": 20, "std": 2.0}, df, symbol="BTCUSDC", timeframe="1m")
        atr = bank.ensure("atr", {"period": 14, "method": "ema"}, df, symbol="BTCUSDC", timeframe="1m")
        indicators = {"bollinger": bb, "atr": atr}

    params = {"entry_z": 1.0, "k_sl": 1.5, "leverage": 1.0, "fees_bps": 10.0}
    engine = BacktestEngine()

    with section_timings(timings, "engine_run"):
        result = engine.run(df, indicators, params=params, symbol="BTCUSDC", timeframe="1m", seed=42, use_gpu=False)

    with section_timings(timings, "performance"):
        summary = summarize(result.trades, result.returns, initial_capital=10_000.0)

    return {"timings": timings, "result": result, "summary": summary}


def main() -> None:
    pr = cProfile.Profile()
    pr.enable()
    out = run_short_backtest()
    pr.disable()

    timings = out["timings"]
    total = sum(timings.values())
    print("Section timings:")
    for name, dur in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        pct = (dur / total) * 100 if total > 0 else 0.0
        print(f" - {name:14s}: {dur:7.3f}s  ({pct:5.1f}%)")
    print(f" - {'TOTAL':14s}: {total:7.3f}s")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(50)
    # Filter to threadx-only lines for readability
    lines = [ln for ln in s.getvalue().splitlines() if "threadx" in ln or ln.startswith("   ncalls")]
    print("\nTop functions by cumulative time (threadx only):")
    for ln in lines[:80]:
        print(ln)


if __name__ == "__main__":
    main()

