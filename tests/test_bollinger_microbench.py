"""
Micro-bench internal steps for Bollinger-like operations (CPU):
- rolling mean/std via pandas
- allocations/copies
- optional numpy<->cupy conversions if available
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


@contextmanager
def t(name: str, results: dict[str, float]):
    start = time.perf_counter()
    try:
        yield
    finally:
        results[name] = results.get(name, 0.0) + (time.perf_counter() - start)


def _microbench(size: int) -> dict[str, float]:
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(size).cumsum().astype(np.float64)
    s = pd.Series(close)
    results: dict[str, float] = {}

    with t("alloc_empty", results):
        tmp = np.empty_like(close)
        _ = tmp  # keep reference

    with t("copy_array", results):
        c2 = close.copy()
        _ = c2

    with t("rolling_mean", results):
        m = s.rolling(window=20, min_periods=20).mean()
        _ = m

    with t("rolling_std", results):
        sd = s.rolling(window=20, min_periods=20).std(ddof=0)
        _ = sd

    # Optional conversions
    try:
        import cupy as cp  # type: ignore

        if hasattr(cp, "asarray"):
            with t("to_gpu", results):
                g = cp.asarray(close)
                _ = g
            with t("to_cpu", results):
                back = cp.asnumpy(g)
                _ = back
    except Exception:
        pass

    return results


def test_bollinger_microbench():
    sizes = [1000, 10000, 100000]
    print("\nBollinger micro-bench (CPU / optional GPU copy):")
    for s in sizes:
        res = _microbench(s)
        total = sum(res.values())
        print(f"size={s}")
        for k, v in sorted(res.items(), key=lambda kv: kv[1], reverse=True):
            pct = (v / total) * 100 if total > 0 else 0.0
            print(f"  {k:14s}: {v:7.4f}s  ({pct:5.1f}%)")

    # Basic sanity
    assert True

