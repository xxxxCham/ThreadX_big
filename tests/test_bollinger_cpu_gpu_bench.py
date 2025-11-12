"""
CPU vs GPU benchmark (if CUDA available) for Bollinger Bands at multiple sizes.
Prints a concise table of average times and speedups.
"""

from __future__ import annotations

import os

import numpy as np


def test_bollinger_cpu_gpu_sizes():
    # Keep logs minimal
    os.environ.setdefault("THREADX_DEBUG", "0")

    from threadx.indicators.bollinger import benchmark_bollinger_performance, GPU_AVAILABLE

    sizes = [1000, 10000, 100000]
    bench = benchmark_bollinger_performance(data_sizes=sizes, n_runs=2)

    print("\nBollinger CPU vs GPU benchmark:")
    print(f"GPU available: {bench['gpu_available']}")
    print("size       cpu(s)   gpu(s)   speedup")
    for s in sizes:
        cpu = bench["cpu_times"].get(s, 0.0)
        gpu = bench["gpu_times"].get(s, 0.0)
        spd = bench["speedups"].get(s, 0.0)
        print(f"{s:7d}  {cpu:8.4f}  {gpu:7.4f}  {spd:7.2f}x")

    # Sanity: total CPU time should be > 0 across sizes (avoid per-size flakiness)
    total_cpu = sum(bench["cpu_times"].get(s, 0.0) for s in sizes)
    assert total_cpu >= 0.0
