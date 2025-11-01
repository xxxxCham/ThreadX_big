# ThreadX Agent Instructions

**Last Updated**: 2024-10-31
**Project Phase**: 10 (Production-ready backtest engine)
**Python Version**: 3.12+

---

## Project Overview

**ThreadX** is a **high-performance GPU-accelerated backtesting framework** for quantitative cryptocurrency trading strategies.

**Key Stats**:
- ~31,000 lines of Python code across 86 modules
- 199,500+ pre-computed indicator files (cached)
- 164 managed dependencies
- Multi-GPU support with intelligent distribution
- Modern Streamlit UI + CLI interface (Typer)

---

## Architecture at a Glance

```
ThreadX Core Subsystems:

backtest/          → Trading strategy execution engine
├── engine.py        (Main orchestrator, Phase 10)
├── performance.py   (Metrics & returns calculation)
└── validation.py    (Anti-overfitting checks)

strategy/          → Trading strategy implementations
├── amplitude_hunter.py   (BB Amplitude Rider)
├── bb_atr.py             (Bollinger Bands + ATR)
├── bollinger_dual.py     (Dual timeframe)
└── model.py              (Base classes & Trade dataclass)

indicators/        → Technical indicators with intelligent caching
├── bank.py         (Central cache with TTL, MD5 validation)
├── engine.py       (Calculation orchestrator)
├── bollinger.py    (Bollinger Bands)
├── xatr.py         (ATR)
├── indicators_np.py (NumPy suite: RSI, MACD, SMA, EMA)
└── gpu_integration.py (CuPy GPU acceleration)

optimization/      → Parameter optimization & sweeps
├── engine.py       (SweepRunner unified compute engine)
├── templates/      (GridOptimizer, MonteCarloOptimizer)
├── scenarios.py    (Parameter grid generation)
└── reporting.py    (Results & metrics)

gpu/              → GPU detection & multi-GPU distribution
├── device_manager.py     (Auto-detect, friendly naming)
├── multi_gpu.py          (Load balancing, NCCL sync)
└── profile_persistence.py (Cache profiles)

ui/               → Streamlit web interface (v2.0)
├── page_config_strategy.py       (Config & strategy)
├── page_backtest_optimization.py (Backtest & optimization)
├── backtest_bridge.py            (Execution coordination)
└── strategy_registry.py          (Strategy discovery)

cli/              → Command-line interface (Typer)
├── main.py        (Typer app entry point)
├── backtest_cmd.py, optimize_cmd.py, data_cmd.py
└── __main__.py    (python -m threadx bootstrap)

bridge/           → Integration components
├── unified_diversity_pipeline.py (Token diversity processing)
├── async_coordinator.py          (Async task coordination)
└── models.py                     (Domain models)

utils/            → Cross-cutting utilities
├── xp.py          (Device-agnostic NumPy/CuPy abstraction)
├── common_imports.py (DRY import centralization)
├── cache.py       (Generic caching utilities)
├── log.py         (Structured logging)
├── timing.py      (Performance decorators)
└── determinism.py (Seed control for reproducibility)

data_access.py    → Data discovery & loading (OHLCV + indicators)
```

---

## Critical Patterns & Conventions

### 1. Device-Agnostic Computing
**Pattern**: Use `utils.xp` instead of directly importing NumPy/CuPy.

```python
from threadx.utils.xp import xp  # Automatically resolves to NumPy or CuPy

# Code works on both CPU and GPU transparently
result = xp.array([1, 2, 3])
```

**Why**: Single code path for both NumPy (CPU) and CuPy (GPU). Fallback to CPU if GPU unavailable.

---

### 2. Indicator Bank Caching
**Pattern**: All indicators computed through `IndicatorBank`, never directly.

```python
from threadx.indicators.bank import IndicatorBank

bank = IndicatorBank()
bb_values = bank.compute_bollinger(
    ohlcv_data,
    period=20,
    std_dev=2.0,
    timeframe='1h',
    symbol='BTC'
)
# Automatically cached with TTL (3600s) and MD5 validation
```

**Why**:
- Disk-persistent cache prevents recomputation
- 199,500+ pre-computed indicator files available
- Automatic registry updates
- Batch processing for optimization runs
- Multi-GPU transparent support

---

### 3. Strategy Implementation Protocol
**Pattern**: All strategies inherit from `Strategy` base or follow the protocol.

```python
from threadx.strategy.model import Strategy, Trade

class MyStrategy(Strategy):
    def generate_signals(self, ohlcv, indicators):
        # Returns list of Trade objects
        return [Trade(...), ...]

    def get_params(self):
        return {"param1": default_value, ...}
```

**Why**: Type-safe, extensible, JSON serializable for UI/CLI integration.

---

### 4. Backtest Engine Entry Point
**Pattern**: Always use `BacktestEngine` for strategy execution, never manual execution.

```python
from threadx.backtest.engine import BacktestEngine, RunResult

engine = BacktestEngine(
    strategy=strategy_instance,
    ohlcv_data=historical_data,
    initial_capital=10000.0
)
result: RunResult = engine.run()

# result.trades → list of Trade objects
# result.metrics → {"sharpe_ratio", "max_drawdown_pct", "win_rate_pct", ...}
# result.returns → per-bar returns
# result.equity_curve → equity progression
```

**Why**: Handles all edge cases, performance calculations, validation.

---

### 5. Data Access Pattern
**Pattern**: Use `data_access.py` for all data discovery and loading.

```python
from threadx.data_access import (
    discover_tokens_and_timeframes,
    get_available_timeframes_for_token,
    load_ohlcv
)

# Discover what data is available
tokens = discover_tokens_and_timeframes()  # {"BTC": ["1h", "4h", "1d"], ...}

# Load OHLCV data
ohlcv = load_ohlcv(
    symbol="BTC",
    timeframe="1h",
    date_start="2024-01-01",
    date_end="2024-10-31"
)
# Data automatically normalized (dates as UTC, lowercase columns)
```

**Why**: Handles multiple formats (JSON, Parquet, CSV), normalization, env var overrides (`THREADX_DATA_DIR`).

---

### 6. Multi-GPU Distribution
**Pattern**: Use `utils.xp` and let the device manager handle distribution.

```python
# In gpu/multi_gpu.py or via utils.xp:
# - Auto-detects available GPUs
# - Default: 75%/25% load balance
# - Configurable in paths.toml

# Your code stays the same:
result = xp.array(...)  # Automatically distributed
```

**Why**: Transparent multi-GPU support without code changes.

---

## Key Data Structures

### Trade (src/threadx/strategy/model.py)
```python
@dataclass
class Trade:
    side: str                      # "LONG" | "SHORT"
    qty: float
    entry_price: float
    entry_time: str               # ISO UTC format
    exit_price: Optional[float]
    exit_time: Optional[str]
    stop: float
    take_profit: Optional[float]
    pnl_realized: Optional[float]
    pnl_unrealized: Optional[float]
    fees_paid: Optional[float]
    meta: Dict[str, Any]          # Indicators, context
```

### RunResult (src/threadx/backtest/engine.py)
```python
@dataclass
class RunResult:
    returns: List[float]          # Per-bar returns
    trades: List[Trade]
    equity_curve: List[float]
    metrics: Dict[str, float]     # sharpe_ratio, max_drawdown_pct, etc.
    metadata: Dict[str, Any]
```

---

## Entry Points for LLMs

### 1. **Understand a Trade Execution**
Start → `backtest/engine.py:BacktestEngine.run()` → logs which Trade objects are created

### 2. **Understand How Indicators Are Cached**
Start → `indicators/bank.py:IndicatorBank.compute_*()` → check TTL logic and MD5 validation

### 3. **Understand Strategy Optimization**
Start → `optimization/engine.py:SweepRunner` → loops through parameter grids, calls BacktestEngine

### 4. **Understand Data Loading**
Start → `data_access.py:load_ohlcv()` → handles normalization and multiple formats

### 5. **Understand GPU Acceleration**
Start → `utils/xp.py` → resolves NumPy vs CuPy → `gpu/device_manager.py` for detection

### 6. **Understand UI Execution**
Start → `streamlit_app.py` → imports from `ui/` → calls `ui/backtest_bridge.py` → calls `backtest/engine.py`

---

## Filesystem Organization

```
d:/ThreadX_big/
├── src/threadx/          ← All production code
│   ├── data/             ← 199,500+ cached indicator files + OHLCV data
│   ├── cache/            ← Runtime caches (streamlit, token_manager, registry)
│   └── [modules listed above]
│
├── tests/                ← Test suite (pytest-based)
├── examples/             ← Usage examples
├── docs/                 ← Documentation
├── tools/                ← Utility tools
│
├── pyproject.toml        ← Package metadata & dependencies
├── setup.cfg             ← Tool configurations (pytest, mypy, flake8, pylint)
├── pytest.ini            ← Pytest settings
├── paths.toml            ← Data paths & GPU configuration
├── requirements.txt      ← 164 pinned dependencies
├── README.md             ← Main documentation
├── AMPLITUDE_HUNTER_README.md
├── OPTIMIZATION_PRESETS_README.md
└── AGENT_INSTRUCTIONS.md ← This file (LLM guidance)
```

---

## Configuration

### paths.toml (Critical for Data & GPU)
```toml
[paths]
data_root = "./data"
indicators = "{data_root}/indicateurs_data_parquet"
cache = "./cache"

[gpu]
enable_gpu = true
devices = ["5090", "2060"]
load_balance = {"5090" = 0.75, "2060" = 0.25}

[trading]
default_timeframe = "1h"
supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

[backtesting]
initial_capital = 10000.0
max_positions = 10
fee_rate = 0.001
```

### Environment Variables
- `THREADX_DATA_DIR`: Override data folder location
- GPU detection: Automatic via CuPy (CUDA 12.x)

---

## Common Tasks for LLMs

### Task: Add a New Indicator
1. Implement computation in `indicators/indicators_np.py` or `indicators/bollinger.py`
2. Register in `indicators/bank.py:_INDICATOR_REGISTRY`
3. Add unit tests in `tests/`
4. Verify caching works: `IndicatorBank.compute_*()` should cache results

### Task: Add a New Strategy
1. Create file in `strategy/` (e.g., `strategy/my_strategy.py`)
2. Inherit from `Strategy` base or follow protocol
3. Register in `ui/strategy_registry.py`
4. Test with `BacktestEngine`

### Task: Optimize a Strategy
1. Use `optimization/engine.py:SweepRunner`
2. Define parameter ranges in `optimization/scenarios.py`
3. Run: `SweepRunner.run(strategy, param_grids)`
4. Review results with `optimization/reporting.py`

### Task: Debug GPU Issues
1. Check detection: `gpu/device_manager.py:DeviceManager.detect_gpu()`
2. Check allocation: `gpu/vector_checks.py`
3. Fallback automatically to NumPy via `utils/xp.py`

---

## Code Style & Conventions

- **Type Hints**: Use throughout (checked with mypy via setup.cfg)
- **Imports**: Use `common_imports.py` to reduce duplication
- **Logging**: Use `utils/log.py` for structured logging
- **Caching**: Use `utils/cache.py` for generic LRU/TTL caches
- **Determinism**: Use `utils/determinism.py` to set seeds
- **Line Length**: 120 characters (configured in .flake8, setup.cfg)
- **Docstrings**: Google-style with type hints

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific marker
pytest tests/ -m "not slow"

# With coverage
pytest tests/ --cov=src/threadx

# Verbose
pytest tests/ -v
```

**Markers** (in pytest.ini):
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.ui`: Streamlit tests

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/threadx/backtest/engine.py` | Core backtest execution (Phase 10, stable) |
| `src/threadx/indicators/bank.py` | Central indicator cache manager |
| `src/threadx/strategy/model.py` | Base strategy & Trade dataclass |
| `src/threadx/optimization/engine.py` | SweepRunner parameter optimization |
| `src/threadx/utils/xp.py` | Device-agnostic NumPy/CuPy |
| `src/threadx/data_access.py` | Data discovery & loading |
| `src/threadx/streamlit_app.py` | Web UI entry point |
| `src/threadx/cli/main.py` | CLI entry point |
| `pyproject.toml` | Dependencies & package metadata |
| `paths.toml` | Configuration (data, GPU, trading) |

---

## Important Notes for LLMs

### ✅ DO:
- Use `utils.xp` for all NumPy/CuPy operations
- Cache indicator computations via `IndicatorBank`
- Use `BacktestEngine` for strategy execution
- Follow existing patterns in similar files
- Add type hints to all functions
- Check `AGENT_INSTRUCTIONS.md` when confused
- Prefer modifying existing files over creating new ones

### ❌ DON'T:
- Import NumPy/CuPy directly (use `xp`)
- Compute indicators outside `IndicatorBank`
- Manually execute strategy trades (use `BacktestEngine`)
- Create new files unless absolutely necessary
- Use relative imports (always absolute from `src/threadx/`)
- Assume GPU is available (always provide CPU fallback)
- Hardcode paths (use `data_access` or `paths.toml`)

---

## Debugging Checklist

**"My code runs differently on GPU vs CPU"**
→ Check `utils/xp.py` is used consistently, not direct NumPy/CuPy imports

**"Indicator computation is slow"**
→ Verify `IndicatorBank` caching is enabled, check disk cache in `cache/` folder

**"Strategy won't load in UI"**
→ Check registration in `strategy_registry.py`, verify it returns list of Trade objects

**"Data won't load"**
→ Check `THREADX_DATA_DIR` env var, verify folder structure, use `discover_tokens_and_timeframes()`

**"Multi-GPU not working"**
→ Check `paths.toml` GPU config, verify CuPy installed, check `gpu/device_manager.py` detection

---

## Version & Dependency Info

- **Python**: 3.12+
- **Streamlit**: 1.49.1
- **Pandas**: 2.1.0
- **NumPy**: 1.26.0
- **CuPy**: 13.6.0 (optional, GPU acceleration)
- **PyTorch**: 2.5.1+cu121 (optional)
- **Typer**: 0.9.0 (CLI)
- **Pydantic**: 2.5.0 (validation)

See `requirements.txt` for complete 164-package dependency list.

---

## Quick Links

- **Main Backtest**: `src/threadx/backtest/engine.py`
- **Indicator System**: `src/threadx/indicators/bank.py`
- **Strategy Base**: `src/threadx/strategy/model.py`
- **Data Access**: `src/threadx/data_access.py`
- **GPU Management**: `src/threadx/gpu/device_manager.py`
- **Web UI**: `src/threadx/streamlit_app.py`
- **CLI**: `src/threadx/cli/main.py`

---

**This document is your LLM cheat sheet. When working on ThreadX, reference this file first.**
