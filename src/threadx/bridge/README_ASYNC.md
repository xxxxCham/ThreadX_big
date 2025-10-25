# ThreadXBridge Async Coordinator - Quick Start

## 🚀 Introduction

ThreadXBridge est l'orchestrateur asynchrone ThreadX pour exécution non-bloquante de backtests, calculs indicateurs, parameter sweeps et validation données.

**Pattern principal**: ThreadPoolExecutor + Queue + Polling pour UI Dash responsive.

---

## 📦 Installation

Aucune dépendance externe requise (utilise stdlib Python uniquement).

```bash
# Imports depuis ThreadX workspace
from threadx.bridge import ThreadXBridge, BacktestRequest
```

---

## 💡 Usage Rapide

### Dash UI (Polling - Non-bloquant)

```python
from threadx.bridge import ThreadXBridge, BacktestRequest

# Initialize
bridge = ThreadXBridge(max_workers=4)

# Dash submit callback
@app.callback(...)
def submit_backtest(n_clicks, symbol, timeframe, strategy):
    req = BacktestRequest(symbol=symbol, timeframe=timeframe, ...)
    bridge.run_backtest_async(req)  # Retour IMMEDIAT
    return "Task submitted"

# Dash polling callback (500ms)
@app.callback(Input("interval", "n_intervals"))
def poll_results(n):
    event = bridge.get_event(timeout=0.1)  # Non-bloquant

    if event and event['type'] == 'backtest_done':
        result = event['payload']
        fig = plot_equity(result.equity_curve)
        return fig

    return dash.no_update
```

**Avantages**:
- UI reste responsive pendant calculs
- Polling léger (100ms timeout)
- Pas de callbacks complexes

---

### CLI (Sync Future - Bloquant)

```python
from threadx.bridge import ThreadXBridge, BacktestRequest

# Initialize
bridge = ThreadXBridge()

# Submit + wait
req = BacktestRequest(symbol='BTCUSDT', timeframe='1h', ...)
future = bridge.run_backtest_async(req)
result = future.result(timeout=300)  # BLOQUE jusqu'à résultat

print(f"Sharpe: {result.sharpe_ratio:.2f}")

# Cleanup
bridge.shutdown(wait=True)
```

**Avantages**:
- API simple (1 ligne submit + 1 ligne wait)
- Timeout configurable
- Callback optionnel pour monitoring

---

## 📚 API Complète

### Méthodes Async (Submit)

```python
# Backtest
future = bridge.run_backtest_async(
    req: BacktestRequest,
    callback: Callable[[BacktestResult | None, Exception | None], None] | None = None,
    task_id: str | None = None
) -> Future[BacktestResult]

# Indicators
future = bridge.run_indicator_async(req: IndicatorRequest, ...) -> Future[IndicatorResult]

# Sweep
future = bridge.run_sweep_async(req: SweepRequest, ...) -> Future[SweepResult]

# Data Validation
future = bridge.validate_data_async(req: DataRequest, ...) -> Future[DataValidationResult]
```

### Polling & State

```python
# Polling événements (non-bloquant)
event = bridge.get_event(timeout: float = 0.1) -> Dict[str, Any] | None
# Returns: {"type": "backtest_done", "task_id": "abc123", "payload": BacktestResult(...)}
#      or: {"type": "error", "task_id": "abc123", "payload": "Error message"}
#      or: None (si queue vide)

# State Bridge
state = bridge.get_state() -> Dict[str, Any]
# Returns: {
#     "active_tasks": 2,
#     "queue_size": 1,
#     "max_workers": 4,
#     "total_submitted": 10,
#     "total_completed": 8,
#     "total_failed": 0,
#     "xp_layer": "numpy"
# }

# Cancel task
cancelled = bridge.cancel_task(task_id: str) -> bool
```

### Lifecycle

```python
# Initialize
bridge = ThreadXBridge(max_workers: int = 4, config: Configuration | None = None)

# Shutdown
bridge.shutdown(wait: bool = True, timeout: float | None = None)
```

---

## 🎓 Exemples Complets

### Dash UI Example
```bash
python examples/async_bridge_dash_example.py
# Accès: http://localhost:8050
```

Features:
- Submit backtest via bouton
- Polling 500ms dcc.Interval
- Graph equity curve Plotly
- Métriques performance

### CLI Example
```bash
python examples/async_bridge_cli_example.py BTCUSDT 1h bb --workers 8 --timeout 300
```

Features:
- Arguments CLI
- Callback monitoring
- Pretty-print résultats
- Timeout configurable

---

## 🔧 Configuration

```python
from threadx.bridge import Configuration, ThreadXBridge

config = Configuration(
    max_workers=8,              # Nombre workers parallèles
    validate_requests=True,     # Validation requêtes avant submit
    xp_layer="numpy",           # Backend calcul ("numpy" | "cupy")
)

bridge = ThreadXBridge(max_workers=8, config=config)
```

---

## 🧵 Thread-Safety

ThreadXBridge est **100% thread-safe**:
- `active_tasks` protégé par `state_lock`
- `Queue` nativement thread-safe
- `Future` nativement thread-safe

✅ **Pas de data races**
✅ **Safe pour Dash multi-callbacks**
✅ **Safe pour CLI multi-threads**

---

## 🐛 Error Handling

### Pattern Polling (Dash)
```python
event = bridge.get_event()
if event and event['type'] == 'error':
    error_msg = event['payload']
    display_error(error_msg)  # UI feedback
```

### Pattern Sync (CLI)
```python
try:
    result = future.result(timeout=300)
except TimeoutError:
    print("Timeout!")
except BridgeError as e:
    print(f"Bridge error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Callbacks
```python
def on_complete(result: BacktestResult | None, error: Exception | None):
    if error:
        logger.error(f"Task failed: {error}")
    else:
        logger.info(f"Task success: {result.sharpe_ratio}")

bridge.run_backtest_async(req, callback=on_complete)
```

---

## 📊 Monitoring

### State Monitoring
```python
state = bridge.get_state()
print(f"Active: {state['active_tasks']}")
print(f"Queue: {state['queue_size']}")
print(f"Completed: {state['total_completed']}")
print(f"Failed: {state['total_failed']}")
```

### Task Tracking
```python
# Submit
future = bridge.run_backtest_async(req)
task_id = list(bridge.active_tasks.keys())[-1]

# Check status
if task_id in bridge.active_tasks:
    print("Still running...")
else:
    print("Completed!")

# Cancel if needed
bridge.cancel_task(task_id)
```

---

## 🚦 Best Practices

### Dash UI
1. **Polling interval**: 500ms recommandé (balance réactivité/overhead)
2. **Timeout get_event()**: 0.1s (100ms) pour non-bloquant
3. **Store task_id**: utiliser dcc.Store pour tracking
4. **Error feedback**: afficher erreurs UI clairement

### CLI
1. **Timeout result()**: configurer selon durée attendue (300s backtest, 600s sweep)
2. **Callbacks**: utiliser pour monitoring long calculs
3. **Shutdown**: toujours appeler `bridge.shutdown(wait=True)` en cleanup

### Général
1. **Workers**: adapter selon CPU (4-8 recommandé)
2. **Validation**: activer `validate_requests=True` en dev
3. **Logging**: configurer niveau selon environnement (DEBUG dev, INFO prod)

---

## 📖 Documentation Complète

- **API Reference**: `docs/PROMPT3_DELIVERY_REPORT.md`
- **Examples**: `examples/async_bridge_*_example.py`
- **Source Code**: `src/threadx/bridge/async_coordinator.py`

---

## 🆘 Troubleshooting

### Import Error
```python
# ❌ ModuleNotFoundError: No module named 'threadx.bridge'
# ✅ Vérifier PYTHONPATH ou run depuis workspace root
cd /path/to/ThreadX
python -c "from threadx.bridge import ThreadXBridge"
```

### Queue Vide (Pas d'événements)
```python
# Vérifier state
state = bridge.get_state()
print(state)  # active_tasks=0 ? queue_size=0 ?

# Augmenter timeout
event = bridge.get_event(timeout=0.5)  # 500ms au lieu de 100ms
```

### Timeout Result
```python
# Augmenter timeout selon durée calcul
result = future.result(timeout=600)  # 10 minutes pour sweeps longs
```

---

## 📝 Changelog

### v0.1.0 (PROMPT 3)
- ✅ Initial release
- ✅ ThreadPoolExecutor + Queue + Lock
- ✅ 4 async methods (backtest, indicator, sweep, data)
- ✅ Polling + state + cancellation
- ✅ Dash + CLI examples
- ✅ Full thread-safety

---

**Questions?** Voir `docs/PROMPT3_DELIVERY_REPORT.md` pour détails complets.
