# ThreadX - Framework Trading Algorithmique

**Framework Python professionnel pour backtesting, optimisation et déploiement de stratégies trading.**

Version: **2025.11.21**  
Python: **3.11+**  
License: Propriétaire

---

## 🚨 IMPORTANT

**👉 Lire obligatoirement:** [`DIRECTIVES_DEV.md`](DIRECTIVES_DEV.md)

Ce fichier centralise **TOUTES** les instructions:
- ✅ Règles consolidation code
- ✅ Architecture générale
- ✅ Conventions nommage
- ✅ Stack technologique
- ✅ Info Netdata MCP Bridge
- ✅ Checklist qualité code

---

## 🎯 QUICKSTART

### Installation
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

### Premier Backtest
```python
from src.threadx.strategy.ma_crossover import MACrossover, MACrossoverParams
import pandas as pd

strategy = MACrossover()
equity, stats = strategy.backtest(
    df=df_1min,
    params=MACrossoverParams(fast_period=10, slow_period=30),
    initial_capital=10000.0
)
print(f"Return: {stats['total_return']:.2f}%")
```

---

## 📊 FRICTIONS RÉALISTES

Backtests SANS frictions = +200% trop optimistes!

**Solution:** `RealisticExecutor` dans `src/threadx/backtest/engine.py`

```python
executor = RealisticExecutor(timeframe="1m", symbol="BTCUSDT")
result = executor.execute_order(
    side="BUY",
    intended_price=50000.0,
    quantity=0.5,
    current_volatility=0.015
)
```

---

## 🌐 NETDATA MCP BRIDGE

Outil monitoring en Go (SÉPARÉ du trading):

```bash
cd tools/netdata-bridge
./build.sh
./nd-mcp ws://localhost:19999/mcp
```

---

## 📚 DOCUMENTATION

- **DIRECTIVES_DEV.md** ← À LIRE EN PRIORITÉ
- **src/threadx/** → Docstrings dans le code
- **tools/** → Scripts développement
- **tests/** → Exemples usage

---

**Avant de coder, lire** [`DIRECTIVES_DEV.md`](DIRECTIVES_DEV.md)
