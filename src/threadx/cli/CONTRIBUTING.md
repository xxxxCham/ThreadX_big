# Contributing to ThreadX CLI

Guide pour contribuer au module CLI de ThreadX.

---

## 🎯 Architecture

### Structure
```
src/threadx/cli/
├── __init__.py          # Module exports
├── __main__.py          # Entry point (python -m threadx.cli)
├── main.py              # Typer app + options globales + version
├── utils.py             # Fonctions partagées (logging, JSON, async, format)
└── commands/
    ├── __init__.py      # Agrégateur de commandes
    ├── data_cmd.py      # Commandes data (validate, list)
    ├── indicators_cmd.py # Commandes indicators (build, cache)
    ├── backtest_cmd.py  # Commandes backtest (run)
    └── optimize_cmd.py  # Commandes optimize (sweep)
```

### Pattern général
```python
# 1. Créer Typer app dans commands/xxx_cmd.py
app = typer.Typer(help="Description du groupe")

# 2. Définir commande
@app.command()
def my_command(
    param1: str = typer.Option(..., help="Description"),
    param2: int = typer.Option(10, help="Description"),
) -> None:
    """
    Docstring de la commande (affiché dans --help).

    Args:
        param1: Description param1
        param2: Description param2
    """
    # 3. Récupérer context pour --json
    ctx = typer.Context.get_current()  # ou injection explicite
    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    # 4. Appeler Bridge
    from threadx.bridge import ThreadXBridge
    bridge = ThreadXBridge()
    request = {"param1": param1, "param2": param2}
    task_id = bridge.run_xxx_async(request)

    # 5. Polling async
    from threadx.cli.utils import async_runner
    event = async_runner(bridge.get_event, task_id, timeout=60.0)

    # 6. Gestion erreur timeout
    if event is None:
        if json_mode:
            print_json({"status": "timeout", "task_id": task_id})
        else:
            typer.echo("⚠️  Command timed out")
        raise typer.Exit(1)

    # 7. Vérifier status
    if event.get("status") == "error":
        error = Exception(event.get("error", "Unknown"))
        handle_bridge_error(error, json_mode)

    # 8. Extraire résultat
    result = event.get("result", {})

    # 9. Afficher résumé
    summary = {
        "param1": param1,
        "param2": param2,
        "output": result.get("output"),
    }
    print_summary("My Command Results", summary, json_mode)

# 10. Enregistrer dans main.py
from .commands import xxx_cmd
app.add_typer(xxx_cmd.app, name="xxx")
```

---

## 🔧 Ajouter une nouvelle commande

### 1. Créer le fichier de commande

**Fichier**: `src/threadx/cli/commands/new_cmd.py`

```python
"""
ThreadX CLI - New Commands
===========================

Description du groupe de commandes.

Usage:
    python -m threadx.cli new action --option value

Author: Your Name
Version: 1.x.x
"""

import logging
from typing import Optional

import typer

from threadx.cli.utils import (
    async_runner,
    format_duration,
    handle_bridge_error,
    print_json,
    print_summary,
)

app = typer.Typer(help="Description du groupe")
logger = logging.getLogger("threadx.cli.new")


@app.command()
def action(
    required_param: str = typer.Option(..., help="Required parameter"),
    optional_param: int = typer.Option(10, "--opt", help="Optional parameter"),
) -> None:
    """
    Action description (shown in --help).

    Detailed description of what the command does,
    its inputs, outputs, and side effects.

    Args:
        required_param: Description of required parameter.
        optional_param: Description of optional parameter.
    """
    try:
        from threadx.bridge import ThreadXBridge
    except ImportError as e:
        logger.error(f"Failed to import ThreadXBridge: {e}")
        typer.echo("❌ Bridge not available. Check installation.")
        raise typer.Exit(1)

    ctx = typer.Context.get_current()
    json_mode = ctx.obj.get("json", False) if ctx.obj else False

    logger.info(f"Running action: {required_param}")

    try:
        # Initialize Bridge
        bridge = ThreadXBridge()

        # Prepare request
        request = {
            "required_param": required_param,
            "optional_param": optional_param,
        }

        # Submit async task
        task_id = bridge.run_new_action_async(request)
        logger.debug(f"Task submitted: {task_id}")

        if not json_mode:
            typer.echo(f"⏳ Processing {required_param}...")

        # Poll for results
        event = async_runner(bridge.get_event, task_id, timeout=60.0)

        if event is None:
            error_msg = "Action timed out"
            if json_mode:
                print_json({"status": "timeout", "task_id": task_id})
            else:
                typer.echo(f"⚠️  {error_msg}")
            raise typer.Exit(1)

        # Check status
        if event.get("status") == "error":
            error = Exception(event.get("error", "Unknown error"))
            handle_bridge_error(error, json_mode)

        # Extract result
        result = event.get("result", {})
        duration = event.get("duration", 0)

        # Prepare summary
        summary = {
            "required_param": required_param,
            "optional_param": optional_param,
            "output": result.get("output"),
            "execution_time": format_duration(duration),
        }

        # Output
        if json_mode:
            output_data = {
                "status": "success",
                "summary": summary,
                "details": result,
            }
            print_json(output_data)
        else:
            print_summary("Action Results", summary)

    except Exception as e:
        handle_bridge_error(e, json_mode)


if __name__ == "__main__":
    app()
```

### 2. Enregistrer dans commands/__init__.py

```python
from . import data_cmd, indicators_cmd, backtest_cmd, optimize_cmd, new_cmd

__all__ = [
    "data_cmd",
    "indicators_cmd",
    "backtest_cmd",
    "optimize_cmd",
    "new_cmd",  # Ajouter ici
]
```

### 3. Enregistrer dans main.py

```python
from .commands import (
    backtest_cmd,
    data_cmd,
    indicators_cmd,
    new_cmd,      # Import
    optimize_cmd,
)

# Add subcommands
app.add_typer(data_cmd.app, name="data")
app.add_typer(indicators_cmd.app, name="indicators")
app.add_typer(backtest_cmd.app, name="backtest")
app.add_typer(optimize_cmd.app, name="optimize")
app.add_typer(new_cmd.app, name="new")  # Enregistrer
```

### 4. Tester

```bash
# Aide
python -m threadx.cli new --help
python -m threadx.cli new action --help

# Exécution
python -m threadx.cli new action --required-param value

# Mode JSON
python -m threadx.cli --json new action --required-param value

# Mode debug
python -m threadx.cli --debug new action --required-param value
```

---

## 🛠️ Fonctions utilitaires (utils.py)

### setup_logger(level: int)
Configure le logging avec format timestamp.

```python
from threadx.cli.utils import setup_logger
import logging

setup_logger(logging.DEBUG)  # Active debug
logger = logging.getLogger("threadx.cli.mycommand")
logger.debug("Debug message")
```

### print_json(data: dict, indent: int = 2)
Sérialisation JSON sécurisée.

```python
from threadx.cli.utils import print_json

data = {"status": "success", "value": 42}
print_json(data)  # {"status": "success", "value": 42}
```

### async_runner(func, task_id, timeout=60.0, poll_interval=0.5)
**Fonction clé**: Polling non bloquant pour Bridge.get_event().

```python
from threadx.cli.utils import async_runner

# Submit task
task_id = bridge.run_action_async(request)

# Poll for completion (0.5s interval, 60s timeout)
event = async_runner(bridge.get_event, task_id, timeout=60.0)

if event is None:
    # Timeout
    typer.echo("⚠️  Timeout")
    raise typer.Exit(1)

if event.get("status") == "error":
    # Error
    error = Exception(event.get("error"))
    handle_bridge_error(error, json_mode)

# Success
result = event.get("result")
```

**Paramètres**:
- `func`: Fonction à appeler (bridge.get_event)
- `task_id`: ID de la tâche Bridge
- `timeout`: Timeout total en secondes (défaut 60.0)
- `poll_interval`: Intervalle de polling en secondes (défaut 0.5)

**Retour**:
- `event` (dict) si succès
- `None` si timeout

### format_duration(seconds: float) -> str
Formatage temps lisible.

```python
from threadx.cli.utils import format_duration

print(format_duration(75.3))  # "1m 15.3s"
print(format_duration(3665.2))  # "1h 1m 5.2s"
```

### print_summary(title: str, data: dict, json_mode: bool)
Affichage tableau texte OU JSON.

```python
from threadx.cli.utils import print_summary

summary = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "total_return": "23.45%",
}

# Mode texte
print_summary("Backtest Results", summary, json_mode=False)
# → Table formatée

# Mode JSON
print_summary("Backtest Results", summary, json_mode=True)
# → {"status": "success", "summary": {...}}
```

### handle_bridge_error(error: Exception, json_mode: bool)
Gestion erreurs centralisée + exit(1).

```python
from threadx.cli.utils import handle_bridge_error

try:
    result = bridge.run_action(request)
except Exception as e:
    handle_bridge_error(e, json_mode)  # Exit automatique
```

---

## 📝 Conventions de code

### Docstrings
```python
def my_function(param1: str, param2: int = 10) -> dict:
    """
    Short description (one line).

    Detailed description explaining what the function does,
    its behavior, side effects, and usage examples.

    Args:
        param1: Description of param1.
        param2: Description of param2 (default: 10).

    Returns:
        Dictionary with keys: key1, key2, key3.

    Raises:
        ValueError: If param1 is empty.
        TimeoutError: If operation exceeds timeout.

    Examples:
        >>> my_function("value", 20)
        {'key1': 'value', 'key2': 20}
    """
    pass
```

### Type Hints
```python
from typing import Optional, List, Dict

def my_command(
    required: str,
    optional: Optional[int] = None,
    items: List[str] = [],
) -> Dict[str, any]:
    pass
```

### Logging
```python
import logging

logger = logging.getLogger("threadx.cli.mycommand")

logger.debug("Debug information")   # --debug only
logger.info("Informational message")  # Always
logger.warning("Warning message")     # Always
logger.error("Error message")         # Always
```

### Error Handling
```python
try:
    # Bridge call
    result = bridge.run_action(request)
except ImportError as e:
    logger.error(f"Import error: {e}")
    typer.echo("❌ Module not found")
    raise typer.Exit(1)
except Exception as e:
    handle_bridge_error(e, json_mode)
```

---

## 🧪 Tests

### Test manuel
```bash
# Test aide
python -m threadx.cli mycommand --help

# Test exécution normale
python -m threadx.cli mycommand --param value

# Test mode JSON
python -m threadx.cli --json mycommand --param value

# Test mode debug
python -m threadx.cli --debug mycommand --param value

# Test erreur (param manquant)
python -m threadx.cli mycommand  # Devrait afficher erreur

# Test timeout (si applicable)
python -m threadx.cli mycommand --param large_dataset  # Timeout >60s
```

### Test automatisé (futur)
```python
# tests/cli/test_mycommand.py
from typer.testing import CliRunner
from threadx.cli.main import app

runner = CliRunner()

def test_mycommand_help():
    result = runner.invoke(app, ["mycommand", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout

def test_mycommand_success():
    result = runner.invoke(app, ["mycommand", "--param", "value"])
    assert result.exit_code == 0
    assert "success" in result.stdout.lower()

def test_mycommand_json():
    result = runner.invoke(app, ["--json", "mycommand", "--param", "value"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["status"] == "success"
```

---

## 📊 Checklist avant commit

### Code
- [ ] Docstrings complets (Google style)
- [ ] Type hints sur toutes les fonctions publiques
- [ ] Logging (pas de `print` pour debug)
- [ ] Error handling avec `handle_bridge_error`
- [ ] Support `--json` mode
- [ ] Timeout approprié pour la commande

### Documentation
- [ ] Docstring de la commande (affiché dans `--help`)
- [ ] Exemples dans README.md
- [ ] Entry dans CHANGELOG.md
- [ ] Update commands/__init__.py
- [ ] Update main.py (app.add_typer)

### Tests
- [ ] Test manuel: `--help`
- [ ] Test manuel: exécution normale
- [ ] Test manuel: mode `--json`
- [ ] Test manuel: mode `--debug`
- [ ] Test manuel: gestion erreur

### Qualité
- [ ] Lint: 0 erreurs fonctionnelles
- [ ] Pattern cohérent avec commandes existantes
- [ ] Code DRY (pas de duplication)
- [ ] Logs appropriés (DEBUG, INFO, ERROR)

---

## 🔗 Ressources

### Documentation
- [Typer Documentation](https://typer.tiangolo.com/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [ThreadXBridge API](../bridge/__init__.py)

### Exemples
- [data_cmd.py](./commands/data_cmd.py) - Simple commands (validate, list)
- [backtest_cmd.py](./commands/backtest_cmd.py) - Complex command (many options)
- [optimize_cmd.py](./commands/optimize_cmd.py) - Long-running command (10min timeout)

### Fichiers clés
- [main.py](./main.py) - Entry point + version
- [utils.py](./utils.py) - Shared utilities
- [README.md](./README.md) - Usage guide

---

## 💡 Tips

### Timeout
Adapter le timeout selon la complexité de la commande:
- Validation rapide: 30s
- Build indicateurs: 120s
- Backtest simple: 300s
- Optimize sweep: 600s

### Options
Utiliser `typer.Option` pour tous les paramètres (meilleure aide):
```python
# ✅ Bon
symbol: str = typer.Option(..., help="Symbol to test")

# ❌ Éviter
symbol: str  # Pas d'aide
```

### Logging
Logger approprié selon le niveau:
```python
logger.debug(f"Task ID: {task_id}")       # Détails techniques
logger.info(f"Processing {symbol}...")     # Progression
logger.warning("Timeout may occur")        # Avertissements
logger.error(f"Failed: {error}")          # Erreurs
```

### JSON Mode
Toujours supporter `--json` pour intégration scripts:
```python
if json_mode:
    print_json({"status": "success", "data": result})
else:
    print_summary("Results", summary)
```

---

**Bonne contribution !** 🚀
