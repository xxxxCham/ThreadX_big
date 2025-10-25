# ThreadX CLI

Interface en ligne de commande pour ThreadX - Framework de backtesting GPU-accéléré

---

## 🚀 Installation rapide

```bash
# Installer dépendances
pip install typer rich

# Vérifier installation
python -m threadx.cli --help
```

---

## 📖 Usage

### Commande de base

```bash
python -m threadx.cli [OPTIONS] COMMAND [ARGS]
```

### Options globales

| Option | Description |
|--------|-------------|
| `--json` | Sortie en format JSON au lieu de texte |
| `--debug` | Active le logging détaillé |
| `--async` | Mode exécution parallèle (expérimental) |
| `--help` | Affiche l'aide |

---

## 📋 Commandes disponibles

### 1. Data - Gestion des datasets

#### Valider un dataset
```bash
python -m threadx.cli data validate <path> [OPTIONS]
```

**Options**:
- `--symbol TEXT`: Symbole du dataset (ex: BTCUSDT)
- `--timeframe TEXT`: Timeframe (1m, 5m, 1h, 1d)

**Exemple**:
```bash
python -m threadx.cli data validate ./data/BTCUSDT_1h.csv \
  --symbol BTCUSDT \
  --timeframe 1h
```

**Sortie**:
```
Data Validation Results:
  path            : ./data/BTCUSDT_1h.csv
  symbol          : BTCUSDT
  timeframe       : 1h
  rows            : 8640
  columns         : 6
  date_range      : 2023-01-01 to 2024-01-01
  quality_score   : 98.5%
  validation_time : 1.2s
```

#### Lister les datasets enregistrés
```bash
python -m threadx.cli data list
```

**Sortie**:
```
Registered Datasets:
Symbol    Timeframe  Rows   Date Range                Status
------------------------------------------------------------
BTCUSDT   1h         8640   2023-01-01 to 2024-01-01  validated
ETHUSDT   1d         365    2023-01-01 to 2024-01-01  validated

Total: 2 datasets
```

---

### 2. Indicators - Indicateurs techniques

#### Construire des indicateurs
```bash
python -m threadx.cli indicators build [OPTIONS]
```

**Options**:
- `--symbol TEXT`: Symbole (requis)
- `--tf TEXT`: Timeframe (défaut: 1h)
- `--ema-period INT`: Période EMA (défaut: 20)
- `--rsi-period INT`: Période RSI (défaut: 14)
- `--bollinger-period INT`: Période Bollinger (défaut: 20)
- `--bollinger-std FLOAT`: Écart-type Bollinger (défaut: 2.0)
- `--force`: Force la reconstruction (ignore le cache)

**Exemple**:
```bash
python -m threadx.cli indicators build \
  --symbol BTCUSDT \
  --tf 1h \
  --ema-period 20 \
  --rsi-period 14 \
  --bollinger-period 20 \
  --bollinger-std 2.0
```

**Sortie**:
```
Indicators Build:
  symbol            : BTCUSDT
  timeframe         : 1h
  indicators_built  : EMA_20, RSI_14, BB_20_2.0
  rows_processed    : 8640
  cache_size_mb     : 12.5
  build_time        : 3.4s
```

#### Voir le cache d'indicateurs
```bash
python -m threadx.cli indicators cache
```

**Sortie**:
```
Indicators Cache:
Symbol    Timeframe  Indicators            Size (MB)  Updated
--------------------------------------------------------------
BTCUSDT   1h         EMA_20, RSI_14, BB    12.5       2024-01-15 10:30
ETHUSDT   1d         EMA_50, RSI_21        8.2        2024-01-15 09:45

Total cache: 20.7 MB
```

---

### 3. Backtest - Exécution de backtests

#### Exécuter un backtest
```bash
python -m threadx.cli backtest run [OPTIONS]
```

**Options**:
- `--strategy TEXT`: Nom de la stratégie (requis)
- `--symbol TEXT`: Symbole à tester (requis)
- `--tf TEXT`: Timeframe (défaut: 1h)
- `--period INT`: Paramètre période stratégie
- `--std FLOAT`: Paramètre écart-type stratégie
- `--start-date TEXT`: Date début (YYYY-MM-DD)
- `--end-date TEXT`: Date fin (YYYY-MM-DD)
- `--initial-capital FLOAT`: Capital initial USD (défaut: 10000)

**Exemple**:
```bash
python -m threadx.cli backtest run \
  --strategy ema_crossover \
  --symbol BTCUSDT \
  --tf 1h \
  --period 20 \
  --initial-capital 10000
```

**Sortie**:
```
Backtest Results:
  strategy        : ema_crossover
  symbol          : BTCUSDT
  timeframe       : 1h
  total_trades    : 45
  win_rate        : 62.22%
  total_return    : 23.45%
  sharpe_ratio    : 1.85
  max_drawdown    : -12.34%
  profit_factor   : 2.15
  final_equity    : 12345.67
  execution_time  : 15.2s

📊 Top 3 Best Trades:
  1. $345.20 (+3.45%) - 2023-03-15
  2. $289.10 (+2.89%) - 2023-06-22
  3. $234.50 (+2.35%) - 2023-09-10

📉 Top 3 Worst Trades:
  1. -$156.80 (-1.57%) - 2023-05-12
  2. -$123.40 (-1.23%) - 2023-08-05
  3. -$98.70 (-0.99%) - 2023-11-20
```

---

### 4. Optimize - Optimisation de paramètres

#### Lancer un sweep de paramètres
```bash
python -m threadx.cli optimize sweep [OPTIONS]
```

**Options**:
- `--strategy TEXT`: Nom de la stratégie (requis)
- `--symbol TEXT`: Symbole à tester (requis)
- `--tf TEXT`: Timeframe (défaut: 1h)
- `--param TEXT`: Paramètre à balayer (requis)
- `--min FLOAT`: Valeur minimale (requis)
- `--max FLOAT`: Valeur maximale (requis)
- `--step FLOAT`: Pas d'incrémentation (défaut: 1.0)
- `--metric TEXT`: Métrique d'optimisation (défaut: sharpe_ratio)
- `--start-date TEXT`: Date début
- `--end-date TEXT`: Date fin
- `--top-n INT`: Nombre de résultats à afficher (défaut: 10)

**Exemple**:
```bash
python -m threadx.cli optimize sweep \
  --strategy bollinger \
  --symbol BTCUSDT \
  --tf 1h \
  --param period \
  --min 10 \
  --max 40 \
  --step 5 \
  --metric sharpe_ratio
```

**Sortie**:
```
Optimization Sweep Results:
  strategy             : bollinger
  symbol               : BTCUSDT
  timeframe            : 1h
  parameter            : period
  range                : [10, 40] (step=5)
  tests_run            : 7
  optimization_metric  : sharpe_ratio
  best_param_value     : 25
  best_sharpe_ratio    : 2.15
  execution_time       : 2m 15.3s

🏆 Top 7 Results (ranked by sharpe_ratio):

Rank   Period          Sharpe Ratio         Total Return    Win Rate
----------------------------------------------------------------------
1      25              2.1500               28.45%          65.5%
2      20              2.0800               26.30%          63.2%
3      30              1.9500               24.10%          61.8%
4      15              1.7200               20.50%          58.9%
5      35              1.6800               19.80%          57.5%
6      10              1.5500               18.20%          55.1%
7      40              1.4200               16.50%          53.4%
```

---

### 5. Version - Information version

```bash
python -m threadx.cli version
```

**Sortie**:
```
ThreadX CLI v1.0.0
Prompt: P9 - CLI Bridge Interface
Python: 3.12.10
```

---

## 🔧 Mode JSON

Toutes les commandes supportent l'output JSON via l'option globale `--json`:

```bash
python -m threadx.cli --json data list
```

**Sortie**:
```json
{
  "status": "success",
  "datasets": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "rows": 8640,
      "date_range": "2023-01-01 to 2024-01-01",
      "status": "validated"
    }
  ]
}
```

Utile pour:
- Intégration dans scripts (Python, bash, PowerShell)
- Pipelines CI/CD
- Parsing automatique des résultats

---

## 🐛 Mode Debug

Activer le logging détaillé:

```bash
python -m threadx.cli --debug indicators build --symbol BTCUSDT --tf 1h
```

**Logs affichés**:
```
2024-01-15 10:23:45 - threadx.cli - DEBUG - CLI initialized: json=False, debug=True
2024-01-15 10:23:45 - threadx.cli.indicators - INFO - Building indicators: BTCUSDT @ 1h
2024-01-15 10:23:45 - threadx.cli.indicators - DEBUG - Task submitted: task_abc123
2024-01-15 10:23:47 - threadx.cli.utils - DEBUG - Polling event (attempt 1/120)
...
```

---

## 🔄 Workflow typique

### 1. Valider dataset
```bash
python -m threadx.cli data validate ./data/BTCUSDT_1h.csv \
  --symbol BTCUSDT --timeframe 1h
```

### 2. Construire indicateurs
```bash
python -m threadx.cli indicators build \
  --symbol BTCUSDT --tf 1h
```

### 3. Backtest simple
```bash
python -m threadx.cli backtest run \
  --strategy ema_crossover \
  --symbol BTCUSDT --tf 1h
```

### 4. Optimiser paramètres
```bash
python -m threadx.cli optimize sweep \
  --strategy ema_crossover \
  --symbol BTCUSDT \
  --param period --min 5 --max 50 --step 5
```

### 5. Backtest avec params optimaux
```bash
python -m threadx.cli backtest run \
  --strategy ema_crossover \
  --symbol BTCUSDT --tf 1h \
  --period 25  # Valeur optimale du sweep
```

---

## 📊 Stratégies disponibles

| Stratégie | Description | Paramètres |
|-----------|-------------|------------|
| `ema_crossover` | Croisement EMA rapide/lente | `period` (EMA rapide) |
| `bollinger_reversion` | Retour à la moyenne Bollinger | `period`, `std` |
| `rsi_oversold` | RSI oversold/overbought | `period` (RSI), `oversold`, `overbought` |

---

## 🎯 Conseils d'utilisation

### Optimisation de paramètres
1. **Commencer large**: `--min 10 --max 50 --step 10` (rapide)
2. **Affiner**: `--min 20 --max 30 --step 2` (précis)
3. **Valider**: Backtest avec params optimaux sur période différente

### Gestion du cache
- `--force`: Utiliser si données source modifiées
- `cache`: Surveiller taille cache (supprimer si >1GB)

### Performance
- Datasets < 10k rows: rapide (< 5s)
- Datasets > 100k rows: patient (> 30s)
- Optimize sweeps: peut prendre plusieurs minutes

---

## 📝 Scripts d'exemple

### Bash - Backtest multiple symboles
```bash
#!/bin/bash
for symbol in BTCUSDT ETHUSDT BNBUSDT; do
  echo "Testing $symbol..."
  python -m threadx.cli backtest run \
    --strategy ema_crossover \
    --symbol $symbol --tf 1h \
    --period 20
done
```

### PowerShell - Export JSON
```powershell
# Backtest + sauvegarde JSON
python -m threadx.cli --json backtest run `
  --strategy ema_crossover `
  --symbol BTCUSDT --tf 1h `
  | Out-File -FilePath results.json

# Parse JSON
$results = Get-Content results.json | ConvertFrom-Json
Write-Host "Sharpe Ratio: $($results.summary.sharpe_ratio)"
```

### Python - Automatisation
```python
import subprocess
import json

# Run backtest
result = subprocess.run(
    ["python", "-m", "threadx.cli", "--json", "backtest", "run",
     "--strategy", "ema_crossover", "--symbol", "BTCUSDT", "--tf", "1h"],
    capture_output=True, text=True
)

# Parse results
data = json.loads(result.stdout)
sharpe = data["summary"]["sharpe_ratio"]
print(f"Sharpe Ratio: {sharpe}")
```

---

## 🔗 Ressources

- [PROMPT9_DELIVERY_REPORT.md](../docs/PROMPT9_DELIVERY_REPORT.md) - Rapport complet
- [PROMPT9_SUMMARY.md](../docs/PROMPT9_SUMMARY.md) - Résumé rapide
- [ThreadXBridge API](../src/threadx/bridge/__init__.py) - Documentation Bridge

---

## 🐛 Troubleshooting

### Erreur: Module non trouvé
```bash
# Solution: Vérifier PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/ThreadX/src"  # Linux/Mac
$env:PYTHONPATH = "D:\ThreadX\src"                      # Windows
```

### Erreur: Import typer non résolu
```bash
# Solution: Installer dépendances
pip install typer rich
```

### Timeout sur commandes
```bash
# Solution: Augmenter timeout (modifier utils.py)
# ou utiliser dataset plus petit
```

---

## 📄 Licence

ThreadX CLI - Partie du framework ThreadX
Auteur: ThreadX Framework
Version: 1.0.0

---

**Prêt à backtester !** 🚀
