# ThreadX Data Ingestion - Guide de Démarrage Rapide

## Installation et Configuration

### 1. Prérequis
```bash
# Python 3.12+ requis
python --version

# Dépendances principales
pip install pandas numpy requests toml pathlib
```

### 2. Configuration
Modifier `paths.toml` si nécessaire :
```toml
[data]
base_data_path = "data"           # Répertoire données
processed_data_path = "data/processed"

[data.api]  
base_url = "https://api.binance.com"
rate_limit_calls = 1200           # Limite API Binance
```

## Utilisation Rapide

### Mode programmation

```python
from threadx.config import get_settings
from threadx.data.ingest import IngestionManager

# 1. Initialisation
settings = get_settings()
manager = IngestionManager(settings)

# 2. Téléchargement simple (1m truth)
df_1m = manager.download_ohlcv_1m("BTCUSDC", "2024-01-01", "2024-01-31")
print(f"Téléchargé: {len(df_1m)} chandeliers 1m")

# 3. Resample vers autres timeframes
df_1h = manager.resample_from_1m_api("BTCUSDC", "1h", "2024-01-01", "2024-01-31")
print(f"Resampleé: {len(df_1h)} chandeliers 1h")

# 4. Validation cohérence
valid = manager.verify_resample_consistency(df_1m, df_1h, "1h")
print(f"Cohérence 1m→1h: {valid}")
```

### Mode interface graphique

```bash
# Lancer l'application ThreadX
python run_tkinter.py

# → Aller dans l'onglet "Data Manager"
# → Sélectionner symboles (Ctrl+clic pour multi-sélection)
# → Configurer dates start/end
# → Cliquer "Download Selected"
# → Suivre progression dans les logs
```

### Mode batch (multi-symboles)

```python
# Téléchargement batch avec timeframes multiples
results = manager.update_assets_batch(
    symbols=["BTCUSDC", "ETHUSDC", "ADAUSDC"],
    start_date="2024-01-01",
    end_date="2024-01-31", 
    target_timeframes=["1h", "4h"],
    max_workers=3  # Parallélisation
)

for symbol, status in results.items():
    print(f"{symbol}: {'✓' if status['success'] else '❌'}")
```

## Tests et Validation

### Tests automatisés
```bash
# Tests unitaires complets
python -m pytest tests/test_legacy_adapter.py -v
python -m pytest tests/test_ingest_manager.py -v

# Démonstration fonctionnelle
python demo_data_ingestion.py

# Validation complète du système
python validate_data_ingestion_final.py
```

### Mode dry-run (simulation)
```python
# Test sans téléchargement réel
results = manager.update_assets_batch(
    symbols=["BTCUSDC"],
    start_date="2024-01-01", 
    end_date="2024-01-31",
    target_timeframes=["1h"],
    dry_run=True  # Simulation uniquement
)
```

## Structure des Données

### Principe "1m Truth"
```
🔄 Flux données ThreadX:

API Binance → DataFrame 1m → Sauvegarde → Resample vers 1h/4h/1d
              (source vérité)            (cohérence garantie)
```

### Format DataFrame
```python
# Colonnes standardisées OHLCV
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Index: datetime UTC
# Valeurs: float64 pour prix, volume
```

### Stockage fichiers
```
data/
├── processed/
│   ├── BTCUSDC_1m.parquet    # Source vérité 1m
│   ├── BTCUSDC_1h.parquet    # Resampleé depuis 1m
│   └── BTCUSDC_4h.parquet    # Resampleé depuis 1m
└── raw/
    └── json/                  # Cache API JSON (optionnel)
```

## Cas d'Usage Typiques

### 1. Première installation
```python
# Setup initial complet
symbols = ["BTCUSDC", "ETHUSDC"]
start_date = "2023-01-01"
end_date = "2024-01-31"

for symbol in symbols:
    print(f"Setup {symbol}...")
    df_1m = manager.download_ohlcv_1m(symbol, start_date, end_date)
    df_1h = manager.resample_from_1m_api(symbol, "1h", start_date, end_date) 
    df_4h = manager.resample_from_1m_api(symbol, "4h", start_date, end_date)
    print(f"✓ {symbol}: {len(df_1m)} 1m, {len(df_1h)} 1h, {len(df_4h)} 4h")
```

### 2. Mise à jour quotidienne
```python
from datetime import datetime, timedelta

# Mise à jour dernières 24h
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.now().strftime("%Y-%m-%d")

results = manager.update_assets_batch(
    symbols=["BTCUSDC", "ETHUSDC"],
    start_date=yesterday,
    end_date=today,
    target_timeframes=["1h", "4h"]
)
```

### 3. Validation données existantes
```python
# Vérifier cohérence données stockées
for symbol in ["BTCUSDC", "ETHUSDC"]:
    try:
        df_1m = pd.read_parquet(f"data/processed/{symbol}_1m.parquet")
        df_1h = pd.read_parquet(f"data/processed/{symbol}_1h.parquet")
        
        valid = manager.verify_resample_consistency(df_1m, df_1h, "1h")
        print(f"{symbol} 1m→1h: {'✓' if valid else '❌'}")
        
    except FileNotFoundError:
        print(f"{symbol}: Données manquantes")
```

## Dépannage

### Problèmes courants

**Timeout API**
```python
# Augmenter timeout dans paths.toml
[data.api]
timeout_seconds = 60.0  # Default: 30.0
```

**Rate limiting Binance**
```python
# Réduire fréquence requêtes
[data.api]  
rate_limit_calls = 600   # Default: 1200
rate_limit_window = 60   # Secondes
```

**Mémoire insuffisante**
```python
# Réduire taille batch
[data.processing]
default_batch_size = 2   # Default: 5
```

**Gaps dans les données**
```python
# Détecter et combler gaps
gaps = adapter.detect_gaps_1m(df)
if gaps:
    print(f"Gaps détectés: {len(gaps)}")
    df_filled = adapter.fill_gaps_conservative(df, gaps)
```

### Logs et debugging

```python
import logging

# Debug complet
logging.getLogger('threadx.data').setLevel(logging.DEBUG)

# Ou via configuration
[logging]
level = "DEBUG"
console_output = true
```

## Support

- **Documentation complète** : `docs/DATA_INGESTION_SYSTEM.md`
- **Tests unitaires** : `tests/test_*.py`
- **Démonstration** : `demo_data_ingestion.py`
- **Validation** : `validate_data_ingestion_final.py`

---
*ThreadX Framework - Phase 8 - Système d'ingestion de données complet*