# Intégration ThreadX - unified_data_historique
## Guide d'Intégration pour Éviter les Téléchargements en Doublon

---

## 1. Vue d'ensemble

Le système **UnifiedDataAdapter** permet d'intégrer le programme de téléchargement `unified_data_historique_with_indicators.py` avec les données existantes de ThreadX.

### Objectifs
- ✅ **Lecture pré-téléchargement** : Vérifier les données existantes avant de télécharger
- ✅ **Détection des gaps** : Identifier uniquement les périodes manquantes
- ✅ **Fusion sans doublon** : Combiner nouvelles et existantes via déduplication par timestamp
- ✅ **Format cohérent** : Assurer nomenclature, structure, type identiques
- ✅ **Environment alignment** : Utiliser variables d'environnement pour répertoires

### Architecture

```
unified_data_historique_with_indicators.py
    ↓ (importe)
UnifiedDataAdapter (src/threadx/data/unified_data_adapter.py)
    ↓ (utilise)
DataCompatibilityManager (src/threadx/data/compatibility.py)
    ↓ (gère)
D:\TradXPro\best_token_DataFrame\
    ├── BTCUSDC_1h.json / BTCUSDC_1h.parquet
    ├── ETHUSDC_1h.json / ETHUSDC_1h.parquet
    └── ...
```

---

## 2. Installation et Setup

### 2.1 Variables d'Environnement

Les deux systèmes utilisent la même variable d'environnement :

```bash
# Définir la variable d'environnement (Windows)
set DATA_FOLDER=D:\TradXPro\best_token_DataFrame

# Ou en Python (avant l'importation)
import os
os.environ["DATA_FOLDER"] = r"D:\TradXPro\best_token_DataFrame"
```

### 2.2 Importer l'Adaptateur

```python
from threadx.data import UnifiedDataAdapter, create_adapter

# Option 1: Créer directement
adapter = UnifiedDataAdapter()

# Option 2: Via factory function
adapter = create_adapter()

# Option 3: Avec chemin personnalisé
adapter = UnifiedDataAdapter(data_folder="/custom/path/best_token_DataFrame")
```

---

## 3. Workflow Intégration - Étape par Étape

### 3.1 Avant le Téléchargement : Détecter les Gaps

```python
from threadx.data import UnifiedDataAdapter

adapter = UnifiedDataAdapter()
symbol = "BTCUSDC"
interval = "1h"

# Étape 1: Vérifier ce qui est manquant
gaps = adapter.get_gaps_to_download(symbol, interval, history_days=365)

print(f"Gaps détectés: {len(gaps)} période(s)")

if not gaps:
    print(f"✅ {symbol}/{interval} est complètement à jour")
else:
    print(f"📥 À télécharger: {len(gaps)} gap(s)")
    for i, (start_ms, end_ms) in enumerate(gaps, 1):
        print(f"  {i}. {start_ms} → {end_ms}")
```

### 3.2 Télécharger Seulement les Gaps

**Au lieu de télécharger la plage complète, télécharger uniquement les gaps :**

```python
from threadx.data import UnifiedDataAdapter
from your_download_module import download_candles_binance

adapter = UnifiedDataAdapter()
symbol = "BTCUSDC"
interval = "1h"

# Récupérer les gaps
gaps = adapter.get_gaps_to_download(symbol, interval)

if gaps:
    print(f"Téléchargement des {len(gaps)} gap(s)...")

    all_new_candles = []

    for start_ms, end_ms in gaps:
        print(f"  Téléchargement: {start_ms} → {end_ms}")

        # Télécharger du API (ex: Binance)
        candles = download_candles_binance(
            symbol=symbol,
            interval=interval,
            start_time=start_ms,
            end_time=end_ms
        )

        if candles:
            all_new_candles.extend(candles)
            print(f"    ✓ {len(candles)} candles téléchargées")

    if all_new_candles:
        print(f"\nFusion de {len(all_new_candles)} candles avec existantes...")
        result = adapter.merge_and_save(symbol, interval, all_new_candles)
        print(f"✅ Fusion complétée: {len(result)} total")
else:
    print("Pas de gaps, données à jour")
```

### 3.3 Charger les Données Existantes

```python
adapter = UnifiedDataAdapter()

# Format liste de dict (compatible unified_data)
candles_list = adapter.load_existing_candles("BTCUSDC", "1h", as_dataframe=False)
if candles_list:
    print(f"Chargé {len(candles_list)} candles")
    print(f"Premier: {candles_list[0]}")
    print(f"Dernier: {candles_list[-1]}")

# OU Format DataFrame (pour calculs internes)
df = adapter.load_existing_candles("BTCUSDC", "1h", as_dataframe=True)
if df is not None:
    print(f"DataFrame shape: {df.shape}")
    print(f"Index: {df.index[0]} → {df.index[-1]}")
```

### 3.4 Valider les Données

```python
adapter = UnifiedDataAdapter()

# Valider les données existantes
is_valid, message = adapter.validate_candles("BTCUSDC", "1h")

if is_valid:
    print(f"✅ {message}")
else:
    print(f"❌ Validation échouée: {message}")
```

### 3.5 Obtenir un Résumé Complet

```python
adapter = UnifiedDataAdapter()

status = adapter.get_data_status("BTCUSDC", "1h")

print(f"Statut pour {status['symbol']}/{status['interval']}:")
print(f"  Existe: {status['exists']}")
print(f"  Nombre de candles: {status['candle_count']}")

if status['exists']:
    print(f"  Plage: {status['date_range'][0]} → {status['date_range'][1]}")
    print(f"  Premier timestamp: {status['first_timestamp']}")
    print(f"  Dernier timestamp: {status['last_timestamp']}")
    print(f"  Gaps manquants: {len(status['gaps'])}")
    print(f"  Validé: {status['is_valid']} - {status['validation_msg']}")
```

---

## 4. Intégration dans unified_data_historique_with_indicators.py

### 4.1 Modification de `download_ohlcv()`

**Avant (télécharge tout) :**
```python
def download_ohlcv(symbol: str, interval: str, start_date: str, end_date: str):
    # ... convertir dates en timestamps
    # Télécharger TOUTE la plage du API
    candles = binance_client.get_historical_klines(symbol, interval, start_time=start_ms, end_time=end_ms)
    return candles
```

**Après (télécharge uniquement les gaps) :**
```python
from threadx.data import UnifiedDataAdapter

adapter = UnifiedDataAdapter()

def download_ohlcv(symbol: str, interval: str, start_date: str = None, end_date: str = None):
    """
    Télécharge OHLCV mais seulement pour les gaps détectés.

    Si start_date/end_date sont fournis, utiliser ces limites.
    Sinon, utiliser la plage par défaut (last 365 days).
    """

    # Déterminer history_days basé sur start_date
    if start_date:
        history_days = calculate_days_between(start_date, end_date)
    else:
        history_days = 365  # défaut

    # Étape 1: Récupérer les gaps existants
    gaps = adapter.get_gaps_to_download(symbol, interval, history_days)

    logger.info(f"Pour {symbol}/{interval}: {len(gaps)} gap(s) à télécharger")

    if not gaps:
        logger.info(f"✅ {symbol}/{interval} est à jour, pas de téléchargement")
        return None

    # Étape 2: Télécharger SEULEMENT les gaps
    all_candles = []

    for start_ms, end_ms in gaps:
        logger.info(f"Téléchargement gap: {start_ms} → {end_ms}")

        try:
            # Télécharger cette période spécifique du API
            candles = binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_ms,
                end_time=end_ms
            )

            if candles:
                all_candles.extend(candles)
                logger.info(f"  ✓ {len(candles)} candles téléchargées")

        except Exception as e:
            logger.error(f"Erreur téléchargement gap {start_ms}→{end_ms}: {e}")
            # Continuer avec le prochain gap
            continue

    return all_candles if all_candles else None
```

### 4.2 Modification de `verify_and_complete()`

**Avant (utilise logique locale) :**
```python
def verify_and_complete():
    # ... logique manuelle de vérification
    for symbol in symbols:
        existing = load_from_local_cache(symbol)
        if not existing:
            # Télécharger tout
            pass
```

**Après (utilise l'adaptateur) :**
```python
from threadx.data import UnifiedDataAdapter

adapter = UnifiedDataAdapter()

def verify_and_complete():
    """
    Vérifie et complète les données pour tous les symboles.
    Utilise UnifiedDataAdapter pour éviter les doublons.
    """

    symbols = ["BTCUSDC", "ETHUSDC", "ADAUSDC"]  # Vos symboles
    interval = "1h"

    for symbol in symbols:
        logger.info(f"Vérification: {symbol}/{interval}")

        # Étape 1: Vérifier l'état des données
        status = adapter.get_data_status(symbol, interval)

        if status['exists']:
            logger.info(f"  Données existantes: {status['candle_count']} candles")
            logger.info(f"  Plage: {status['date_range'][0]} → {status['date_range'][1]}")

            if status['gaps']:
                logger.info(f"  Gaps détectés: {len(status['gaps'])}")
            else:
                logger.info(f"  Aucun gap: données complètes")
                continue
        else:
            logger.info(f"  Aucune donnée existante: téléchargement complet requis")

        # Étape 2: Télécharger et fusionner
        new_candles = download_ohlcv(symbol, interval)

        if new_candles:
            logger.info(f"  Fusion en cours...")
            result = adapter.merge_and_save(symbol, interval, new_candles)

            if result:
                logger.info(f"  ✅ Total: {len(result)} candles")
            else:
                logger.error(f"  ❌ Erreur fusion")
```

### 4.3 Modification de la Conversion JSON→Parquet

**Avant :**
```python
def convert_to_parquet(symbol: str, interval: str):
    # Charger JSON
    df = pd.read_json(f"{symbol}_{interval}.json")
    # Convertir et sauvegarder Parquet
    df.to_parquet(f"{symbol}_{interval}.parquet")
```

**Après (utilise le format unifié) :**
```python
from threadx.data import UnifiedDataAdapter

adapter = UnifiedDataAdapter()

def convert_to_parquet(symbol: str, interval: str):
    """
    Convertit JSON→Parquet pour un symbole/intervalle.
    Utilise le système de normalisation ThreadX.
    """

    # Charger les données actuelles
    df = adapter.load_existing_candles(symbol, interval, as_dataframe=True)

    if df is None:
        logger.warning(f"Aucune donnée pour {symbol}/{interval}")
        return

    # Sauvegarder au format Parquet (automatiquement normalisé)
    saved_path = adapter.manager.save_data(symbol, interval, df, format="parquet")

    if saved_path:
        logger.info(f"Converti → Parquet: {saved_path}")

        # Optionnel: Supprimer l'ancien JSON
        json_path = adapter.manager.data_folder / f"{symbol}_{interval}.json"
        if json_path.exists():
            json_path.unlink()
            logger.info(f"JSON supprimé: {json_path}")
```

### 4.4 Ajout de Flags CLI

**Modification du main :**
```python
import argparse
from threadx.data import UnifiedDataAdapter

def main():
    parser = argparse.ArgumentParser()

    # Flags existants
    parser.add_argument("--symbols", default="BTCUSDC,ETHUSDC")
    parser.add_argument("--interval", default="1h")

    # NOUVEAUX flags pour la compatibilité ThreadX
    parser.add_argument(
        "--compatible-mode",
        action="store_true",
        help="Utiliser le mode compatible (évite doublons, lit avant de télécharger)"
    )
    parser.add_argument(
        "--validate-after-download",
        action="store_true",
        help="Valider les données après téléchargement"
    )
    parser.add_argument(
        "--convert-to-parquet",
        action="store_true",
        help="Convertir JSON→Parquet après fusion"
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        help="Chemin personnalisé au dossier de données (défaut: DATA_FOLDER env)"
    )

    args = parser.parse_args()

    # Initialiser l'adaptateur
    adapter = UnifiedDataAdapter(data_folder=args.data_folder)

    symbols = args.symbols.split(",")
    interval = args.interval

    for symbol in symbols:
        symbol = symbol.strip()

        if args.compatible_mode:
            # Mode compatib : utiliser l'adaptateur
            logger.info(f"\n📥 Mode compatible: {symbol}/{interval}")

            gaps = adapter.get_gaps_to_download(symbol, interval)

            if gaps:
                new_candles = download_ohlcv(symbol, interval)
                if new_candles:
                    result = adapter.merge_and_save(symbol, interval, new_candles)
                    logger.info(f"✅ Fusionné: {len(result)} total")

            # Validation optionnelle
            if args.validate_after_download:
                is_valid, msg = adapter.validate_candles(symbol, interval)
                logger.info(f"Validation: {msg}")

            # Conversion optionnelle
            if args.convert_to_parquet:
                convert_to_parquet(symbol, interval)
        else:
            # Mode standard (ancien comportement)
            download_all_and_save(symbol, interval)

if __name__ == "__main__":
    main()
```

**Utilisation :**
```bash
# Mode compatible avec validation et conversion
python unified_data_historique_with_indicators.py \
    --symbols BTCUSDC,ETHUSDC \
    --interval 1h \
    --compatible-mode \
    --validate-after-download \
    --convert-to-parquet

# Avec chemin personnalisé
python unified_data_historique_with_indicators.py \
    --compatible-mode \
    --data-folder /custom/path/best_token_DataFrame
```

---

## 5. Formats de Données

### 5.1 Format Liste de Dict (unified_data)

```python
# Format retourné par load_existing_candles(as_dataframe=False)
# et attendu par merge_and_save()

candles = [
    {
        "timestamp": 1697289600000,  # ms Unix timestamp
        "open": "27500.00",          # string (peut aussi être float)
        "high": "27650.50",
        "low": "27450.00",
        "close": "27600.25",
        "volume": "125.5",           # volume en token/satoshi
    },
    {
        "timestamp": 1697293200000,
        "open": "27600.25",
        # ...
    }
]
```

### 5.2 Format DataFrame (interne ThreadX)

```python
# Format utilisé en interne par load_existing_candles(as_dataframe=True)

# Index: DatetimeIndex en UTC
# Colonnes: open, high, low, close, volume (tous float)

#                     open      high       low     close    volume
# timestamp
# 2023-10-14 16:00 27500.00 27650.500 27450.00 27600.25  125.5000
# 2023-10-14 17:00 27600.25 27700.000 27550.00 27650.75  135.2500
# 2023-10-14 18:00 27650.75 27800.000 27600.00 27750.00  140.0000
```

### 5.3 Nomenclature des Fichiers

```
D:\TradXPro\best_token_DataFrame\
├── BTCUSDC_1h.json          # ou .parquet
├── BTCUSDC_5m.json
├── BTCUSDC_1d.json
├── ETHUSDC_1h.json
├── ETHUSDC_5m.json
└── ...
```

Patterns supportés :
- `{SYMBOL}_{INTERVAL}.json`
- `{SYMBOL}_{INTERVAL}.parquet`

Exemple : `BTCUSDC_1h`, `ETHUSDC_5m`, `ADAUSDC_1d`

---

## 6. Cas d'Usage Courants

### 6.1 Mise à Jour Quotidienne (Cron Job)

```python
import schedule
import time
from threadx.data import UnifiedDataAdapter

def daily_update():
    """Télécharge et fusionne les données manquantes chaque jour."""

    adapter = UnifiedDataAdapter()
    symbols = ["BTCUSDC", "ETHUSDC", "ADAUSDC"]
    interval = "1h"

    for symbol in symbols:
        try:
            gaps = adapter.get_gaps_to_download(symbol, interval)

            if gaps:
                print(f"Mise à jour {symbol}: {len(gaps)} gap(s)")
                new_candles = download_ohlcv(symbol, interval)

                if new_candles:
                    result = adapter.merge_and_save(symbol, interval, new_candles)
                    print(f"✅ {len(result)} candles total")
            else:
                print(f"✅ {symbol} à jour")

        except Exception as e:
            print(f"❌ Erreur {symbol}: {e}")
            # Continuer avec le suivant

# Schedule tous les jours à 00:00
schedule.every().day.at("00:00").do(daily_update)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 6.2 Chargement de Données pour Backtesting

```python
from threadx.data import UnifiedDataAdapter
from threadx.strategy import AmplitudeHunterStrategy

adapter = UnifiedDataAdapter()

# Charger les données
df = adapter.load_existing_candles("BTCUSDC", "1h", as_dataframe=True)

if df is not None:
    # Valider
    is_valid, msg = adapter.validate_candles("BTCUSDC", "1h")
    print(f"Validation: {msg}")

    if is_valid:
        # Utiliser avec une stratégie
        strategy = AmplitudeHunterStrategy("BTCUSDC", "1h")
        signals, stats = strategy.backtest(df, params_dict, initial_capital=10000)
        print(f"Sharpe: {stats.sharpe_ratio:.3f}")
```

### 6.3 Migration depuis JSON vers Parquet

```python
from threadx.data import UnifiedDataAdapter

adapter = UnifiedDataAdapter()

# Lister tous les fichiers JSON
json_files = [f for f in adapter.list_available_files() if f.endswith(".json")]

print(f"Migration de {len(json_files)} fichiers JSON vers Parquet...")

for filename in json_files:
    # Extraire symbol et interval
    symbol, interval = filename.replace(".json", "").rsplit("_", 1)

    # Charger et convertir
    df = adapter.load_existing_candles(symbol, interval, as_dataframe=True)

    if df is not None:
        saved = adapter.manager.save_data(symbol, interval, df, format="parquet")
        print(f"✅ {symbol}/{interval}: {saved}")
```

---

## 7. Troubleshooting

### 7.1 "Fichier de données non trouvé"

```
⚠️ DataCompatibilityManager initialisé: D:\TradXPro\best_token_DataFrame
❌ Aucune donnée existante pour BTCUSDC/1h
```

**Solution :**
- Vérifier que `DATA_FOLDER` pointe vers le bon répertoire
- Vérifier que les fichiers existent avec la bonne nomenclature
- Vérifier les permissions d'accès

```python
import os
from pathlib import Path

folder = os.environ.get("DATA_FOLDER")
print(f"DATA_FOLDER: {folder}")

path = Path(folder)
if path.exists():
    files = list(path.glob("*.json")) + list(path.glob("*.parquet"))
    print(f"Fichiers trouvés: {len(files)}")
    for f in files[:5]:
        print(f"  - {f.name}")
```

### 7.2 "Validation échouée: NaN dans colonnes critiques"

Les données contiennent des valeurs manquantes.

**Solution :**
```python
adapter = UnifiedDataAdapter()
df = adapter.load_existing_candles("BTCUSDC", "1h", as_dataframe=True)

# Vérifier les NaN
print(df.isna().sum())

# Supprimer les NaN
df = df.dropna()

# Resauvegarder
adapter.manager.save_data("BTCUSDC", "1h", df, format="json")
```

### 7.3 "Gaps détectés vides"

```
get_gaps_to_download("BTCUSDC", "1h") retourne []
```

**Cause :**
- Les données existantes couvrent toute la plage requise
- C'est normal ! Pas besoin de télécharger

**Vérifier :**
```python
status = adapter.get_data_status("BTCUSDC", "1h")
print(f"Plage: {status['date_range']}")
print(f"Gaps: {status['gaps']}")
```

### 7.4 "Deduplication issue: doublon timestamps"

Si les données fusionnées ont des doublons après merge_and_save():

```python
adapter = UnifiedDataAdapter()
df = adapter.load_existing_candles("BTCUSDC", "1h", as_dataframe=True)

# Vérifier les doublons
duplicates = df.index.duplicated()
if duplicates.any():
    print(f"Doublons trouvés: {duplicates.sum()}")

    # Nettoyer (garder le premier)
    df = df[~df.index.duplicated(keep='first')]

    # Resauvegarder
    adapter.manager.save_data("BTCUSDC", "1h", df, format="json")
    print("✅ Nettoyage complété")
```

---

## 8. Performance et Optimisations

### Gap Detection Performance
- **Chargement DataFrame** : ~50-200ms (selon taille)
- **Détection gaps** : ~10-50ms
- **Total avant téléchargement** : ~100-300ms

### Merge and Save Performance
- **Concat + deduplicate** : ~50-150ms
- **Validation** : ~20-50ms
- **Sauvegarde JSON** : ~100-500ms (selon taille)
- **Sauvegarde Parquet** : ~50-200ms
- **Total** : ~300-900ms pour ~1000 candles

### Optimisations

1. **Batch Downloads** : Télécharger plusieurs symbols en parallèle
```python
from concurrent.futures import ThreadPoolExecutor

def download_batch(symbols):
    adapter = UnifiedDataAdapter()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for symbol in symbols:
            gaps = adapter.get_gaps_to_download(symbol, "1h")
            if gaps:
                future = executor.submit(download_ohlcv, symbol, "1h")
                futures[symbol] = future

        for symbol, future in futures.items():
            candles = future.result()
            if candles:
                adapter.merge_and_save(symbol, "1h", candles)
```

2. **Cache DataFrame** : Éviter de recharger si déjà chargé
```python
cache = {}

def load_with_cache(symbol, interval):
    key = f"{symbol}/{interval}"
    if key not in cache:
        cache[key] = adapter.load_existing_candles(symbol, interval, as_dataframe=True)
    return cache[key]
```

---

## 9. Architecture Résumée

```
┌─────────────────────────────────────────┐
│  unified_data_historique_with_...py     │
│  (Programme de téléchargement)          │
└────────────────┬────────────────────────┘
                 │
                 ↓ importe
┌─────────────────────────────────────────┐
│  UnifiedDataAdapter                     │
│  - get_gaps_to_download()               │
│  - load_existing_candles()              │
│  - merge_and_save()                     │
│  - validate_candles()                   │
└────────────────┬────────────────────────┘
                 │
                 ↓ utilise
┌─────────────────────────────────────────┐
│  DataCompatibilityManager               │
│  - get_missing_time_ranges()            │
│  - load_existing_data()                 │
│  - merge_with_existing()                │
│  - validate_data()                      │
└────────────────┬────────────────────────┘
                 │
                 ↓ gère
┌─────────────────────────────────────────┐
│  D:\TradXPro\best_token_DataFrame\      │
│  ├── BTCUSDC_1h.json/.parquet           │
│  ├── ETHUSDC_1h.json/.parquet           │
│  └── ...                                │
└─────────────────────────────────────────┘
```

---

## 10. Summary

L'intégration fournit une solution complète pour :

✅ **Éviter les doublons** : Détection gaps + déduplication timestamp
✅ **Téléchargement efficace** : Seulement les périodes manquantes
✅ **Format unifié** : JSON et Parquet, nomenclature cohérente
✅ **Variables d'environnement** : DATA_FOLDER compatible
✅ **Validation automatique** : Vérification intégrité post-fusion
✅ **API simple** : 4 méthodes clés (gaps, load, merge, validate)

Le système est **produit et maintenable**, avec logs détaillés pour le debugging.

---

**Documentation Version**: 1.0.0
**Date**: Octobre 2025
**Framework**: ThreadX Data Compatibility Layer
