# Intégration unified_data_historique ↔ ThreadX
## Configuration et Stockage Partagé des Données OHLCV

---

## 1. Architecture Finale

```
D:\my_soft\mise_a_jour_dataframe\
└── unified_data_historique_with_indicators.py  ← Programme de téléchargement
    ├── Mode: GUI (Tkinter) ou CLI headless
    ├── Télécharge les tokens depuis Binance API
    ├── Détecte les données existantes AVANT téléchargement
    ├── Remplit les gaps manquants (pas de doublon)
    └── Stocke dans: D:\ThreadX_big\src\threadx\data
                      ↓ (chemin centralisé)
D:\ThreadX_big\src\threadx\data\
├── BTCUSDC_1h.json        ← Données OHLCV (ou .parquet)
├── BTCUSDC_5m.json
├── ETHUSDC_1h.json
├── ETHUSDC_5m.json
└── ...
    ↓
    Utilisé par ThreadX pour backtesting/analyse
    ├── AmplitudeHunterStrategy
    ├── Optimization avec presets
    └── Backtests Monte Carlo
```

---

## 2. Modification Effectuée

### Avant (Ancien Chemin)
```python
DEFAULT_DATA_DIR = r"D:\TradXPro\best_token_DataFrame"
```

### Après (Nouveau Chemin ThreadX)
```python
DEFAULT_DATA_DIR = r"D:\ThreadX_big\src\threadx\data"
DEFAULT_IDB_ROOT = r"D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet"
```

**Fichier modifié:** `D:\my_soft\mise_a_jour_dataframe\unified_data_historique_with_indicators.py`

---

## 3. Fonctionnement Actuel

Le programme `unified_data_historique_with_indicators.py` exécute déjà:

### ✅ Détection des données existantes
```python
# Avant de télécharger, le programme vérifie:
# - Si le fichier BTCUSDC_1h.json existe déjà
# - Quelle est la dernière bougie téléchargée
# - À partir de quand il faut télécharger les nouvelles données
```

**Fonctions clés:**
- `detect_missing()` → Identifie les gaps temporels
- `verify_and_complete()` → Remplit les gaps manquants
- `fetch_klines()` → Télécharge uniquement les gaps (pas tout)

### ✅ Pas de doublon
```python
# Les données sont fusionnées et dédupliquées par timestamp:
merged = list({c["timestamp"]: c for c in (data + add)}.values())
merged.sort(key=lambda x: x["timestamp"])
```

### ✅ Format unifié
```python
# Nomenclature cohérente:
# - BTCUSDC_1h.json  ← Format utilisé partout
# - ETHUSDC_5m.json
# - ADAUSDC_1d.json
```

---

## 4. Stockage des Fichiers

### Emplacement Central: `D:\ThreadX_big\src\threadx\data\`

```
D:\ThreadX_big\src\threadx\data\
├── BTCUSDC_1h.json          (Téléchargées par unified_data_historique)
├── BTCUSDC_1h.parquet       (Conversion JSON→Parquet automatique)
├── BTCUSDC_5m.json
├── BTCUSDC_5m.parquet
├── ETHUSDC_1h.json
├── ETHUSDC_1h.parquet
├── ETHUSDC_5m.json
├── ETHUSDC_5m.parquet
└── ... (autres tokens)
```

**Structure:**
- Les fichiers OHLCV sont stockés **directement** dans ce dossier
- Format: `{SYMBOL}_{INTERVAL}.{json|parquet}`
- Exemple: `BTCUSDC_1h.json` ou `BTCUSDC_1h.parquet`

---

## 5. Variables d'Environnement (Optionnel)

Tu peux override les chemins par défaut avec des variables d'environnement:

### Dans le fichier `.env`
```bash
# Dans D:\my_soft\mise_a_jour_dataframe\.env
DATA_FOLDER=D:\ThreadX_big\src\threadx\data
INDICATORS_DB_ROOT=D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet
JSON_PATH=D:\my_soft\mise_a_jour_dataframe\resultats_choix_des_100tokens.json
LOG_FILE=D:\my_soft\mise_a_jour_dataframe\unified_data_historique.log
```

### En Python (avant import)
```python
import os
os.environ["DATA_FOLDER"] = r"D:\ThreadX_big\src\threadx\data"
os.environ["INDICATORS_DB_ROOT"] = r"D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet"
```

---

## 6. Utilisation du Programme

### Mode GUI (Défaut)
```bash
cd D:\my_soft\mise_a_jour_dataframe
python unified_data_historique_with_indicators.py
```

L'interface affichera:
1. **Sélection des étapes** (Complet, Download, Vérification, etc.)
2. **Checkbox "Inclure RSI/MACD/VWAP/OBV/EMA200"**
3. **Bouton "Convertir maintenant (JSON→Parquet)"**
4. **Affichage du progrès** en temps réel
5. **Logs détaillés**

### Mode CLI Headless (si Tkinter indisponible)
```bash
# Télécharger et vérifier
python unified_data_historique_with_indicators.py --mode full

# Seulement conversion JSON→Parquet
python unified_data_historique_with_indicators.py --mode convert

# Seulement calcul d'indicateurs (avec RSI/MACD/VWAP/OBV)
python unified_data_historique_with_indicators.py --mode indicators --include-core

# Test hors-ligne
python unified_data_historique_with_indicators.py --selftest
```

---

## 7. Étapes du Pipeline

### 1️⃣ Sélection des Tokens (Mode "select")
```
CoinGecko API  → Top 100 par market cap
Binance API    → Top 100 par volume (24h)
Fusion         → Fichier JSON (resultats_choix_des_100tokens.json)
```

### 2️⃣ Téléchargement OHLCV (Mode "download")
```
Pour chaque token:
  - Vérifier si {SYMBOL}_{INTERVAL}.json existe
  - Si oui: Déterminer la dernière bougie
  - Télécharger UNIQUEMENT les données après la dernière bougie
  - Si non: Télécharger les 365 derniers jours
  - Fusionner + dédupliquer + sauvegarder
```

### 3️⃣ Vérification et Complétude (Mode "verify")
```
Pour chaque fichier dans DATA_FOLDER:
  - Charger les bougies
  - Détecter les gaps temporels
  - Pour chaque gap:
    - Télécharger les données manquantes
    - Fusionner avec les existantes
    - Sauvegarder
```

### 4️⃣ Conversion JSON→Parquet (Mode "convert")
```
Pour chaque {SYMBOL}_{INTERVAL}.json:
  - Convertir en DataFrame pandas
  - Vérifier que le .parquet est à jour (timestamp check)
  - Sauvegarder au format Parquet (compression snappy)
  - Effectué en parallèle (MAX_WORKERS = 4+)
```

### 5️⃣ Calcul d'Indicateurs (Mode "indicators")
```
Pour chaque {SYMBOL}_{INTERVAL}.{json|parquet}:
  - Charger les OHLCV (préfère Parquet si disponible)
  - Calculer les indicateurs demandés:
    • Bollinger Bands (periods: 10, 20, 50 / stds: 1.5, 2.0, 2.5)
    • ATR (periods: 14, 21)
    • EMA (periods: 20, 50, 200)
    • Vortex (periods: 14, 21)
    • RSI, MACD, VWAP, OBV (optionnel avec --include-core)
  - Sauvegarder chaque indicateur en Parquet
    Chemin: D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet\
            {SYMBOL}\{INTERVAL}\{INDICATOR}.parquet
```

---

## 8. Vérification de la Configuration

### Créer les répertoires
```bash
# Assure que le dossier de stockage existe
mkdir "D:\ThreadX_big\src\threadx\data"
mkdir "D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet"
```

### Test simple
```bash
# Vérifier que le programme démarre
cd D:\my_soft\mise_a_jour_dataframe
python unified_data_historique_with_indicators.py --selftest
# Résultat attendu: "SELFTEST OK"
```

### Vérifier les chemins en code
```python
# Dans D:\my_soft\mise_a_jour_dataframe\unified_data_historique_with_indicators.py
# Ligne ~72-84: Vérifier que DEFAULT_DATA_DIR pointe vers:
# r"D:\ThreadX_big\src\threadx\data"

import os
data_dir = os.path.normpath(os.getenv("DATA_FOLDER", r"D:\ThreadX_big\src\threadx\data"))
print(f"Stockage OHLCV: {data_dir}")
```

---

## 9. Première Utilisation

### Scénario A: Pas de données existantes
```
1. Lancer unified_data_historique_with_indicators.py
2. Mode: "Complet"
3. Cliquer "Play"
4. Attendre:
   - Sélection (tokens depuis API)
   - Téléchargement (365 jours pour chaque token/intervalle)
   - Vérification (checks d'intégrité)
   - Conversion (JSON→Parquet)
   - Indicateurs (si coché)
5. Les fichiers apparaissent dans D:\ThreadX_big\src\threadx\data\
```

### Scénario B: Mise à jour quotidienne
```
1. Les données existent déjà en D:\ThreadX_big\src\threadx\data\
2. Lancer le programme
3. Mode: "Complet"
4. Le programme:
   - Détecte BTCUSDC_1h.json (existe déjà)
   - Lit la dernière bougie (ex: timestamp = 2025-10-27 12:00)
   - Télécharge UNIQUEMENT depuis 2025-10-27 13:00 à maintenant
   - Fusionne (pas de doublon)
   - Sauvegarde
5. Résultat: mise à jour minimale, très rapide
```

---

## 10. Intégration avec ThreadX

### Utiliser les données dans ThreadX
```python
from threadx.data import UnifiedDataAdapter

# Les données téléchargées par unified_data_historique
# sont automatiquement accessibles à ThreadX

adapter = UnifiedDataAdapter()

# Charger les données
df = adapter.load_existing_candles("BTCUSDC", "1h", as_dataframe=True)

# Utiliser pour backtesting
strategy = AmplitudeHunterStrategy("BTCUSDC", "1h")
signals = strategy.generate_signals(df)
stats = strategy.backtest(df, params_dict)
```

### Synchronisation automatique
```
Modification en unified_data_historique
    ↓
Stockage en D:\ThreadX_big\src\threadx\data\{SYMBOL}_{INTERVAL}.json
    ↓
ThreadX lit automatiquement les mises à jour
    ↓
Backtesting utilise les données à jour
```

---

## 11. Troubleshooting

### "Le dossier D:\ThreadX_big\src\threadx\data\ n'existe pas"
```bash
# Solution: Créer le répertoire
mkdir "D:\ThreadX_big\src\threadx\data"
mkdir "D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet"
```

### "Permission denied: D:\ThreadX_big\src\threadx\data\"
```bash
# Vérifier les permissions
# Assurer que l'utilisateur peut lire/écrire dans ce dossier
```

### "Les fichiers se téléchargent toujours en entier"
```python
# Vérifier que le programme trouve bien les fichiers existants:
import os
folder = r"D:\ThreadX_big\src\threadx\data"
files = os.listdir(folder)
print(f"Fichiers trouvés: {len(files)}")
for f in files[:5]:
    print(f"  - {f}")

# Si rien n'apparaît, vérifier le chemin DATA_FOLDER
```

### "Erreur API Binance: rate limit"
```python
# Le programme a un délai de 0.2s entre les appels
# Si tu as toujours des erreurs rate limit:
time.sleep(1)  # Augmenter le délai dans fetch_klines()
```

---

## 12. Résumé des Changements

| Aspect | Avant | Après |
|--------|-------|-------|
| **Stockage OHLCV** | `D:\TradXPro\best_token_DataFrame` | `D:\ThreadX_big\src\threadx\data` |
| **Indicateurs** | `D:\TradXPro\indicators_db` | `D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet` |
| **Détection gaps** | ✅ Déjà existant | ✅ Fonctionnel |
| **Pas de doublon** | ✅ Déjà existant | ✅ Fonctionnel |
| **Format nomenclature** | `{SYMBOL}_{INTERVAL}.json` | ✅ Identique |
| **Synchronisation ThreadX** | ❌ Chemin différent | ✅ Chemin centralisé |

---

## 13. Prochaines Étapes

1. **Créer les répertoires**
   ```bash
   mkdir "D:\ThreadX_big\src\threadx\data"
   mkdir "D:\ThreadX_big\src\threadx\data\indicateurs_data_parquet"
   ```

2. **Lancer le premier téléchargement**
   ```bash
   cd D:\my_soft\mise_a_jour_dataframe
   python unified_data_historique_with_indicators.py
   ```

3. **Vérifier que les fichiers apparaissent**
   ```bash
   dir "D:\ThreadX_big\src\threadx\data\"
   # Devrait voir: BTCUSDC_1h.json, ETHUSDC_1h.json, etc.
   ```

4. **Utiliser dans ThreadX**
   ```python
   from threadx.data import UnifiedDataAdapter
   adapter = UnifiedDataAdapter()
   df = adapter.load_existing_candles("BTCUSDC", "1h", as_dataframe=True)
   # → Les données fraîches sont chargées automatiquement
   ```

---

**Date:** Octobre 2025
**Version:** 2.3 (ThreadX Integration)
**Status:** ✅ Production Ready
