# 🔧 Fix: Cache Python Streamlit

## ❌ Problème

Les modifications de `ma_crossover.py` ne sont pas prises en compte car:
- Streamlit ne recharge pas les modules automatiquement
- Python utilise les `.pyc` compilés en cache
- Les 100 runs d'optimisation ont tous échoué

## ✅ Solution Rapide

### Option 1: Script PowerShell (RECOMMANDÉ)

```powershell
cd D:\ThreadX_big
.\restart_streamlit.ps1
```

Ce script:
1. Nettoie tous les `__pycache__`
2. Supprime le cache Streamlit
3. Tue les processus existants
4. Relance Streamlit proprement

### Option 2: Commandes Manuelles

**Dans PowerShell:**

```powershell
# 1. Arrêter Streamlit (Ctrl+C dans le terminal)

# 2. Nettoyer __pycache__
cd D:\ThreadX_big
Get-ChildItem -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# 3. Vider cache Streamlit
Remove-Item -Path "$env:USERPROFILE\.streamlit\cache" -Recurse -Force -ErrorAction SilentlyContinue

# 4. Redémarrer Streamlit
streamlit run src\threadx\ui\app.py
```

---

## 🎯 Vérification Après Redémarrage

### Test 1: Vérifier Signature dans Python

```powershell
cd D:\ThreadX_big
python -c "import sys; sys.path.insert(0, 'src'); import importlib; import threadx.strategy.ma_crossover as mac; importlib.reload(mac); import inspect; print(inspect.signature(mac.MACrossoverStrategy.backtest))"
```

**Doit afficher:**
```
(..., fee_bps: float | None = None, slippage_bps: float | None = None, precomputed_indicators: dict | None = None) -> ...
```

### Test 2: Run Optimisation Rapide

**Dans Streamlit UI:**
1. Stratégie: MA_Crossover
2. Mode: Optimization
3. Params:
   ```
   fast_period: 10 (fixe)
   slow_period: 30 (fixe)
   stop_loss_pct: 2.0 (fixe)
   take_profit_pct: 4.0 (fixe)
   risk_per_trade: 0.01, 0.015, 0.02 (3 valeurs)
   ```
4. Total: **3 combinaisons** (~10 secondes)

**Résultat attendu:**
```
✅ 3/3 runs successful
Final equity: 8500-9500 USDC
Trades: 200-300
```

Si encore des erreurs → Problème plus profond (voir ci-dessous).

---

## 🐛 Troubleshooting Avancé

### Cas 1: Erreur Persiste Après Nettoyage

**Vérifier fichier source:**
```powershell
Select-String -Path "D:\ThreadX_big\src\threadx\strategy\ma_crossover.py" -Pattern "precomputed_indicators" | Select-Object -First 5
```

**Doit trouver:**
```
line 407:         precomputed_indicators: dict | None = None,
line 418:         precomputed_indicators: Indicateurs précalculés...
```

Si absent → Fichier non sauvé ou éditeur a annulé les changements.

### Cas 2: Import Depuis Mauvais Emplacement

**Vérifier emplacement module:**
```python
python -c "import sys; sys.path.insert(0, 'src'); from threadx.strategy import ma_crossover; print(ma_crossover.__file__)"
```

**Doit afficher:**
```
D:\ThreadX_big\src\threadx\strategy\ma_crossover.py
```

Si autre chemin → Conflit avec une autre installation.

### Cas 3: ProcessPoolExecutor Cache

L'optimization engine utilise `ProcessPoolExecutor`. Chaque worker a son propre cache.

**Solution:** Redémarrer **complètement** le terminal PowerShell:
```powershell
# Fermer le terminal
# Rouvrir nouveau terminal
cd D:\ThreadX_big
.\restart_streamlit.ps1
```

---

## 📊 Analyse Post-Fix

Une fois l'optimisation relancée avec succès, analyse:

### Vérifications Système

```python
import pandas as pd

# Charger résultats
df = pd.read_csv("CSV/[TIMESTAMP]_export.csv")

# Check 1: Success rate
total = len(df)
success = df['error'].isna().sum()
print(f"Success rate: {success}/{total} ({100*success/total:.1f}%)")

# Check 2: Stats présentes
has_stats = df['stats'].apply(lambda x: len(eval(x)) > 0).sum()
print(f"Configs avec stats: {has_stats}/{total}")

# Check 3: Range PnL
import json
pnls = []
for idx, row in df[df['error'].isna()].iterrows():
    stats = json.loads(row['stats'].replace("'", '"'))
    if 'total_pnl' in stats:
        pnls.append(stats['total_pnl'])

if pnls:
    print(f"PnL range: {min(pnls):.2f} → {max(pnls):.2f} USDC")
```

### Métriques Attendues

**Si moteur fonctionne:**
- Success rate: 100%
- PnL range: -2000 → +1000 USDC (-20% → +10%)
- Trades: 150-350 par config
- Drawdown: -10% → -30% (jamais > -50%)

**Si problème persiste:**
- Success rate < 100% → Autres bugs dans code
- Drawdown > -50% → Bug dans logic Numba
- 0 trades → Filtres trop restrictifs

---

## 🎓 Prévention Future

### Hook Pre-commit

Ajouter vérification signature:

```python
# .git/hooks/pre-commit
import inspect
from threadx.strategy import MACrossoverStrategy, BBAtrStrategy

required_params = ['fee_bps', 'slippage_bps', 'precomputed_indicators']

for strategy_class in [MACrossoverStrategy, BBAtrStrategy]:
    sig = inspect.signature(strategy_class.backtest)
    params = list(sig.parameters.keys())

    for req in required_params:
        if req not in params:
            print(f"❌ {strategy_class.__name__}.backtest() manque: {req}")
            exit(1)

print("✅ Signatures OK")
```

### Pytest Automatique

```python
# tests/test_strategy_interface.py
def test_backtest_signature():
    from threadx.strategy import MACrossoverStrategy
    import inspect

    sig = inspect.signature(MACrossoverStrategy.backtest)
    params = list(sig.parameters.keys())

    assert 'precomputed_indicators' in params
    assert 'fee_bps' in params
    assert 'slippage_bps' in params
```

---

**Étape suivante:** Exécute `restart_streamlit.ps1` et relance l'optimisation ! 🚀
