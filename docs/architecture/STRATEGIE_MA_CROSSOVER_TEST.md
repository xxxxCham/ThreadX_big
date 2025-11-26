# Stratégie MA Crossover - Validation Système

## 🎯 Objectif

Stratégie **simple et éprouvée** pour **valider le moteur de backtest**, pas pour optimiser les profits.

Si cette stratégie montre aussi un **drawdown -99%** ou des **incohérences**, alors le problème est **dans le moteur**, pas dans la logique de trading.

---

## 📋 Description de la Stratégie

### Règles d'Entrée

**LONG:**
- SMA rapide (10) croise **au-dessus** de SMA lente (30)

**SHORT:**
- SMA rapide (10) croise **en-dessous** de SMA lente (30)

### Règles de Sortie

**Exit sur:**
1. **Stop loss:** -2% du prix d'entrée (fixe)
2. **Take profit:** +4% du prix d'entrée (fixe)
3. **Signal inverse:** SMA rapide recroise dans l'autre sens
4. **Max hold:** 100 bars (~6h en 15m)

### Risk Management

- **Position sizing:** 1% du capital risqué par trade
- **Leverage:** 1.0x (PAS de levier)
- **Frais:** 4.5 bps (0.045%)
- **Slippage:** 0 bps

---

## ✅ Points de Validation

### 1. Génération de Trades

✅ **Attendu:** Entre 20-100 trades sur 2 mois (BTC 15m)
❌ **Problème:** 0 trades ou > 200 trades

### 2. Drawdown Maximum

✅ **Attendu:** Entre -10% et -30% maximum
⚠️  **Suspect:** -30% à -50%
❌ **BUG:** > -50% ou > -90%

**Pourquoi?**
- Stop loss fixe à -2%
- Risk 1% par trade
- Leverage 1.0x
- **Impossible** de perdre -99% avec ces paramètres!

### 3. Cohérence Capital

✅ **Attendu:**
```
Capital final = Capital initial + Total PnL
```

❌ **BUG si:**
```
|Capital final - (Initial + PnL)| > 1 USDC
```

### 4. Stops Loss Respectés

✅ **Attendu:** Aucun trade ne doit perdre plus de ~2.5%
- 2% stop loss
- 0.045% frais × 2 = 0.09%
- Marge slippage: 0.5%
- **Maximum théorique: -2.59%**

❌ **BUG si:** Des trades perdent -5%, -10% ou plus

### 5. Win Rate

✅ **Attendu:** 30-50% (stratégie MA classique)
❌ **BUG:** 0% ou 100%

---

## 🚀 Utilisation

### Exécuter le Test

```bash
cd D:\ThreadX_big
python test_ma_crossover.py
```

### Résultats Attendus

```
📊 RÉSULTATS BACKTEST
=========================

💰 Capital:
  Initial:        10,000.00 USDC
  Final:           9,500.00 USDC  ← -5% à +15% acceptable
  PnL:              -500.00 USDC (-5.00%)

📈 Performance:
  Total trades:          45  ← 20-100 OK
  Win rate:            40%  ← 30-50% OK
  Max DD:          -1,200 USDC (-12%)  ← < -30% OK

✅ Le moteur de calcul semble FONCTIONNEL
```

### Résultats Problématiques

```
📊 RÉSULTATS BACKTEST
=========================

💰 Capital:
  Initial:        10,000.00 USDC
  Final:              50.00 USDC  ← ❌ -99.5% !!
  PnL:            -9,950.00 USDC

⚠️  Max DD:       -9,980 USDC (-99.8%)  ← ❌ IMPOSSIBLE

❌ Des problèmes ont été détectés
```

---

## 🔍 Debugging

### Si DD > -50%

1. **Analyser les trades individuels:**
```python
trades_df = pd.read_csv("D:/ThreadX_big/CSV/trades.csv")
print(trades_df.sort_values("pnl").head(10))  # Pires pertes
```

2. **Vérifier les stops:**
```python
# Chaque trade devrait avoir:
# - stop_price != 0
# - |entry_price - stop_price| / entry_price ≈ 2%
```

3. **Vérifier la fermeture des positions:**
```python
# Durée maximale devrait être <= 100 bars
max_duration = (exit_time - entry_time).max()
```

### Si Capital Incohérent

```python
# Vérifier accumulation PnL
equity_cumsum = trades_df["pnl"].cumsum()
expected_capital = 10000 + equity_cumsum.iloc[-1]
```

### Si Win Rate = 0%

- Les stops sont probablement **jamais déclenchés**
- Ou la logique d'entrée est **trop restrictive**

---

## 📂 Fichiers Créés

```
ThreadX_big/
├── src/threadx/strategy/
│   └── ma_crossover.py          ← Stratégie MA Crossover
├── test_ma_crossover.py         ← Script de test
├── STRATEGIE_MA_CROSSOVER_TEST.md  ← Ce document
└── CSV/
    ├── test_ma_crossover_results.csv   ← Stats backtest
    └── test_ma_crossover_equity.csv    ← Courbe équité
```

---

## 🎓 Interprétation des Résultats

### Scénario 1: Tests Passent ✅

```
✅ Des trades générés
✅ Drawdown raisonnable (< 50%)
✅ Cohérence capital validée

Score: 3/3 checks passés
```

**Conclusion:** Le moteur de calcul fonctionne correctement.

**Action:** Le problème vient de la stratégie BB+ATR, pas du moteur.
→ Revoir les paramètres, les filtres, la logique de stops

---

### Scénario 2: Tests Échouent ❌

```
❌ Drawdown excessif (> 50%)
❌ Incohérence capital détectée

Score: 1/3 checks passés
```

**Conclusion:** Bug dans le moteur de backtest.

**Priorité investigation:**
1. **Position sizing:** `_backtest_loop_numba()` ligne ~210
2. **Stop loss check:** `_backtest_loop_numba()` ligne ~129
3. **Cash management:** Vérifier déductions/additions

---

## 🔧 Améliorations Futures

1. **Export détaillé des trades:**
   - Ajouter sauvegarde CSV avec tous les trades
   - Colonnes: entry_time, exit_time, side, pnl, stop_hit, tp_hit

2. **Calculs manuels de référence:**
   - Pour 1-2 trades, calculer manuellement le PnL attendu
   - Comparer avec les résultats du backtest

3. **Tests unitaires:**
   - Test avec 1 seul trade LONG
   - Test avec 1 seul trade SHORT
   - Test stop loss déclenché
   - Test take profit déclenché

---

## 📞 Support

Si les tests échouent et que tu as besoin d'aide pour identifier le bug:

1. **Envoyer les fichiers:**
   - `CSV/test_ma_crossover_results.csv`
   - `CSV/test_ma_crossover_equity.csv`
   - Logs de sortie du script

2. **Informations à inclure:**
   - Drawdown observé
   - Nombre de trades générés
   - Win rate
   - Message d'erreur éventuel

---

**Version:** 1.0.0
**Date:** 2025-11-13
**Auteur:** Claude Code Assistant
